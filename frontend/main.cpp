// frontend/main.cpp - UnbiasedStrata's front end: tokenizer, chat template and an OpenAI-style HTTP endpoint in
// front of the unmodified Strata engine.
//
// One binary, two roles.  Started normally it is the front end: it starts a copy of itself with `--engine`, which
// runs the engine's own program (frontend/engine_entry.cpp) in serve mode, and talks to it over a pipe in the
// engine's line protocol (token ids in, token ids out).  That is the process model upstream's Python server uses,
// so the engine runs exactly as upstream tested it.
//
// The tokenizer and the chat template come from llama.cpp (the pinned submodule): the GGUF's own vocabulary and
// its own Jinja template.  Nothing here opens an outgoing connection.
#include "engine.hpp"
#include "memory_plan.hpp"
#include "model.hpp"
#include "vision.hpp"
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

int strata_engine_main(int argc, char** argv);

namespace {

struct Options {
    std::string model;                  // shard 1 of the GGUF
    std::string pack;                   // the prepared pack directory
    std::string mtp;                    // the prepared draft-head directory
    std::string expert_profile;         // empty: the pack's expert-profile.bin
    std::string mmproj;
    std::string chat_template;
    std::string table;                   // the n-gram table (shard 2) when it is NOT beside the model: the
                                        // table is the same file for every quant of the model, so a trunk
                                        // from one quant can be served with another quant's table
    int vision_tokens = 1024;
    bool vision_gpu = false;            // --vision-gpu on: encode images on the GPU (needs a CUDA build)
    std::string vision_evict = "off";   // --vision-evict: off | immediate | N (free GPU VRAM after N s idle)
    int vision_cache_mb = 128;          // --vision-cache-mb: embedding cache cap in MiB (host RAM)
    long long vision_min_free_mib = 0;  // --vision-min-free-mib: GPU encode only above this free VRAM (0: off)
    long long vision_shed_mib = 1300;   // --vision-shed-mib: #533.1, with an elastic engine: how much VRAM the
                                        // expert cache sheds before an image batch (encode peak ~1.08 GiB + margin)
    std::string host = "127.0.0.1";
    int port = 8080;
    long long ctx = 32768;
    int threads = 0;                    // CPU expert workers; 0 = the engine's default (every physical core)
    std::string ple_io = "auto";        // the n-gram table: auto | direct | mmap | ram (memory_plan.hpp)
    std::string low_ram = "auto";       // the experts: auto | on (--resident-experts) | off
    std::vector<std::string> extra;     // after `--`: passed to the engine as they are
};

void usage() {
    std::fprintf(stderr,
        "usage: unbiased-strata --model SHARD1.gguf --pack DIR --mtp DIR [options] [-- engine options]\n\n"
        "  --model PATH    the model's first GGUF shard (...-00001-of-00002.gguf); the second must be beside it\n"
        "  --table PATH    the n-gram table (shard 2) when it is not beside the model: the table is the\n"
        "                  same file for every quant of this model, so one quant's trunk can be served\n"
        "                  with another quant's table\n"
        "  --pack DIR      the prepared pack for that model\n"
        "  --mtp DIR       the prepared draft head\n"
        "  --expert-profile PATH  the expert profile (default: the pack's expert-profile.bin)\n"
        "  --mmproj PATH   enable image input using this vision encoder (CPU by default)\n"
        "  --chat-template PATH  override the GGUF's Jinja chat template with this file\n"
        "  --vision-tokens N  maximum tokens per image (default 1024)\n"
        "  --vision-gpu on|off  encode images on the GPU (default off; needs a CUDA build;\n"
        "                  falls back to CPU with a warning when GPU init fails)\n"
        "  --vision-evict MODE  GPU VRAM policy (default off: resident; immediate: free the encoder\n"
        "                  after each request's images are encoded; N: free it after N s idle;\n"
        "                  needs --vision-gpu on; no-op for CPU encoding)\n"
        "  --vision-cache-mb N  embedding cache cap in MiB of host RAM (default 128)\n"
        "  --vision-min-free-mib N  encode on GPU only above N MiB free VRAM (default 0: off;\n"
        "                  below it, or with no GPU device, this batch uses CPU)\n"
        "  --vision-shed-mib N  with an elastic engine (its args after -- hold --vram-elastic): the headroom in\n"
        "                  MiB the expert cache sheds before an image batch and reclaims after the encoder is\n"
        "                  evicted (default 1300; 0: never send the engine's VRAM command)\n"
        "  --ctx N         context length in tokens (default 32768)\n"
        "  --threads N     CPU threads for the experts the GPU does not hold (default: every physical core,\n"
        "                  or half the CPU quota inside a container that has one)\n"
        "  --ple-io MODE   how the 29 GB n-gram table is read: auto (default: direct, or ram for a model on a\n"
        "                  rotational disk with RAM to spare), direct (from the SSD), mmap, or ram (not on Windows)\n"
        "  --low-ram MODE  auto (default: on with less than ~53 GB of RAM), on (--resident-experts: the GPU holds\n"
        "                  what it can and the rest is copied into RAM), or off (every expert in RAM)\n"
        "  --host ADDR     address to listen on (default 127.0.0.1)\n"
        "  --port N        port to listen on (default 8080)\n\n"
        "Serves POST /v1/chat/completions (OpenAI style, with streaming), GET /v1/models and GET /health.\n");
}

bool file_exists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);   // UTF-8 on Windows too: the manifest sets the process code page
}

// ---------------------------------------------------------------------------------------------------- the engine

using strata_frontend::Engine;
using strata_frontend::GenResult;

// ------------------------------------------------------------------------------------------------ text handling

/// The length of the longest prefix of `s` that does not end inside a UTF-8 character.
size_t utf8_complete_prefix(const std::string& s) {
    size_t i = s.size();
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char) s[i - 1];
        ++back;
        if ((c & 0xC0) == 0x80) { --i; continue; }             // a continuation byte: keep looking for its lead
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? i - 1 : s.size();
    }
    return s.size();
}

std::string make_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    char b[40];
    std::snprintf(b, sizeof b, "chatcmpl-%016llx", (unsigned long long) rng());
    return b;
}

long long now_s() {
    return (long long) std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void json_error(httplib::Response& res, int status, const std::string& message, const char* type) {
    common_json e = common_json::object();
    e["error"] = common_json::object();
    e["error"]["message"] = message;
    e["error"]["type"] = type;
    res.status = status;
    res.set_content(e.dump_safe(), "application/json");
}

// ------------------------------------------------------------------------------------------------- the request

struct Prepared {
    std::shared_ptr<strata_frontend::ImageFile> images;
    std::vector<llama_token> ids;
    std::string header;        // "GEN <max_new> key=value ... " - the ids follow
    std::string request_id;
    common_chat_parser_params parser;   // how to split the output into reasoning, answer and tool calls
    bool stream = false;
    long long max_new = 0;
};

struct Server {
    Options opt;
    const strata_frontend::ModelInfo* info = nullptr;   // the model being served (model.hpp)
    Engine engine;
    strata_frontend::Vision vision;
    // #533.1: an engine started with --vram-elastic (in its args) resizes its expert cache on the VRAM
    // command; the front end sheds it before an image batch and restores it after the encoder is evicted.
    bool vram_elastic = false;
    long long vram_base_reserve = 700;   // the engine's --vram-reserve-mib (what an image batch sheds from / restores to)
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    common_chat_templates_ptr tmpls;

    /// #533.1: one VRAM command to an elastic engine.  Errors are logged, never fatal: the cache simply stays
    /// smaller (or grows less), and the request runs as it would without the command.
    void vram_control(long long reserve_mib, const char* why) {
        const std::string e = engine.vram_command(reserve_mib);
        if (e.empty())
            std::fprintf(stderr, "unbiased-strata: expert cache %s for the image batch (VRAM %lld)\n", why, reserve_mib);
        else
            std::fprintf(stderr, "unbiased-strata: the engine's %s for the image batch failed: %s\n", why, e.c_str());
    }

    /// #533.1: whether the request carries images (scanned before the encode, to shed the cache ahead of it).
    static bool has_image_url(const common_json& messages) {
        for (auto& message : messages) {
            if (!message.contains("content") || !message.at("content").is_array()) continue;
            for (auto& part : message["content"])
                if (part.value("type", "") == "image_url") return true;
        }
        return false;
    }

    /// Parses an OpenAI chat request, renders the model's chat template and tokenizes the result.
    bool prepare(const common_json& body, Prepared& p, std::string& err) {
        if (!body.is_object() || !body.contains("messages")) { err = "the request needs a \"messages\" array"; return false; }
        common_chat_templates_inputs in;
        common_json messages = body.at("messages");
        std::vector<std::string> markers;
        std::vector<std::shared_ptr<strata_frontend::EncodedImage>> images;
        {
            std::lock_guard<std::mutex> lock(engine.mu);
            // #533.1: with an elastic engine the expert cache is shed around the encode - the encoder needs
            // ~1.1 GiB, and once it is evicted the cache grows back into the freed VRAM, so the generation
            // that follows runs with more of the experts on the GPU.  Both commands run under engine.mu,
            // between requests; without --vram-elastic in the engine's args nothing here runs.
            const bool shed = vram_elastic && opt.vision_shed_mib > 0 && has_image_url(messages);
            if (shed) vram_control(vram_base_reserve + opt.vision_shed_mib, "shed");
            try {
                for (auto& message : messages) {
                    if (!message.contains("content") || !message.at("content").is_array()) continue;
                    for (auto& part : message["content"]) {
                        if (part.value("type", "") != "image_url") continue;
                        if (images.size() >= 32) throw std::invalid_argument("at most 32 images per request");
                        const std::string url = part.at("image_url").at("url").get<std::string>();
                        images.push_back(vision.encode(url));
                        markers.push_back("UNBIASED_IMAGE_" + make_id() + "_" + std::to_string(markers.size()));
                        part = common_json::object();
                        part["type"] = "media_marker"; part["text"] = markers.back();
                    }
                }
                // Immediate evict: the embeddings are extracted above as plain vectors (and the
                // file is written from them below), so the encoder VRAM can go back before generation.
                vision.after_batch(!images.empty());
            } catch (...) {
                if (shed) vram_control(vram_base_reserve, "restore");
                throw;
            }
            if (shed) vram_control(vram_base_reserve, "restore");
        }
        in.messages = common_chat_msgs_parse_oaicompat(messages);
        if (body.contains("tools") && body.at("tools").is_array() && body.at("tools").size() > 0) {
            in.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
            if (body.contains("tool_choice") && body.at("tool_choice").is_string())
                in.tool_choice = common_chat_tool_choice_parse_oaicompat(body.at("tool_choice").get<std::string>());
            const auto caps = common_chat_templates_get_caps(tmpls.get());
            in.parallel_tool_calls = body.value("parallel_tool_calls", caps.at("supports_parallel_tool_calls"));
        }
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;   // reasoning goes to "reasoning_content"
        in.enable_thinking = true;
        // thinking: OpenAI's reasoning_effort, or the llama.cpp / vLLM convention chat_template_kwargs.enable_thinking
        if (body.contains("reasoning_effort") && body.at("reasoning_effort").is_string()) {
            const std::string effort = body.at("reasoning_effort").get<std::string>();
            if (effort == "none") in.enable_thinking = false;
            else in.chat_template_kwargs["reasoning_effort"] = common_json(effort).dump();
        }
        if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
            const common_json& kw = body.at("chat_template_kwargs");
            if (kw.contains("enable_thinking") && kw.at("enable_thinking").is_boolean())
                in.enable_thinking = kw.at("enable_thinking").get<bool>();
            if (kw.contains("reasoning_effort") && kw.at("reasoning_effort").is_string())
                in.chat_template_kwargs["reasoning_effort"] = kw.at("reasoning_effort").dump();
        }
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        std::string rendered = cp.prompt;
        if (images.empty()) {
            p.ids = common_tokenize(vocab, rendered, false, true);
        } else {
            const auto pad = common_tokenize(vocab, "<|image_pad|>", false, true);
            const auto literal_pad = common_tokenize(vocab, "<|image_pad|>", false, false);
            const auto start = common_tokenize(vocab, "<|vision_start|>", false, true);
            const auto end = common_tokenize(vocab, "<|vision_end|>", false, true);
            if (pad.size() != 1 || start.size() != 1 || end.size() != 1)
                throw std::runtime_error("model does not have the expected image special tokens");
            auto text_ids = [&](const std::string& text) {
                for (auto id : common_tokenize(vocab, text, false, true)) {
                    if (id == pad[0]) p.ids.insert(p.ids.end(), literal_pad.begin(), literal_pad.end());
                    else p.ids.push_back(id);
                }
            };
            size_t offset = 0; rendered.clear();
            for (size_t i = 0; i < images.size(); ++i) {
                auto at = cp.prompt.find(markers[i], offset);
                if (at == std::string::npos || cp.prompt.find(markers[i], at+markers[i].size()) != std::string::npos)
                    throw std::invalid_argument("chat template did not preserve each image exactly once");
                auto text = cp.prompt.substr(offset, at-offset);
                text_ids(text); rendered += text + "<|vision_start|>";
                p.ids.push_back(start[0]);
                p.ids.insert(p.ids.end(), images[i]->tokens, pad[0]);
                for (int j = 0; j < images[i]->tokens; ++j) rendered += "<|image_pad|>";
                p.ids.push_back(end[0]); rendered += "<|vision_end|>";
                offset = at + markers[i].size();
            }
            auto tail = cp.prompt.substr(offset); text_ids(tail); rendered += tail;
            p.images = std::make_shared<strata_frontend::ImageFile>();
            strata_frontend::write_images(*p.images, images);
        }
        if (p.ids.empty()) { err = "the prompt is empty"; return false; }

        p.parser = common_chat_parser_params(cp);
        p.parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        p.parser.parse_tool_calls = true;
        if (!cp.parser.empty()) p.parser.parser.load(cp.parser);

        // The engine refuses a request unless prompt + max_new + 8 <= context (generate.cpp, the GEN handler), so
        // the slots it keeps back are not ours to promise.  Without this a client that asks for "as much as fits"
        // (pi sends max_tokens 32768 on a 64K card) was refused on every prompt longer than ctx - max_tokens.
        constexpr long long kEngineReserve = 8;
        const long long n = (long long) p.ids.size();
        if (n + 1 + kEngineReserve > opt.ctx) {
            err = "the prompt is " + std::to_string(n) + " tokens, which does not fit the context of " +
                  std::to_string(opt.ctx);
            return false;
        }
        long long max_new = opt.ctx - n - kEngineReserve;
        for (const char* key : {"max_completion_tokens", "max_tokens"})
            if (body.contains(key) && body.at(key).is_number()) {
                const long long want = body.at(key).get<long long>();
                if (want >= 1) max_new = std::min(max_new, want);
                break;
            }
        p.max_new = max_new;
        p.stream = body.value("stream", false);

        // sampling: greedy unless the request sets a temperature above zero
        std::string keys;
        const double temperature = body.contains("temperature") && body.at("temperature").is_number()
                                       ? body.at("temperature").get<double>() : 0.0;
        if (temperature > 0.0) {
            const double top_p = body.contains("top_p") && body.at("top_p").is_number() ? body.at("top_p").get<double>() : 0.95;
            long long top_k = body.contains("top_k") && body.at("top_k").is_number() ? body.at("top_k").get<long long>() : 20;
            top_k = std::max(1LL, std::min(64LL, top_k));       // the engine's sampler takes 1..64
            unsigned long long seed;
            if (body.contains("seed") && body.at("seed").is_number()) seed = (unsigned long long) body.at("seed").get<long long>();
            else seed = std::random_device{}();
            char b[160];
            std::snprintf(b, sizeof b, " temperature=%g top_p=%g top_k=%lld seed=%llu", temperature, top_p, top_k, seed);
            keys += b;
            if (body.contains("min_p") && body.at("min_p").is_number()) {
                std::snprintf(b, sizeof b, " min_p=%g", body.at("min_p").get<double>());
                keys += b;
            }
        }
        for (const auto& kv : {std::pair<const char*, const char*>{"presence_penalty", "penalty_present"},
                               std::pair<const char*, const char*>{"frequency_penalty", "penalty_freq"},
                               std::pair<const char*, const char*>{"repeat_penalty", "penalty_repeat"}})
            if (body.contains(kv.first) && body.at(kv.first).is_number()) {
                char b[64];
                std::snprintf(b, sizeof b, " %s=%g", kv.second, body.at(kv.first).get<double>());
                keys += b;
                if (keys.find("penalty_last_n") == std::string::npos) keys += " penalty_last_n=64";
            }
        p.header = (p.images ? "GENI " : "GEN ") + std::to_string(max_new) + keys + " " +
                   (p.images ? p.images->path + " " : "");
        // Opt-in provenance capture: the exact first rendered request this
        // process serves, including token IDs and the engine sampling header.
        if (const char* path = std::getenv("SUGMEOFF_CAPTURE_FIRST_PROMPT")) {
            if (FILE* f = std::fopen(path, "wx")) {
                common_json capture = common_json::object();
                capture["request"] = body;
                capture["rendered_prompt"] = rendered;
                capture["token_ids"] = p.ids;
                capture["engine_header"] = p.header;
                const std::string text = capture.dump_safe();
                if (std::fwrite(text.data(), 1, text.size(), f) != text.size())
                    std::fprintf(stderr, "unbiased-strata: prompt capture write failed: %s\n", path);
                std::fclose(f);
            }
        }
        return true;
    }

    common_json usage_json(const GenResult& r) const {
        common_json u = common_json::object();
        u["prompt_tokens"] = r.prompt_n;
        u["completion_tokens"] = r.produced;
        u["total_tokens"] = r.prompt_n + r.produced;
        return u;
    }

    common_json timings_json(const GenResult& r) const {
        common_json t = common_json::object();
        t["prompt_n"] = r.prompt_n;
        t["prompt_ms"] = r.prompt_ms;
        t["prompt_per_second"] = r.prompt_ms > 0 ? 1000.0 * (double) r.prompt_n / r.prompt_ms : 0.0;
        t["predicted_n"] = r.produced;
        t["predicted_ms"] = r.decode_ms;
        t["predicted_per_second"] = r.decode_ms > 0 ? 1000.0 * (double) r.produced / r.decode_ms : 0.0;
        t["draft_n"] = r.draft_offered;
        t["draft_n_accepted"] = r.draft_accepted;
        return t;
    }

    /// Runs one request.  The text produced so far is parsed by llama.cpp's chat parser after every token, and
    /// `emit` gets what changed (reasoning, answer text, tool-call pieces); it returns false when the client has
    /// gone.  `final_msg` is the whole parsed message.  `keepalive`, when given, is called whenever 15 s pass
    /// without an emit (content and tool calls are published only once the message is complete, so a long answer
    /// is silent for minutes): the stream writes an SSE comment, which keeps clients with an idle timeout (pi:
    /// 5 minutes) attached and notices a client that has gone, so the generation is cancelled instead of
    /// running to the end for nobody.
    GenResult run(const Prepared& p, common_chat_msg& final_msg,
                  const std::function<bool(const std::vector<common_chat_msg_diff>&)>& emit,
                  const std::function<bool()>& keepalive = {}) {
        std::lock_guard<std::mutex> lk(engine.mu);
        std::fprintf(stderr, "unbiased-strata: request %s start: prompt=%zu header=%s\n",
                     p.request_id.c_str(), p.ids.size(), p.header.c_str());
        std::string text, pending;
        common_chat_msg msg;
        std::vector<std::string> call_ids;
        bool client_ok = true;
        auto last_write = std::chrono::steady_clock::now();
        auto update = [&](bool partial) {
            common_chat_msg now;
            try {
                now = common_chat_parse(text, partial, p.parser);
            } catch (const std::exception&) {
                if (partial) return;                            // not parseable yet: wait for more text
                now = common_chat_msg();
                now.role = "assistant";
                now.content = text;
            }
            // Partial tool JSON is reparsed and normalized as tokens arrive.
            // Escapes and incomplete arguments can change already-parsed text,
            // which cannot be retracted from an OpenAI stream. Publish content
            // and tool calls once, after the full message is parsed; reasoning
            // can still stream. This changes delivery, never generated tokens.
            if (partial) {
                now.content.clear();
                now.content_parts.clear();
                now.tool_calls.clear();
            }
            if (now.empty()) return;
            now.set_tool_call_ids(call_ids, [] { return "call_" + make_id().substr(9); });
            const std::vector<common_chat_msg_diff> diffs = common_chat_msg_diff::compute_diffs(msg, now);
            msg = now;
            if (client_ok && !diffs.empty()) {
                client_ok = emit(diffs);
                last_write = std::chrono::steady_clock::now();
            }
        };
        GenResult r = engine.generate(p.header, p.ids, [&](llama_token id) {
            if (llama_vocab_is_eog(vocab, id)) return client_ok;
            pending += common_token_to_piece(vocab, id, /*special=*/true);
            const size_t n = utf8_complete_prefix(pending);
            if (n > 0) {
                text.append(pending, 0, n);
                pending.erase(0, n);
                update(true);
            }
            if (client_ok && keepalive) {
                const auto now = std::chrono::steady_clock::now();
                if (now - last_write >= std::chrono::seconds(15)) {
                    client_ok = keepalive();
                    last_write = now;
                }
            }
            return client_ok;
        });
        text += pending;
        try {
            update(false);
        } catch (const std::exception& ex) {
            r.ok = false;
            r.error = std::string("response formatter failed: ") + ex.what();
            std::fprintf(stderr, "unbiased-strata: final response formatter failed: %s\n", ex.what());
        }
        final_msg = msg;
        if (final_msg.role.empty()) final_msg.role = "assistant";
        if (final_msg.tool_calls.empty() && text.find("<tool_call>") != std::string::npos) {
            // The parser accepts a message with no tool call even when the text holds one it could not
            // match (see third_party/llama.cpp/common/parsers/qwen3-coder.cpp); say so, with the text.
            std::fprintf(stderr, "unbiased-strata: request %s: the output contains <tool_call> but no tool call was "
                                 "parsed; the client sees finish=stop. raw-output-begin\n%s\n"
                                 "unbiased-strata: request %s raw-output-end\n",
                         p.request_id.c_str(), text.c_str(), p.request_id.c_str());
        }
        if (r.error.find("response formatter failed") == 0) {
            std::fprintf(stderr, "unbiased-strata: request %s formatter-input-begin\n%s\n"
                                 "unbiased-strata: request %s formatter-input-end\n",
                         p.request_id.c_str(), text.c_str(), p.request_id.c_str());
        }
        std::fprintf(stderr, "unbiased-strata: request %s end: ok=%d prompt=%lld generated=%lld finish=%s error=%s\n",
                     p.request_id.c_str(), r.ok ? 1 : 0, r.prompt_n, r.produced, r.finish.c_str(), r.error.c_str());
        vision.sweep();   // still under engine.mu: frees GPU VRAM when the encoder idled past its TTL
        return r;
    }

    static const char* finish_reason(const GenResult& r, const common_chat_msg& msg) {
        if (!msg.tool_calls.empty()) return "tool_calls";
        return r.finish == "stop" ? "stop" : "length";
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        if (!engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        const common_json body = common_json::parse_no_throw(req.body);
        if (body.is_discarded()) return json_error(res, 400, "the request body is not valid JSON", "invalid_request_error");
        auto prep = std::make_shared<Prepared>();
        std::string err;
        try {
            if (!prepare(body, *prep, err)) {
                // prepare may have encoded images before refusing: reclaim their VRAM in
                // immediate mode. Taken under engine.mu, like every other Vision call.
                std::lock_guard<std::mutex> lk(engine.mu);
                vision.after_batch(true);
                return json_error(res, 400, err, "invalid_request_error");
            }
        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lk(engine.mu);
            vision.after_batch(true);
            return json_error(res, 400, ex.what(), "invalid_request_error");
        }
        const std::string id = make_id();
        prep->request_id = id;
        const long long created = now_s();

        if (!prep->stream) {
            common_chat_msg msg;
            const GenResult r = run(*prep, msg, [](const std::vector<common_chat_msg_diff>&) { return true; });
            if (!r.ok) return json_error(res, 500, r.error, "server_error");
            common_json choice = common_json::object();
            choice["index"] = 0;
            choice["message"] = msg.to_json_oaicompat();
            choice["finish_reason"] = finish_reason(r, msg);
            common_json out = common_json::object();
            out["id"] = id;
            out["object"] = "chat.completion";
            out["created"] = created;
            out["model"] = info->id;
            out["choices"] = common_json::array();
            out["choices"].push_back(choice);
            out["usage"] = usage_json(r);
            out["timings"] = timings_json(r);
            res.set_content(out.dump_safe(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [this, prep, id, created](size_t, httplib::DataSink& sink) {
            auto chunk = [&](const common_json& delta, const char* finish, const GenResult* done) {
                common_json choice = common_json::object();
                choice["index"] = 0;
                choice["delta"] = delta;
                if (finish) choice["finish_reason"] = finish; else choice["finish_reason"] = nullptr;
                common_json out = common_json::object();
                out["id"] = id;
                out["object"] = "chat.completion.chunk";
                out["created"] = created;
                out["model"] = info->id;
                out["choices"] = common_json::array();
                out["choices"].push_back(choice);
                if (done) { out["usage"] = usage_json(*done); out["timings"] = timings_json(*done); }
                const std::string s = "data: " + out.dump_safe() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            common_json first = common_json::object();
            first["role"] = "assistant";
            first["content"] = "";
            bool ok = chunk(first, nullptr, nullptr);
            common_chat_msg msg;
            const GenResult r = run(*prep, msg, [&](const std::vector<common_chat_msg_diff>& diffs) {
                for (const common_chat_msg_diff& df : diffs) {
                    common_json d = common_json::object();
                    if (!df.reasoning_content_delta.empty()) d["reasoning_content"] = df.reasoning_content_delta;
                    if (!df.content_delta.empty()) d["content"] = df.content_delta;
                    if (df.tool_call_index != std::string::npos) {
                        common_json fn = common_json::object();
                        if (!df.tool_call_delta.name.empty()) fn["name"] = df.tool_call_delta.name;
                        fn["arguments"] = df.tool_call_delta.arguments;
                        common_json tc = common_json::object();
                        tc["index"] = (long long) df.tool_call_index;
                        if (!df.tool_call_delta.id.empty()) { tc["id"] = df.tool_call_delta.id; tc["type"] = "function"; }
                        tc["function"] = fn;
                        d["tool_calls"] = common_json::array();
                        d["tool_calls"].push_back(tc);
                    }
                    if (d.size() == 0) continue;
                    ok = ok && chunk(d, nullptr, nullptr);
                }
                return ok;
            }, [&]() {
                static const std::string ka = ": keepalive\n\n";   // an SSE comment: every client ignores it
                ok = ok && sink.write(ka.data(), ka.size());
                return ok;
            });
            if (ok) {
                if (r.ok) {
                    chunk(common_json::object(), finish_reason(r, msg), &r);
                } else {
                    common_json e = common_json::object();
                    e["error"] = common_json::object();
                    e["error"]["message"] = r.error;
                    e["error"]["type"] = "server_error";
                    const std::string s = "data: " + e.dump_safe() + "\n\n";
                    sink.write(s.data(), s.size());
                }
                static const std::string end = "data: [DONE]\n\n";
                sink.write(end.data(), end.size());
            }
            sink.done();
            return true;
        });
    }
};

Server* g_server = nullptr;
httplib::Server* g_http = nullptr;

void on_signal(int) {
    if (g_http) g_http->stop();
}

#if defined(_WIN32)
// Ctrl+C, Ctrl+Break, the console closing, logoff, shutdown.  Called on a thread of its own, so stopping the HTTP
// server here is safe.  On a close, logoff or shutdown Windows ends the process ~5 s after this returns, before
// Engine::quit's 10 s wait is over; the engine's Job Object then ends the engine with it.
BOOL WINAPI on_console_event(DWORD) {
    on_signal(0);
    return TRUE;
}
#endif

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// The CPU expert workers to ask the engine for when --threads is not given: 0 lets the engine use every physical
// core it sees.  Inside a container it sees the host's cores, not its own share, and that many workers on a small
// quota throttle one another; so with a cgroup CPU quota below the visible CPUs the quota decides.  The quota
// counts hardware threads and the engine wants cores, hence the halving when SMT is on.  Windows: always the engine's
// own count (src/kernels/cpu/pool.cpp counts physical cores there).
int default_threads() {
#if defined(_WIN32)
    return 0;
#else
    long long quota = 0, period = 0;
    char q[32] = "";
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {                       // cgroup v2
        if (std::fscanf(f, "%31s %lld", q, &period) == 2 && std::strcmp(q, "max") != 0) quota = std::atoll(q);
        std::fclose(f);
    } else if (FILE* f1 = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {  // cgroup v1
        if (std::fscanf(f1, "%lld", &quota) != 1) quota = 0;
        std::fclose(f1);
        if (FILE* f2 = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(f2, "%lld", &period) != 1) period = 0;
            std::fclose(f2);
        }
    }
    if (quota <= 0 || period <= 0) return 0;
    const long long cpus = (quota + period / 2) / period;
    if (cpus < 1 || cpus >= sysconf(_SC_NPROCESSORS_ONLN)) return 0;
    int smt = 0;
    if (FILE* f = std::fopen("/sys/devices/system/cpu/smt/active", "r")) {
        if (std::fscanf(f, "%d", &smt) != 1) smt = 0;
        std::fclose(f);
    }
    return (int) std::max<long long>(smt ? cpus / 2 : cpus, 1);
#endif
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "unbiased-strata: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") o.model = next("--model");
        else if (a == "--mmproj") o.mmproj = next("--mmproj");
        else if (a == "--table") o.table = next("--table");
        else if (a == "--chat-template") o.chat_template = next("--chat-template");
        else if (a == "--vision-tokens") o.vision_tokens = std::atoi(next("--vision-tokens").c_str());
        else if (a == "--vision-gpu") {
            const std::string v = next("--vision-gpu");
            if (v == "on") o.vision_gpu = true;
            else if (v == "off") o.vision_gpu = false;
            else { std::fprintf(stderr, "unbiased-strata: --vision-gpu must be on or off\n\n"); return false; }
        }
        else if (a == "--vision-evict") o.vision_evict = next("--vision-evict");
        else if (a == "--vision-cache-mb") o.vision_cache_mb = std::atoi(next("--vision-cache-mb").c_str());
        else if (a == "--vision-min-free-mib") o.vision_min_free_mib = std::atoll(next("--vision-min-free-mib").c_str());
        else if (a == "--vision-shed-mib") o.vision_shed_mib = std::atoll(next("--vision-shed-mib").c_str());
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--host") o.host = next("--host");
        else if (a == "--port") o.port = std::atoi(next("--port").c_str());
        else if (a == "--ctx") o.ctx = std::atoll(next("--ctx").c_str());
        else if (a == "--threads") o.threads = std::atoi(next("--threads").c_str());
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--low-ram") o.low_ram = next("--low-ram");
        else if (a == "--") {
            for (++i; i < argc; ++i) {
                const std::string e = argv[i];
                // one GPU only: the engine's multi-GPU and remote-expert modes are not part of this fork
                for (const char* no : {"--gpus", "--layer-split", "--split-device", "--peer-device", "--peer-slots",
                                       "--remote-expert", "--vision", "--batch"})
                    if (e.rfind(no, 0) == 0) {
                        std::fprintf(stderr, "unbiased-strata: %s is not supported here (use --mmproj for image input)\n", e.c_str());
                        std::exit(2);
                    }
                o.extra.push_back(e);
            }
        }
        else if (a == "-h" || a == "--help") return false;
        else { std::fprintf(stderr, "unbiased-strata: unknown option %s\n\n", a.c_str()); return false; }
    }
    return !o.model.empty() && !o.pack.empty() && !o.mtp.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--engine") == 0) {
#if defined(_WIN32)
        // the engine reads its requests with _read(0): in text mode a 0x1A byte would end them
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        return strata_engine_main(argc - 1, argv + 1);
    }

    Server srv;
    Options& o = srv.opt;
    if (!parse_args(argc, argv, o)) { usage(); return 2; }

    std::string chat_template_text;
    if (!o.chat_template.empty()) {
        FILE* f = std::fopen(o.chat_template.c_str(), "rb");
        if (f == nullptr) { std::fprintf(stderr, "unbiased-strata: cannot read chat template: %s\n", o.chat_template.c_str()); return 2; }
        char buf[8192];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) chat_template_text.append(buf, n);
        const bool ok = std::ferror(f) == 0;
        std::fclose(f);
        if (!ok) { std::fprintf(stderr, "unbiased-strata: cannot read chat template: %s\n", o.chat_template.c_str()); return 2; }
    }

    // every file the engine needs must already be here: nothing is fetched
    std::string shard2 = o.model;
    const size_t at = shard2.rfind("-00001-of-00002");
    if (at == std::string::npos) {
        std::fprintf(stderr, "unbiased-strata: --model must be the first shard (...-00001-of-00002.gguf)\n");
        return 2;
    }
    shard2.replace(at, 15, "-00002-of-00002");
    // The n-gram table is the SAME file for every quant of this model - the three quant repos ship a
    // byte-identical 28,800,138,432 B shard 2 - so a trunk from one quant can be served with another
    // quant's table.  --table says where it is; without it, the table is the one beside the model.
    if (!o.table.empty()) shard2 = o.table;
    const std::string profile = o.expert_profile.empty() ? o.pack + "/expert-profile.bin" : o.expert_profile;
    for (const std::string& f : {o.model, shard2, o.pack + "/index.txt", profile, o.mtp + "/experts.bin",
                                 o.mtp + "/draft_vocab.bin"})
        if (!file_exists(f)) { std::fprintf(stderr, "unbiased-strata: missing file: %s\n", f.c_str()); return 2; }
    if (o.ctx < 1024) { std::fprintf(stderr, "unbiased-strata: --ctx must be at least 1024\n"); return 2; }

    // which model this is, from the GGUF's header (the tensors are not read)
    {
        gguf_init_params gp{true, nullptr};
        auto gg = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_from_file(o.model.c_str(), gp), gguf_free);
        const int64_t kn = gg ? gguf_find_key(gg.get(), "general.name") : -1;
        const int64_t ke = gg ? gguf_find_key(gg.get(), "qwen4exp.expert_count") : -1;
        const std::string name = kn >= 0 ? gguf_get_val_str(gg.get(), kn) : "";
        const int64_t experts = ke >= 0 ? (int64_t) gguf_get_val_u32(gg.get(), ke) : 0;
        srv.info = strata_frontend::identify_model(name, experts, o.model);
        if (srv.info == nullptr) {
            std::fprintf(stderr, "unbiased-strata: %s (%lld experts) is not a model this server supports: Qwen3.8-Flash-Next "
                                 "IQ3_XXS, IQ3_S or the Coder IQ1_M\n", name.empty() ? o.model.c_str() : name.c_str(), (long long) experts);
            return 2;
        }
        std::fprintf(stderr, "unbiased-strata: model: %s\n", srv.info->about);
    }

    if (!o.mmproj.empty() && (!file_exists(o.mmproj) || o.vision_tokens < 16 || o.vision_tokens > 4096)) {
        std::fprintf(stderr, "unbiased-strata: --mmproj must exist and --vision-tokens must be 16..4096\n"); return 2;
    }
    strata_frontend::Vision::EvictMode evict_mode = strata_frontend::Vision::EvictMode::Resident;
    long long evict_secs = 0;
    if (o.vision_evict == "off") {
    } else if (o.vision_evict == "immediate") {
        evict_mode = strata_frontend::Vision::EvictMode::Immediate;
    } else {
        evict_secs = std::atoll(o.vision_evict.c_str());
        if (evict_secs < 1 || evict_secs > 86400 || o.vision_evict.empty() ||
            o.vision_evict.find_first_not_of("0123456789") != std::string::npos) {
            std::fprintf(stderr, "unbiased-strata: --vision-evict must be off, immediate or 1..86400\n"); return 2;
        }
        evict_mode = strata_frontend::Vision::EvictMode::Ttl;
    }
    if (evict_mode != strata_frontend::Vision::EvictMode::Resident && !o.vision_gpu) {
        std::fprintf(stderr, "unbiased-strata: --vision-evict needs --vision-gpu on (CPU encoding holds no VRAM)\n");
        return 2;
    }
    if (o.vision_cache_mb < 16 || o.vision_cache_mb > 4096) {
        std::fprintf(stderr, "unbiased-strata: --vision-cache-mb must be 16..4096\n"); return 2;
    }
    if (o.vision_min_free_mib < 0 || o.vision_min_free_mib > 65536) {
        std::fprintf(stderr, "unbiased-strata: --vision-min-free-mib must be 0..65536\n"); return 2;
    }
    if (o.vision_shed_mib < 0 || o.vision_shed_mib > 65536) {
        std::fprintf(stderr, "unbiased-strata: --vision-shed-mib must be 0..65536\n"); return 2;
    }
    strata_frontend::MemoryPlan mem;
    std::string merr;
    std::error_code ec;
    const double table_gb = (double) std::filesystem::file_size(shard2, ec) / 1e9;   // shard 2 is the n-gram table
    if (!strata_frontend::plan_memory(strata_frontend::probe_machine(shard2), srv.info->experts_gb, ec ? 0.0 : table_gb,
                                      o.ctx, o.ple_io, o.low_ram, mem, merr)) {
        std::fprintf(stderr, "unbiased-strata: %s\n", merr.c_str()); return 2;
    }
    for (const std::string& n : mem.notes) std::fprintf(stderr, "unbiased-strata: %s\n", n.c_str());
    // the engine, started first: loading takes minutes, and on POSIX it must be forked before any thread exists
    std::vector<std::string> eargs = {
        "unbiased-strata", "--engine", "--serve", "--pack", o.pack, "--native", o.model, "--ple-gguf", shard2,
        "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", srv.info->spec_min_p,
        "--mtp", o.mtp, "--max-context", std::to_string(o.ctx), "--ple-io", mem.ple_io};
    if (mem.resident_experts) eargs.push_back("--resident-experts");
    if (mem.kv_resident > 0) { eargs.push_back("--kv-resident"); eargs.push_back(std::to_string(mem.kv_resident)); }
    for (const char* const* a = srv.info->engine_args; *a != nullptr; ++a) eargs.push_back(*a);
    if (!o.mmproj.empty()) eargs.push_back("--vision");
    if (o.ctx > 8192) { eargs.push_back("--kv"); eargs.push_back("int8"); }
    if (o.threads <= 0) {
        o.threads = default_threads();
        if (o.threads > 0)
            std::fprintf(stderr, "unbiased-strata: this container's CPU quota gives %d expert threads (--threads N overrides)\n",
                         o.threads);
    }
    if (o.threads > 0) { eargs.push_back("--pool-workers"); eargs.push_back(std::to_string(o.threads)); }
    for (const std::string& e : o.extra) eargs.push_back(e);
    // #533.1: does the engine run elastic?  Only then does the front end drive its expert cache around image
    // batches; without --vram-elastic nothing below ever runs and the front end behaves exactly as before.
    for (size_t i = 0; i < eargs.size(); ++i) {
        if (eargs[i] == "--vram-elastic") srv.vram_elastic = true;
        else if (eargs[i] == "--vram-reserve-mib" && i + 1 < eargs.size())
            srv.vram_base_reserve = std::atoll(eargs[i + 1].c_str());
    }
    if (!srv.vram_elastic && o.vision_shed_mib != 1300)
        std::fprintf(stderr, "unbiased-strata: --vision-shed-mib has no effect: the engine's args (after --) need "
                             "--vram-elastic\n");
    if (srv.vram_elastic)
        std::fprintf(stderr, "unbiased-strata: elastic expert cache: image batches shed %lld MiB and restore to a "
                             "%lld MiB reserve (--vision-shed-mib, --vram-reserve-mib)\n",
                     o.vision_shed_mib, srv.vram_base_reserve);
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);   // a write to an engine that has gone fails instead (Windows has no SIGPIPE)
#endif
    std::string err;
    if (!srv.engine.start(eargs, err)) { std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str()); return 1; }

    // the tokenizer and the chat template, from the GGUF itself
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    srv.model = llama_model_load_from_file(o.model.c_str(), mp);
    if (srv.model == nullptr) {
        std::fprintf(stderr, "unbiased-strata: could not read the tokenizer from %s\n", o.model.c_str());
        srv.engine.quit();
        return 1;
    }
    srv.vocab = llama_model_get_vocab(srv.model);
    srv.tmpls = common_chat_templates_init(srv.model, chat_template_text);
    if (!o.mmproj.empty()) {
        try { srv.vision.init(o.mmproj, srv.model, o.threads, o.vision_tokens, o.vision_gpu, evict_mode, evict_secs,
                              o.vision_cache_mb, o.vision_min_free_mib); }
        catch (const std::exception& ex) {
            std::fprintf(stderr, "unbiased-strata: %s\n", ex.what()); srv.engine.quit(); return 1;
        }
    }

    std::fprintf(stderr, "unbiased-strata: loading the model (this takes a few minutes)...\n");
    if (!srv.engine.wait_ready(err)) {
        std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str());
        srv.engine.quit();
        return 1;
    }

    httplib::Server http;
    g_server = &srv;
    g_http = &http;
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = srv.engine.alive() ? 200 : 503;
        res.set_content(srv.engine.alive() ? "{\"status\":\"ok\"}" : "{\"status\":\"error\"}", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        if (!srv.engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        common_json m = common_json::object();
        m["id"] = srv.info->id;
        m["object"] = "model";
        m["owned_by"] = "local";
        m["context_length"] = srv.opt.ctx;
        m["input_modalities"] = common_json::array();
        m["input_modalities"].push_back("text");
        if (srv.vision.enabled()) m["input_modalities"].push_back("image");
        common_json out = common_json::object();
        out["object"] = "list";
        out["data"] = common_json::array();
        out["data"].push_back(m);
        res.set_content(out.dump_safe(), "application/json");
    });
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        srv.handle_chat(req, res);
    });
    http.set_payload_max_length(64 * 1024 * 1024);
    http.set_read_timeout(600, 0);
    http.set_write_timeout(600, 0);
#if defined(_WIN32)
    SetConsoleCtrlHandler(on_console_event, TRUE);
#else
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#endif

    if (!http.bind_to_port(o.host, o.port)) {
        std::fprintf(stderr, "unbiased-strata: could not listen on %s:%d\n", o.host.c_str(), o.port);
        srv.engine.quit();
        return 1;
    }
    std::fprintf(stderr, "unbiased-strata: ready on http://%s:%d (context %lld tokens)\n", o.host.c_str(), o.port, o.ctx);
    // The engine watchdog intentionally aborts a stuck child. Exit the HTTP
    // parent too so systemd's Restart=on-failure reloads the complete service.
    // Do not fork a replacement from this multithreaded HTTP process.
    std::atomic<bool> shutting_down{false};
    std::thread supervisor([&] {
        while (!shutting_down.load()) {
            std::string why;
            if (srv.engine.reap_exit(why)) {
                std::fprintf(stderr, "unbiased-strata: %s; exiting frontend for service restart\n", why.c_str());
                std::fflush(stderr);
                std::_Exit(1);
            }
            if (!srv.engine.alive()) {
                std::fprintf(stderr, "unbiased-strata: engine protocol failed; exiting frontend for service restart\n");
                std::fflush(stderr);
                std::_Exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    http.listen_after_bind();
    shutting_down.store(true);
    supervisor.join();

    std::fprintf(stderr, "unbiased-strata: stopping\n");
    srv.engine.quit();
    llama_model_free(srv.model);
    return 0;
}
