#pragma once
#include "process.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace strata_frontend {
struct GenResult {
    bool ok = false;
    std::string error;
    std::string finish = "length";
    long long produced = 0, prompt_n = 0, draft_accepted = 0, draft_offered = 0;
    double prompt_ms = 0, decode_ms = 0;
};

class Engine {
public:
    /// Starts `args` (args[0] is its argv[0]) as the engine; an empty `executable` is this program.
    bool start(const std::vector<std::string>& args, std::string& err, const std::string& executable = "") {
        return proc_.start(args, executable, err);
    }

    /// Reads the engine's start-up lines until it says READY.  Loading the model takes minutes.
    bool wait_ready(std::string& err) {
        std::string line;
        while (read_line(line)) {
            if (line.rfind("READY", 0) == 0) return true;
            if (line.rfind("ERR", 0) == 0) { err = line; return false; }
        }
        err = "the engine exited before it was ready (its messages are above)";
        return false;
    }

    /// One request.  `on_token` returns false when the client has gone: the engine is told to stop and the rest of
    /// its output for this request is read and dropped, so the next request starts clean.
    GenResult generate(const std::string& header, const std::vector<int32_t>& ids,
                       const std::function<bool(int32_t)>& on_token) {
        GenResult r;
        if (!alive_) { r.error = "the engine is not running"; return r; }
        std::string req = header;
        req.reserve(header.size() + ids.size() * 7 + 2);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) req += ',';
            req += std::to_string(ids[i]);
        }
        req += '\n';
        if (!send(req)) { r.error = "the engine is not running"; return r; }
        bool stopped = false;
        std::string callback_error;
        std::string line;
        while (read_line(line)) {
            if (line.size() > 2 && line[0] == 'T' && line[1] == ' ') {
                if (!stopped) {
                    bool keep_going = false;
                    try {
                        keep_going = on_token((int32_t) std::atoi(line.c_str() + 2));
                    } catch (const std::exception& ex) {
                        callback_error = ex.what();
                        std::fprintf(stderr, "unbiased-strata: response formatter failed; stopping and draining request: %s\n", ex.what());
                    }
                    if (!keep_going) {
                        stopped = true;
                        send("STOP\n");
                    }
                }
            } else if (line.rfind("DONE ", 0) == 0) {
                char finish[32] = {0};
                std::sscanf(line.c_str(), "DONE %lld %lld %lf %lf %31s %lld %lld", &r.produced, &r.prompt_n, &r.prompt_ms,
                            &r.decode_ms, finish, &r.draft_accepted, &r.draft_offered);
                r.finish = finish;
                if (r.prompt_n != (long long) ids.size()) {
                    r.error = "engine response prompt size mismatch: submitted " + std::to_string(ids.size()) +
                              ", returned " + std::to_string(r.prompt_n);
                    std::fprintf(stderr, "unbiased-strata: %s; refusing this response\n", r.error.c_str());
                    alive_ = false;
                    return r;
                }
                r.ok = callback_error.empty();
                if (!r.ok) r.error = "response formatter failed: " + callback_error;
                return r;
            } else if (line.rfind("ERR", 0) == 0) {
                r.error = line.size() > 4 ? line.substr(4) : "the engine reported an error";
                return r;
            }
            // PP (prompt progress), RESUME, REUSED and INFO lines need no action here
        }
        alive_ = false;
        r.error = "the engine exited during the request";
        return r;
    }

    /// Asks the engine to stop, and ends it after 10 s if it has not.
    void quit() {
        if (!proc_.started()) return;
        send("QUIT\n");
        proc_.close_stdin();
        if (!proc_.wait(10000)) proc_.kill();
    }

    /// #533.1: the engine's `VRAM [reserve_mib]` command between requests (elastic builds only; the front end
    /// decides).  The engine answers on stdout - "VRAM ..." with the new state, or "ERR ..." - read here so the
    /// reply never lands in the next request's token stream (generate() skips stray lines anyway, but this way
    /// the caller learns whether it worked).  Empty string: done; otherwise the error, never fatal.
    std::string vram_command(long long reserve_mib) {
        const std::string cmd = reserve_mib < 0 ? "VRAM\n" : "VRAM " + std::to_string(reserve_mib) + "\n";
        if (!send(cmd)) return "the engine is not running";
        std::string line;
        for (int i = 0; i < 64; ++i) {   // INFO/PP lines would be read and dropped here, as generate() does
            if (!read_line(line)) return "the engine exited during the VRAM command";
            if (line.rfind("VRAM", 0) == 0) return "";
            if (line.rfind("ERR", 0) == 0) return line.size() > 4 ? line.substr(4) : "the engine reported an error";
        }
        return "no VRAM answer from the engine";
    }

    bool reap_exit(std::string& why) {
        if (proc_.running(why)) return false;
        alive_ = false;
        return true;
    }

    bool alive() const { return alive_; }
    std::mutex mu;   // one request at a time: the engine holds one conversation

private:
    bool send(const std::string& s) {
        size_t off = 0;
        while (off < s.size()) {
            const long n = proc_.write(s.data() + off, s.size() - off);
            if (n <= 0) { alive_ = false; return false; }
            off += (size_t) n;
        }
        return true;
    }

    /// One line of the engine's output, without its line ending.
    bool read_line(std::string& line) {
        for (;;) {
            const size_t nl = rbuf_.find('\n', rpos_);
            if (nl != std::string::npos) {
                line.assign(rbuf_, rpos_, nl - rpos_);
                rpos_ = nl + 1;
                while (!line.empty() && line.back() == '\r') line.pop_back();
                return true;
            }
            rbuf_.erase(0, rpos_);
            rpos_ = 0;
            char chunk[65536];
            const long n = proc_.read(chunk, sizeof chunk);
            if (n <= 0) { alive_ = false; return false; }
            rbuf_.append(chunk, (size_t) n);
        }
    }

    ChildProcess proc_;
    std::string rbuf_;      // read from the engine, not yet returned as lines
    size_t rpos_ = 0;
    std::atomic<bool> alive_{true};
};

}
