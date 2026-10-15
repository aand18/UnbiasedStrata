#pragma once
#include "common.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#include "process.hpp"
#include <filesystem>
#else
#include <unistd.h>
#endif

namespace strata_frontend {
#if defined(_WIN32)
// Where the image embedding files go: `images` beside this program.  The engine takes the file's path in a
// space-separated request line (GENI, src/program/generate.cpp:5765), and the usual temp folder is under the user's
// name, which can hold a space; so the folder is one this server picks, as its 8.3 short form if the long one has a
// space.  Throws when neither form is free of spaces.
inline const std::wstring& image_dir() {
    static const std::wstring dir = [] {
        std::wstring exe(32768, L'\0');
        exe.resize(GetModuleFileNameW(nullptr, exe.data(), (DWORD) exe.size()));
        std::filesystem::path d = std::filesystem::path(exe).parent_path() / L"images";
        std::error_code ec;
        std::filesystem::create_directories(d, ec);
        std::wstring w = d.wstring();
        if (w.find(L' ') != std::wstring::npos) {
            std::wstring s(32768, L'\0');
            s.resize(GetShortPathNameW(w.c_str(), s.data(), (DWORD) s.size()));
            if (!s.empty()) w = s;
        }
        return w;
    }();
    if (dir.find(L' ') != std::wstring::npos)
        throw std::runtime_error("image input needs a folder without spaces in its path for its temporary files, and " +
                                 narrow(dir) + " has one: move unbiased-strata.exe to such a folder");
    return dir;
}

// Private per-request file, retained through streaming completion/cancellation.
struct ImageFile {
    std::string path;
    ImageFile() {
        wchar_t name[MAX_PATH];
        if (GetTempFileNameW(image_dir().c_str(), L"usi", 0, name) == 0)
            throw std::runtime_error("cannot create image embedding file");
        path = narrow(name);
    }
    ~ImageFile() { if (!path.empty()) DeleteFileW(widen(path).c_str()); }
    ImageFile(const ImageFile&) = delete;
    ImageFile& operator=(const ImageFile&) = delete;
};
#else
// Private per-request file, retained through streaming completion/cancellation.
struct ImageFile {
    std::string path;
    ImageFile() {
        char name[] = "/tmp/unbiased-strata-image-XXXXXX";
        int fd = mkstemp(name);
        if (fd < 0) throw std::runtime_error("cannot create image embedding file");
        close(fd); path = name;
    }
    ~ImageFile() { if (!path.empty()) unlink(path.c_str()); }
    ImageFile(const ImageFile&) = delete;
    ImageFile& operator=(const ImageFile&) = delete;
};
#endif

inline std::vector<unsigned char> image_data(const std::string& url) {
    const size_t comma = url.find(',');
    if (url.rfind("data:image/", 0) != 0 || comma == std::string::npos ||
        comma < 7 || url.substr(comma - 7, 7) != ";base64")
        throw std::invalid_argument("image_url must contain a base64 data:image URL (PNG, JPEG, BMP or GIF)");
    const std::string data = url.substr(comma + 1);
    if (data.empty() || data.size() > 32 * 1024 * 1024 || data.size() % 4)
        throw std::invalid_argument("empty, oversized or malformed base64 image");
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::array<int, 256> table; table.fill(-1);
    for (size_t i = 0; i < alphabet.size(); ++i) table[(unsigned char) alphabet[i]] = (int) i;
    std::vector<unsigned char> out; out.reserve(data.size() / 4 * 3);
    for (size_t i = 0; i < data.size(); i += 4) {
        int a = table[(unsigned char)data[i]], b = table[(unsigned char)data[i+1]];
        bool p2 = data[i+2] == '=', p3 = data[i+3] == '=';
        int c = p2 ? 0 : table[(unsigned char)data[i+2]], d = p3 ? 0 : table[(unsigned char)data[i+3]];
        if (a < 0 || b < 0 || c < 0 || d < 0 || (p2 && !p3) ||
            ((p2 || p3) && i + 4 != data.size()) || (p2 && (b & 15)) || (!p2 && p3 && (c & 3)))
            throw std::invalid_argument("malformed base64 image");
        out.push_back((unsigned char) ((a << 2) | (b >> 4)));
        if (!p2) out.push_back((unsigned char) ((b << 4) | (c >> 2)));
        if (!p3) out.push_back((unsigned char) ((c << 6) | d));
    }
    return out;
}
struct EncodedImage {
    int32_t tokens, nx, ny, width;
    std::vector<float> values;
};

// CPU encoding leaves the engine's full VRAM expert cache available; GPU encoding
// holds the ~1-2 GB vision weights in VRAM instead and encodes each image faster.
// Its callers take engine.mu, serializing image encoding and generation just like
// upstream, so no internal locking here: every method runs under engine.mu.
class Vision {
    mtmd_context* ctx_ = nullptr;
    int width_ = 0;
    int max_tokens_ = 1024;
    size_t cache_bytes_ = 0;
    std::map<std::string, std::shared_ptr<EncodedImage>> cache_;
    std::string path_;
    const llama_model* model_ = nullptr;   // owned by Server, outlives Vision
    int threads_ = 8;
    bool want_gpu_ = false;
    bool gpu_active_ = false;
public:
    enum class EvictMode { Resident, Immediate, Ttl };   // VRAM policy for the GPU encoder
private:
    EvictMode evict_mode_ = EvictMode::Resident;
    long long evict_secs_ = 0;             // Ttl only: free VRAM when idle this long
    std::chrono::steady_clock::time_point last_use_{};
    void free_ctx() {
        if (ctx_) { mtmd_free(ctx_); ctx_ = nullptr; gpu_active_ = false; }
    }
    void load_locked() {
        auto cp = mtmd_context_params_default();
        cp.print_timings = false; cp.warmup = false;
        cp.n_threads = threads_ > 0 ? threads_ : 8;
        cp.image_max_tokens = max_tokens_;
        gpu_active_ = false;
        if (want_gpu_) {
            cp.use_gpu = true;
            try {
                ctx_ = mtmd_init_from_file(path_.c_str(), model_, cp);
            } catch (const std::exception& ex) {
                ctx_ = nullptr;
                std::fprintf(stderr, "unbiased-strata: vision GPU init threw (%s), falling back to CPU\n", ex.what());
            }
            // mtmd falls back to CPU silently when no GPU backend is registered; only claim
            // GPU when one actually exists.
            if (ctx_ && mtmd_support_vision(ctx_)) {
                ggml_backend_t probe = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                if (!probe) probe = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
                if (probe) {
                    ggml_backend_free(probe);
                    gpu_active_ = true;
                } else {
                    std::fprintf(stderr, "unbiased-strata: vision GPU requested but no GPU backend: using CPU\n");
                }
            } else {
                if (ctx_) { mtmd_free(ctx_); ctx_ = nullptr; }
                std::fprintf(stderr, "unbiased-strata: vision GPU init failed, falling back to CPU\n");
            }
        }
        if (!ctx_) {
            cp.use_gpu = false;
            ctx_ = mtmd_init_from_file(path_.c_str(), model_, cp);
        }
        if (!ctx_ || !mtmd_support_vision(ctx_)) throw std::runtime_error("cannot load vision encoder: " + path_);
        last_use_ = std::chrono::steady_clock::now();
        std::string mode;
        if (want_gpu_ && evict_mode_ == EvictMode::Immediate) mode = " (evict after each batch)";
        if (want_gpu_ && evict_mode_ == EvictMode::Ttl) mode = " (evict after " + std::to_string(evict_secs_) + " s idle)";
        std::fprintf(stderr, "unbiased-strata: vision encoder on %s%s\n", gpu_active_ ? "GPU" : "CPU", mode.c_str());
    }
public:
    ~Vision() { if (ctx_) mtmd_free(ctx_); }
    bool enabled() const { return ctx_ != nullptr || !path_.empty(); }
    bool using_gpu() const { return gpu_active_; }
    void init(const std::string& path, const llama_model* model, int threads, int max_tokens,
              bool want_gpu = false, EvictMode mode = EvictMode::Resident, long long evict_secs = 0) {
#if defined(_WIN32)
        (void) image_dir();   // refuse at start, not on the first image, when the folder cannot be used
#endif
        gguf_init_params gp{true, nullptr};
        auto gg = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_from_file(path.c_str(), gp), gguf_free);
        auto key = gg ? gguf_find_key(gg.get(), "clip.vision.projection_dim") : -1;
        if (key < 0) throw std::runtime_error("vision encoder has no projection_dim");
        width_ = (int) gguf_get_val_u32(gg.get(), key);
        if (width_ != 2560) throw std::runtime_error("vision projector does not match Flash-Next's 2560 embedding width");
        max_tokens_ = max_tokens;
        path_ = path; model_ = model; threads_ = threads;
        want_gpu_ = want_gpu; evict_mode_ = mode; evict_secs_ = evict_secs < 0 ? 0 : evict_secs;
        load_locked();
    }
    // Free the GPU/CPU context but keep the embedding cache (sysRAM floats, backend
    // independent) so a reload still skips identical images.
    void evict() { free_ctx(); }
    // End of an encode batch (or a failed prepare) under engine.mu: in Immediate mode the
    // embeddings are already extracted as plain vectors, so the VRAM can go back at once.
    // No-op for Resident/Ttl and when nothing is loaded.
    void after_batch(bool had_images) {
        if (evict_mode_ != EvictMode::Immediate || !had_images || !ctx_ || !gpu_active_) return;
        std::fprintf(stderr, "unbiased-strata: vision batch done, evicting encoder from VRAM\n");
        free_ctx();
    }
    // Called after each request under engine.mu: frees VRAM when the encoder has been
    // idle past its TTL, or whenever anything is still loaded in Immediate mode (a path
    // the batch hook missed). No-op for Resident mode and CPU fallback.
    void sweep() {
        if (!ctx_ || !gpu_active_) return;
        if (evict_mode_ == EvictMode::Immediate) {
            std::fprintf(stderr, "unbiased-strata: vision sweep evicting encoder from VRAM\n");
            free_ctx();
            return;
        }
        if (evict_mode_ != EvictMode::Ttl) return;
        auto idle = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - last_use_).count();
        if (idle >= evict_secs_) {
            std::fprintf(stderr, "unbiased-strata: vision encoder idle %lld s, evicting from VRAM\n", (long long) idle);
            free_ctx();
        }
    }
    std::shared_ptr<EncodedImage> encode(const std::string& url) {
        if (path_.empty()) throw std::invalid_argument("image input requires starting UnbiasedStrata with --mmproj PATH");
        if (!ctx_) load_locked();   // reloaded after an eviction; the embedding cache below survives it
        auto bytes = image_data(url);
        auto t0 = std::chrono::steady_clock::now();
        auto bw = mtmd_helper_bitmap_init_from_buf(ctx_, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(bw.bitmap, mtmd_bitmap_free);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
        if (!bitmap || mtmd_bitmap_is_audio(bitmap.get())) throw std::invalid_argument("cannot decode image; use PNG, JPEG, BMP or GIF");
        const char* hash = mtmd_bitmap_get_id(bitmap.get());
        if (!hash || !*hash) throw std::runtime_error("image decoder did not supply a content hash");
        std::string id(hash);
        auto cached = cache_.find(id);
        if (cached != cache_.end()) return cached->second;
        auto chunks = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>(mtmd_input_chunks_init(), mtmd_input_chunks_free);
        std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bm[] = {bitmap.get()};
        if (mtmd_tokenize(ctx_, chunks.get(), &txt, bm, 1)) throw std::invalid_argument("image preprocessing failed");
        const mtmd_input_chunk* image = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto ch = mtmd_input_chunks_get(chunks.get(), i);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (image) throw std::invalid_argument("multiple image chunks are unsupported for this model");
                image = ch;
            }
        }
        if (!image) throw std::invalid_argument("no image tokens produced");
        size_t n = mtmd_input_chunk_get_n_tokens(image);
        if (!n || n > (size_t)max_tokens_) throw std::invalid_argument("image token count exceeds configured limit");
        if (mtmd_encode_chunk(ctx_, image)) throw std::runtime_error("vision encoding failed");
        auto last = mtmd_image_tokens_get_decoder_pos(mtmd_input_chunk_get_tokens_image(image), 0, n-1);
        int nx = (int)last.x + 1, ny = (int)last.y + 1;
        if ((size_t)nx * ny != n) throw std::runtime_error("image grid is not rectangular");
        auto result = std::make_shared<EncodedImage>();
        result->tokens = (int32_t)n; result->nx = nx; result->ny = ny; result->width = width_;
        auto values = mtmd_get_output_embd(ctx_);
        if (!values) throw std::runtime_error("encoder returned no image embeddings");
        result->values.assign(values, values+n*width_);
        size_t size = result->values.size()*sizeof(float);
        if (cache_bytes_ + size > 128*1024*1024) { cache_.clear(); cache_bytes_ = 0; }
        cache_[id] = result; cache_bytes_ += size;
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count();
        last_use_ = std::chrono::steady_clock::now();
        std::fprintf(stderr, "unbiased-strata: encoded image %.0f ms, %d tokens (%dx%d), %s\n", ms, result->tokens, nx, ny,
                     gpu_active_ ? "GPU" : "CPU");
        return result;
    }
};
inline void write_images(const ImageFile& file, const std::vector<std::shared_ptr<EncodedImage>>& images) {
    struct Closer { void operator()(FILE* p) const { std::fclose(p); } };
    auto f = std::unique_ptr<FILE, Closer>(std::fopen(file.path.c_str(), "wb"));
    if (!f) throw std::runtime_error("cannot open image embedding file");
    for (auto& im : images) {
        int32_t hdr[] = {0x31455653, im->tokens, im->nx, im->ny, im->width};
        if (std::fwrite(hdr, sizeof(hdr), 1, f.get()) != 1 ||
            std::fwrite(im->values.data(), sizeof(float), im->values.size(), f.get()) != im->values.size())
            throw std::runtime_error("short write to image embedding file");
    }
    if (std::fflush(f.get())) throw std::runtime_error("cannot flush image embedding file");
}
} // namespace strata_frontend
