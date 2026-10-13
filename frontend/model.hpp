// frontend/model.hpp - the models this front end serves, and which one a GGUF is.
//
// Two releases of Qwen3.8-Flash-Next, told apart by the GGUF itself: the Coder keeps 256 of the 512 experts and
// comes in one size (IQ1_M); the full model is served at IQ3_XXS and IQ3_S, which only the file name says.
// Anything else is refused, as before: the engine runs other packs, but nothing here was checked against them.
#pragma once
#include <cstdint>
#include <string>

namespace strata_frontend {

struct ModelInfo {
    const char* id;            // the name the API reports
    const char* about;
    double experts_gb;         // upstream's setup.py MODELS arena_gb: the experts the engine holds in RAM
    const char* spec_min_p;    // the draft probability a speculative window needs
    const char* const* engine_args;   // more engine arguments, nullptr-terminated
};

// IQ3_XXS: the values this fork was measured with.  IQ3_S: upstream setup.py MODELS (arena_gb 50.3, ram_gb 62:
// needs a 64 GB PC with little else running); the draft cutoff is carried over from IQ3_XXS, unmeasured.
// Coder IQ1_M: upstream's tested config on an RTX 5070 Ti, where
// the expert cache that adapts every round, slowly forgetting, gave 84.0 tok/s against the engine defaults' 72.6
// (8 prompts of 400 tokens, 32K context); --spec-min-p 0.5 measured the same as 0.70 (83.3).
inline constexpr const char* kNoArgs[] = {nullptr};
inline constexpr const char* kCoderArgs[] = {"--adapt-every", "1", "--adapt-swaps", "80", "--adapt-decay", "0.97", nullptr};
inline constexpr ModelInfo kFlashNextIq3Xxs{"qwen3.8-flash-next-iq3_xxs", "Qwen3.8-Flash-Next IQ3_XXS", 42.9, "0.5",
                                            kNoArgs};
inline constexpr ModelInfo kFlashNextIq3S{"qwen3.8-flash-next-iq3_s", "Qwen3.8-Flash-Next IQ3_S", 50.3, "0.5",
                                          kNoArgs};
inline constexpr ModelInfo kCoderIq1M{"qwen3.8-flash-next-coder-iq1_m", "Qwen3.8-Flash-Next Coder IQ1_M", 23.4, "0.70",
                                      kCoderArgs};

// `name` is general.name, `experts` qwen4exp.expert_count, `file` shard 1's path.  nullptr: not a served model.
inline const ModelInfo* identify_model(const std::string& name, int64_t experts, const std::string& file) {
    if (experts == 256 && name.find("Coder") != std::string::npos) return &kCoderIq1M;
    if (experts == 512 && name.find("Coder") == std::string::npos) {
        if (file.find("-IQ3_S-") != std::string::npos) return &kFlashNextIq3S;
        if (file.find("-IQ3_XXS-") != std::string::npos) return &kFlashNextIq3Xxs;
    }
    return nullptr;
}

}  // namespace strata_frontend
