#include "memory_plan.hpp"
#include "model.hpp"
#include <cassert>
#include <cstdio>
#include <utility>

using strata_frontend::Machine;
using strata_frontend::MemoryPlan;
using strata_frontend::ModelInfo;
using strata_frontend::plan_memory;

namespace {
struct Sizes { double experts, table; };
constexpr Sizes kIq3{42.9, 28.8};       // Flash-Next IQ3_XXS
constexpr Sizes kIq3S{50.3, 28.8};      // Flash-Next IQ3_S (upstream setup.py MODELS arena_gb)
constexpr Sizes kCoder{23.4, 28.8};     // Coder IQ1_M

MemoryPlan plan(const Machine& m, Sizes s = kIq3, const char* ple_io = "auto", const char* low_ram = "auto",
                long long ctx = 32768) {
    MemoryPlan p;
    std::string err;
    assert(plan_memory(m, s.experts, s.table, ctx, ple_io, low_ram, p, err) && err.empty());
    assert(p.notes.size() == (ctx > 32768 ? 3u : 2u));   // the KV cache is only mentioned above 32K
    return p;
}
bool refused(const Machine& m, const char* ple_io, const char* low_ram) {
    MemoryPlan p;
    std::string err;
    return !plan_memory(m, kIq3.experts, kIq3.table, 32768, ple_io, low_ram, p, err) && !err.empty();
}
}  // namespace

int main() {
    Machine big;                        // the RTX 4090 host: 82 GB holds experts + table, as the README measured
    big.ram_gb = 80.5;
    assert(plan(big).ple_io == "ram" && !plan(big).resident_experts);
    big.slow_disk = "network filesystem";   // the RunPod volume: still in RAM, read once
    assert(plan(big).ple_io == "ram");

    Machine host64;                     // the RTX 5070 Ti host: 64 GB, model on NVMe
    host64.ram_gb = 63.9;
    MemoryPlan p = plan(host64);
    assert(p.ple_io == "direct" && !p.resident_experts && p.notes[0].find("WARNING") == std::string::npos);
    assert(!plan(host64, kIq3S).resident_experts);  // IQ3_S: 60.3 GB fits 64 GB, with little else running
    assert(plan(host64, kCoder).ple_io == "ram");    // the Coder's 23.4 + 28.8 GB fit 64 GB on Linux
    host64.windows = true;
    p = plan(host64);
    assert(p.ple_io == "direct" && !p.resident_experts);
    p = plan(host64, kCoder);
    assert(p.ple_io == "direct" && !p.resident_experts);

    Machine winbig = big;               // never `ram` on Windows, asked or not
    winbig.windows = true;
    assert(plan(winbig).ple_io == "direct");
    assert(refused(winbig, "ram", "auto"));
    assert(plan(winbig, kIq3, "mmap").ple_io == "mmap");

    Machine hdd;                        // too little RAM for the table, slow disk: direct with a warning
    hdd.ram_gb = 64;
    hdd.slow_disk = "rotational disk (sdb)";
    p = plan(hdd);
    assert(p.ple_io == "direct" && p.notes[0].find("WARNING") != std::string::npos);

    Machine small;                      // below the experts plus 10 GB
    small.ram_gb = 48;
    assert(plan(small).resident_experts);           // IQ3_XXS: 52.9 GB
    assert(plan(small, kIq3S).resident_experts);    // IQ3_S: 60.3 GB
    assert(!plan(small, kCoder).resident_experts);  // Coder: 33.4 GB
    assert(!plan(small, kIq3, "auto", "off").resident_experts);
    assert(plan(host64, kIq3, "auto", "on").resident_experts);
    Machine wsl;                        // WSL's default half of 64 GB
    wsl.ram_gb = 31.3;
    assert(plan(wsl, kCoder).resident_experts);

    Machine unknown;                    // no RAM figure: the normal mode, the table on disk
    p = plan(unknown);
    assert(p.ple_io == "direct" && !p.resident_experts);

    // KV streaming above 32K positions: where RAM holds it, not in the low-RAM mode, not under WSL
    assert(plan(host64, kCoder).kv_resident == 0);                                   // 32K: nothing to stream
    assert(plan(host64, kCoder, "auto", "auto", 65536).kv_resident == 32768);        // the 5070 Ti host at 64K
    assert(plan(host64, kIq3, "auto", "auto", 131072).kv_resident == 32768);         // 42.9 + 10 + 1.8 GB fit
    assert(plan(small, kIq3, "auto", "auto", 65536).kv_resident == 0);               // low-RAM mode
    Machine wsl64 = host64;
    wsl64.windows = false;
    wsl64.wsl = true;
    assert(plan(wsl64, kCoder, "auto", "auto", 65536).kv_resident == 0);
    Machine tight;                      // the IQ3_XXS experts fit, the KV cache beside them does not
    tight.ram_gb = 53.5;
    assert(!plan(tight).resident_experts && plan(tight, kIq3, "auto", "auto", 262144).kv_resident == 0);
    assert(plan(unknown, kIq3, "auto", "auto", 65536).kv_resident == 0);

    assert(refused(host64, "fast", "auto"));
    assert(refused(host64, "auto", "maybe"));

    // which model a GGUF is: its general.name and expert count, and for the full model the file name
    using strata_frontend::identify_model;
    const std::string coder = "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf";
    const std::string full = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    const std::string full_s = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf";
    assert(identify_model("Qwen3.8-Flash-Next GSQ-RCO Coder", 256, coder) == &strata_frontend::kCoderIq1M);
    assert(identify_model("Qwen3.8-Flash-Next GSQ-RCO", 512, full) == &strata_frontend::kFlashNextIq3Xxs);
    assert(identify_model("Qwen3.8-Flash-Next GSQ-RCO", 512, full_s) == &strata_frontend::kFlashNextIq3S);
    assert(identify_model("Qwen3.8-Flash-Next GSQ-RCO", 512,
                          "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf") == nullptr);   // another size
    assert(identify_model("Qwen3.8-Flash-Next GSQ-RCO", 256, coder) == nullptr);              // not the Coder
    assert(identify_model("Qwen3.8-Flash-Next GSQ-RCO Coder", 512, full) == nullptr);

    const Machine here = strata_frontend::probe_machine(".");
    std::printf("this machine: %.1f GB of RAM%s%s\n", here.ram_gb, here.slow_disk.empty() ? "" : ", ",
                here.slow_disk.c_str());
    const std::pair<const ModelInfo*, Sizes> models[] = {{&strata_frontend::kFlashNextIq3Xxs, kIq3},
                                                          {&strata_frontend::kFlashNextIq3S, kIq3S},
                                                          {&strata_frontend::kCoderIq1M, kCoder}};
    for (const auto& [mi, s] : models) {
        std::printf("  %s:\n", mi->about);
        for (const std::string& n : plan(here, s).notes) std::printf("    %s\n", n.c_str());
    }
    std::puts("PASS: n-gram table and low-RAM rules, model identification");
}
