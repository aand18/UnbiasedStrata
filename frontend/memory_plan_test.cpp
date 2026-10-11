#include "memory_plan.hpp"
#include <cassert>
#include <cstdio>

using strata_frontend::Machine;
using strata_frontend::MemoryPlan;
using strata_frontend::plan_memory;

namespace {
MemoryPlan plan(const Machine& m, const char* ple_io = "auto", const char* low_ram = "auto") {
    MemoryPlan p;
    std::string err;
    assert(plan_memory(m, ple_io, low_ram, p, err) && err.empty());
    assert(p.notes.size() == 2);
    return p;
}
bool refused(const Machine& m, const char* ple_io, const char* low_ram) {
    MemoryPlan p;
    std::string err;
    return !plan_memory(m, ple_io, low_ram, p, err) && !err.empty();
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
    host64.windows = true;
    p = plan(host64);
    assert(p.ple_io == "direct" && !p.resident_experts);

    Machine winbig = big;               // never `ram` on Windows, asked or not
    winbig.windows = true;
    assert(plan(winbig).ple_io == "direct");
    assert(refused(winbig, "ram", "auto"));
    assert(plan(winbig, "mmap").ple_io == "mmap");

    Machine hdd;                        // too little RAM for the table, slow disk: direct with a warning
    hdd.ram_gb = 64;
    hdd.slow_disk = "rotational disk (sdb)";
    p = plan(hdd);
    assert(p.ple_io == "direct" && p.notes[0].find("WARNING") != std::string::npos);

    Machine small;                      // below the experts' 42.9 GB + 10 GB
    small.ram_gb = 48;
    assert(plan(small).resident_experts);
    assert(!plan(small, "auto", "off").resident_experts);
    assert(plan(host64, "auto", "on").resident_experts);

    Machine unknown;                    // no RAM figure: the normal mode, the table on disk
    p = plan(unknown);
    assert(p.ple_io == "direct" && !p.resident_experts);

    assert(refused(host64, "fast", "auto"));
    assert(refused(host64, "auto", "maybe"));

    const Machine here = strata_frontend::probe_machine(".");
    std::printf("this machine: %.1f GB of RAM%s%s\n", here.ram_gb, here.slow_disk.empty() ? "" : ", ",
                here.slow_disk.c_str());
    for (const std::string& n : plan(here).notes) std::printf("  %s\n", n.c_str());
    std::puts("PASS: n-gram table and low-RAM rules");
}
