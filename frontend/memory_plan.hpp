// frontend/memory_plan.hpp - two engine arguments the front end chooses for this machine: where the n-gram table is
// read from (--ple-io) and whether the experts fit in RAM (the low-RAM mode, --resident-experts).
//
// The low-RAM rule is upstream's (setup.py:3267 low_ram_needed).  The table rule is not: upstream keeps the table in
// RAM only for a rotational disk (setup.py:5365), but this fork's measured hosts hold it in RAM, and one of them reads
// the model from a network volume, where a table read from the file on every token would crawl.  So the table goes
// to RAM wherever it fits beside the experts, and is read from the disk (`direct`) everywhere else, with a warning
// when that disk is slow.  Windows has no `ram` mode (the engine refuses it: src/program/generate.cpp:1758).
//
// The sizes come from the model (frontend/model.hpp and the n-gram shard's size).  RAM is in GiB, as upstream's ram_gb()
// counts it.
#pragma once
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

namespace strata_frontend {

// The room the low-RAM rule leaves beside the experts for the OS, the engine's other buffers and this server
// (upstream's LOW_RAM_HEADROOM_GB); the room left beside experts + table, which the fork's 82 GB host runs with.
constexpr double kLowRamHeadroomGB = 10.0;
constexpr double kTableRamHeadroomGB = 8.0;

struct Machine {
    double ram_gb = 0;          // 0 = not known
    std::string slow_disk;      // why the n-gram table's disk is slow ("rotational disk sdb", "network filesystem")
    bool windows = false;
};

struct MemoryPlan {
    std::string ple_io;                 // direct | mmap | ram
    bool resident_experts = false;
    std::vector<std::string> notes;     // one log line each
};

inline std::string gb(double v) { return std::to_string((int) (v + 0.5)) + " GB"; }

// `ple_io` is auto|direct|mmap|ram and `low_ram` auto|on|off, as given on the command line.
// `experts_gb` and `table_gb` are the model's experts and n-gram table.
inline bool plan_memory(const Machine& m, double experts_gb, double table_gb, const std::string& ple_io,
                        const std::string& low_ram, MemoryPlan& p, std::string& err) {
    p = MemoryPlan{};
    if (ple_io == "direct" || ple_io == "mmap" || ple_io == "ram") {
        // say so now, not after the minutes the engine takes to reach the same refusal
        if (ple_io == "ram" && m.windows) { err = "--ple-io ram is not available on Windows; use direct"; return false; }
        p.ple_io = ple_io;
        p.notes.push_back("n-gram table: --ple-io " + ple_io + " (as asked)");
    } else if (ple_io != "auto") {
        err = "--ple-io must be auto, direct, mmap or ram";
        return false;
    } else if (!m.windows && m.ram_gb >= experts_gb + table_gb + kTableRamHeadroomGB) {
        p.ple_io = "ram";
        p.notes.push_back("n-gram table: --ple-io ram (" + gb(m.ram_gb) + " of RAM holds the experts and the " +
                          gb(table_gb) + " table)");
    } else {
        p.ple_io = "direct";
        if (m.slow_disk.empty())
            p.notes.push_back("n-gram table: --ple-io direct (read from the disk as needed)");
        else    // read from a slow disk at random, the table can stall a prompt for minutes (upstream #605)
            p.notes.push_back("n-gram table: --ple-io direct; WARNING: the model is on a " + m.slow_disk +
                              ", so prompts can stall for minutes: move it to an SSD");
    }

    if (low_ram == "on") {
        p.resident_experts = true;
        p.notes.push_back("experts: low-RAM mode, --resident-experts (as asked)");
    } else if (low_ram == "off") {
        p.notes.push_back("experts: all in RAM (low-RAM mode off, as asked)");
    } else if (low_ram != "auto") {
        err = "--low-ram must be auto, on or off";
        return false;
    } else if (m.ram_gb <= 0) {
        p.notes.push_back("experts: all in RAM (this machine's RAM is not known; --low-ram on if it runs short)");
    } else if (m.ram_gb < experts_gb + kLowRamHeadroomGB) {
        p.resident_experts = true;
        p.notes.push_back("experts: low-RAM mode, --resident-experts (" + gb(m.ram_gb) + " of RAM is less than the " +
                          gb(experts_gb) + " of experts plus " + gb(kLowRamHeadroomGB) + ")");
    } else {
        p.notes.push_back("experts: all in RAM (" + gb(m.ram_gb) + " of RAM)");
    }
    return true;
}

#if defined(_WIN32)
inline Machine probe_machine(const std::string&) {
    Machine m;
    m.windows = true;
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof s;
    if (GlobalMemoryStatusEx(&s)) m.ram_gb = (double) s.ullTotalPhys / 1073741824.0;
    return m;   // the disk only matters for a warning on Windows, and is not looked at yet
}
#else
namespace detail {
inline long long read_ll(const char* path) {
    long long v = -1;
    if (FILE* f = std::fopen(path, "r")) {
        if (std::fscanf(f, "%lld", &v) != 1) v = -1;   // cgroup v2 writes "max" for no limit
        std::fclose(f);
    }
    return v;
}
inline bool exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}
}  // namespace detail

// The RAM: the machine's, or a container's memory limit when that is smaller.  The disk: a network filesystem by
// its statfs type, a rotational disk from sysfs as upstream's rotational_disk() finds it; neither under WSL, whose
// virtual disk says rotational whatever is behind it.
inline Machine probe_machine(const std::string& ple_path) {
    Machine m;
    const long pages = ::sysconf(_SC_PHYS_PAGES), page = ::sysconf(_SC_PAGE_SIZE);
    double bytes = pages > 0 && page > 0 ? (double) pages * (double) page : 0;
    for (const char* limit : {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"}) {
        const long long v = detail::read_ll(limit);
        if (v > 0 && (bytes <= 0 || (double) v < bytes)) bytes = (double) v;
    }
    m.ram_gb = bytes / 1073741824.0;

    if (detail::exists("/proc/sys/fs/binfmt_misc/WSLInterop") || detail::exists("/dev/dxg")) return m;
    struct statfs fs;
    if (::statfs(ple_path.c_str(), &fs) == 0) {
        switch ((unsigned long) fs.f_type) {
            case 0x6969: case 0xFF534D42: case 0xFE534D42: case 0x517B: case 0x65735546: case 0x564C: case 0x6B414653:
                m.slow_disk = "network filesystem";   // NFS, CIFS, SMB2, SMB, FUSE (s3fs, gcsfuse...), NCP, AFS
                return m;
            default: break;
        }
    }
    struct stat st;
    if (::stat(ple_path.c_str(), &st) != 0) return m;
    const std::string dev = "/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev));
    char real[PATH_MAX];
    if (::realpath(dev.c_str(), real) == nullptr) return m;
    std::string q = real;
    for (int up = 0; up < 2; ++up) {    // a partition has its disk's queue
        const long long r = detail::read_ll((q + "/queue/rotational").c_str());
        if (r >= 0) {
            if (r == 1) m.slow_disk = "rotational disk (" + q.substr(q.rfind('/') + 1) + ")";
            break;
        }
        q.erase(q.rfind('/'));
    }
    return m;
}
#endif

}  // namespace strata_frontend
