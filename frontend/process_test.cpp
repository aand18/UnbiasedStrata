// The engine's child process (process.hpp): arguments arrive as they were sent, an exit is reported with its code,
// and on Windows the engine dies with the front end that started it.  No model or GPU.
#include "process.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using strata_frontend::ChildProcess;

namespace {
std::string read_all(ChildProcess& p) {
    std::string out;
    char buf[4096];
    for (long n; (n = p.read(buf, sizeof buf)) > 0;) out.append(buf, (size_t) n);
    std::string clean;   // Windows' text-mode stdout writes \r\n
    for (char c : out) if (c != '\r') clean += c;
    return clean;
}

bool exits_soon(ChildProcess& p, std::string& why) {
    for (int i = 0; i < 100; ++i) {
        if (!p.running(why)) return true;
        p.wait(50);
    }
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string role = argc > 1 ? argv[1] : "";
    if (role == "--echo") {                 // one argument per line
        for (int i = 2; i < argc; ++i) std::printf("%s\n", argv[i]);
        return 0;
    }
    if (role == "--exit") return std::atoi(argv[2]);
    if (role == "--sleep") {                // until killed, or 60 s
        for (int i = 0; i < 600; ++i) {
#if defined(_WIN32)
            Sleep(100);
#else
            usleep(100 * 1000);
#endif
        }
        return 0;
    }
#if defined(_WIN32)
    if (role == "--middle") {               // a front end that starts an engine, says its id and dies without a word
        ChildProcess engine;
        std::string err;
        if (!engine.start({argv[0], "--sleep"}, "", err)) return 2;
        std::printf("%lu\n", engine.id());
        std::fflush(stdout);
        TerminateProcess(GetCurrentProcess(), 9);   // no destructors, no quit(): as a crash
    }
#endif

    // arguments with spaces, quotes, backslashes and non-ASCII text arrive unchanged
    const std::vector<std::string> sent = {"plain", "with space", "C:\\path with\\trailing\\", "say \"hi\"",
                                           "back\\\\\"slash", "", "ünïcödé"};
    std::vector<std::string> args = {argv[0], "--echo"};
    args.insert(args.end(), sent.begin(), sent.end());
    {
        ChildProcess p;
        std::string err;
        assert(p.start(args, "", err));
        p.close_stdin();
        std::string expect;
        for (const std::string& s : sent) expect += s + "\n";
        const std::string got = read_all(p);
        if (got != expect) std::fprintf(stderr, "sent:\n%sgot:\n%s", expect.c_str(), got.c_str());
#if defined(_WIN32)
        // without the UTF-8 manifest the child's argv is in the ANSI code page: compare the ASCII ones only
        if (GetACP() != CP_UTF8) assert(got.substr(0, got.rfind("\n", got.size() - 2) + 1) ==
                                        expect.substr(0, expect.rfind("\n", expect.size() - 2) + 1));
        else
#endif
        assert(got == expect);
        std::string why;
        assert(exits_soon(p, why) && why == "engine exited with code 0");
    }

    // an exit is reported with its code
    {
        ChildProcess p;
        std::string err, why;
        assert(p.start({argv[0], "--exit", "17"}, "", err));
        assert(exits_soon(p, why));
        std::puts(why.c_str());
        assert(why == "engine exited with code 17");
        assert(!p.started());
    }

    // quit's last resort ends a child that does not stop by itself
    {
        ChildProcess p;
        std::string err, why;
        assert(p.start({argv[0], "--sleep"}, "", err));
        assert(p.running(why) && !p.wait(200));
        p.kill();
        assert(!p.running(why));
    }

#if defined(_WIN32)
    // an engine dies with its front end, even one that crashed: the Job Object's handle closes with the process
    {
        ChildProcess middle;
        std::string err, why;
        assert(middle.start({argv[0], "--middle"}, "", err));
        const unsigned long engine_id = std::strtoul(read_all(middle).c_str(), nullptr, 10);
        assert(engine_id != 0 && exits_soon(middle, why));
        std::printf("front end: %s\n", why.c_str());
        const HANDLE engine = OpenProcess(SYNCHRONIZE, FALSE, (DWORD) engine_id);
        assert(engine == nullptr || WaitForSingleObject(engine, 5000) == WAIT_OBJECT_0);
        if (engine != nullptr) CloseHandle(engine);
    }
    std::puts("PASS: arguments, exit codes, the forced stop, and the engine ending with a crashed front end");
#else
    std::puts("PASS: arguments, exit codes and the forced stop");
#endif
    return 0;
}
