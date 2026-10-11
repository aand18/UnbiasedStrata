# Porting UnbiasedStrata to Windows

Oct 12, 2026 · @Rob · supersedes `windows-port.md`

## Summary

A native Windows build is a modest port. Almost all of the Linux-only code is in the small C++ frontend (about 1,300 lines with its tests); the engine and its CMake build still carry upstream Strata's Windows support (`_WIN32` paths in 26 files under `src/` and `include/`, MSVC flags), and upstream ships and runs that engine on Windows today. The work is:

- replace the frontend's POSIX process handling (`fork`, pipes, `getline`, `waitpid`, signals) with Win32 equivalents
- move the image embeddings temp file off `/tmp` and make its path safe in the engine protocol
- stop the frontend passing `--ple-io ram`, which the engine refuses on Windows (the 5070 Ti plan already changes this)
- add a PowerShell build script in place of the GNU `Makefile`
- bundle the two cuBLAS DLLs

This plan depends on step 4 of `rtx-5070-ti-16gb-support.md` (the `--ple-io` rule). Do that first, or at the same time.

## What is Linux-only today

All of it is in the frontend, the build script and the fork's one engine addition. The engine core is already cross-platform: it reads stdin with `_read` on Windows (`src/program/generate.cpp:5625`), strips `\r`, and keeps stdout unbuffered (`generate.cpp:1245`), so the pipe protocol needs no engine-side change beyond binary mode (step 3).

| Where | Linux-only code | Windows replacement |
| --- | --- | --- |
| `frontend/engine.hpp` | `pipe`, `fork`, `execv("/proc/self/exe")`, `dup2`, `write`, `fdopen`, `getline`, `kill(SIGKILL)`, `waitpid`, `usleep` | `CreatePipe`, `CreateProcessW` on `GetModuleFileNameW`, `WriteFile`, a `ReadFile` line reader, `TerminateProcess`, `WaitForSingleObject` |
| `frontend/child_watch.hpp` | `waitpid(WNOHANG)`, `WIFSIGNALED` | `GetExitCodeProcess`; a Job Object so the engine dies with the frontend |
| `frontend/vision.hpp:16-28` `ImageFile` | `mkstemp("/tmp/…")`, `close`, `unlink` | `GetTempPathW` + `GetTempFileNameW` (or the short path), `DeleteFileW`; see step 3 on spaces in the path |
| `frontend/main.cpp` | `SIGPIPE`, `SIGINT`/`SIGTERM` handlers, `stat`, `sys/wait.h`, `fcntl.h`, `unistd.h` | `SetConsoleCtrlHandler`; no SIGPIPE on Windows; `std::filesystem::exists` |
| `frontend/main.cpp:513` `default_threads()` | reads cgroup files and `/sys/.../smt/active` | return `0`: the engine already counts physical cores on Windows (`src/kernels/cpu/pool.cpp:39`) |
| `frontend/main.cpp:602` | always passes `--ple-io ram` | the rule from the 5070 Ti plan, step 4; `direct` on Windows |
| `frontend/*_test.cpp` | `fork`, `getline`, `sys/resource.h`, `unistd.h`; `chat_parser_test` links `libmtmd.a` | port `engine_test` and `engine_backpressure_test`; keep `child_watch_test` Linux-only (step 7) |
| `Makefile` | GNU make, `nproc`, shell tests, applies the llama.cpp patch | `build.ps1` (pwsh) calling CMake with MSVC |
| `include/strata/platform/robust_read.hpp`, `src/kernels/ngram.cpp:266` | the fork's retrying reads and its own-memory n-gram table, both guarded `!_WIN32` | Windows keeps upstream's plain reads and file mapping; a known gap, below |
| `README.md`, supervision | `systemd` `Restart=on-failure` | a service wrapper with restart-on-failure (step 8) |

## What upstream Strata already does on Windows

Upstream runs the same engine natively on Windows, including your tested 5070 Ti config (`D:\Strata\engine\strata.exe`). Its Python server drives the engine over stdin/stdout pipes, the same protocol this fork's C++ frontend uses. What carries over:

- **Build:** CMake with MSVC from Visual Studio 2019 or 2022 (CUDA 13.0–13.2 accept only those; VS 2026 needs CUDA 13.3+), found through `vcvars64.bat` (upstream `setup.py:1390`). `STRATA_PORTABLE=ON` links the MSVC runtime statically (`CMakeLists.txt:47`), so no redistributable is needed.
- **Static linking throughout:** the CUDA runtime is linked statically by default (`CMakeLists.txt:127-132`; only the experimental SM60 build uses the shared one), and llama.cpp/ggml are forced static (`CMakeLists.txt:959-960`). So the only DLLs to ship are cuBLAS's.
- **CPU kernels:** MSVC `/arch:AVX2` and `/arch:AVX512` are set per source file (`CMakeLists.txt:933-934`), and `/W4 /WX` is already on for MSVC (`CMakeLists.txt:196-205`).
- **Memory locking:** `src/platform/memory.cpp` raises the working-set minimum and uses `VirtualLock` in place of `mlock`.
- **CPU pinning and core count:** `src/kernels/cpu/pool_affinity_win.hpp` is in the fork, and `src/kernels/cpu/pool.cpp` counts physical cores with `GetLogicalProcessorInformationEx`.
- **N-gram table:** `direct` and `mmap` modes work on Windows (`ngram.cpp` uses `PrefetchVirtualMemory` on the mapping). Only `ram` is refused (`generate.cpp:1758-1763`).
- **Runtime DLLs:** upstream loads the CUDA 13 DLLs from the pip `nvidia` package (`lib_dirs` in your config).

Two upstream-only headers (`platform/aux_cpus.hpp`, `platform/host_limits.hpp`) are newer than the fork's base and aren't needed.

What upstream does not cover: its Python server owns the engine's stdio from the Python side, so the engine never sets binary mode on its own stdin. The C++ frontend needs that (step 3).

## Work plan

1. **Engine-only build first.** Build the `strata` engine target with MSVC + CUDA 13 (`CUDA_ARCHS=120`) and run its parity tests on the 5070 Ti. This shows whether the fork's engine changes broke upstream's Windows paths before any frontend work. The fork's additions that compile on Windows are the n-gram own-memory path (guarded out) and the retrying reads (guarded out), so expect few failures; fix any `/WX` warnings.
   A trial of the engine alone needs no frontend change: run `strata.exe --serve … --ple-io direct` by hand, as the 5070 Ti plan's "trial" section does.
2. **Build llama.cpp parts.** Build the pinned `third_party/llama.cpp` (`llama`, `llama-common`, `mtmd`, `cpp-httplib`) with the fork's patch applied. `mtmd` is needed: the frontend takes image input through it (`frontend/vision.hpp`, `--mmproj`). llama.cpp supports MSVC, so little is expected.
3. **Process layer and temp files.** Split `frontend/engine.hpp` and `child_watch.hpp` into a small interface with POSIX and Win32 versions:
   - start: `CreatePipe` + `CreateProcessW` on the program's own path with `--engine`.
     - Use `STARTUPINFOEX` with `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` so the child inherits only its own pipe ends. If it inherits the parent's write end of its stdin, closing stdin never reaches it as EOF.
     - Set `hStdError` to the frontend's stderr with `STARTF_USESTDHANDLES`, so the engine's loading messages still show.
     - Build the command line with `CommandLineToArgvW` quoting rules; model paths can contain spaces.
     - Create the child with `CREATE_SUSPENDED`, assign it to the Job Object, then `ResumeThread`, so nothing it starts escapes the job.
   - binary mode: the engine calls `_setmode(0, _O_BINARY)` on Windows at its entry. Its stdin reader uses `_read(0, …)`, which in text mode treats `0x1A` as end of file. stdout can stay in text mode: the frontend's line reader already strips `\r`.
   - write and read: `WriteFile` on the stdin pipe; a `ReadFile`-based line reader on the stdout pipe in place of `fdopen` + `getline`, which MSVC doesn't have.
   - watch: `GetExitCodeProcess`; the Job Object has kill-on-close, so the engine can't outlive the frontend.
   - stop: send `QUIT`, close stdin, wait 10 s, then `TerminateProcess`.
   - The "fork before any thread exists" rule (`main.cpp:599`) doesn't apply on Windows; keep the order and guard the comment.
   - `ImageFile` (`vision.hpp:16`): create the embeddings file under `GetTempPathW`. The engine's `GENI <max_new> <file> <ids>` line is split on spaces (`generate.cpp:5765`), and the default temp folder is `C:\Users\<name>\AppData\Local\Temp`, which contains a space whenever the user name does. Either pass the 8.3 short path (`GetShortPathNameW`) or, better, put the file in a fixed directory without spaces beside the executable, or teach the engine to take a quoted path. Decide this one before writing code; it's the one protocol change in the port.
4. **Signals, threads and arguments** in `frontend/main.cpp`:
   - `SetConsoleCtrlHandler` for Ctrl+C and close. It runs on its own thread, so calling `http.stop()` there is safe. On `CTRL_CLOSE_EVENT` Windows kills the process about 5 s after the handler returns, before `quit()`'s 10 s wait ends; the Job Object then stops the engine.
   - `default_threads()` returns `0` on Windows.
   - `--ple-io` from the 5070 Ti plan's rule; on Windows that always gives `direct`.
   - UTF-8 arguments: add an `activeCodePage` UTF-8 manifest (or use `wmain`), so model paths with non-ASCII characters reach the engine and `llama_model_load_from_file` intact.
5. **Build script.** `build.ps1` (pwsh): find `vcvars64.bat` with `vswhere`, apply the llama.cpp patch, run CMake with `STRATA_PORTABLE=ON`, then copy `cublas64_13.dll` and `cublasLt64_13.dll` beside `unbiased-strata.exe`. Confirm the list with `dumpbin /dependents` after the first build; the CUDA runtime and ggml are static, so nothing else is expected.
6. **Windows defaults** in the frontend's engine arguments (next section).
7. **Tests.** Port `engine_test` and `engine_backpressure_test` to the new process layer (their fake engines use `getline` on stdin too). Add a CMake target for `chat_parser_test` (it links `mtmd`). `child_watch_test` tests `fork`, `SIGABRT` and the "killed by signal" message, none of which exist on Windows; keep it Linux-only and add a Windows twin that checks the exit code and the Job Object kill. Run one image request with `--mmproj` on the 5070 Ti.
8. **Supervision.** The frontend exits with `_Exit(1)` when the engine dies (`main.cpp:683-700`) and relies on something outside to restart it. On Windows, run it as a service under a wrapper (WinSW or NSSM) with restart-on-failure, and document a pwsh restart loop for running by hand.
9. **CI.** A `windows-latest` GitHub Actions job (pwsh). The runner has no CUDA toolkit, and the `unbiased-strata` target only exists with CUDA enabled (`CMakeLists.txt:1254`). So either:
   - build only the header-only tests (`engine_test`, `engine_backpressure_test`) with MSVC: fast, catches process-layer regressions; or
   - install the CUDA toolkit in the job and compile the whole frontend without running it: slower, catches MSVC build breaks.
10. **Docs.** Add Windows to the README's requirements, build and run sections, including the slow-store gap below. Update `UPSTREAM.md`'s "Build" row.

## Windows runtime differences for this host

Native Windows avoids WSL's two big limits: all 64 GB of RAM is usable, and KV streaming works (your tested config uses `--kv-resident 32768` on Windows).

| Setting | Linux fork today | Windows | Reason |
| --- | --- | --- | --- |
| `--ple-io` | `ram` (hard-coded at `main.cpp:602`) | `direct` | The engine refuses `ram` on Windows (`generate.cpp:1758-1763`), so this change is required. `direct` is the engine's default and what the 5070 Ti plan already wants on an SSD. `mmap`, which the engine's error message suggests, maps the 28.8 GB table through the file cache, where it competes with the 53 GB of experts for the same RAM |
| Expert memory | locked with `mlock` | working-set minimum + `VirtualLock` | Already in `src/platform/memory.cpp`; needs a system-managed page file large enough for the commit |
| Expert mode | normal | normal first, `--resident-experts` if Windows' own use leaves too little | IQ3\_XXS needs about 53 GB with headroom; Windows itself uses more than a server Linux |
| `--kv-resident` | not set | 32768 above 32K context | Works natively, unlike WSL |
| Tuned flags | engine defaults | `--pcie-frac 0.35`, `--adapt-every 1`, `--adapt-swaps 80`, `--adapt-decay 0.97` | Measured on this card under Windows |
| VRAM | whole card | less any desktop use | If a monitor is on the 5070 Ti, Windows takes some VRAM; `--expert-cache auto` adjusts |
| Threads | cgroup quota, else engine's choice | engine's choice | The engine counts physical cores itself on Windows |
| Image temp file | `/tmp` | a path without spaces (step 3) | The `GENI` line is space-separated |

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| The fork's engine changes were never built with MSVC | Step 1 builds the engine alone before frontend work |
| MSVC rejects GCC-only code in the frontend or the fork's patch | Fix under `/W4 /WX`; keep the same code compiling with g++ |
| An engine left running after the frontend crashes holds VRAM | Job Object with kill-on-close, assigned before the child runs (step 3) |
| The engine never sees EOF on stdin because a stray handle keeps the pipe open | Explicit handle list on `CreateProcessW` (step 3) |
| An image request fails because the temp path has a space | Decided in step 3 before any code |
| Pipe backpressure behaves differently on Windows | Port `engine_backpressure_test` (step 7) |
| Windows takes too much RAM for the normal mode | `--resident-experts` |
| Model on a network share or slow drive: Windows has none of the fork's retrying reads (`robust_read.hpp`) or own-memory table (`ngram.cpp:266`), both `!_WIN32` | Keep the model on a local SSD and document it; port the retries later if needed |

- [ ] Should the Linux build stay the main target, with Windows as a second one, or the other way round?
- [ ] Is a GitHub Actions Windows job wanted, and which kind (step 9)?
- [ ] Should the engine run as a Windows service, or start by hand?
- [ ] Should Windows get `--ple-io ram` back (read the table into its own memory and `VirtualLock` it, as `ngram.cpp:266` does on Linux with `mlock` and `memory.cpp` already does for experts), or stay on `direct` for good? This decides whether the engine's Windows guard is a fix to make or a limit to document.
- [ ] Image temp file: short path, a fixed directory, or a quoted path in the `GENI` line?

## Changes from `windows-port.md`

- `vision.hpp` is not "portable with an unused include": `ImageFile` uses `mkstemp` on `/tmp` and `unlink`, and the engine reads the file's path from a space-separated line. Added to the table, step 3, the risks and the open questions.
- Stdio: the engine already uses `_read` and strips `\r` on Windows, and keeps stdout unbuffered. Binary mode is needed on stdin only. Stated as fact instead of inferred.
- DLLs: the CUDA runtime and ggml are linked statically, so the list is the two cuBLAS DLLs, not three.
- Tied the `--ple-io` change to the 5070 Ti plan's step 4 instead of describing it twice.
- The `mmap` vs `direct` reason is now specific (page cache against expert RAM).
- Step 7 says what `child_watch_test` tests and why it stays Linux-only.
- Step 1 notes the trial that needs no code.
