# Porting UnbiasedStrata to Windows

Oct 11, 2026 · @Rob · revised Oct 12, 2026

> Superseded by `fable-windows-port.md`, which has the status.

## Summary

A native Windows build is a modest port, because almost all of the Linux-only code is in the small C++ frontend (about 1,300 lines with its tests), not the engine. The engine and its CMake build still carry upstream Strata's Windows support (`_WIN32` paths in 26 files under `src/` and `include/`, MSVC flags), and upstream ships and runs that engine on Windows today. The work is:

- replace the frontend's POSIX process handling (`fork`, pipes, `getline`, `waitpid`, signals) with Win32 equivalents
- add a PowerShell build script in place of the GNU `Makefile`
- bundle the CUDA 13 runtime DLLs
- stop the frontend passing `--ple-io ram`, which the engine refuses on Windows

## What is Linux-only today

The Linux-only code is all in the frontend, the build script and the fork's one engine addition. The engine core is already cross-platform.

| Where | Linux-only code | Windows replacement |
| --- | --- | --- |
| `frontend/engine.hpp` | `pipe`, `fork`, `execv("/proc/self/exe")`, `dup2`, `write`, `fdopen`, `getline`, `kill(SIGKILL)`, `waitpid`, `usleep` | `CreatePipe`, `CreateProcessW` on `GetModuleFileNameW`, `WriteFile`, a `ReadFile` line reader, `TerminateProcess`, `WaitForSingleObject` |
| `frontend/child_watch.hpp` | `waitpid(WNOHANG)`, `WIFSIGNALED` | `GetExitCodeProcess`; a Job Object so the engine dies with the frontend |
| `frontend/main.cpp` | `SIGPIPE`, `SIGINT`/`SIGTERM` handlers, `stat`, `sys/wait.h`, `fcntl.h`, `unistd.h` | `SetConsoleCtrlHandler`; no SIGPIPE on Windows; `std::filesystem::exists` |
| `frontend/main.cpp:513` `default_threads()` | reads cgroup files and `/sys/.../smt/active` | return `0`: the engine already counts physical cores on Windows (`src/kernels/cpu/pool.cpp:39`) |
| `frontend/main.cpp:602` | always passes `--ple-io ram` | pass `--ple-io direct` on Windows (see below) |
| `frontend/vision.hpp` | an unused `unistd.h` include | drop the include; the vision code itself is portable |
| `frontend/*_test.cpp` | `sys/resource.h`, `unistd.h`; `chat_parser_test` links `libmtmd.a` | port, or limit to Linux; CMake targets for the ones kept |
| `Makefile` | GNU make, `nproc`, shell tests | `build.ps1` (pwsh) calling CMake with MSVC |
| `include/strata/platform/robust_read.hpp` | the fork's retrying reads, guarded `!_WIN32` | Windows keeps upstream's plain reads; see the known gap below |
| `README.md`, supervision | `systemd` `Restart=on-failure` | a service wrapper with restart-on-failure (step 8) |

## What upstream Strata already does on Windows

Upstream runs the same engine natively on Windows, including your tested 5070 Ti config (`D:\Strata\engine\strata.exe`). Its Python server drives the engine over stdin/stdout pipes, the same protocol this fork's C++ frontend uses. What carries over:

- **Build:** CMake with MSVC from Visual Studio 2019 or 2022 (CUDA 13.0–13.2 accept only those; VS 2026 needs CUDA 13.3+), found through `vcvars64.bat` (upstream `setup.py:1390`). `STRATA_PORTABLE=ON` links the MSVC runtime statically, so no redistributable is needed.
- **CPU kernels:** MSVC `/arch:AVX2` and `/arch:AVX512` are set per source file in the fork's own `CMakeLists.txt`.
- **Memory locking:** `src/platform/memory.cpp` raises the working-set minimum and uses `VirtualLock` in place of `mlock`.
- **CPU pinning and core count:** `src/kernels/cpu/pool_affinity_win.hpp` is in the fork, and `src/kernels/cpu/pool.cpp` counts physical cores with `GetLogicalProcessorInformationEx`.
- **Runtime DLLs:** upstream loads the CUDA 13 DLLs from the pip `nvidia` package (`lib_dirs` in your config).

Two upstream-only headers (`platform/aux_cpus.hpp`, `platform/host_limits.hpp`) are newer than the fork's base and aren't needed.

What upstream does not cover: its Python server reads the engine's stdout through Python's own I/O, so the engine never sets binary mode on its stdio. The C++ frontend needs that (step 3).

## Work plan

1. **Engine-only build first.** Build the `strata` engine target with MSVC + CUDA 13 (`CUDA_ARCHS=120`) and run its parity tests on the 5070 Ti. This shows whether the fork's engine changes broke upstream's Windows paths before any frontend work. Expect fixes for MSVC `/WX` warnings.
2. **Build llama.cpp parts.** Build the pinned `third_party/llama.cpp` (`llama`, `llama-common`, `mtmd`, `cpp-httplib`) with the fork's patch applied. `mtmd` is needed: the frontend takes image input through it (`frontend/vision.hpp`, `--mmproj`). llama.cpp supports MSVC, so little is expected.
3. **Process layer.** Split `frontend/engine.hpp` and `child_watch.hpp` into a small interface with POSIX and Win32 versions:
   - start: `CreatePipe` + `CreateProcessW` on the program's own path with `--engine`.
     - Use `STARTUPINFOEX` with `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` so the child inherits only its own pipe ends. If it inherits the parent's write end of its stdin, closing stdin never reaches it as EOF.
     - Set `hStdError` to the frontend's stderr with `STARTF_USESTDHANDLES`, so the engine's loading messages still show.
     - Build the command line with `CommandLineToArgvW` quoting rules; model paths can contain spaces.
     - Create the child with `CREATE_SUSPENDED`, assign it to the Job Object, then `ResumeThread`, so nothing it starts escapes the job.
   - binary mode: the engine calls `_setmode(_fileno(stdin), _O_BINARY)` and the same for stdout on Windows, at its entry. In text mode the CRT treats `0x1A` on stdin as EOF.
   - write and read: `WriteFile` on the stdin pipe; a `ReadFile`-based line reader on the stdout pipe in place of `fdopen` + `getline`, which MSVC doesn't have.
   - watch: `GetExitCodeProcess`; the Job Object has kill-on-close, so the engine can't outlive the frontend.
   - stop: send `QUIT`, close stdin, wait 10 s, then `TerminateProcess`.
   - The "fork before any thread exists" rule (`main.cpp:599`) doesn't apply on Windows; keep the order and guard the comment.
4. **Signals, threads and arguments** in `frontend/main.cpp`:
   - `SetConsoleCtrlHandler` for Ctrl+C and close. It runs on its own thread, so calling `http.stop()` there is safe. On `CTRL_CLOSE_EVENT` Windows kills the process about 5 s after the handler returns, before `quit()`'s 10 s wait ends; the Job Object then stops the engine.
   - `default_threads()` returns `0` on Windows.
   - `--ple-io direct` in place of `ram` on Windows.
   - UTF-8 arguments: add an `activeCodePage` UTF-8 manifest (or use `wmain`), so model paths with non-ASCII characters reach the engine intact.
5. **Build script.** `build.ps1` (pwsh): find `vcvars64.bat` with `vswhere`, apply the llama.cpp patch, run CMake with `STRATA_PORTABLE=ON`, then copy the CUDA 13 DLLs beside `unbiased-strata.exe`. Take the DLL list from the link line (`dumpbin /dependents`) rather than a fixed list; the expected set is `cublas64_13` and `cublasLt64_13` (prefill GEMM), plus `cudart64_13` unless the CUDA runtime is linked statically.
6. **Windows defaults** in the frontend's engine arguments (next section).
7. **Tests.** Port `engine_test` and `engine_backpressure_test` to the new process layer. Add a CMake target for `chat_parser_test` (it links `mtmd`). Keep `child_watch_test` Linux-only, or add a Windows twin. Run one image request with `--mmproj` on the 5070 Ti.
8. **Supervision.** The frontend exits with `_Exit(1)` when the engine dies (`main.cpp:683-700`) and relies on something outside to restart it. On Windows, run it as a service under a wrapper (WinSW or NSSM) with restart-on-failure, and document a pwsh restart loop for running by hand.
9. **CI.** A `windows-latest` GitHub Actions job (pwsh). The runner has no CUDA toolkit, and the `unbiased-strata` target only exists with CUDA enabled (`CMakeLists.txt:1254`). So either:
   - build only the header-only tests (`engine_test`, `engine_backpressure_test`) with MSVC: fast, catches process-layer regressions; or
   - install the CUDA toolkit in the job and compile the whole frontend without running it: slower, catches MSVC build breaks.
10. **Docs.** Add Windows to the README's requirements, build and run sections, including the slow-store gap below.

## Windows runtime differences for this host

Native Windows avoids WSL's two big limits: all 64 GB of RAM is usable, and KV streaming works (your tested config uses `--kv-resident 32768` on Windows).

| Setting | Linux fork today | Windows | Reason |
| --- | --- | --- | --- |
| `--ple-io` | `ram` (hard-coded at `main.cpp:602`) | `direct` | The engine refuses `ram` on Windows (no `mlock`, `src/program/generate.cpp:1758-1763`), so this change is required. `direct` is the engine's default and what the 5070 Ti plan already wants; `mmap`, which the engine's error message suggests, would put the table in the page cache and compete with expert memory |
| Expert memory | locked with `mlock` | working-set minimum + `VirtualLock` | Already in `src/platform/memory.cpp`; needs a system-managed page file large enough for the commit |
| Expert mode | normal | normal first, `--resident-experts` if Windows' own use leaves too little | IQ3\_XXS needs about 53 GB with headroom; Windows itself uses more than a server Linux |
| `--kv-resident` | not set | 32768 above 32K context | Works natively, unlike WSL |
| Tuned flags | engine defaults | `--pcie-frac 0.35`, `--adapt-every 1`, `--adapt-swaps 80`, `--adapt-decay 0.97` | Measured on this card under Windows |
| VRAM | whole card | less any desktop use | If a monitor is on the 5070 Ti, Windows takes some VRAM; `--expert-cache auto` adjusts |
| Threads | cgroup quota, else engine's choice | engine's choice | The engine counts physical cores itself on Windows |

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| The fork's engine changes were never built with MSVC | Step 1 builds the engine alone before frontend work |
| MSVC rejects GCC-only code in the frontend or the fork's patch | Fix under `/W4 /WX`; keep the same code compiling with g++ |
| An engine left running after the frontend crashes holds VRAM | Job Object with kill-on-close, assigned before the child runs (step 3) |
| The engine never sees EOF on stdin because a stray handle keeps the pipe open | Explicit handle list on `CreateProcessW` (step 3) |
| Pipe backpressure behaves differently on Windows | Port `engine_backpressure_test` (step 7) |
| Windows takes too much RAM for the normal mode | `--resident-experts` |
| Model on a network share or slow drive: Windows has none of the fork's retrying reads (`robust_read.hpp` is `!_WIN32`) | Keep the model on a local SSD and document it; port the retries later if needed |

- [ ] Should the Linux build stay the main target, with Windows as a second one, or the other way round?
- [ ] Is a GitHub Actions Windows job wanted, and which kind (step 9)?
- [ ] Should the engine run as a Windows service, or start by hand?
- [ ] Should Windows get `--ple-io ram` back (lock the n-gram table with `VirtualLock`, as expert memory already is), or stay on `direct` for good? This decides whether the engine's Windows guard is a fix to make or a limit to document.
