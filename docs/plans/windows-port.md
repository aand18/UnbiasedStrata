# Porting UnbiasedStrata to Windows

Oct 11, 2026 · @Rob

## Summary

A native Windows build is a modest port, because almost all of the Linux-only code is in the small C++ frontend (about 1,300 lines), not the engine. The engine and its CMake build still carry upstream Strata's Windows support (`_WIN32` paths in 32 files, MSVC flags), and upstream ships and runs that engine on Windows today. The work is:

- replace the frontend's POSIX process handling (`fork`, pipes, `waitpid`, signals) with Win32 equivalents
- add a PowerShell build script in place of the GNU `Makefile`
- bundle the CUDA 13 runtime DLLs
- adjust two runtime defaults for Windows memory rules

## What is Linux-only today

The Linux-only code is all in the frontend, the build script and the fork's one engine addition. The engine core is already cross-platform.

| Where | Linux-only code | Windows replacement |
| --- | --- | --- |
| `frontend/engine.hpp` | `pipe`, `fork`, `execv("/proc/self/exe")`, `dup2`, `write`, `kill(SIGKILL)`, `waitpid`, `usleep` | `CreatePipe`, `CreateProcessW` on `GetModuleFileNameW`, `WriteFile`, `TerminateProcess`, `WaitForSingleObject` |
| `frontend/child_watch.hpp` | `waitpid(WNOHANG)`, `WIFSIGNALED` | `GetExitCodeProcess`; a Job Object so the engine dies with the frontend |
| `frontend/main.cpp` | `SIGPIPE`, `SIGINT`/`SIGTERM` handlers, `sys/wait.h`, `fcntl.h` | `SetConsoleCtrlHandler`; no SIGPIPE on Windows |
| `frontend/main.cpp:513` `default_threads()` | reads cgroup files and `/sys/.../smt/active` | `GetLogicalProcessorInformationEx` for physical cores |
| `frontend/vision.hpp` | `unistd.h` | vision isn't needed; drop the include or guard it |
| `frontend/*_test.cpp` | `sys/resource.h`, `unistd.h` | port or limit to Linux |
| `Makefile` | GNU make, `nproc`, shell tests | `build.ps1` (pwsh) calling CMake with MSVC |
| `include/strata/platform/robust_read.hpp` | the fork's retrying reads, guarded `!_WIN32` | Windows keeps upstream's plain reads; no change needed |
| `README.md`, supervision | `systemd` `Restart=on-failure` | a Windows service wrapper or a pwsh restart loop |

## What upstream Strata already does on Windows

Upstream runs the same engine natively on Windows, including your tested 5070 Ti config (`D:\Strata\engine\strata.exe`). Its Python server drives the engine over stdin/stdout pipes, the same protocol this fork's C++ frontend uses. What carries over:

- **Build:** CMake with MSVC from Visual Studio 2019 or 2022 (CUDA 13.0–13.2 accept only those; VS 2026 needs CUDA 13.3+), found through `vcvars64.bat` (upstream `setup.py:1390`). `STRATA_PORTABLE=ON` links the MSVC runtime statically, so no redistributable is needed.
- **CPU kernels:** MSVC `/arch:AVX2` and `/arch:AVX512` are set per source file in the fork's own `CMakeLists.txt`.
- **Memory locking:** `src/platform/memory.cpp` raises the working-set minimum and uses `VirtualLock` in place of `mlock`.
- **CPU pinning:** `src/kernels/cpu/pool_affinity_win.hpp` is in the fork.
- **Runtime DLLs:** upstream loads the CUDA 13 DLLs from the pip `nvidia` package (`lib_dirs` in your config).

Two upstream-only headers (`platform/aux_cpus.hpp`, `platform/host_limits.hpp`) are newer than the fork's base and aren't needed.

## Work plan

1. **Engine-only build first.** Build the `strata` engine target with MSVC + CUDA 13 (`CUDA_ARCHS=120`) and run its parity tests on the 5070 Ti. This shows whether the fork's engine changes broke upstream's Windows paths before any frontend work. Expect fixes for MSVC `/WX` warnings.
2. **Build llama.cpp parts.** Build the pinned `third_party/llama.cpp` (`llama`, `llama-common`, `cpp-httplib`; skip `mtmd`, since vision isn't needed). llama.cpp supports MSVC, so little is expected.
3. **Process layer.** Split `frontend/engine.hpp` and `child_watch.hpp` into a small interface with POSIX and Win32 versions:
   - start: `CreatePipe` + `CreateProcessW` on the program's own path with `--engine`; stdin/stdout in binary mode
   - write and read: `WriteFile` / `ReadFile` on the pipe handles
   - watch: `GetExitCodeProcess`; put the engine in a Job Object with kill-on-close, so it can't outlive the frontend
   - stop: close stdin, wait 10 s, then `TerminateProcess`
4. **Signals and threads** in `frontend/main.cpp`: `SetConsoleCtrlHandler` for Ctrl+C and close; a Windows `default_threads()` counting physical cores.
5. **Build script.** `build.ps1` (pwsh): find `vcvars64.bat` with `vswhere`, apply the llama.cpp patch, run CMake with `STRATA_PORTABLE=ON`, then copy the CUDA 13 DLLs (`cudart64_13`, `cublas64_13`, `cublasLt64_13`) beside `unbiased-strata.exe`.
6. **Windows defaults** in the frontend's engine arguments (next section).
7. **Tests.** Port `engine_test` and `engine_backpressure_test` to the new process layer; keep `child_watch_test` Linux-only, or add a Windows twin.
8. **Supervision.** Document a restart loop in pwsh, or a service wrapper, to replace `systemd` `Restart=on-failure`.
9. **CI.** A `windows-latest` GitHub Actions job (pwsh) that compiles the frontend and runs its tests without a GPU.
10. **Docs.** Add Windows to the README's requirements, build and run sections.

## Windows runtime differences for this host

Native Windows avoids WSL's two big limits: all 64 GB of RAM is usable, and KV streaming works (your tested config uses `--kv-resident 32768` on Windows).

| Setting | Linux fork today | Windows | Reason |
| --- | --- | --- | --- |
| `--ple-io` | `ram` | `direct` | The engine refuses `ram` on Windows (no `mlock`, `src/program/generate.cpp:1759`); `direct` is upstream's default and what the 5070 Ti plan already wants |
| Expert memory | locked with `mlock` | working-set minimum + `VirtualLock` | Already in `src/platform/memory.cpp`; needs a system-managed page file large enough for the commit |
| Expert mode | normal | normal first, `--resident-experts` if Windows' own use leaves too little | IQ3\_XXS needs about 53 GB with headroom; Windows itself uses more than a server Linux |
| `--kv-resident` | not set | 32768 above 32K context | Works natively, unlike WSL |
| Tuned flags | engine defaults | `--pcie-frac 0.35`, `--adapt-every 1`, `--adapt-swaps 80`, `--adapt-decay 0.97` | Measured on this card under Windows |
| VRAM | whole card | less any desktop use | If a monitor is on the 5070 Ti, Windows takes some VRAM; `--expert-cache auto` adjusts |

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| The fork's engine changes were never built with MSVC | Step 1 builds the engine alone before frontend work |
| MSVC rejects GCC-only code in the frontend or the fork's patch | Fix under `/W4 /WX`; keep the same code compiling with g++ |
| An engine left running after the frontend crashes holds VRAM | Job Object with kill-on-close (step 3) |
| Pipe backpressure behaves differently on Windows | Port `engine_backpressure_test` (step 7) |
| Windows takes too much RAM for the normal mode | `--resident-experts` |

- [ ] Should the Linux build stay the main target, with Windows as a second one, or the other way round?
- [ ] Is a GitHub Actions Windows job wanted (step 9)?
- [ ] Should the engine run as a Windows service, or start by hand?
