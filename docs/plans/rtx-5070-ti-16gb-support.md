# RTX 5070 Ti 16 GB support plan

Oct 11, 2026 · @Rob

## Summary

UnbiasedStrata can run on an RTX 5070 Ti 16 GB host with 64 GB RAM, but the frontend's fixed engine arguments must change so the model fits in RAM. The GPU side needs no porting: sm\_120 is already built (`CUDA_ARCHS=86;89;120`), and the engine sizes its expert cache from measured free VRAM (`--expert-cache auto`). Upstream Strata has already run this engine on an RTX 5070 12 GB and an RTX 5070 Ti.

The blocker is RAM, and upstream Strata already handles it. This fork always passes `--ple-io ram` (`frontend/main.cpp:603`), which locks the 28.8 GB n-gram table in RAM beside 42.9 GB of IQ3\_XXS experts. Upstream's setup defaults to `--ple-io direct` (reads from SSD). It uses `ram` only for a rotational disk with RAM to spare (`setup.py:5585`). With that default, upstream treats IQ3\_XXS as fitting a 64 GB PC without low-RAM mode.

The tested settings (`strata-coder-iq1_m.json`) come from upstream on Windows with the Coder model (IQ1\_M, 23.4 GB of experts). This fork is Linux-only and supports only Qwen3.8-Flash-Next IQ3\_XXS, so only the card- and host-dependent settings carry over (step 5).

If the host stays on Windows, WSL2 works but changes the defaults: resident low-RAM mode from the start, no KV streaming, and 32K context (see Running under WSL2).

## Target host and constraints

| Part | Target | Currently supported | Effect |
| --- | --- | --- | --- |
| GPU | RTX 5070 Ti, 16 GB, sm\_120 | 24 GB+, RTX 30/40/50 | Smaller expert cache (about 700 experts per GB); more expert work falls to the CPU |
| RAM | 64 GB | 128 GB | N-gram table and the experts can't both be locked in RAM |
| CPU | Ryzen 9 3900XT, 12 cores, Zen 2 | x86-64 with AVX2 | AVX2 present, no AVX-512; DDR4 bandwidth limits CPU expert speed |
| Vision | Not required | Optional (`--mmproj`) | No vision VRAM reserve; run without `--mmproj` |
| Storage | Must be NVMe SSD (to confirm) | Any; all read into RAM at start | N-gram table read from disk per token |

## Running under WSL2

WSL2 is feasible but forces the low-RAM mode and drops KV streaming; native Linux is faster. Upstream says WSL works (Ubuntu 24.04 tested, upstream `docs/DETAILS.md:550`). The closest measured case is an RTX 5060 Ti 16 GB under WSL2 with a Ryzen 7 3700X on the larger IQ3\_S model: about 48 tokens/s generating, 598 tokens/s on a 20K prompt ([upstream benchmark](https://github.com/mtnrbq/Strata/blob/main/bench/results/2026-10-09-community-rtx-5060ti-wsl2-iq3_s-0141/README.md); 128 GB host, 78 GB given to WSL).

| WSL limit | Effect on the plan |
| --- | --- |
| WSL gets half the host RAM by default (32 GB) | Set `memory=56GB` in `.wslconfig`, leaving \~8 GB for Windows |
| 56 GB is too tight for the normal mode (42.9 + 10 GB headroom) | Start with `--resident-experts`: \~11 GB of experts on the GPU, \~32 GB in RAM |
| The NVIDIA driver locks only \~1 GB of RAM | No KV streaming (`--kv-resident`), so KV must fit VRAM: favours 32K context; experts copied from unlocked RAM, so slower prompts |
| Windows drives (`/mnt/d`) are slow and report as rotational | Keep the model on WSL's own ext4 disk for `--ple-io direct` |
| GPU driver comes from Windows | Install the WSL-Ubuntu CUDA toolkit 13.x; no Linux NVIDIA driver inside WSL |

## Plan

1. **Build.** Use CUDA 13.0+ (advised for RTX 50; under WSL, the WSL-Ubuntu toolkit) with `make build CUDA_ARCHS=120`. Build the tests (`STRATA_BUILD_TESTS=ON`) and run the parity tests on the card, especially the sm\_120 paths:
   - `native_flash_attn.cu` (pinned sm120a binary)
   - the cluster paths in `sampler.cu` and `qsa_select.cu`, checked by `decode_cluster_parity`
   - the `qsa_*` and `fused_gr` parity tests
2. **N-gram table: follow upstream.** Default to `--ple-io direct` (unbuffered SSD reads). Use `ram` only when the model is on a rotational disk and RAM is at least 60 + 28.8 + 4 GB (upstream `setup.py:5585`). A slow disk can stall prompts for minutes (upstream #605), so the model must be on NVMe (under WSL, on WSL's own ext4 disk, not /mnt/d).
3. **Experts: follow upstream's low-RAM rule** (upstream `setup.py:3465-3490`). Low-RAM mode applies only when RAM is less than the experts plus 10 GB.
   - IQ3\_XXS: 42.9 + 10 = 52.9 GB. On native Linux (64 GB) the normal mode, with all experts in RAM, comes first. Under WSL (56 GB VM) it doesn't fit.
   - `--resident-experts`: the fallback on native Linux, the first choice under WSL. The GPU then holds about 11 GB of experts (VRAM less 5 GB at 32K context); the other \~32 GB stays in RAM.
   - Confirm the fallback works on the native IQ3\_XXS pack without `experts.bin` (upstream engine 0.1.31 says it does).
4. **Frontend change** (`frontend/main.cpp`). Add a `--low-ram` flag, or automatic detection, for step 3. Replace the fixed --ple-io ram with the rule in step 2. Log what was chosen. A trial needs no code: flags after -- come last, and the engine keeps the last --ple-io it is given (src/program/generate.cpp:1326).
5. **Carry over the tested settings** (`strata-coder-iq1_m.json`, upstream on Windows, CUDA 13 runtime, Coder IQ1\_M). Add the card- and host-dependent arguments to the frontend's fixed list (`frontend/main.cpp:600-605`), then re-measure on IQ3\_XXS:

   | Argument | Tested value | Fork today | Action |
   | --- | --- | --- | --- |
   | `--pcie-frac` | 0.35 | engine default | Adopt; measured for this card's PCIe link |
   | `--adapt-every` / `--adapt-swaps` / `--adapt-decay` | 1 / 80 / 0.97 | engine default | Adopt; lets the small expert cache follow the conversation |
   | `--kv-resident` | 32768 (at 64K context) | not set | Native Linux only, above 32K context; not possible under WSL |
   | `--spec-min-p` | 0.70 | 0.5 | Try both; may depend on the model |
   | `--ple-io` | not set (engine default, direct) | ram | Matches step 2 |
   | `--max-context` | 65536 | 32768 | Decide in open questions |

   Leave out the model-specific ones: pack, GGUF shards, `expert-profile-coder.bin` and the MTP path.
6. **VRAM fit.** Start at 8K and 32K context and record the engine's budget: dense weights, int8 KV, GDN state, MTP draft head, expert-cache slots. Upstream estimates about 11 GB of experts (\~26%) on the GPU at 32K.
   - If the draft head doesn't fit (upstream #474), fall back to `--draft-vocab en`.
   - Consider KV streaming (`--kv-resident`) at long context, on native Linux only.
7. **CPU.** Check that the worker pool's CPU pinning (`pool_affinity_linux`) is sensible on two CCDs / four CCXs. Compare 12 workers against 11.
8. **Validate.**
   - Run the README speed method: the mean of 8 prompts of 400 generated tokens, and 2 prompts of about 20K tokens.
   - Run a long soak test with tool calls and long prompts, watching available RAM and RSS.
   - Check the stall watchdog and supervisor restart.
9. **Docs.** Add this configuration to the README's requirements and Status tables.

## Expected outcome and risks

It should run, but well below the README's 107 tokens/s on the 24 GB cards. Upstream estimates about 56–66 tokens/s for an RTX 5060 Ti 16 GB with IQ3\_XXS, assuming everything in RAM. With the table read from SSD and the experts in DDR4, expect lower; only a measurement will tell.

| Risk | Fallback |
| --- | --- |
| RAM tight in normal mode: 42.9 GB of experts locked, plus OS and buffers | `--resident-experts` (step 3) |
| Under WSL, Windows left with \~8 GB of RAM | Lower `.wslconfig` memory and use plain mapped mode (`--mmap-experts`); slower |
| Prompts stall on n-gram table reads | Faster NVMe; tune `--ple-inflight` |
| Draft head doesn't fit at long context | `--draft-vocab en`, a shorter `--ctx`, or `--kv-resident` (native Linux only) |
| Tested settings were tuned on a different model and OS | Re-measure on IQ3\_XXS before adopting |
| sm\_120 kernel paths untested in this fork | Parity tests in step 1 before any serving |

## First step and open questions

First, a trial run with no code changes:

```
./unbiased-strata --model <shard 1> --pack <pack> --mtp <mtp> --ctx 32768 -- \
    --ple-io direct --pcie-frac 0.35 --adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97
```

The engine keeps the last value of a repeated flag, so the `--ple-io direct` after `--` overrides the frontend's `ram`. Under WSL, add `--resident-experts` from the start; on native Linux, only if RAM runs short.

- [ ] Native Linux or WSL2? WSL2 works but is slower (see Running under WSL2).
- [ ] Is the model on an NVMe SSD?
- [ ] Which context length is needed: 32K (the fork's default) or 64K (as tested)?
- [ ] What speed is acceptable?
