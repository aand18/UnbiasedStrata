# Track D - the hyper-connection path in the verify window

The performance plan attributes 3.84 ms of the GPU's 15.86 ms/window (24.2%) to "the hyper-connection path"
(`docs/plans/specter-performance.md` §2.3). This file answers what that 3.84 ms *is*, what it can be made of,
what was implemented, and what the A/B measured. It is written against the branch `spectre-moe` at `f4de37d`;
the change in §4 was measured and then **reverted** (§6), so the tree carries none of it.

Everything below is measured on the box (RTX 4090, `C:\Users\yoho\Downloads\UnbiasedStrata`) with the best
configuration the pin measurement found:

```
--low-ram off --ple-io direct --ctx 32768 -- --kv int8 --resident-budget-gib 32
STRATA_PARTIAL_PIN=1 STRATA_PARTIAL_PIN_GIB=8 STRATA_ARENA_LOCK_GIB=8 STRATA_ARENA_PIN_GIB=4
STRATA_DECODE_TIMING=1 STRATA_RESIDENT_HEADROOM_GIB=6 STRATA_VERIFY_PROFILE=1
```

---

## 1. What the 3.84 ms is

The model's residual is a stack of `hc = 4` streams, each `n_embd = 2560` wide
(`include/strata/kernels/gr.hpp:5-13`). The geometry is fixed in the kernel: `N = 2560`, `HC = 4`,
`D = N * HC = 10240`, `LR = 320` (`src/kernels/cuda/fused_gr.cu:20-23`), and `fused_gr_supported()` refuses any
other shape, so a run in which the hyper-connection path executes at all is a run at that geometry. The engine
confirms the variant it picked on this card: `strata hc: CUDA0: the hyper-connection read runs as staged (...)
checked bit for bit against the plain read on this card`.

Per layer the verify window does **two** hyper-connection reads:

| call | site | what it collapses |
| --- | --- | --- |
| `gr_read_group(0, ...)` | `src/core/verify.cpp:696` (inside the lambda at `:682-695`) | the 4 streams -> the one vector the attention/GDN mixer consumes, plus a 4-wide `inject`, with the **previous layer's** FFN write folded in (`apply = pending`) |
| `gr_read_group(1, ...)` | `src/core/verify.cpp:902` | the same for the FFN mixer, with **this** layer's attention write folded in (`apply = true`) |

Each call is `fused_gr_read_multi()` -> `launch_multi()` (`src/kernels/cuda/fused_gr.cu:785`), which issues
**three** kernels (`:791`, `:821`, `:831`):

1. `gr_norm_split_kernel` - one block per (token, stream), i.e. 8-12 blocks: the per-stream RMSNorm
   (`src/kernels/cuda/fused_gr.cu:516`). Two passes over the stream with a block reduction between them.
2. `gr_down_staged_kernel` - `DOWN_BLOCKS + 1 = 41` blocks, one warp per row of `w_down` (320 + 4 inject rows),
   `cp.async`-staged activations (`src/kernels/cuda/fused_gr.cu:614`).
3. `gr_up_multi_kernel` - `UPM_BLOCKS = 160` blocks, 64 rows per block, one warp per row of `w_up`
   (`src/kernels/cuda/fused_gr.cu:258`).

`G` (the group split) is 1 here: `split_` is the opt-in `--spec-split` (`include/strata/core/verify.hpp:330`), so
there are **96 reads, 288 launches** per window (plus 2 router launches per layer, `src/core/verify.cpp:918-920`).

The profile's stage names are at `src/core/verify.cpp:1178`; the `hc0 norm/down/up` sub-stages are the stamps
`gr_read_group` passes into the read (`src/core/verify.cpp:693`, indices 27/28) and are derived at
`src/core/verify.cpp:1379-1384` - they cover **half 0 only**, so `hc-read1+router` is a whole read *plus* the
router GEMV, top-10 and plan publish.

### The measured split (baseline arm, `logs/ab-hc.csv`, profiled regime)

| stage | ms/window | per call | what moves | achieved |
| --- | ---: | ---: | ---: | ---: |
| `hc0 norm` | 0.31 | 6.5 us | 0.131 MB | **20 GB/s** |
| `hc0 down` | 0.65 | 13.5 us | 6.554 MB | 485 GB/s |
| `hc0 up` | 0.53 | 11.0 us | 6.554 MB | 596 GB/s |
| `hc-read1+router` | 1.96 | 40.8 us | 13.23 MB + 2.62 MB | 391 GB/s |
| **the path** | **3.45** | | **1.396 GB** | **405 GB/s** |

`hc-read1+router` minus the same read's work (norm+down+up = 31.0 us) leaves **~9.8 us per layer for the router's
GEMV alone** - 2.62 MB at 267 GB/s. The sweep's unprofiled baseline gives the same shape (norm 0.32, down 0.73,
up 0.59, `hc-read1+router` 2.12 = 3.76 ms of 15.75), so this is not a profiling artefact.

### The byte arithmetic

| tensor | shape | bytes/call | calls/window | bytes/window |
| --- | --- | ---: | ---: | ---: |
| `w_down` | `LR x D` bf16 | 6,553,600 | 96 | 629.1 MB |
| `w_up` | `D x LR` bf16 | 6,553,600 | 96 | 629.1 MB |
| `w_inject` | `HC x D` bf16 | 81,920 | 96 | 7.9 MB |
| `w_norm` | `D` f32 | 40,960 | 96 | 3.9 MB |
| `ffn_gate_inp` (router) | `512 x N` bf16 | 2,621,440 | 48 | 125.8 MB |
| | | | | **1.396 GB** |

**Comparison with the expert math** (`VRAM hits` + `PCIe grp`, 3.06 ms/window): the tier has 8408 slots in
15.95 GiB, i.e. ~1.94 MB per resident expert blob, and the window takes 27.86 VRAM hits per layer-window, so the
hit path moves `48 x 27.86 x 1.94 MB = 2.59 GB` in 3.06 ms - **~850 GB/s**. The hyper-connection path therefore
moves **half the expert math's bytes and takes more time**: 1.40 GB at 405 GB/s against 2.59 GB at 850 GB/s.

### So what is it, in one paragraph

It is **not** a weighted mix of streams, a split/merge or a cast that can be removed: it is 96 dense bf16
matrix-vector products per window - two per layer, one on each side of the attention - each of which must read
13.1 MB of hyper-connection weights (`w_down` 320x10240 and `w_up` 10240x320) because the weights do not fit
anywhere but DRAM (1.27 GB of them per window against 72 MB of L2). The bytes are irreducible at bf16; what is
not is the **rate**: the two big kernels run at 485 and 596 GB/s where the expert math reaches ~850, and the
norm's 96 launches move 131 KB each in 6.5 us (**20 GB/s**, i.e. almost entirely launch and dependency latency,
not bandwidth). The path is thus *latency-bound at the kernel level and bandwidth-bound only in principle*: on
this card `fused_gr_bench` reads the same 13.2 MB in ~31 us **with the weights already hot in L2**, so the
limiter is not DRAM at all.

---

## 2. Ranked candidates

Expected gains are `(per-call saving) x (calls per window)`, from the numbers above.

| # | candidate | mechanism | expected | risk |
| --- | --- | --- | --- | --- |
| **C1** | **prefetch the up projection's rows** (`gr_up_fast_kernel`) | the kernel issues one row's two `__ldg`s, consumes them, then the next row - 8 rows deep per warp, with the epilogue's scattered `R`/`w_norm` loads on the same chain (`src/kernels/cuda/fused_gr.cu:258-300`). The 8-lane-per-row form puts **both rows of a group and their epilogue inputs** in flight before either is used: 64 rows in flight per block instead of 8. | up 11.0 -> 6-8 us, **-0.3 to -0.5 ms/window** | none numerically: the same xor tree (see §4) |
| C2 | register-resident norm | `gr_norm_split_kernel` re-reads `R` and `bo_prev` after the reduction (`:530-560`); each thread owns 2-3 `float4` of its stream and could keep them. | norm 6.5 -> 4-5 us, -0.12 to -0.2 ms/window | bitwise-safe only if the element->thread mapping and the tree are kept exactly |
| C3 | more blocks for the down projection | 41 blocks x 8 warps = 41 of 128 SMs busy (`DOWN_BLOCKS = 40 + 1`, `:26`). Repackaging the same one-warp-per-row mapping into fewer warps per block raises the block count without touching the per-row summation order. | down 13.5 -> 10-11 us, -0.25 to -0.35 ms/window | each block re-stages the activation tile in its own smem, so the redundant staging grows with the block count |
| C4 | fuse the two reads into one | - | impossible: the attention's write lands between them (`src/core/verify.cpp:696` vs `:902`) | - |

C1 was built and measured (it is the change in §4): the prediction above is wrong in sign - the up projection got
11% slower, not 0.3-0.5 ms faster (§6). C2 and C3 are still predictions.

Two things that were checked and are **not** levers:

- **The launch structure.** `STRATA_HC_SPLIT=0|1|2` (plain / split / staged) measure the same on this card:
  `fused_gr_bench`, T 3, whole read: 30.7 / 30.7 / 31.5 us. The variants differ in *which* kernels run, not in
  how many bytes they move, and none of them is faster.
- **The K-split down (`STRATA_GR_V3=1`).** It exists, is opt-in, and the file's own comment explains why
  (`src/kernels/cuda/fused_gr.cu:318-323`: "The norm kernel runs on only T blocks (~16 us of pure latency per
  call) and `down` on 41 blocks (~280 GB/s)"). It is a different summation order, so it cannot be the default.

---

## 3. What the numbers say about the order

C1 was implemented because it is the largest single kernel with the clearest mechanism and a *bitwise-identical*
rewrite that already exists in the tree for gfx906. C2 is smaller than the noise floor on its own and C3 trades
one unmeasured cost (redundant staging) for another.

---

## 4. What was implemented

`src/kernels/cuda/fused_gr.cu`, behind `STRATA_GR_UP_PREFETCH=1` (**default off**) - built, checked bitwise,
measured, and reverted (§6):

- `gr_up_fast_kernel` and its `xor8` helper move out of the `#if defined(STRATA_HIP_GFX906)` block so they
  compile on CUDA; `xor8`'s shuffle width becomes `kGrShfl` (64 = the wavefront on gfx906, 8 = the warp's
  8-lane segments on CUDA) and uses `__shfl_xor_sync` on CUDA (the legacy `__shfl_xor` is gone in CUDA 13).
- `launch_multi` picks the kernel through one predicate, `up_prefetch()`: on gfx906 it is the existing
  `STRATA_GR_FAST` (default **on**, behaviour unchanged); on CUDA it is `STRATA_GR_UP_PREFETCH` (default
  **off**), overridable by `fused_gr_set_fast()` so `fused_gr_bench` can compare the two in one process.
- No new allocation: the kernel uses the same two shared arrays as the kernel it replaces
  (`lo[kFusedGrMaxT][LR]` + `g[kFusedGrMaxT][HC][UPM_COLS]`, 4.1 KiB + 2 KiB). It does cost registers: the
  prefetch keeps both rows' weights and the epilogue inputs live across the two dot loops, and no
  `-Xptxas -v` pass was run to put a number on it (§7).

**Why it is bitwise.** The old kernel reduces each row with `warp_sum`, an xor tree over 32 lanes:
stage 16 pairs lanes `j` and `j+16`, stage 8 pairs `j` and `j+8`, then 4, 2, 1 inside the 8. The new one gives
lane `j` the values of old lanes `j, j+8, j+16, j+24` and computes `xor8((p0+p2) + (p1+p3))`: stage 16 is
`p0+p2` (fp addition is commutative, so the lane pair's result is the same value), stage 8 is
`(p0+p2)+(p1+p3)`, and 4/2/1 are shuffles inside the 8. The per-lane accumulator is also unchanged -
`dot8(wa) + dot8(wb)` in both. This is the argument the file already carries at `fused_gr.cu:912-918`, and it is
checked, not asserted:

- `fused_gr_bench` (the tree's own bench) compares every output (`R`, `lo`, `rs`, `inject`, `mixed`) of the two
  kernels for T = 1..6 x apply x inject = 24 combinations: **"bitwise equal" in all 24**.
- `gr_parity --selftest` compares the multi-token read against T single-token reads.
- The A/B's generated text is hashed per run (`out_sha` in the CSV) and is identical across arms.

---

## 5. The A/B

`ab-hc.ps1` at the box root (cloned from `ab-pin.ps1`; neither that file nor `ab-forecast-pin.ps1` was modified)
runs the best configuration above with **one** difference between the arms - `STRATA_GR_UP_PREFETCH` unset vs `1` -
three runs per arm, **interleaved** (off, on, off, on, off, on) so a drift over the session cannot land on one arm. The
floors (2,048 MB available RAM, 200 MB free VRAM) are sampled every 5 s by a watchdog that kills the `serve.ps1`
parent first and then the engine; the teardown waits for both to be gone and port 8099 to be free before the next
run. Both arms carry `STRATA_VERIFY_PROFILE=1`, which is what makes the per-kernel comparison possible; it also
turns the shared-expert stream fork off (`src/core/verify.cpp:907`), so the absolute ms/window here is ~2 ms above
the unprofiled baseline in §1 (15.75 ms). The delta is the measurement.

### The six runs

| arm | run | tok/s | ms/window | `hc0 norm` | `hc0 down` | `hc0 up` | `hc-read1+router` | GPU total | RAM min | VRAM min |
| --- | --: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1 | 136.6 | 17.80 | 0.31 | 0.65 | **0.53** | 1.96 | 14.84 | 13,393 | 821 |
| off | 2 | 136.9 | 17.76 | 0.31 | 0.65 | **0.53** | 1.96 | 14.79 | 13,514 | 821 |
| off | 3 | 137.1 | 17.74 | 0.30 | 0.65 | **0.53** | 1.96 | 14.79 | 13,542 | 843 |
| on | 1 | 135.1 | 17.99 | 0.31 | 0.65 | **0.59** | 2.01 | 14.89 | 13,371 | 797 |
| on | 2 | 133.9 | 18.15 | 0.30 | 0.65 | **0.59** | 2.01 | 15.02 | 13,367 | 797 |
| on | 3 | 135.8 | 17.90 | 0.30 | 0.65 | **0.59** | 2.02 | 14.92 | 13,263 | 795 |

Stage columns are ms per window; `hc0 up` is stamped over 48 calls (half 0 only), so 0.53 -> 0.59 is
**11.0 -> 12.3 us per call**. `tokens_per_window` (2.43) and `hit_rate_pct` (95.1) are identical in all six runs,
and so is the output hash - the two arms did the same work and produced the same tokens.

| | off | on | delta |
| --- | ---: | ---: | ---: |
| tok/s, mean (spread) | 136.87 (0.5) | 134.93 (1.9) | **-1.94 (-1.4%)** |
| ms/window, mean (spread) | 17.767 (0.06) | 18.013 (0.25) | **+0.247 (+1.4%)** |
| `hc0 up` | 0.530 (0.00) | 0.590 (0.00) | **+0.060 (+11.3%)** |
| `hc-read1+router` | 1.960 (0.00) | 2.013 (0.03) | +0.053 (+2.7%) |
| `hc0 norm` | 0.307 | 0.303 | -0.004 |
| `hc0 down` | 0.650 | 0.650 | 0.000 |
| GPU total | 14.807 (0.05) | 14.943 (0.13) | +0.137 (+0.9%) |
| output sha | `ADA8FF3BFAB30C26` | `ADA8FF3BFAB30C26` | identical |

Floors: available RAM never fell below 13,263 MB (floor 2,048) and free VRAM never below 795 MB (floor 200) -
no floor violation, so no run is excluded for that reason.

The internal control is the point of the stage columns. The flag changes **one** kernel, and exactly one stage
moves: `hc0 up` +0.060 ms, dead flat within each arm (0.53 x 3 against 0.59 x 3), while `hc0 norm` and `hc0 down` -
same read, same launches, untouched by the flag - are unchanged to the hundredth. `hc-read1+router`, the other
half's read, moves with it (+0.053), which is what a slower `up` in a serial stream should do to the span that
follows it.

---

## 6. Verdict

**It does not pay, and the change is reverted.** The rule set before the run was that it pays only if the window
falls by more than the noise floor (~0.3 ms). It rose: **+0.247 ms/window (+1.4%)**, **-1.94 tok/s (-1.4%)**. The
decisive number is the kernel itself: `hc0 up` went **0.530 -> 0.590 ms/window**, 11.0 -> 12.3 us per call, **+11.3%**,
with no overlap between the arms - 0.53 in all three off runs, 0.59 in all three on runs.

**How much the whole-window number carries.** Less than the stage columns, and that should be said plainly. One
off-arm run was made to smoke-test the harness before the scored sequence and was discarded as a cold start; it
measured 18.02 ms and 134.9 tok/s (`logs/ab-hc-probe.csv`) - as slow as the on arm - while its stamps were off-arm
values (`hc0 up` 0.53, `hc-read1+router` 1.96, GPU total 14.91). A session's first run can therefore cost the whole
0.25 ms in the *unstamped* part of the window, so at n=3 the wall-clock delta alone would not have separated the
arms. What does separate them is that the flag changes exactly one kernel and exactly one stage moves, flat within
each arm, while `hc0 norm` and `hc0 down` - same read, same launches, untouched by the flag - do not move at all.

**Why it is slower, as far as these numbers say.** The premise in §2 was that the old kernel's one-row-at-a-time
chain left a warp with only two loads outstanding. The new kernel does put 64 rows in flight per block instead of 8,
and it is slower - so the old kernel was not load-latency-starved at this block count. `UPM_BLOCKS = 160` already
puts more blocks on the card than it has SMs, so a warp's shallow chain is covered by other blocks' chains; adding
depth *inside* a warp buys what inter-block concurrency was already providing, and it is paid for in registers, which
is what decides how many of those blocks stay resident. There is a second reason to distrust the port specifically:
the gfx906 form was written for a 64-lane wavefront, where an 8-lane group is an eighth of the wave; a CUDA warp is
32 lanes, so the same 8-lane grouping gives each warp four groups and the in-flight shape the AMD comment describes
does not transfer. Neither explanation is measured - no occupancy or counter read was taken (§7) - but both are
consistent with a kernel that got 11% slower moving identical bytes with an identical launch count.

**What this removes.** C1 is dead as implemented: row prefetch in the up projection is now a measured negative on
this card, not an untried idea. That leaves C2 (norm, -0.12 to -0.2 ms) and C3 (down, -0.25 to -0.35 ms), both still
arithmetic from §1 rather than results. The more useful reading is about the shape of the path: at 405 GB/s against
the expert math's ~850, with the same 13.2 MB read taking ~31 us L2-hot in the bench, the gap is not per-warp load
depth. If the up projection is revisited it should be attacked at the block and occupancy level - a register-capped
variant (`__launch_bounds__` with a min-blocks argument) or the same rows spread over more blocks - and only with an
occupancy read in hand.

The tree state: `src/kernels/cuda/fused_gr.cu` is back at `f4de37d` and the box copy was rebuilt from it, so the
engine on the box and in the branch match, and neither carries the flag. The only change this track leaves behind is
this document, the harness and the logs on the box.

---

## 7. What this does not establish

- The up kernel's time is not measured in isolation on the GPU: the engine's stamps give it per call, but the
  only standalone instrument (`fused_gr_bench`) times the whole three-kernel read, and its two variants differ by
  0-3.8 us run to run at T 3 with L2-hot weights.
- The expert math's ~850 GB/s is inferred (hits x slot size from `8408 slots, 15.95 GiB`), not read from a counter.
- C2 and C3 were not implemented or measured; their expected gains are arithmetic from §1, not results.
- The A/B is in the profiled regime. The same delta in the unprofiled regime is not measured here, and the regime
  matters for this stage: the unprofiled sweep in §1 puts `hc0 up` at 0.59, the profiled baseline here at 0.53, so the
  0.06 ms gap the flag produces is the same size as the profiled-vs-unprofiled difference of that stage. The delta is
  only valid *within* the profiled regime.
- No occupancy, register or hardware-counter read was taken, so §6's mechanism is a reading of the timings and of the
  kernel's shape, not a measurement of what the SMs did.
- The wall-clock arm separation (+0.247 ms) rests on the stage stamps, not on throughput: a discarded cold-start run
  of the faster arm was as slow as the slower arm (§6), so n=3 whole-window numbers do not establish the 0.25 ms on
  their own. The stage stamps do establish the 0.06 ms on the kernel the flag changes.
- Nothing here says the up projection is at its floor. It says one specific rewrite of it is worse.
