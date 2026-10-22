# SPECTER in UnbiasedStrata - the feasibility study

Written 2026-10-21 against this tree's `spectre-moe` branch (HEAD `9605b4f`). The question: can the
SPECTER paper (<https://arxiv.org/abs/2511.14102>, "Specter: Fast MoE Inference via Quantization-Enabled
Multi-Step Speculative Offloading", ASPLOS '27) be implemented in this engine for Qwen3.8-Flash-Next?

The paper's own notes, with every claim traced to the arXiv HTML or the authors' repository, are in
[`spectre-paper-notes.md`](spectre-paper-notes.md). This document is the mapping of that paper onto this
engine and the decision that follows from it.

---

## 0. Status: measured on the target box, 2026-10-21

**The sibling study exists and is canonical for the other engine.** [`aand18/exllamav3` PR #3](https://github.com/aand18/exllamav3/pull/3)
("Specter feasibility for Qwen3.8-Flash-Next in exllamav3", branch `feat/specter-feasibility`, with
`doc/specter-feasibility.md`, `doc/specter-share.md` and `eval/specter_budget.py`) answers the same question
for the exllamav3 engine. Its verdict and this document's agree on both load-bearing points, reached
independently: **do not port the artifact; do port the predictor.** This file is the UnbiasedStrata-side
twin - the engine here has a different execution model (a CPU expert pool, not a host-RAM arena the GPU
streams from), so the numbers differ.

**What was measured here** (the gate: host cost against bus cost). RTX 4090 24 GB, Ryzen 9 7950X3D
(16 cores / 32 threads), 63.6 GB RAM, PCIe **4.0 x16** (26.8 GB/s probed), Qwen3.8-Flash-Next at IQ3_S
(native pack, 2.66 MB largest blob), the engine's own production serve path (`--low-ram on --ple-io direct`,
33.55 GiB of experts locked in RAM, 7,998 experts resident in VRAM), with `STRATA_DECODE_TIMING=1` and
`STRATA_VERIFY_PROFILE=1`. Warm tier, 65 windows, 2.47 tokens/window:

| term | ms/window | share |
| --- | ---: | ---: |
| window cost | 19.56 | |
| **GPU-reach wait** (the host waits for the GPU) | **9.90** | **51%** |
| per-layer host | 4.01 | 21% |
| &nbsp;&nbsp;of which the CPU expert pool | 3.53 | 18% |
| draft (MTP + suffix) | 2.05 | 10% |
| commit/emit | 0.07 | 0.4% |
| GPU's own expert math (`VRAM hits`) | 2.66 | 14% |
| GPU idle waiting for the host (`waitCPU`) | 1.70 | 9% |
| **GPU's PCIe work (`PCIe grp`)** | **0.19** | **1%** |
| per layer-window: 28.25 VRAM hits, 1.68 CPU experts, **0.00 PCIe** | | |

Three numbers decide the direction of this document's advice:

1. **PCIe is 0.19 ms/window and 0.00 experts.** There is no weight transfer to hide, so **the paper's
   mechanism has nothing to hide in this engine** - measured, not argued.
2. **The CPU pool is 3.53 ms of 19.56 (18%)**, so the paper's *idea* (predict the routing, get the
   experts ready earlier) addresses that 18% - and **the whole 18% is the prize**, not a fraction of
   it: the 5.9 points of reads the tier misses *are* the pool's work, so a tier that covered the
   window's expert set would take the pool to zero and the decode from 126 to ~150 tok/s (+19.5%).
   Whether a predictor can cover the tier's *missed* set (rather than the window's hot set, where the
   cheap predictors are worse than the tier) is the open measurement; see
   `docs/plans/specter-performance.md` §2.2.
3. **The host's cost per offloaded expert is 44 us** here (3.53 ms / 48 layers / 1.68 experts). PR #3's
   exllamav3 model assumes 186 us for the same quantity. The gap is ~4.2x and it is *engine quality*, not
   the box: Strata's hand-tuned AVX2/AVX-512 pool is far ahead of a general CPU-MoE path. Any design that
   compares "stream the weights to the GPU" against "let the host compute them" must use **44 us**, not
   186 us, or it will prefer the bus for the wrong reason.

**The smaller quant, measured (the IQ2_XS rig, 2026-10-22).** The IQ3_S regime above is a *RAM-starved*
regime: 50.3 GB of experts against 63.6 GB of RAM, so the resident copy is capped and the misses come
off the disk. The IQ2_XS trunk (~36 GB of experts) with the **same n-gram table** (the table is the same
file for every quant) runs with **19.94 GiB resident and 0 blob reads from the file** - the disk tier is
never touched. Three short requests (59-token prompt, 32768 context, `STRATA_ARENA_LOCK_GIB=8`):

| request | generated | tok/s | ms/window | tokens/window | hit rate | CPU experts/layer-window | PCIe/layer-window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (cold tiers) | 221 | 124.5 | 16.75 | 2.08 | 93.1% | 1.36 | 0.94 |
| 2 | 301 | 173.2 | 14.13 | 2.45 | 97.2% | 0.77 | 0.39 |
| 3 | 159 | 179.7 | 12.82 | 2.30 | 98.3% | 0.48 | 0.19 |

The floors held (RAM min 13,562 MB, VRAM min 797 MB). Three readings matter more than the tok/s:
- **The GPU-reach wait is unchanged: 9.06-9.15 ms, against 9.90 ms for IQ3_S.** It is the same term in
  both regimes, so it is *model-size independent* - it is Track D's 24% of the GPU's time, and no quant
  buys it away.
- **The per-layer host cost falls 3.30 -> 1.27 ms** (the CPU pool 2.64 -> 0.86 ms) because the tier
  misses less: fewer experts per layer-window, and the ones that miss come from RAM, not the disk.
- **The tok/s comparison with the IQ3_S baseline is not like-for-like** (short prompts, warm-up, a
  different context), so the rig answers the question it was built for - *the box's RAM economy* - not
  the question of which quant is faster.

**And the biggest measured lever is not the paper's at all: the draft's acceptance rate.** The window
costs 19.56 ms and yields 2.47 tokens, so 7.92 ms/token. Most of the window cost is *per window* (the
GPU's per-layer expert set, attention, GDN, router, head), not per token - so the tokens a window
*commits* is the first lever to pull. The engine's MTP plus suffix drafter reaches 2.47; see §5.6.

**But the sweep of §7's Track A corrected the obvious form of that lever** (the sweep's own numbers are
in `docs/plans/specter-performance.md` §2.1): **widening the window loses, and so does lowering the
acceptance gate.** On a noise floor of +-1.1 tok/s, `--spec 6` (T 3.08 -> 4.07) went 131.6 -> 122.9
tok/s and `--spec-min-p 0.2` went to 119.9: the window's cost is ~linear in T (T +32%, cost +31%), the
per-draft acceptance *falls* as the window widens (69% -> 66%), the CPU pool grows with the wider
window (2.90 -> 5.04 ms) and the hit rate falls (94.1% -> 92.9%). The lever is therefore the
acceptance **rate at a fixed window**: at T = 3.08 and 18.4 ms, 100% acceptance would be 3.08
tokens/window (167 tok/s, +27%) and 85% acceptance 2.77 tokens/window (150 tok/s, +14%). That is the
number a better drafter has to beat - not the published acceptance *lengths*, which are a different
quantity on a different harness (§5.6).

---

**The answer, in one paragraph.** The paper cannot be implemented as described, and the reason is not
incidental: its draft model *is* the target model, quantized to INT4, resident in VRAM - 62.8 GB for this
model's routed experts, on cards with 12-24 GB. Its premise is also wrong for this engine, and §0
measures it rather than arguing it: the PCIe path carries **0.00 experts per layer-window** and 0.19 ms
of a 19.56 ms window, because a CPU pool reads the expert bytes out of pinned host RAM and computes
them. What the paper's *idea* is worth - use the speculation phase to know which experts the next window
will want, and have them in VRAM before the window starts - is worth re-implementing here, but not with
the paper's draft model, and not because the paper says so. The mechanism that has to be built is a
**predictor of the next verify window's per-layer expert sets**, and the engine already holds every piece
except that predictor: the MTP drafter that knows the window's tokens before the window runs, the
target's own routers as BF16 host copies, the per-layer VRAM tier, the adaptive tier's async copy
engine, and the one-layer-ahead `RouterLookahead` that already does the predict-and-warm trick for the
file tier. The first thing to do is not code: it is an offline measurement, from a routing trace, of how
much of the tier's budget a predictor could win - because the tier's static frequency profile plus its
adaptive tier already answers 72-94% of the reads, and a predictor has to beat *that*, not a blank
cache. Section 7 has the experiment; it costs an afternoon and it decides the project.

**One thing the measurement changed.** The ceiling is not what §5 first estimated. On the target box the
CPU pool is 18% of the window and the GPU's own wait dominates (51%), so the paper's idea is worth at
most -18% here - while the **acceptance length** (2.47 tokens/window) is the bigger lever, and
[DFlash 2](#56-dflash-2-a-better-drafter-and-free-routing-features) (§5.6) is a 0.6 B drop-in for exactly
that.

---

## 1. The paper, in the terms this engine uses

### 1.1 What it is

SPECTER is a **single-GPU, single-request, batch-1** system for MoE models whose experts do not fit in
VRAM. Its three components (v2 §3.3, the repository's `docs/design.md`):

| Component | What it does |
| --- | --- |
| **SIC** - Speculative Inference Core | Runs an INT4 copy of the target model on the GPU, drafting K tokens. |
| **PIO** - Predictive I/O Orchestrator | Reads the draft's router output per layer, keeps per-layer counters of which experts the drafted tokens would select, ranks the top-M per layer, and issues the H2D transfers of the experts the target is about to need. |
| **SEE** - Streamlined Execution Engine | Verifies the K drafted tokens in the target, with the MoE reordered expert-wise (gather tokens by expert, one dense GEMM per expert, scatter back) so that cache-resident experts are computed first and the missing ones' transfers overlap the compute. |

The target's experts live in **pinned host RAM** (256 GB on their A100 workstation); only the draft, the
trunk, the shared experts and a small per-layer resident tier are in VRAM. The speedup comes from turning
an exposed PCIe stall into a hidden one: their per-iteration breakdown (v2 Table 5a) reports a hidden-H2D
ratio of 55.5% (DeepSeekV2-Lite), 60.5% (Qwen1.5-MoE), 22.4% (Phi-3.5-MoE), and the headline is an
average **1.50x** TPOT over **SpecMoEOff** (their own retained implementation of the strongest baseline),
up to 1.88x, on one A100-40GB over PCIe 4.0 x16 (31.5 GB/s).

### 1.2 The two facts that decide everything

**(a) The draft is the target, quantized.** v2 §1: *"we discover that a low-bit quantized version (e.g.,
4-bit integer (INT4)) of the target MoE model itself serves as an ideal, zero-training-cost predictor."*
No distillation, no auxiliary loss - the fidelity of the routing prediction is an empirical property of
4-bit PTQ: 90.8% mean top-4 expert-ID match for Qwen1.5-MoE, 96.3-96.5% for Phi-3.5-MoE and
DeepSeekV2-Lite, and 95.7% even for GGUF Q2 (`spectre-paper-notes.md` §B.1-§B.2). This is the single
most important fact for a port: the paper's layer correspondence is 1:1 *because the draft is the target*,
and the paper says explicitly that without that correspondence the problem is unsolved (§2.2: *"lack of
routing correspondence ... making it impossible to directly map its expert activations to the target
model"*).

**(b) The prediction is not the bottleneck.** v2 Table 6a varies the prefetch accuracy from 1.000 down to
0.410 and the TPOT moves from 63.0 ms to 84.2 ms - and the *oracle* (accuracy 1.000) is 63.0 ms against
their own 69.2 ms, i.e. **perfect prediction is worth 1.10x on top of their system**. The rest of their
1.50x comes from the fused MoE draft kernel and from the residency manager's scheduling (v2 Table 5b:
full 69.2 ms -> without PIO 83.5 ms (1.21x) -> without PIO+SIC 94.4 ms (1.36x) -> without all three
130.4 ms (1.88x)). Their own natural recall is 0.930-0.961 (§4.6), i.e. they are not near the ceiling of
the predictor either.

Read together, (a) and (b) say: *the paper's contribution is the runtime, not the predictor.* A port
that can only bring a predictor - and not the runtime restructuring - is porting the 1.10x part.

### 1.3 What a faithful port would need from this tree

| Requirement (v2 / the repository) | This engine | Verdict |
| --- | --- | --- |
| INT4 GPTQ (sym, g128) copy of the target's routed experts, all of it in VRAM | the experts are IQ3_XXS in GGUF/pack form; no GPTQ path, no W4A16 GEMM (the engine's expert formats are the IQ family plus Q8_0 dense) | **missing** |
| Routers in FP16 on device, shared with the draft | the routers are BF16 (`ffn_gate_inp`, `Bf16InF32`) and already copied to the host for `RouterLookahead` (`generate.cpp:3692-3699`) | **present** |
| Pinned host expert pool for async H2D | `PinnedArena` + `cudaHostRegister`; `ArenaExpertSource::device_alias` gives the device alias (`expert_source.hpp:656-712`) | **present** |
| Dedicated I/O stream, CUDA events, non-blocking H2D, per-slot readiness/last-use | the adaptive tier already does exactly this for its swaps: `adapt_stream`, `adapt_ev`, a `pending` list, `cudaMemcpyAsync` (`generate.cpp:7961-8060`) | **present** |
| One contiguous byte blob per expert, so a prefetch is one copy | the canonical pack's blob is 1,382,400 B (`expert_source.hpp:403`); the native pack's is per-layer variable, ~1.746 MB average at IQ3_XXS | **present** |
| Per-layer resident slots, LRU eviction, slot protection while in use | per-layer admission and per-layer swaps exist (`expert_cache.hpp:161-164`, `generate.cpp:7961+`); the eviction policy is frequency-with-decay, not LRU | **present, different policy** |
| Full speculative decoding: draft tokens, target verification, accept/reject, KV rollback | `src/spec/`, `src/core/mtp.cpp`, `src/core/verify.cpp`, `src/core/coupled_draft.hpp` | **present** |
| A draft model with 48 layers of the target's own architecture | the MTP drafter is **one** layer - the model's own `nextn` block, with its own 512-expert MoE in VRAM (`mtp.hpp:9-25`) | **the blocker, §4.1** |
| A draft that proposes a whole block in one pass (SPECTER needs multi-token lookahead, and the paper's own draft is AR) | **DFlash 2**: 0.6 B, 5 dense sliding-attention layers, `is_causal: false`, `use_cache: false` - no recurrence, one parallel pass over the block; 1.2 GB BF16; no port exists here yet | **exists upstream (§5.6)** |

Everything except the draft model is already here. That is the good news and the bad news: the paper's
runtime is largely reimplementable, and its one indispensable asset does not fit.

---

## 2. What this engine's decode path actually is

### 2.1 The round

`src/program/generate.cpp:7188-7260`. One round is:

1. **Verify window** - `ver.run(T, ...)` (`verify.cpp:1390`). T <= 6 in the fork's serving path
   (`frontend/main.cpp:805` passes `--spec 4`; the suffix drafter adds two, `generate.cpp:1790-1792`).
   For each of the 48 layers: the router (`layer.cpp:355-373`, top-10 of 512), then the split of every
   (token, expert) entry into *VRAM hit* / *PCIe* / *peer GPU* / *CPU* (`expert_source.cpp:2128-2131`),
   then the CPU pool computes the misses (`expert_source.cpp:2233-2236`), then the GPU's grouped kernels
   compute the hits and the PCIe share, then `moe_combine`.
2. **Commit** - the accepted prefix is written to the session (`verify.cpp:1699`).
3. **MTP draft** - `mtp.draft(T, ...)` (`mtp.cpp`), up to 4 tokens from the model's own MTP layer, plus up
   to two from the suffix drafter.

The draft phase runs **after** the window it belongs to has been verified and committed, and **before**
the next window starts. That is the slot a SPECTER-style prefetch would have to use, and it is worth
about 1.2 ms per draft token by the cost model's own prior (`include/strata/spec/controller.hpp:34`).

### 2.2 Where the time goes

The engine has measured this, and the numbers are the whole argument:

| Measurement | Number | Source |
| --- | --- | --- |
| Expert bytes read per token (no VRAM tier) | 663.6 MB = 48 layers x 10 experts x 1,382,400 B | `expert_cache.hpp:4` |
| At ~40 GB/s of host DRAM | 16.2 ms, on a token of ~53 ms | `expert_cache.hpp:4-5` |
| CPU pool's own time, and how much of it overlaps | 19.035 ms, of which **1.055 ms is hidden** - 96% exposed, "because the residual chain is strictly serial" | `expert_cache.hpp:5-7` |
| GPU's expert kernel vs the CPU's | 159 GB/s vs 35 GB/s | `expert_source.hpp:311` |
| Measured effect of the VRAM tier | `--expert-cache 4096 --expert-cache-per-layer`: **54.4% hits**, CPU pool drain **19.1 -> 10.3 ms/token**, **-2.7 ms/token** end to end | `generate.cpp:672-675` |
| What the tier's per-layer budget is worth | 21.4% hits at 8 slots/layer, 70.4% at 64 | `generate.cpp:690-696` |
| What a static frequency profile is worth | 72% of the expert reads hit the GPU (Coder, 12 GB card) | `docs/upstream/DETAILS.md:263` |
| This fork's measured decode | 107 tok/s on an RTX 3090 with IQ3_XXS (9.3 ms/token) | `README.md:174` |

Two consequences, and they point in opposite directions:

- **The CPU pool is the engine's bottleneck.** On the fork's own numbers, 107 tok/s is 9.3 ms/token, of
  which the CPU pool's expert reads are the largest single term. Whatever does not reduce the CPU pool's
  work cannot move the token time much.
- **The CPU pool's work is set by the VRAM tier's hit rate**, and the tier is small: the measured free
  VRAM at steady state was 6,414 MiB (`expert_cache.hpp:13`), which is ~3,700 experts of the IQ3_XXS
  pack (1.746 MB each) out of 24,576 - **about 15% of the expert space**. The other 85% of the routed
  expert bytes are read by the CPU out of host RAM and computed there. That is structural for this
  model on this class of card, and no amount of scheduling changes it.

### 2.3 Where the bytes come from

Three tiers, in the order the engine prefers them:

| Tier | What it is | Where the bytes are read | Prediction? |
| --- | --- | --- | --- |
| VRAM tier (`ExpertCache`) | ~3,700 expert blobs resident on the GPU, filled at start from a frequency-ranked profile and then swapped by the adaptive tier | VRAM | no (frequency, from the past) |
| PCIe share (`--pcie-frac`, default 0.2 for the Q2_0 pack, 0.55 for native packs) | a share of the misses, read by the GPU over PCIe: DMA staging, a mapped-arena alias, or a copy kernel inside the graph (`expert_source.hpp:216-231`) | PCIe, overlapped with the CPU pool of the *same* layer (`expert_source.cpp:2124`) | no |
| CPU pool | everything else | pinned host RAM at ~40-44 GB/s, computed by AVX2/AVX-512 kernels | no |

**There is no predictor in the default path.** `RouterLookahead` - the one prediction mechanism in the
engine - is wired only when the source is the *GGUF-in-place* file tier: `generate.cpp:3689` requires
`srcp == &src && src.warms()`, and `warms()` is true only for the native GGUF-in-place source without
unbuffered reads (`expert_source.hpp:536`). It predicts **one layer ahead** (layer l+1's router applied to
layer l's MoE input, `expert_source.hpp:169-173`, `expert_source.cpp:1191-1214`) and it only **warms OS
pages** for the file tier (`expert_source.cpp:1130`). In the default RAM-resident mode there is no
prefetching of experts at all.

---

## 3. Component-by-component mapping

SPECTER's mechanism, and this engine's counterpart:

| SPECTER | This engine | Same thing? |
| --- | --- | --- |
| INT4 draft of the target, in VRAM, drafting K tokens | the model's own MTP layer (one layer), in VRAM, drafting <= 4 tokens | **no** - one layer, not 48; and it is the model's `nextn` block, not a quantized copy of the trunk |
| Draft's per-layer router output -> the prefetch plan | the target's own routers, as host-side BF16 copies, applied to whatever hidden state is at hand (`RouterLookahead`) | **partly** - the right instrument (the router), the wrong source (no layer-wise draft hidden states) |
| Per-layer expert counters (ELB), top-M per layer, one batched D2H per checkpoint | `ExpertDispatch::usage`, a decayed per-(layer, expert) count of the routing so far, consumed by the adaptive tier | **the same shape, a different signal**: counts of the past, not a prediction of the future |
| Per-layer resident slots, LRU, slot protection | per-layer admission and per-layer swaps; protection via the `pending` list and `adapt_ev` | **yes, in mechanism** |
| Async H2D on a dedicated stream, events for readiness and reuse | `adapt_stream`, `adapt_ev`, `cudaMemcpyAsync` for the swaps; the verify path's `fetch_dma` for the PCIe share | **yes** |
| Three-phase prefetch planner (cold start / overlap / cleanup), chunk boundary chosen so draft compute ~= H2D time | none | **missing** |
| Expert-wise verification reorder (gather by expert, dense GEMM, scatter) | the grouped hit kernels already group by expert per layer (`moe_hit_grouped_s2`), and the CPU pool's multi-token jobs group the misses by expert too (`expert_source.cpp:2233-2236`) | **already there in a different form** |
| Draft kernel fusion (one fused MoE launch for the draft) | the MTP layer's 512 experts run through the grouped hit kernels (`mtp.hpp:19-25`) | **already there** |
| Target-authoritative KV, shared prefix, rollback on rejection | the session owns the KV; the MTP drafter keeps its own ring (`mtp.hpp`, `coupled_draft.hpp`) | **already there** |
| Offline profiling of K | `DraftPolicy` learns the window size from measured round times (`src/spec/draft_policy.cpp`) | **already there, and better adapted** |

**Summary of the mapping.** Of SPECTER's nine mechanisms, five are already in this engine (in a
different but equivalent shape), one is impossible (§4.1), one is missing but cheap (the prefetch
planner), and one - the predictor - exists in a form that only works for the file tier and only one layer
ahead. The paper's *system* is therefore about 70% present. Its *asset* is not.

---

## 4. The blockers

### 4.1 The draft model does not fit, and cannot be made to fit

SPECTER's draft is the target model with its routed experts quantized to INT4 (GPTQ, symmetric, g128).
For Qwen3.8-Flash-Next:

| Quantity | Number | Source |
| --- | --- | --- |
| Routed experts | 48 layers x 512 experts | `layout.hpp:29, 52` |
| Parameters per expert | 3 x 2560 x 640 = 4,915,200 (gate, up, down) | `layout.hpp:28, 53` |
| Routed expert parameters | **120.8e9** | computed |
| Experts at IQ3_XXS | 42.9 GB -> 2.84 bits/param | `README.md:134` |
| The same experts at INT4 g128 (4 bits + fp16 scale/128 + packed zero) | 4.16 bits/param -> **62.8 GB** | computed |
| The tier the engine can hold, for comparison | 4,105 slots x 1,382,400 B = 5.67 GB (measured 5,413 MiB free) | `expert_cache.hpp:13-15` |

62.8 GB of VRAM for the draft alone, on cards with 12-24 GB (`README.md:20-25`). Even sharing the
trunk, the routers, the shared experts and the KV cache - the paper's own sharing ladder, worth 43% of
the footprint there - the routed experts are the irreducible part: **62.8 GB**.

The reason is scale, and the paper's numbers show it is the model, not the format: SPECTER's draft of
Qwen1.5-MoE costs +6.13 GB (v2 Table 4), which matches its 12.4e9 routed parameters at 4.16 bits. This
model has **9.7x** that. The paper's smallest model is a 14.3B MoE with 60 experts of 2048x1408; this is
a 125B MoE with 512 experts of 2560x640.

Three ways out, and what each costs:

- **A shallower draft.** A draft of layers 0..N-1 predicts layers 0..N-1's routing *exactly* (the
  hidden state at layer l depends only on layers 0..l-1), so the correspondence problem does not bite -
  but the router at layer l reads a hidden state the target computes with *its* quantization, so a
  quantized draft drifts (the paper's own numbers: 90.8% for Qwen at INT4, and 84.7% for the best
  published *one-layer-ahead* predictor). VRAM: 1.31 GB per drafted layer, so N=2 is 2.6 GB and N=4 is
  5.2 GB. But a prefix draft of N layers has to *run* those N layers over the K speculated tokens, and
  for N >= 2 that means reading N x K x 10 expert blobs - the same bytes the verify pass will read. For
  N=2, K=6: 120 blobs x 1.746 MB = 210 MB, ~5 ms at 40 GB/s, against a 9.3 ms token. The prediction
  costs more than the layer it predicts.
- **A smaller expert set in the draft** (drop experts, keep the routers). This changes the routing
  space - the paper's §2.2 argument against a different draft model applies verbatim.
- **A draft on the CPU** with the CPU's own kernels. Then the draft is not cheaper than the target, and
  the CPU is the thing that is already saturated (§2.2).

Verdict: **the paper's draft is not implementable here**, and the variants that are implementable are
either inexact (quantization drift) or more expensive than what they save.

### 4.2 The premise - a PCIe-bound MoE - does not hold for this engine

SPECTER's gains are hiding H2D stalls (their exposed stall: 108.3 ms of a 243.4 ms H2D load for
DeepSeekV2-Lite, v2 Table 5a). In this engine, in the default configuration:

- the experts live in **pinned host RAM**, read by the **CPU** at 40-44 GB/s, and computed by the CPU -
  the bytes never cross PCIe (`ArenaExpertSource`, `expert_source.hpp:656-712`);
- the PCIe share exists only as a tunable fraction (`--pcie-frac`), and the engine *measures* the link
  and picks the share that is faster (`generate.cpp:2051-2063`) - because for a 1.746 MB expert blob the
  CPU's 40 us of DRAM read beats a PCIe 4.0 transfer's 70 us, even before the CPU's arithmetic;
- that PCIe share is **already overlapped** with the CPU pool's work for the same layer
  (`expert_source.cpp:2124`, "the copy engine, beside the CPU's work").

So the thing SPECTER hides is not this engine's exposed cost. This engine's exposed cost is the CPU
pool: 96% of 19.035 ms, because "the residual chain is strictly serial" (`expert_cache.hpp:6-7`).

There is one configuration where the paper's premise does apply: when the experts are behind the PCIe
link rather than behind the CPU - i.e. a `pcie_frac` near 1, which needs a fast link (PCIe 5.0) and a
slow CPU, or the file tier behind an SSD. In the file tier the engine already warms one layer ahead
(`RouterLookahead`), which is the same idea as SPECTER's PIO with a one-layer horizon.

### 4.3 The baseline is not this engine, so the paper's numbers do not transfer

The 1.50x is against **SpecMoEOff**, the authors' retained PyTorch implementation of a
speculative-I/O-pipelined offloading system (repository `baselines/specmoeoff/PROVENANCE.md`). The
paper's own ablation says the bulk of the gain is the fused draft kernel and the residency manager, not
the predictor (§1.2 (b) above): an *oracle* predictor buys 1.10x on top of their system.

In this engine the predictor would have to beat a static frequency profile that already answers 72% of
the reads with 15% of the experts resident. That is a much higher bar than the paper's, and it is an
empirical question about this model's routing, not about the paper.

---

## 5. What is actually worth building

Here is the shape of the thing that would pay, if the measurement in §7 says the predictor can win.

### 5.1 The observation the whole thing rests on

**A verify window's whole expert working set is about the size of the VRAM tier.**

- A window of T <= 6 tokens routes 10 experts per token per layer: **<= 60 distinct experts per layer**,
  **<= 2,880 (layer, expert) pairs per window** over 48 layers.
- At IQ3_XXS's 1.746 MB per expert, that is **5.0 GB**.
- The tier measured 6,414 MiB free at steady state (`expert_cache.hpp:13`): ~3,700 slots, ~6.4 GB.

So a predictor that knew the next window's expert set exactly could put **all of it** in VRAM, and the
CPU pool would have nothing to do: the GPU's grouped kernels run the expert bytes at 159 GB/s against the
CPU's 35 (`expert_source.hpp:311`). The measured anchor for what the tier is worth is
**-2.7 ms/token at 54.4% hits** on a 53 ms canonical-pack token (`generate.cpp:672-675`).

The ceiling is bounded by the pool's whole cost - 16.2 ms of that 53 ms token (`expert_cache.hpp:4-5`) -
so on the canonical pack a perfect predictor is worth at most about **1.4x**, and less on this fork's
9.3 ms IQ3_XXS token (`README.md:174`), where the pool's share of the token is smaller. The prize is
real but it is a 1.2-1.4x class of win on the decode, not the paper's 1.50x, and it only materializes if
the predictor beats the frequency profile for the same slots.

**Measured on the target box, the ceiling is much lower (§0):** the CPU pool is 3.53 ms of a 19.56 ms
window, so a predictor that moved every miss into the tier would buy **-18%**, not 1.4x. The reason is
the tier's shape on this box: 7,998 resident experts (29% of the 24,576) and a decode hit rate that climbs
to **94.2%** within three requests, against the 72% the Coder reaches on a 12 GB card. The frequency
profile plus the adaptive tier already do most of what a predictor could do here.

### 5.2 What the predictor has to beat

The tier is not a blank slate: it is filled at start from a frequency-ranked profile and then maintained
by the adaptive tier, which swaps in the most-routed missing experts every `adapt_every` rounds
(default 4, 96 swaps, decay 0.97 - `generate.cpp:474-497`, `7961-8060`). That policy is *frequency*,
which is the marginal distribution of the routing over the whole run - a strong predictor, since MoE
routing is heavily skewed (15% of the experts answer 72% of the reads).

A next-window predictor is a *conditional* predictor. It wins only where the conditional distribution
differs from the marginal one, i.e. where the routing is not stationary - at a topic change, a code
block, a tool call, a long structured output. **The measurement in §7 is therefore not a formality: it
decides whether the entire idea has a subject.**

### 5.3 The three candidate predictors, cheapest first

| # | Predictor | Cost | Fidelity | Where it plugs in |
| --- | --- | --- | --- | --- |
| **A** | **The previous window's routing for the same layer.** The engine already counts it: `ExpertDispatch::usage` is a decayed per-(layer, expert) count of the routing so far, and a routing trace has the previous window's exact set. A window's expert set for layer l is the natural prior for the next window's. | zero - it is already counted | unknown; measurable offline (§7) | the adaptive tier's ranking: replace "most routed ever" with "most routed, weighted by recency and by agreement with the previous window" |
| **B** | **Layer l+1's router on layer l's input, for the *next* window's tokens.** The mechanism is `RouterLookahead`'s (`expert_source.hpp:169-173`); the extension is from "this window's tokens" to "the next window's", whose tokens the MTP drafter has already produced (`mtp.draft`, `generate.cpp:7231`). | **not free, and this is where the paper's draft earns its keep**: the next window's layer-l MoE input is only known after running the target's layers 0..l-1 over the drafted tokens - a prefix forward that re-reads the very experts the verify pass will read (§4.1: N=2, K=6 is ~5 ms against a 9.3 ms token). In its cheap form - the *current* window's tokens - it is free, and the engine already does it (§2.3). | unknown; the paper's own comparison puts a one-layer-ahead predictor at 84.7% vs 90.9% for a full quantized draft | the file tier's page warm (today, one layer ahead) and the tier's admission (new) |
| **C** | **The MTP layer's own router, read out for the drafted tokens.** The MTP block has its own 512-expert MoE with its own router (`mtp.hpp:9-25`), and its top-10 for a drafted token is a *free* signal about that token. But it is the router of the `nextn` block - the model's deepest layer - so it speaks for layer 47, not layer 0. | zero - the grouped kernels already compute the router | probably weak for early layers, unknown | a per-layer prior, weighted by how well layer l's routing correlates with layer 47's |

Predictor **A** is the one to measure first and the one most likely to matter: it is free, it is already
computed, and it is the only one that needs no new machinery at all.

Predictor **B** has a shape worth spelling out, because it is the one the paper's draft model replaces.
`RouterLookahead` predicts layer l+1 during layer l, using layer l's MoE input: the residual stream
changes little from one layer to the next, so layer l+1's router applied to layer l's input is a good
guess. The horizon can be pushed further - predicting layer l+2 during layer l+1 needs layer l+1's MoE
input, which exists by then - at the cost of one layer of lead time per step and of a growing drift.
Pushing it to the *next window* is the step that needs the paper's draft, and without it the prefix
forward costs more than the layer it predicts.

Predictor **C** is the free one nobody has looked at: the MTP block has 512 routed experts of its own
and its router is already computed for every drafted token, so its top-10 is a signal about the token
that costs nothing. It is the router of the model's deepest layer, so it is a prior for layer 47, not for
layer 0 - but the *shape* of a token's expert affinity may well carry across layers.

### 5.4 The runtime piece, if the measurement says yes

The engine's own primitives cover most of SPECTER's PIO:

- the tier's slots are per-layer already (`expert_cache.hpp:161-164`; `--expert-cache-per-layer` exists
  and is *off* in the fork's serving path - `frontend/main.cpp:805` does not pass it);
- the adaptive tier's copy path is exactly a SPECTER-style async prefetch: a dedicated stream, an event,
  a pending list, `cudaMemcpyAsync` from the pinned host blob into the slot (`generate.cpp:8020-8040`);
- the swap decision is made once per layer, on the layer's ids, and shared by both sides
  (`ExpertDispatch::is_hit`, `expert_source.hpp:297-299`) - so a prediction-driven decision has a home.

What is missing is the *decision function*: instead of ranking candidates by decayed frequency, rank them
by the predictor, and let the frequency be the tiebreak. That is a change in one ranking function, not a
new subsystem - which is why the honest answer to "how hard is this" is: the hard part is the
prediction quality, and the easy part is everything else.

### 5.6 DFlash 2: a better drafter, and free routing features

[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (Inco AI,
Apache-2.0, published 2026-10-19) is a **0.6 B-parameter block-diffusion draft model for exactly this
target**. It is not a language model: it runs inside a speculative server and proposes tokens for the
target to verify. From its `config.json`: 5 layers, hidden 2560, sliding attention (all 5 layers),
`is_causal: false`, `use_cache: false`, block size 8, a selector that keeps the top 16 candidates per
position and traces one coherent path, and two-tap dynamic convolutions. Its acceptance length beats the
model's own MTP on every task measured (GSM8K 3.26 vs 3.08, HumanEval 2.94 vs 2.52, MT-Bench 2.83 vs
2.50).

Three things follow, and they are not the same thing:

1. **It is the cheapest possible way to raise the acceptance length** - which §0 shows is the *first*
   lever on this box, because most of the window's cost is per window, not per token. The engine's MTP
   plus suffix drafter reaches 2.47 tokens/window here; DFlash 2's published range is 2.83-3.26 on a
   different target and harness, so the number to beat is 2.47, not 3.26. It costs 1.2 GB of VRAM
   (BF16) - affordable next to a 13.4 GiB expert tier.
2. **It dissolves the sibling study's blocker #2.** PR #3 rules out an in-family draft because
   exllamav3 asserts the draft model has no recurrent state (`generator.py:162`) and this target's
   Gated DeltaNet layers are recurrent. DFlash 2's draft is **dense and non-recurrent** - 5
   sliding-attention layers, no KV cache, `is_causal: false` - so there is no recurrence to forbid, and
   it drafts a whole block in one pass. That is also, not coincidentally, what the paper wants: SPECTER
   needs *multi-step inter-token* lookahead, and a block-diffusion drafter produces the whole block at
   once.
3. **Its capture points are the paper's predictor, for free.** The SGLang capture patch
   (`patches/qwen38-dflash2-capture.patch`) makes the target hand the draft its **completed hidden states
   at layers 5, 15, 25, 35, 45**, averaged over the four hyper-connection streams - 5 x 2560 = 12,800
   features per position, from layers the target computes anyway. That is exactly the input the
   "Speculating Experts" style lightweight estimator wants (see the paper notes' §B.2 reference to a
   neural estimator over internal representations): a predictor of `P(routing_l | captured features)`
   costs well under a megabyte of VRAM and can be trained from a routing trace. So the paper's *idea*
   gets a feature source it did not have in the paper, where the draft had to be a 62.8 GB copy of the
   target.

**What DFlash 2 does not do:** its own routing is useless for the paper's ELB - it is dense, it has no
router at all - so it does not predict the target's experts. The routing predictor still has to be built,
from a trace (§7 Phase 0) and from the captured features. And the integration is not free: the patch
targets SGLang's `qwen4_exp.py`, DFlash's own repo lists SGLang, vLLM, oMLX and a llama.cpp PR
(ggml-org/llama.cpp#27342), but neither this engine nor the exllamav3 fork has a DFlash path. The
draft loop (one masked parallel pass, the selector, the capture) is the work.

### 5.7 What the port would look like if the engine ever became GPU-only

The paper's design becomes relevant the day the CPU pool stops being the executor - i.e. if the engine
moves to a regime where every routed expert is computed on the GPU, with the bytes fetched over PCIe or
from the SSD. In that regime the paper's three components map one-to-one onto this engine's
`ExpertSource`/`ExpertCache`/`mtp` split, the draft's role is taken by the MTP layer plus a *quantized
prefix* of the target (N layers, 1.31 GB each, §4.1), and the three-phase planner is the missing piece.
This is worth writing down because it is the condition under which the paper stops being a curiosity.

---

## 6. What the paper teaches that is worth keeping regardless

Three things in SPECTER that this engine does not have and should:

1. **Measure routing fidelity as a first-class metric.** The paper's Table 1 and Figure 2 make "how often
   does the predictor agree with the router" a number the system is sized against, with a threshold
   effect: at 0.575 accuracy their TPOT is 1.31x, at 0.410 it is 1.34x. This engine measures the
   *tier's* hit rate but never the *predictor's* accuracy. Add it.
2. **Separate "the prediction is wrong" from "the prediction was not used".** The paper's Table 6a is an
   accuracy sweep with everything else fixed; that is the experiment §7 is built on.
3. **Spend the speculation phase's wall clock on the next window, not on the current one.** This engine's
   draft phase is currently a pure token producer. It costs about 1.2 ms per draft token by the cost
   model's own prior (`controller.hpp:34`), and the CPU pool - the engine's exposed cost - is idle
   throughout it. That idle CPU is the free resource, and it is idle for a reason that has nothing to do
   with the paper: nothing has been asked of it yet.

---

## 7. The plan

### Phase 0 - measure, before writing any C++ (an afternoon)

**The question:** for a window of T <= 6 tokens and a per-layer slot budget of B slots, what fraction of
the window's expert *reads* does each candidate tier cover?

**The instrument:** the engine already writes the routing trace: `--dump-routing FILE`
(`generate.cpp:610`, `3797-3805`), and `tools/make_profile.py` already reads such traces and ranks
(layer, expert) pairs. A short script in `tools/` reads a trace and computes, per window:

| Arm | What it answers |
| --- | --- |
| the shipped profile's top-B per layer | the number to beat (this is what runs today) |
| the previous window's expert set per layer | predictor **A**'s score |
| the previous window's set, re-ranked by the profile's frequency as a tiebreak | a cheap hybrid |
| the window's own set, truncated to B | the **oracle**: the ceiling of any predictor |
| the union of the previous two windows | how fast the routing decorrelates |

**The acceptance criterion:** the oracle's read coverage minus the profile's, on the *same* budget, is
the size of the prize. If the oracle is not much above the profile (say under 15 points), **stop** - the
paper's own ablation says a predictor is worth 1.10x at best on a system built around it, and this
engine does not have that system. If the oracle is far above the profile but the previous-window arm is
not, the idea is dead in this form and only the file tier's warm (predictor **B**) survives.

The oracle number also sizes the project honestly: it is the difference between "the CPU pool stops being
the bottleneck" and "nothing changes".

### Phase 1 - the predictor as a measurement, not a mechanism

Two cheap moves on the one predictor that exists, both on the file tier, where the pages genuinely have
to be fetched and the warm is free:

1. Push `RouterLookahead`'s horizon from one layer to two - predict layer l+2 while layer l+1 computes
   (`expert_source.cpp:1191-1214` already has the layer's input in hand). The lead time per layer
   shrinks and the drift grows; the warm-hit rate it reports (`generate.cpp:8224-8228`) says whether that
   is a trade worth making.
2. Wire it in the **RAM-resident** mode too, where it is currently absent (`generate.cpp:3689` requires
   the GGUF-in-place file tier). There the "warm" is a no-op - the bytes are in RAM - but the same
   prediction is the input a VRAM admission policy would need (§5.4), so measuring it there is how the
   predictor's fidelity on this model gets established at all.

Neither move needs a draft model; both produce the number that decides Phase 2.

### Phase 2 - prediction-driven admission (only if Phase 0 says yes)

Change the adaptive tier's ranking function from "decayed frequency" to "predicted use", with the
frequency as the tiebreak and the frequency tier as the fallback when the predictor has no opinion.
Everything it needs exists: the per-layer budgets, the async copy path, the pending list, the one-place
decision (`ExpertDispatch::is_hit`). The measurement is the same as the tier's own: the hit rate and the
CPU pool drain, from `STRATA_DECODE_TIMING=1` and `--stats`.

### Phase 3 - the prefetch planner (only if the engine moves to the PCIe-bound regime)

The paper's three-phase planner, over the draft phase's ~1.2 ms per token. Not worth building while the
CPU pool is the exposed cost.

### Risks

| Risk | Why it is not fatal | Fallback |
| --- | --- | --- |
| The predictor does not beat the frequency profile | Phase 0 decides this before any code | the file tier's warm (Phase 1) is worth keeping on its own |
| The routing is non-stationary (a topic change) | the frequency tier absorbs the average; the predictor only has to help in the stationary case | keep both, weight by the predictor's recent accuracy |
| A predictor-driven tier evicts a hot expert | the frequency fallback and the `pending`/event machinery keep the copy correct | the swap is already conditional on a gain (`generate.cpp:7993`) |
| The GPU and CPU round the same expert differently | already true today, measured and documented (`generate.cpp:3443-3449`, 95-98% same top-1, equal perplexity) | unchanged - the tier already changes the tokens a little |
| The whole thing is a 1.10x-class win at best | the paper says so itself | then it is a 1.10x on the decode, which is still worth an afternoon |

### What would change the verdict

- The engine moving to a GPU-only executor (the CPU pool gone) - then the paper's system matters and
  this document should be rewritten around §5.7.
- A `pcie_frac` near 1 on a PCIe 5.0 host with a slow CPU - then the exposed cost becomes the transfer,
  and the paper's premise holds.
- A model with far fewer routed parameters per layer, where an INT4 draft would fit - then §4.1 dissolves
  and the faithful port becomes a project.

---

## 8. Open questions

1. **What is the temporal correlation of this model's routing?** Consecutive tokens, and consecutive
   windows, per layer. Phase 0 answers it; nothing else can be planned before it.
2. **Is the tier's budget better spent per layer or globally?** The engine has both (`--expert-cache-per-layer`,
   off in the fork's serving path) and has measured the difference for a *static* profile
   (2.97% global vs 21.4%/70.4% per layer). For a *predictor*, the right shape may be different again -
   the window's working set is per-layer uniform by construction.
3. **Does the paper's "exposed H2D" have an analogue here?** The engine's `STRATA_DECODE_TIMING=1` split
   (GPU-reach wait, per-layer host, CPU pool, stage) is the closest instrument; it has never been run
   against a predictor, so the baseline for the whole project is missing.
4. **Would the MTP drafter's own router (predictor C) be a useful layer-47 prior?** It is free, and the
   MTP layer is already resident. Nobody has looked.

---

## 9. Sources

- The paper and its artifact: [`spectre-paper-notes.md`](spectre-paper-notes.md) (primary sources, with
  per-claim citations), <https://arxiv.org/abs/2511.14102>, <https://github.com/wwwwwf/specter_asplos>.
- The sibling study for the other engine, canonical there: [`aand18/exllamav3` PR #3](https://github.com/aand18/exllamav3/pull/3)
  (`doc/specter-feasibility.md`, `doc/specter-share.md`, `eval/specter_budget.py`).
- The draft model of §5.6: [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
  its `config.json` and `patches/qwen38-dflash2-capture.patch`; method: `z-lab/dflash` (MIT), paper
  arXiv:2602.06036.
- The gate measurement of §0: this box's own run - `C:\Users\yoho\Downloads\UnbiasedStrata\logs\gate-*.err.log`
  (`STRATA_DECODE_TIMING=1`, `STRATA_VERIFY_PROFILE=1`), driven by `gate-measure.ps1` and
  `gate-request.ps1` in that tree; the engine's counters (`expert_cache.hpp`, `generate.cpp`) and
  `docs/upstream/DETAILS.md:182-191` (the tier counters) and `:263` (the Coder's 72% read hit rate).
- This engine's own measurements: `include/strata/core/expert_cache.hpp:3-17` (the CPU pool's cost and the
  tier's size), `src/program/generate.cpp:668-700` (the tier's measured effect), `README.md:169-181`
  (this fork's decode speed), `docs/upstream/DETAILS.md:182-191` (the tier counters) and `:263` (the
  Coder's 72% read hit rate).
- The prediction machinery that exists: `include/strata/core/expert_source.hpp:169-205` (`RouterLookahead`),
  `src/core/expert_source.cpp:1130` (the page warm), `:1191-1214` (the predictor's thread),
  `src/program/generate.cpp:3688-3705` (where it is wired - and where it is not).
