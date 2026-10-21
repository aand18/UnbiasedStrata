# Plan: performance of UnbiasedStrata on Qwen3.8-Flash-Next, from the SPECTER study

Written 2026-10-21 on the `spectre-moe` branch, after `docs/SPECTER-FEASIBILITY.md` (the study) and
`aand18/exllamav3` PR #3 (the same question for the sibling engine, canonical there).

## 0. The destination, and the one measurement it turns on

**Destination:** more tokens/second out of UnbiasedStrata on the Windows 4090 box (63.6 GB RAM,
Ryzen 9 7950X3D 16C/32T, RTX 4090 24 GB, PCIe 4.0 x16 / 26.8 GB/s), serving Qwen3.8-Flash-Next at
IQ3_S through the fork's own front end. The exllamav3 side of the same question is a **separate
session** (Track E below); this document covers this engine.

**Measured baseline** (production serve path: `--low-ram on --ple-io direct --ctx 262144 --kv fp16
--kv-resident 32768`, `STRATA_DECODE_TIMING=1` + `STRATA_VERIFY_PROFILE=1`, warm tier, 65 windows):

| term | ms/window | share |
| --- | ---: | ---: |
| window (2.47 tokens/window, T = 3.00) | 19.56 | |
| **GPU-reach wait** (host waits for the GPU) | **9.90** | **51%** |
| per-layer host, of which the CPU expert pool | 4.01 / **3.53** | 21% / **18%** |
| draft (MTP + suffix) | 2.05 | 10% |
| commit/emit | 0.07 | 0.4% |
| GPU stages: VRAM hits / waitCPU / PCIe grp | 2.66 / 1.70 / **0.19** | |
| per layer-window: 28.25 VRAM hits, 1.68 CPU experts, **0.00 PCIe** | | |

126 tok/s. Tier: 7,998 resident experts of 24,576 (29%); decode hit rate 46.9% -> 94.2% over three
requests. **Host cost per offloaded expert: 44 us.** Bus cost of the same expert: 2.66 MB / 26.8 GB/s
= **99 us**.

**The one measurement that decides the paper's direction** is already in hand: `t_bus` (99 us) >
`t_host` (44 us) by 2.25x. PR #3's own decision rule (§5) says: when the host is faster than the bus,
keep the host-compute path and the look-ahead has no target. So **the paper's mechanism - hide the
PCIe transfer - cannot pay in this engine**, and the 0.00 PCIe experts per layer-window is the
measurement saying so from the other side. Everything below is what *does* pay.

## 1. The ranked levers, with their ceilings

| # | lever | ceiling on the decode | cost | gated by |
| --- | --- | --- | --- | --- |
| **L1** | **Tokens per window** (2.47 now): the window costs 19.56 ms and returns 2.47 tokens | **+40-60%** if the window can be widened and the acceptance held (see §2.1 for the arithmetic) | L1a: config only. L1b: a DFlash 2 port | A (measure the ceiling first) |
| **L2** | **The CPU pool** (3.53 ms, 18%): the paper's *idea* - predict the routing, get the misses into the VRAM tier | **-18%** (126 -> ~155 tok/s), and only if the predictor beats the frequency profile, which already answers 94% of the reads | a trace, an oracle, then a predictor | B (Phase 0 kills it if the oracle does not beat the profile) |
| **L3** | **The GPU-reach wait** (9.90 ms, 51%): the dominant term, and *not* the GPU's expert math (2.66 ms) | unknown, and it is the biggest term | profiling + engine work | D (not the paper's idea; a separate line of attack) |
| **L4** | **The tier's shape and the adaptive tier's knobs** (`--expert-cache-per-layer` is off in the fork's serving path; the Coder's tuned `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97` are not applied to IQ3_S) | unknown, cheap to find out | config only | A |

**The paper's mechanism is not on this list** - measured out (§0). The paper's *idea* is L2, and L2 is
second to L1.

## 2. The tracks

### Track A - free wins: configuration only, one afternoon of runs

Every item is a flag change against the measurement harness. The harness is
`gate-measure.ps1` + `gate-request.ps1` in `C:\Users\yoho\Downloads\UnbiasedStrata` (they start the
server with the production path and the decode timing on, then drive three requests). **Measurement
discipline** (from the sibling project's hard-won lesson): this box cannot be A/B'd across processes -
identical code and identical command have produced 39.3 vs 50.8 tok/s - so every comparison must be
**single-process interleaved**, and the tier's state (which warms across requests) must be held equal.
Read the per-request block: `decode timing`, `decode GPU stages`, `decode expert cache hit rate`,
`drafts accepted`.

**A1 - the drafter sweep (L1).** Sweep `--spec` (4 now; the window is `min(spec+2, 8)`), `--spec-min-p`
(0.5 now), `--suffix-draft`. The measured average T is 3.00 against a 6-row window, so the drafter is
not filling its window and there is headroom without any new code. Record per setting: tokens/window,
window cost, acceptance, and the distinct-experts-per-layer-window count. **The hypothesis to test:**
the window's cost grows *sub-linearly* in T (distinct experts per layer saturate), so tokens/window
grows faster than the cost. The measurement must confirm or kill it - if the cost grows linearly, L1 is
worth much less than the table says.

**A2 - `--expert-cache-per-layer` on** (L4). The flag exists (`generate.cpp:3441`) and the fork's
serving path does not pass it (`frontend/main.cpp:805`). The engine's own numbers for the tier's shape:
2.97% global vs 21.4% at 8 slots/layer and 70.4% at 64 - but that was measured with a static profile
and a 4096-slot tier, not this profile-filled 7,998-slot one, so the answer here is unknown.

**A3 - the adaptive tier's knobs on IQ3_S** (L4). `kFlashNextIq3S` in `frontend/model.hpp:31` carries
`kNoArgs`; the Coder carries the tuned `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97`. Sweep the
same values on IQ3_S.

**A4 - `--pcie-frac`** (L4, and a check on §0). The engine probes the link and picks 0.55 for native
packs, yet no PCIe group ran in the warm state. Sweep 0 / 0.55 / 1.0 and confirm the prediction: with
`t_bus` (99 us) at 2.25x `t_host` (44 us), the PCIe share cannot win, and the engine's own probe should
agree.

### Track B - the paper's idea, behind a gate

**B0 - the oracle (the gate, offline, no GPU, no code in the engine).** Take a routing trace
(`--dump-routing`) and compute, per layer and per window, the read coverage of: (i) the shipped
profile's top-B per layer (what runs today, 94%), (ii) the previous window's expert set (the cheapest
predictor), (iii) the union of two windows, (iv) **the oracle** - the window's own set, truncated to the
tier's budget. The oracle is the ceiling of *any* predictor.

**Acceptance criterion: the oracle must beat the profile by a margin worth the pool's 18%.** If the
oracle's coverage is not meaningfully above 94%, **stop** - the tier is already doing the work, and the
predictor has no headroom. The oracle is also what sizes the prize: the difference between the oracle
and the profile, times the pool's 18%, is L2's real ceiling.

**B1 - the predictor (only if B0 passes).** The routing-transition model PR #3 sketches -
`P(routing_L(t+1) | routing_L(t))`, learnable from a trace, well under a megabyte of VRAM. Plugged
into the adaptive tier's ranking (`generate.cpp:7961+`), with the frequency ranking as the fallback
and tiebreak. The engine's existing hooks make this a ranking-function change, not a new subsystem:
per-layer budgets (`expert_cache.hpp:161`), the async copy path with its stream and event, and the
one-place hit decision (`ExpertDispatch::is_hit`).

**B2 - the capture points.** DFlash 2's capture (target layers 5, 15, 25, 35, 45, averaged over the
four hyper-connection streams, 12,800 features per position) is a free feature source for a predictor of
the form `P(routing_l | captured features)` - the "Speculating Experts" estimator shape. It needs the
capture in this engine, which the SGLang patch shows is a few lines at five layers.

### Track C - DFlash 2, gated by A1

`incoai/Qwen3.8-Flash-Next-DFlash2`: 0.6 B, Apache-2.0, 5 dense sliding-attention layers,
`is_causal: false`, `use_cache: false`, block size 8, a top-16 selector, and an acceptance length that
beats the model's own MTP on every task published (GSM8K 3.26 vs 3.08, HumanEval 2.94 vs 2.52,
MT-Bench 2.83 vs 2.50). ~1.2 GB BF16.

**Gate: A1 must show that the existing drafter cannot reach the same acceptance.** If tuning gets
tokens/window to ~2.8 on its own, DFlash 2's marginal value is small and the port is not worth the
draft loop.

**If it passes**, the port is: load the draft (1.2 GB, a new model type in this engine), the block
diffusion pass (one masked parallel forward over the block), the selector, and the target's capture at
five layers. That is a new subsystem in an engine that has none of it - the capture patch upstream
targets SGLang's `qwen4_exp.py`, and `z-lab/dflash` lists SGLang, vLLM, oMLX and a llama.cpp PR, but
nothing here.

**Note what DFlash 2 is not:** it is dense, it has no router, so it predicts *tokens*, not experts. It
is L1's lever and B2's feature source, not L2's predictor.

### Track D - the elephant (not the paper's idea)

The GPU-reach wait is 51% of the window and the pipeline is close to serial: the GPU idles 1.70 ms per
window waiting for the host's misses (`waitCPU`), and the host waits 9.90 ms for the GPU. The GPU's own
9.90 ms is not mostly expert math (2.66 ms) - it is attention, GDN, the hyper-connection reads, the
router, the head and the QSA indexer. Two questions, both measurable with the harness already in
place: (i) how much of the GPU's time is the router and the hyper-connection read (1.96 ms/window across
the two layer kinds), and (ii) can the hit path start before the misses land, given that the hits do not
depend on the misses - only the combine does.

This is the biggest term and it is not the paper's problem. It is listed because the paper's study
redirected attention here, and because Track A will not move it.

### Track E - the exllamav3 session (separate)

The sibling question, canonical in `aand18/exllamav3` PR #3 (updated 2026-10-21 with the gen4 x16
regime named, the DFlash 2 note, and the host-cost cross-check). Its scope:

1. **M1, the gate:** measure exllamav3's own host cost per offloaded expert. This study assumes
   186 us; the Strata engine measures **44 us** for the same bytes on the same CPU. The gap is engine
   quality, not the box - so exllamav3's number decides whether "stream the weights to the GPU" wins
   there (it wins by ~2.8x on the offloaded part at gen4 x16 *if* the look-ahead hides 60%; it loses if
   the host is as fast as Strata's).
2. **M4, the router export** (the primitive every predictor, profiler and prefetch needs).
3. **Routing autocorrelation** - temporal and cross-layer - which is the predictor's input data and
   also the routing trace that B0 needs.
4. Then the streaming path and the look-ahead, in PR #3's phases.

The EXL3 checkpoint is not a prerequisite: `turboderp/Qwen3.8-Flash-Next-exl3` at 2.05 bpw and 3.05 bpw
are already on the box (`D:\llms\Qwen3.8-Flash-Next-exl3-*`).

## 3. Order of work, and what each step is allowed to assume

1. **A1-A4** (config only, one afternoon). No code, no engine change, no risk to the production path.
   Output: the drafter's ceiling (L1), the tier's shape (L4), and the pcie_frac check.
2. **B0** (offline, one script, one trace). Output: L2's real ceiling, or a stop.
3. **B1/B2** if B0 says yes. Output: a predictor beating the profile on the same budget.
4. **C** if A1 says the existing drafter saturates. Output: a bigger acceptance length.
5. **D** in parallel with 3-4, since it is independent and it owns the largest term.
6. **E** in its own session.

## 4. Risks

| Risk | Why it is survivable | Fallback |
| --- | --- | --- |
| The box's A/B noise (39.3 vs 50.8 tok/s for identical code, across processes) | every comparison is single-process interleaved, and the harness gives the per-request split | repeat the pair; if the intervals overlap, the effect is below the noise floor and the item is dropped |
| A wider window costs as much as it returns | A1 measures the cost curve as well as the tokens/window | if the cost is linear in T, L1 shrinks to the acceptance-only lever |
| The tier's warm state differs between runs (the hit rate climbs 47% -> 94% over three requests) | every configuration is measured from the same warm state, and the tier's state is reported per request | discard the first request; compare requests 2-3 |
| A predictor evicts a hot expert | the frequency fallback and the swap's own gain condition (`generate.cpp:7993`) keep the copy correct | the frequency ranking stays the tiebreak |
| DFlash 2's acceptance does not transfer to this engine | A1 bounds the existing drafter first; the port is gated on it | keep the MTP |
| The measurement harness itself perturbs the decode (`STRATA_DECODE_TIMING` adds event overhead) | the harness's numbers are compared against harness numbers, never across configurations | the profile mode's own cost is reported and subtracted |

## 5. Open questions

1. **What is the acceptance-length ceiling here?** A1 answers it, and it bounds C.
2. **Why is the GPU-reach wait 9.90 ms when the GPU's expert math is 2.66 ms?** The GPU stage profile
   breaks it down; the answer decides whether Track D is worth a week.
3. **Does the paper's *idea* have any headroom in this engine at all?** B0's oracle says yes or no, and
   it is cheap.
4. **Would the paper's mechanism work in exllamav3?** E1 says yes or no, with the same 44 us / 186 us
   question.
5. **Is the routing predictor's input the trace, the capture, or both?** The trace is what exists; the
   capture is what would be free once DFlash 2 is in.

## 6. Artifacts

- `docs/SPECTER-FEASIBILITY.md` - the study, with the measured status block (§0) and the DFlash 2
  section (§5.6). Canonical for this engine.
- `docs/spectre-paper-notes.md` - the paper's primary sources with per-claim citations.
- `aand18/exllamav3` PR #3 - canonical for the sibling engine; this plan's Track E is its §6 and §7.
- `C:\Users\yoho\Downloads\UnbiasedStrata\gate-measure.ps1`, `gate-request.ps1` - the harness; logs in
  `logs\gate-*.err.log`.
