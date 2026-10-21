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

### 0.1 The box's floors, and the production configuration's breach of them

The box's rule is **never below 2 GB available system RAM, never below 200 MB free VRAM**. Measured
while the production configuration runs (`--low-ram on --ctx 262144 --kv fp16`):

| | measured |
| --- | --- |
| available RAM during the run | **0.0 GB** (the engine locks 33,557 MiB of experts and 6.00 GiB of pinned KV) |
| available RAM after the engine is stopped | 55.9 GB of 63.6 GB - the box recovers completely |
| free VRAM during the run | 747-865 MiB (above the floor) |
| commit limit / pagefile | 115 GB / 52.6 GB allocated, 4.7 GB in use |
| the mapped mode (`--low-ram off`) | still locks the arena's working set: **22,227 MiB**, and the unbuffered load then hit the commit limit: *"the expert load was refused: short unbuffered read of layer 0 ... (error 1453)"* |

So the production configuration **does not satisfy the floor**, and the box has no RAM to spare for
this model: the expert bytes (50.3 GB) and the n-gram table (28.8 GB) together are 83.6 GB against
63.6 GB of RAM, so every mode either locks a large working set or leans on the page cache. That is also
the neighbourhood of issue #1's hang ("a page-in while the verify graph spins on a host flag stalls the
request for good", `generate.cpp:5667-5672`).

**The compliant configuration**, measured to hold 3.9 GB of available RAM and 747 MB of free VRAM
through a full request:

| knob | value | why |
| --- | --- | --- |
| `--low-ram off` | mapped mode | no second copy of the experts in RAM |
| `STRATA_ARENA_PIN_GIB` | **4** | caps the arena's pinning/working-set lock; the rest stays an ordinary evictable mapping |
| `--ctx 32768`, `--kv int8` | the front end's defaults | the pinned KV is ~0.4 GiB instead of 6.00 GiB |
| `--ple-io direct` | unbuffered table reads | the table never enters the page cache |

**Every measurement in this plan's Track A runs in that regime**, and the numbers are therefore
*relative* comparisons inside it. The absolute production baseline is the 19.56 ms/window figure above.
The floor is enforced, not hoped for: the sweep's watchdog samples available RAM and free VRAM every
five seconds and kills the server (the `serve.ps1` parent first, or it restarts the engine) if either
floor is crossed, recording the run as FLOOR-VIOLATION.

### 0.2 The models available locally

| model | size | engine | note |
| --- | --- | --- | --- |
| `D:\llms\IQ3_S` | 54.8 GB weights + 28.8 GB table | Strata/UnbiasedStrata | **the only Strata-servable quant on the box** - the front end refuses any other (`frontend/model.hpp:31-43`), and the Coder's IQ1_M pack is not on the box. The sweep's subject. |
| `D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw` | 33.93 GB | exllamav3 | **the smallest model on the box**; Track E's subject |
| `D:\llms\Qwen3.8-Flash-Next-exl3-3.05bpw` | ~52 GB | exllamav3 | deployed through TabbyAPI |
| `D:\llms\mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 0.9 GB | both | the vision encoder |

The two paths are separate engines, so "the smallest quant locally" means IQ3_S for this engine and
2.05 bpw for the sibling. Note the consequence for Track E: at 2.05 bpw the expert blobs are ~1.06 MB
rather than 1.5 MB, so the bus cost of an expert falls with them - the conclusion "the host is faster
than the bus" gets *stronger*, not weaker, on the smaller model.


**The one measurement that decides the paper's direction** is already in hand: `t_bus` (99 us) >
`t_host` (44 us) by 2.25x. PR #3's own decision rule (§5) says: when the host is faster than the bus,
keep the host-compute path and the look-ahead has no target. So **the paper's mechanism - hide the
PCIe transfer - cannot pay in this engine**, and the 0.00 PCIe experts per layer-window is the
measurement saying so from the other side. Everything below is what *does* pay.

## 1. The ranked levers, with their ceilings

| # | lever | ceiling on the decode | cost | gated by |
| --- | --- | --- | --- | --- |
| **L1** | **Tokens committed per window** (2.42 now): the window costs 18.4 ms and returns 2.42 tokens | **+14%** at 85% acceptance, **+27%** at 100% - and **nothing** from widening the window (measured: §2.1) | L1a: config only (done, negative). L1b: a DFlash 2 port | A (measured) |
| **L2** | **The CPU pool** (3.53 ms, 18%): the paper's *idea* - predict the routing, get the misses into the VRAM tier | **-18%** (126 -> ~155 tok/s), and only if the predictor beats the frequency profile, which already answers 94% of the reads | a trace, an oracle, then a predictor | B (Phase 0 kills it if the oracle does not beat the profile) |
| **L3** | **The GPU-reach wait** (9.90 ms, 51%): the dominant term, and *not* the GPU's expert math (2.66 ms) | unknown, and it is the biggest term | profiling + engine work | D (not the paper's idea; a separate line of attack) |
| **L4** | **The tier's shape and the adaptive tier's knobs** (`--expert-cache-per-layer` is off in the fork's serving path; the Coder's tuned `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97` are not applied to IQ3_S) | unknown, cheap to find out | config only | A |

**The paper's mechanism is not on this list** - measured out (§0). The paper's *idea* is L2, and L2 is
second to L1.

## 2. The tracks

### Track A - free wins: configuration only, one afternoon of runs

Every item is a flag change against the measurement harness. The harness is
`gate-measure.ps1` + `gate-request.ps1` in `C:\Users\yoho\Downloads\UnbiasedStrata` (they start the
server with the production path and the decode timing on, then drive three requests), extended into
`sweep.ps1` (the same harness, ten configurations, the floor watchdog of §0.1 and a measured noise
floor). **Measurement discipline** (from the sibling project's hard-won lesson): this box cannot be
A/B'd across processes - identical code and identical command have produced 39.3 vs 50.8 tok/s - so
every comparison must be **single-process interleaved**, and the tier's state (which warms across
requests) must be held equal. Read the per-request block: `decode timing`, `decode GPU stages`, `decode
expert cache hit rate`, `drafts accepted`.

**A0 - the RAM-safe configuration (new, and the operationally important one).** §0.1: the production
configuration drives available RAM to 0.0 GB on this box, and the mapped mode still locks 22.2 GiB and
exhausts the commit limit. The compliant configuration is `--low-ram off` + `STRATA_ARENA_PIN_GIB=4` +
`--ctx 32768 --kv int8`, measured at 3.9 GB available. The open question is whether the *production*
path can be made compliant as well (the resident budget's knob is `--resident-budget-gib N`, and
`STRATA_RESIDENT_HEADROOM_GIB=N` sets the headroom instead of the default 4) - and what that costs the
decode, since a smaller resident set means more expert reads from the file. This is not a performance
item; it is a stability item, and it sits next to issue #1.

**A1 - the drafter sweep (L1).** Sweep `--spec` (4 now; the window is `min(spec+2, 8)`), `--spec-min-p`
(0.5 now), `--suffix-draft`. The measured average T is 3.00 against a 6-row window, so the drafter is
not filling its window and there is headroom without any new code. Record per setting: tokens/window,
window cost, acceptance, and the distinct-experts-per-layer-window count. **The hypothesis to test:**
the window's cost grows *sub-linearly* in T (distinct experts per layer saturate), so tokens/window
grows faster than the cost. The measurement must confirm or kill it - if the cost grows linearly, L1 is
worth much less than the table says. *(Tracker: issue #2.)*

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
agree. *(A2-A4: issue #3.)*

### 2.1 The sweep's results (Track A, measured 2026-10-21, ten configurations)

Harness: `sweep.ps1` on the box, the RAM-safe regime of §0.1, `STRATA_DECODE_TIMING=1` and
`STRATA_VERIFY_PROFILE=1`, three requests per configuration (the first is cold; the table is the warm
third). **No run crossed a floor** - every row is `ok`, with 3.4-4.6 GB of available RAM and 747-785 MB
of free VRAM throughout.

**The noise floor: 131.7 / 132.1 / 131.0 tok/s (mean 131.6, spread 1.1 = 0.8%).** Only a difference
larger than that is a difference.

| config | tok/s | vs base | ms/window | tokens/window | avg T | hit % | CPU pool ms | draft ms | drafts ok/offered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **C0 baseline** (x3) | **131.6** | - | 18.4 | 2.42 | 3.10 | 94.1 | 3.01 | 1.86 | 224/328 |
| C1 `--spec 6` | 122.9 | **-6.6%** | 24.1 | 2.96 | 4.07 | 92.9 | 5.04 | 2.79 | 109/166 |
| C2 `--spec-min-p 0.2` | 119.9 | **-8.9%** | 21.7 | 2.60 | 3.83 | 93.2 | 4.96 | 2.12 | 235/413 |
| C3 `--expert-cache-per-layer` | 111.2 | **-15.5%** | 21.3 | 2.36 | 2.95 | 88.6 | 5.25 | 1.89 | 101/144 |
| C4 `--adapt-every 1 --adapt-swaps 80 --adapt-decay 0.97` | 130.6 | -0.8% (noise) | 18.9 | 2.46 | 3.22 | 94.4 | 3.28 | 1.94 | 178/269 |
| C5 `--pcie-frac 1.0` | 130.6 | -0.8% (noise) | 19.8 | 2.58 | 3.26 | 94.0 | 3.20 | 1.90 | 218/310 |
| C6 `--pcie-frac 0.0` | 122.7 | **-6.8%** | 20.1 | 2.47 | 3.03 | 92.3 | 3.76 | 1.91 | 94/130 |
| C7 `--suffix-draft 0` | 132.5 | +0.7% (noise) | 18.6 | 2.47 | 2.97 | 93.6 | 3.24 | 1.78 | 94/126 |

**What the sweep settled.**

1. **The window's cost is ~linear in T, not sub-linear - the hypothesis is dead.** C1 widened the
   window (T 3.10 -> 4.07, +31%) and the window cost grew 31% (18.4 -> 24.1 ms). Tokens/window grew
   only 22% (2.42 -> 2.96), because the per-draft acceptance *fell* as the window widened (69% -> 66%),
   and the CPU pool grew with it (3.01 -> 5.04 ms) while the hit rate fell (94.1% -> 92.9%). The net is
   **-6.6%**. C2 did the same thing the other way (a lower acceptance gate admits more drafts: 413
   offered against 328) and lost **-8.9%**. **So the "tokens per window" lever is not a window-size
   lever.** It is the *acceptance rate at a fixed window*: at T = 3.08 and 18.4 ms, 100% acceptance
   would be 3.08 tokens/window = 167 tok/s (+27%), and 85% acceptance 2.77 tokens/window = 150 tok/s
   (+14%). That is the number a better drafter has to beat, and it is a third of what this plan
   originally assumed.
2. **`--expert-cache-per-layer` is -15.5%** (hit rate 94.1% -> 88.6%, CPU pool 3.01 -> 5.25 ms). The
   flag's own numbers (2.97% global vs 21.4%/70.4% per layer) were measured with a 4096-slot tier and
   a static profile; against *this* profile-filled 7,998-slot tier the per-layer form is worse. **The
   fork's serving path is right not to pass it** - the question is closed, and the answer is "no".
3. **The PCIe share is worth ~7%, and the probe's choice is right.** `--pcie-frac 0` loses 6.8%;
   `1.0` is indistinguishable from the probed 0.55. So the PCIe path is worth having (it takes the
   misses off the CPU) but it cannot be improved by turning it up - consistent with §0's
   `t_bus` (99 us) against `t_host` (44 us).
4. **The Coder's tuned adaptive knobs do nothing here** (-0.8%, inside the noise floor) and **the suffix
   drafter does nothing** (-0.7% to +0.7%): C7's own log line says "suffix drafts: 1 window, 1 of 3
   drafts accepted". The MTP head does the drafting; the suffix drafter is nearly idle.

**Consequence for the plan.** Track A is done: it found no free win, and it *narrowed* L1 from
"+40-60%" to "+14% at 85% acceptance" and *identified* the only form of the lever that pays. The
configuration work is over - the engine's defaults are at a local optimum - so the remaining levers
are the ones that need code or a new model: B0/B1 (the predictor, gated), C (the drafter), D (the
GPU wait). *(Tracker: issue #2, updated with these numbers.)*
### Track B - the paper's idea, behind a gate

**B0 - the oracle (the gate, offline, no GPU, no code in the engine).** Take a routing trace
(`--dump-routing`) and compute, per layer and per window, the read coverage of: (i) the shipped
profile's top-B per layer (what runs today, 94%), (ii) the previous window's expert set (the cheapest
predictor), (iii) the union of two windows, (iv) **the oracle** - the window's own set, truncated to the
tier's budget. The oracle is the ceiling of *any* predictor.

**Acceptance criterion: the oracle must beat the profile by a margin worth the pool's 18%.** If the
oracle's coverage is not meaningfully above 94%, **stop** - the tier is already doing the work, and the
predictor has no headroom. The oracle is also what sizes the prize: the difference between the oracle
and the profile, times the pool's 18%, is L2's real ceiling. *(Tracker: issue #4.)*

**B1 - the predictor (only if B0 passes).** The routing-transition model PR #3 sketches -
`P(routing_L(t+1) | routing_L(t))`, learnable from a trace, well under a megabyte of VRAM. Plugged
into the adaptive tier's ranking (`generate.cpp:7961+`), with the frequency ranking as the fallback
and tiebreak. The engine's existing hooks make this a ranking-function change, not a new subsystem:
per-layer budgets (`expert_cache.hpp:161`), the async copy path with its stream and event, and the
one-place hit decision (`ExpertDispatch::is_hit`). *(Tracker: issue #5.)*

**B2 - the capture points.** DFlash 2's capture (target layers 5, 15, 25, 35, 45, averaged over the
four hyper-connection streams, 12,800 features per position) is a free feature source for a predictor of
the form `P(routing_l | captured features)` - the "Speculating Experts" estimator shape. It needs the
capture in this engine, which the SGLang patch shows is a few lines at five layers.

### 2.2 Track B's gate, answered: **STOP** (measured, 2026-10-21)

`tools/specter_oracle.py` (the B0 implementation, with a self-test) was run against a real routing
trace from this box - 4,234 records, 971 (layer, window) pairs, the prompt's positions and part of a
decode - at a 64-slot per-layer budget:

| arm | read coverage | what it is |
| --- | ---: | --- |
| the shipped profile (static) | **28.4%** | what a cold tier would hold |
| the previous window's set | 51.1% | the cheapest possible predictor |
| the union of two windows | 58.9% | a slightly smarter version |
| **the oracle** (the window's own set) | **100%** | the ceiling of any predictor |
| **the engine's running tier** | **94.1%** | profile **+ the adaptive tier's swaps**, measured |

**The working set fits: 27.8 distinct experts per layer per window on average, 59 at worst - inside a
64-slot budget.** So the shape of the paper's design is fine here; the *predictor* is what fails.

**The verdict: STOP.** Against the bar that matters - the engine's **running** tier at 94.1%, not the
static profile at 28.4% - the oracle's headroom is **5.9 points**, which at the pool's 18% share is
**1.1% of the window** (126 -> ~127 tok/s), an order of magnitude under the 15-point threshold. And
the best *achievable* arm, the previous window's set at 51.1%, is **42 points below** the running tier:
the engine's retrospective frequency tier already tracks the conversation better than any cheap
predictor does. The naive comparison (oracle vs the *static* profile: +71.6 points, "GO") is the trap -
the static profile is not what the engine runs.

**So the paper's idea has no headroom in this engine**, and Track B stops here: no predictor is built,
and B2's capture points lose their consumer here (they remain a legitimate idea for the sibling
engine, whose offloaded part is 89.3 ms/token instead of 3.0). *(Tracker: issue #4.)*

What survives from the paper is not the predictor but the *runtime* idea - and the runtime's transfer
is to the sibling engine, which is Track E.

### Track C - DFlash 2, gated by A1

`incoai/Qwen3.8-Flash-Next-DFlash2`: 0.6 B, Apache-2.0, 5 dense sliding-attention layers,
`is_causal: false`, `use_cache: false`, block size 8, a top-16 selector, and an acceptance length that
beats the model's own MTP on every task published (GSM8K 3.26 vs 3.08, HumanEval 2.94 vs 2.52,
MT-Bench 2.83 vs 2.50). ~1.2 GB BF16.

**Gate: A1 must show that the existing drafter cannot reach the same acceptance.** If tuning gets
tokens/window to ~2.8 on its own, DFlash 2's marginal value is small and the port is not worth the
draft loop. *(Tracker: issue #6.)*

**The gate is answered, and the port does not pay on the published evidence**
(`docs/DFLASH2-INTEGRATION.md` §5, §6, §7):

- **The engine's drafter already matches DFlash 2's average published acceptance.** Measured here:
  68.3-68.8% per draft. DFlash 2's published rates, converted to this engine's window (gamma = 2.08
  rows, not the published gamma = 3), run 61.0-75.3% - **mean 68.4%, which is the engine's own number**.
  DFlash 2's *best* published case (GSM8K, 75.3%) converts to 2.57 tokens/window = 139 tok/s:
  **+6%**, against the **+14%** the 85%-acceptance arithmetic names. The +14% bar sits ten points
  above anything DFlash 2 publishes.
- **The VRAM does not fit.** 1.12 GiB of draft against ~750 MiB free at boot: it must displace the MTP
  (708 MB of experts + 111 MiB dense, i.e. roughly VRAM-neutral if it *replaces* the MTP) or ~670 tier
  slots (8.4% of the tier - and the sweep measured what a smaller tier costs: the hit rate fell
  94.1% -> 88.6% for -15.5% of tok/s).
- **The capture needs a new kernel.** Upstream takes a *plain mean* over the four hyper-connection
  streams; the engine's only reduction is the hyper-connection's *learned weighted* mean
  (`include/strata/kernels/fused_gr.hpp:13`). The tap point is fixed (layer 5's capture in `pre(6)`),
  but equivalence to SGLang's `hidden_states` is unproven.

**So Track C's first step is a measurement, not a port**: record the per-position acceptance bits
`src/program/generate.cpp:8113` already computes and discards, at `--spec 4` (gamma = 3), and read the
engine's per-task acceptance against DFlash 2's published per-task rates. That is an afternoon, it needs
no new model, and it decides whether the +6% best case is even reachable here.

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
redirected attention here, and because Track A will not move it. *(Tracker: issue #7.)*

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

1. ~~**A1-A4**~~ **done** (2026-10-21): ten configurations, no free win, the engine's defaults are at
   a local optimum, and the acceptance rate at a fixed window is the only form of L1 that pays
   (§2.1). A0 - the RAM-safe production path - is still open, and it is the one that matters
   operationally.
2. ~~**B0**~~ **done: STOP** (§2.2). No predictor is built; the paper's idea has no headroom in this
   engine and the transfer is to the sibling engine (Track E).
3. **C** - the DFlash 2 port, now gated on the *narrowed* ceiling: it must lift the per-draft
   acceptance from 69% to 85% (or the window's committed tokens from 2.42 to 2.77) to be worth a new
   subsystem. `docs/DFLASH2-INTEGRATION.md` has the seams.
4. **D** - the GPU-reach wait (51% of the window), in parallel: the two questions are the GPU's
   9.90 ms attribution and whether the hit path can start before the misses land.
5. **E** in its own session, in the sibling engine.

So: one track left that needs a new model (C), one that needs profiling (D), one that needs
measurement in the other engine (E), and one operational item (A0).

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
- The tracker (GitHub issues on this repo):
  [#2 A1 the drafter sweep](https://github.com/aand18/UnbiasedStrata/issues/2) ·
  [#3 A2-A4 tier shape, adaptive knobs, PCIe share](https://github.com/aand18/UnbiasedStrata/issues/3) ·
  [#4 B0 the routing oracle](https://github.com/aand18/UnbiasedStrata/issues/4) ·
  [#5 B1-B2 the predictor and the capture points](https://github.com/aand18/UnbiasedStrata/issues/5) ·
  [#6 C the DFlash 2 port](https://github.com/aand18/UnbiasedStrata/issues/6) ·
  [#7 D the GPU-reach wait](https://github.com/aand18/UnbiasedStrata/issues/7) ·
  [#8 Track E, the sibling engine's gate](https://github.com/aand18/UnbiasedStrata/issues/8) (the
  exllamav3 repository has issues disabled, so its track is tracked here and planned in PR #3).
- The harness, on the box: `C:\Users\yoho\Downloads\UnbiasedStrata\gate-measure.ps1`,
  `gate-request.ps1` (one measured run) and `sweep.ps1` (the ten-configuration sweep with the floor
  watchdog and the noise floor); logs in `logs\gate-*.err.log` and `logs\sweep-summary.csv`.
