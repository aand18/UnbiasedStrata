# Plan: performance of UnbiasedStrata on Qwen3.8-Flash-Next, from the SPECTER study

Written 2026-10-21 on the `spectre-moe` branch, after `docs/SPECTER-FEASIBILITY.md` (the study) and
`aand18/exllamav3` PR #3 (the same question for the sibling engine, canonical there).

## How to read the numbers in this document

Every figure below comes from one of these, and each is defined here so that any single paragraph can
be read on its own:

| term | what it means | where it comes from |
| --- | --- | --- |
| **the window** | one verification step: the engine verifies T tokens of the target model against the drafts, then commits the accepted prefix. A *window* here is 18.34-18.50 ms and yields 2.42 committed tokens (T = 3.08 rows verified, ~69% of the drafts accepted). | the engine's per-request log line `strata decode timing:` |
| **tok/s** | committed output tokens per second. 131.6 in the RAM-safe regime, 126 in the production regime. | the same log line |
| **the noise floor** | **131.6 ± 1.1 tok/s (0.8%)**, from three identical baseline runs. Only a difference larger than this is a difference - the box cannot be A/B'd across processes (identical code has produced 39.3 vs 50.8 tok/s in the sibling project). | the sweep's three `C0` rows |
| **the running tier** | the engine's *actual* resident expert set at decode time: the shipped frequency profile's fill **plus** the adaptive tier's swaps. It measures a **94.1% decode hit rate**. | the log line `decode expert cache hit rate` |
| **the static profile** | the shipped frequency ranking alone, with no swaps - the tier a cold start would have. On this model it covers only **28.4%** of the reads. | `tools/specter_oracle.py` |
| **the CPU pool** | the host's expert computation for the tier's misses: **3.01 ms of the 18.41 ms window** (18%), 1.68 distinct experts per layer-window. | the same log line, per layer-window |
| **the bar for a new drafter** | the per-draft acceptance a drafter must reach to be worth building: **85%**, which would take the window from 2.42 to 2.77 committed tokens and the decode from 131.6 to ~150 tok/s (**+14%**). The engine's drafter is at **68.3-68.8%** today. | the acceptance arithmetic in 2.1 |
| **the floors** | the box's rule: never below **2 GB available system RAM**, never below **200 MB free VRAM**. | the operator's rule; enforced by the sweep's watchdog |


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
| **L2** | **The CPU pool** (3.53 ms, 18%): the paper's *idea* - predict the routing, get the misses into the VRAM tier | **-18%** in theory, **~1% in practice** - the gate measured it (§2.2): the oracle's headroom over the engine's running tier is 5.9 points | the gate: a trace, an oracle, a stop | B0 **done: STOP** |
| **L3** | **The GPU-reach wait** (9.90 ms, 51%): the dominant term, and *not* the GPU's expert math (2.66 ms) | the GPU's 15.86 ms is 24% hyper-connection, 20% expert math, 16% dense projections, 16% waits; the serialization inside it is worth **≤5%** (§2.3) | kernel work, not scheduling | D (attribution done) |
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

**The verdict, corrected by measurement: GO, and the prize is +4.9% to +19.4%.**

*Two corrections, both mine.* The first version of this document sized the prize as "the headroom in
points x the pool's share of the window" and wrote 1.1%. Wrong: the 5.9 points of reads the tier misses
**are** the pool's work (3.01 ms of an 18.41 ms window), so a tier covering the window's expert set
takes the pool to **zero**: 131.5 -> 157.1 tok/s, **+19.5%**. The second correction is the threshold:
15 points of read coverage was *my* policy, not a measured constant.

**The measurement that settles it** (`tools/specter_miss_coverage.py`, on a real run's `--dump-misses`
and `--dump-routing`: 318 sweeps, 33,770 missed (layer, expert) pairs):

| arm | coverage of the tier's MISSES | what it would buy |
| --- | ---: | --- |
| the shipped profile's top-64 per layer | 5.3% | ~0.9% of the window |
| **the previous window's routed set** | **22.6%** | ~3.9% of the window |
| **the union of two windows' routed sets** | **28.4%** | **~4.9% of the window (+4.9%, 131.6 -> ~138 tok/s)** |
| the previous window's MISS set | 14.6% | ~2.5% of the window |
| the union of two windows' MISS sets | 19.0% | ~3.3% of the window |
| the oracle (the window's own set) | 100% | **+19.4% (131.6 -> 157 tok/s)** |

So the gate is **GO**: the cheapest predictor - the previous window's routing, which the engine
already counts - would cover **22.6% of the tier's misses**, and two windows 28.4%, which is ~5% of the
window, **six times the box's noise floor** (+-1.1 tok/s). A better predictor (the captured features of
§5.3's estimator shape) sits between that and the oracle, and the whole band is worth 5-19%.

**The surprising part, and it decides the implementation**: the tier's own recent **misses** are a
*worse* forecast (14.6%) than its recent **routing** (22.6%). The tail churns - the experts the tier
lacked last window are not the ones it lacks this window - so the forecast has to come from what was
*routed*, not from what was *missing*. That also says the current policy is not far off the right
shape: `drive.d.usage` already counts the routing; what it lacks is *recency* (it is diluted over the
whole run) rather than a new signal.

**The track is therefore open, and its next step is a ranking function**: the predictor's set has to
beat the frequency ranking *for the misses*, and the frequency ranking is what the adaptive tier uses
today (`generate.cpp:7961+`). Nothing else about the engine changes: the tier's slots, the copy path
and the one-place hit decision are already there.

**What this does not change**: the paper's *mechanism* is still out (0.00 PCIe experts, `t_bus` 99 us >
`t_host` 44 us), and the *cheap* predictor is worth 5%, not the 12.9% the first (static-profile)
comparison suggested.

### 2.3 Track D's first question, answered from the profile the sweep already produced

The GPU's own stage profile (the sweep's baseline run, `STRATA_VERIFY_PROFILE=1`) attributes its
**15.86 ms/window** exactly:

| | ms/window | share |
| --- | ---: | ---: |
| **the hyper-connection path** (hc-read + router + the three hc0 stages) | **3.84** | **24.2%** |
| the expert math (VRAM hits + PCIe groups) | 3.10 | 19.5% |
| the dense projections (q8 gemv, out-proj, q+q-idx) | 2.47 | 15.6% |
| **the waits** (waitA, waitB, waitCPU, gap) | **2.46** | 15.5% |
| the GDN recurrence (conv, ab, z, rec) | 1.24 | 7.8% |
| the shared expert and the activation quantisation | 0.83 | 5.2% |
| the head | 0.73 | 4.6% |
| the QSA attention (attention, k/v, kv-idx, resolve) | 0.70 | 4.4% |
| copy + combine | 0.39 | 2.5% |

**Two things fall out, and the first kills the second question.**

1. **The "51% GPU-reach wait" is mostly real GPU work, not serialization.** The host waits 10.6 ms of
   an 18.4 ms window; the GPU's own work is 15.86 - 2.46 (its own waits) = **13.4 ms**. The largest
   single block of that work is **the hyper-connection plumbing at 24.2%** - more than the expert math
   (19.5%), and three times the GDN recurrence. That is ordinary kernel work on the residual stream,
   and it is where the largest remaining term on this box actually lives.
2. **"Can the hit path start before the misses land?" is worth at most 5%.** The serialization's cost
   is `waitCPU` - the GPU idling for the host's misses - which measures **0.97 ms/window** (0.76 GDN +
   0.21 QSA), i.e. **5.3% of the window**. The hits do not depend on the misses, only the combine does,
   so reordering is worth at most that 0.97 ms - and it is a real change to the plan's stream ordering
   for a fifth of a percent.

So Track D is not a serialization fix; it is a **kernel** target, and the paper's idea is not involved
either way. *(Tracker: issue #7.)*

### 2.4 Track A0, answered: the compliant path is the *mapped* mode, and it is faster

Four runs of the production path with one bound changed at a time, the same watchdog (2 GB available
RAM, 200 MB free VRAM):

| run | what it bounds | verdict | tok/s | ms/window | hit % | RAM min |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| P0 | the production path, unbounded | **FLOOR-VIOLATION** (the watchdog fired) | - | - | - | 0 GB |
| P1 | `--resident-budget-gib 16` | ok | 32.9 | 74.4 | 93.9 | 25.7 GB |
| P2 | `--resident-budget-gib 24` | ok | 58.6 | 40.4 | 94.1 | 17.5 GB |
| P3 | `--ctx 32768 --kv int8` | **FLOOR-VIOLATION** (ram 1827 MB) | - | - | - | - |

**Reading it:**

- **The unbounded production path breaches the floor** - now demonstrated *under the watchdog*, so the
  box is never left at 0 GB.
- **Capping the resident budget makes it compliant, but it is expensive**: 126 tok/s (the production
  measurement) -> **58.6** at 24 GiB and **32.9** at 16 GiB. The experts outside the budget come from
  the SSD, and the window's cost triples (18.4 -> 40.4 ms). This is the *disk regime* the paper's idea
  would love - and it is a 54% loss against the RAM-resident path.
- **Shrinking the context does not fix it** (P3 still breached it): the resident copy, not the KV, is
  what takes the RAM.

**But the sweep already found the fix, and it is free.** The RAM-safe regime of §0.1 - `--low-ram off`
(mapped mode) with `STRATA_ARENA_PIN_GIB=4` and the front end's default `--ctx 32768 --kv int8` -
measures **131.6 tok/s with 3.9-4.6 GB of available RAM**: *faster* than the production regime's 126
tok/s, and compliant. The mapped mode's experts come through the OS page cache instead of a locked
copy, so the OS can evict them and the box keeps its headroom; the cost is that a cold cache or a long
session can fall back to the disk, which is worth watching.

**So A0's answer: switch the production path to the mapped mode with a capped arena pin.** It is
strictly better on this box - 4.5% faster, and 4 GB of RAM back - with the caveat that the disk's
latency now shows up when the cache is cold. *(Tracker: issue #9.)*

**What "switch the production path" means concretely**, because the two modes are not what their names
suggest:

| mode | what the engine does | measured here |
| --- | --- | --- |
| the **arena** mode (the engine's default; what the front end's `auto` picks) | the expert arena is read into RAM and its working set is **locked** (22,227 MiB measured) | 131.6 tok/s, 3.9-4.6 GB available |
| the **resident** mode (`--low-ram on` -> `--resident-experts`, which the operator's `start-logged.ps1` forces) | the experts the VRAM tier does not hold are **copied** into RAM and locked (33.55 GiB measured) | 126 tok/s, **0.0 GB available** |

So the production switch is: **drop `--low-ram on` from the invocation** (the front end's `auto` then
picks the arena mode, which is both faster and compliant) and set `STRATA_ARENA_PIN_GIB=4` in the
server's environment. Two measurements still stand between that and the change, and both are in the
harness: the arena mode at the *production* context and KV (262,144 / fp16 - the sweep measured it at
32K / int8, and the KV's 6.00 GiB of pinned RAM is 5.6 GiB more than the sweep's), and the arena mode
with a tighter lock (`STRATA_ARENA_PIN_GIB=2`). The configurations are `P4-arena-prod` and
`P5-arena-prod-pin` in `a0-measure.ps1`.

**And the engine change that makes it robust** is issue #10: the arena's lock is not bounded by any
configuration today (`STRATA_ARENA_PIN_GIB` caps only the driver's `cudaHostRegister`), so the load
dips 1.6 GB under the floor on a cold cache and every measurement dies mid-load. *Implemented:
`STRATA_ARENA_LOCK_GIB=N` (N bounds the lock, 0 locks nothing, unset = today's behaviour), with the
truncation stated in the note the engine prints. Committed as `e484e83`.*

### 2.5 The box's memory is ONE pool: the arena, the tier and the KV all come out of 63.6 GB

The lock cap is in, and the measurement says the lock is **half** the cause. With
`STRATA_ARENA_LOCK_GIB=8` the load *finishes* (46.84 GiB at 4.65 GiB/s) instead of being killed
mid-load - but the box still dips to 1,525 MB. With the lock disabled entirely (`=0`) it still dips,
to 1,692 MB. So the constraint is structural, and it is this:

**Under WDDM the VRAM tier IS host RAM.** The expert tier (13.4-16 GiB) is allocated from the same
63.6 GB as the expert arena, the pinned KV cache and the OS. So the configuration has to satisfy

```
arena_resident + expert_tier + kv_pinned + the OS + the floor  <=  63.6 GB
```

and the terms move together: the engine sizes the tier from the *free VRAM*, which under WDDM is the
free RAM - so a box that is short on RAM gets a *bigger* tier, which makes it shorter on RAM. The
measured arms of the same session:

| arm | the arena | the tier | the KV | available RAM | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| the sweep's regime (32K, int8) | 22.2 GiB locked | 13.41 GiB (7,072 slots) | 0.4 GiB | **3.9-4.6 GB** | 131.6 tok/s, compliant by ~4 GB |
| the same with a bigger tier (8,408 slots) | 22.2 GiB | 15.95 GiB | 0.4 GiB | 1,525 MB | the load finished, then the tier's allocation tripped the watchdog |
| the production path (262,144, fp16) | 33.55 GiB copied and locked | 15.95 GiB | 6.00 GiB | **0.0 GB** | breaches |
| a bounded resident copy (24 GiB) + 32K/int8 | 24 GiB | 13.41 GiB | 0.4 GiB | **17.5-22.2 GB** | 58.6-67 tok/s, compliant with room |

### 2.6 The production configuration, measured: eight arms, one answer

All eight arms ran the production path (IQ3_S, 262,144 context, fp16 KV with 32,768 resident cells) except
where the table says otherwise; the floors were hard (2,048 MB available RAM, 200 MB free VRAM) and the
watchdog killed any arm that crossed them.

| arm | the bound it changes | verdict | tok/s | ms/window | hit % | RAM min |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| P0 | nothing (today's script) | **FLOOR-VIOLATION** | - | - | - | - |
| P1 | `--resident-budget-gib 16` | ok | 33.5 | 67.7 | 93.9 | 25.1 GB |
| **P7** | **`--resident-budget-gib 24`** | **ok** | **65.0** | **39.0** | **95.3** | **19.2 GB** |
| P2 | the same at 32K / int8 | ok | 59.8 | 39.5 | 94.1 | 17.0 GB |
| P3 | the resident mode at 32K / int8 | **FLOOR-VIOLATION** | - | - | - | - |
| P4 | the arena mode (no `--low-ram on`) | **FLOOR-VIOLATION** | - | - | - | - |
| P5 | the same with a tighter pin | **FLOOR-VIOLATION** | - | - | - | - |
| P6 | the arena mode with the tier capped at 2,800 slots | **FLOOR-VIOLATION** | - | - | - | - |

**The answer, and the cost.** At the production context the *only* compliant configuration is the
resident mode with a **bounded** copy: **65.0 tok/s with 19.2 GB of RAM to spare**, against today's
unbounded 126 tok/s at 0.0 GB. So complying costs **~48% of the decode** - the experts outside the
24 GiB budget stream from the file. The arena mode cannot be made compliant at this context at any tier
size (P4, P5, P6 all breach): the arena's 46.84 GiB plus the KV's 6.00 GiB plus the OS and the floor
already exceed the box before the tier is counted.

**And the fast compliant configuration is the sweep's**: the arena mode at **32,768 / int8** measures
**131.6 tok/s with 3.9-4.6 GB available** - faster than today's 126 tok/s *and* compliant, at a context
8x smaller. So the operator's real choice is not "speed or compliance" but **"which context"**:

| | tok/s | available RAM | context |
| --- | ---: | ---: | ---: |
| today's script | 126 | **0.0 GB** (breaches) | 262,144 |
| **the compliant production configuration** (P7: `--resident-budget-gib 24`) | **65.0** | 19.2 GB | 262,144 |
| the compliant fast configuration (the arena mode at 32K / int8) | **131.6** | 3.9-4.6 GB | 32,768 |

`start-logged.ps1` now carries the P7 configuration (with the alternatives in a comment, one flag away
each). Which of the three to run is the operator's call - the floor says P7 or the 32K one, and the
262,144-context speed is what P7 costs.

### 2.7 Track B1, implemented: the forecast in the tier's candidate ranking

**The flag: `--adapt-forecast N`** (a float, **0 = off**, which is today's behaviour exactly - the branch
is never taken, so the candidate set, the scores and the gain condition are unchanged).

What it does, in three pieces (`src/program/generate.cpp`):

1. **`Forecast`** - two planes of "routed recently", per layer: plane 0 is the current window, plane 1 the
   one before it. 48 x 512 bytes per plane, 48 KB in total, and it is filled by the pool callbacks
   (`drive_pool`, `drive_pool_multi`), which already receive each layer's routed ids.
2. **The rotation** at the start of every decode window: plane 1 becomes what plane 0 held, plane 0 is
   cleared. So the predictor is exactly "routed in the last two windows" - the arm the measurement liked
   (28.4% of the misses).
3. **The bonus** in the adaptive tier's candidate ranking: a candidate that the forecast predicts scores
   `N` more than its whole-run count says, and an expert that the forecast predicts is a candidate even
   if its count is below the usual threshold of 2.0. The victims' ranking is untouched, so the gain
   condition (`cand >= vict + 1.5`) still compares like with like.

**Why this shape and not a new predictor.** The measurement said the signal is *recency*, not a new
signal: `drive.d.usage` already counts the routing, but it is diluted over the whole run and thresholded.
The forecast is the same count with a two-window memory, and the bonus lets it outrank the diluted one.

**The measurement, in the A/B** (`ab-forecast.ps1`): three baseline runs and three forecast runs in the
P7 regime (the one that completes under the RAM floor), the floors hard, the same prompt stream.

**The verdict: the flag does not beat the noise floor, so it stays off.** But the mechanism is real, and
why the decode does not show it is the more valuable result.

Measured in the quiet regime (`--resident-budget-gib 32`, so the misses come from the RAM copy and the
instrument is not dominated by the disk; six runs per arm, the warm request):

| | baseline | forecast | noise floor |
| --- | ---: | ---: | ---: |
| **tok/s** | 130.47 | 129.87 | **4.60 (3.5%)** |
| ms/window | 18.92 | 18.68 (-1.3%) | 0.65 |
| tokens/window | 2.47 | 2.43 (-1.6%) | 0.00 |
| **hit rate** | 92.30% | **94.40% (+2.1 points)** | 0.00 |
| **CPU pool ms/window** | 3.60 | **2.52 (-30%)** | 0.26 |
| **CPU experts/layer-window** | 2.16 | **1.47 (-32%)** | 0.00 |
| the adapt round's wait | 3.58 | 4.49 (+25%) | 0.35 |

1. **The ranking change works as designed**: the tier's misses fall 32%, the pool's time falls 30%, the
   hit rate rises 2.1 points - identically in all six runs (sd 0.00). The forecast *is* a better
   predictor, exactly as the offline measurement said.
2. **The tok/s is a wash because the tier's contents change the drafter.** Which experts the GPU
   computes and which the CPU does changes the logits in the last bits (the fp16-vs-fp32 activation
   scales, `expert_source.hpp:281-293`), so the drafter's window shrinks (avg T 3.03 -> 2.80) and the
   tokens/window falls 1.6% - precisely what the 1.3% window gain was worth. **This is a measurement
   caveat for all of Track B: any change to the tier's contents is confounded through the drafter.** A
   tok/s A/B on this engine cannot read a tier change that is worth less than ~2% of the window.
3. **The tier's churn is not free - and it is the round's BYTES, not its schedule.** Measured with the
   new `STRATA_TRACE_ADAPT=1` trace (the quiet regime, three runs per arm):

   | per window | baseline | forecast |
   | --- | ---: | ---: |
   | swaps per round | 60.5 | 75.3 |
   | the round's work (the adapt thread) | 2.82 ms | 3.46 ms |
   | … copies back to RAM (D2H, 33 GB/s) | 1.23 | 1.53 |
   | … pageable H2D (the driver's staging) | 1.55 | 1.94 |
   | apply_pending: the landing wait | 0.06 ms/request | 0.06 |
   | apply_pending: the exchange's memcpy | 1.04 | 1.29 |
   | **the churn tax** | **3.86** | **4.75 (+0.89)** |

   The forecast's bonus enters the *gain* condition, so 24% more candidates qualify, 14.8 more swaps
   per round, and **each marginal swap costs 240 us to save 276 ± 19 us of pool time - break-even**. The
   landing is not the cost: the copies land during the round.

   **It is not removable by scheduling.** A round moves 320 MB over a link that does not add its
   directions: a probe of 160 MB each way concurrently takes 12.2 ms, the same as the two one after the
   other, and a pageable H2D's submission blocks for the whole copy (so it waits for the copies back
   too). The chunked staging - the next chunk's D2H issued before the current chunk's H2D - was
   implemented and **reverted**: the round stayed 11.2 ms and the window 18.9. The round is bound by its
   bytes, and no schedule of the copies removes that.

   **So the next lever is bytes per swap, not the forecast.** A swap copies the victim's bytes back
   into RAM (5.32 MB) and the new expert into the slot (2.66 MB). Releasing the victim to the file
   tier instead would halve the round's bytes (5.32 -> 2.66 MB) - at the price of the misses coming
   off the disk. That is the change to price next.
4. **The bonus's scale matters, and small is better.** Replaying the real dumps offline (318 windows),
   the simulated misses fall **2.6% at a bonus of 0.2-0.5** and only **0.5% at 5**: a big bonus pushes
   every forecast-hit candidate to the same score, so the 96-swap cap is spent on an arbitrary
   tie-break. The A/B used 0.5.

**The flag's shape, after the review** (`--adapt-forecast N`, `--adapt-forecast-windows K`, default
N = 0 = off): the forecast is maintained **only when the flag is on**, so the off path allocates
nothing, rotates nothing and marks nothing; K is a flag rather than hardcoded; the prompt path rotates
too (otherwise the prompt's routing piles into plane 0 and lands in the first decode window's
forecast); and `mark()` bounds the layer, because the index is `layer * n_expert + e` and a layer past
the model's would write past the plane. *(Tracker: issue #5.)*

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
   (§2.1). A0 - the RAM-safe production path - is implemented (the operator's `start-logged.ps1` now
   carries the compliant configuration); the choice of *context* is still the operator's.
2. ~~**B0**~~ **done: GO** (§2.2). The gate measured the predictor's prize against the tier's actual
   misses: the previous window's routed set covers 22.6% of them, two windows 28.4%, against 5.3% for
   the frequency profile. The paper's idea does have headroom here.
3. ~~**B1**~~ **implemented and measured** (§2.7): the forecast's ranking change works - the tier's
   misses fall 32%, the pool's time 30%, the hit rate +2.1 points - but the tok/s does not move,
   because the adapt round's own cost rose ~0.9 ms/window and because **any change to the tier's
   contents is confounded through the drafter** (a tok/s A/B here cannot read a change worth less than
   ~2% of the window). The flag **stays off** by default.
4. **B2 - the churn tax: done, and it is the round's bytes** (§2.7 item 3). The forecast's swaps are
   break-even (240 us each to save 276 ± 19 us), the tax is not removable by scheduling (the link does
   not add its directions; the chunked staging was implemented and reverted), so the flag **stays off**.
5. **B3 - the bytes per swap: researched** (`docs/CUDA-MEMORY-OPTIONS.md`, 660 lines). The round costs
   3.57 ms/window (the join 2.53 + the exchange's memcpy 1.04), and the options rank:
   **(1) pin the complement's prefix - `cudaHostRegister` with the knobs that already exist
   (`STRATA_PARTIAL_PIN`, `STRATA_PARTIAL_PIN_GIB`): 0.98-1.58 ms/window (5-8%), zero build.** The
   complement's 30.89 GiB `cudaHostAlloc` was **refused** on this box ("page-locking refused (out of
   memory)"), so the host-to-device source is pageable and its submission blocks for the whole copy -
   which is exactly what a prefix registration removes. **(2) release the victim to the file tier:
   2.27 ms/window (12%), ~60-100 lines**, with the bill (the victims' re-route rate) unmeasured.
   **(3) `cudaMemcpyBatchAsync`/3D: <= 0.05 ms. (4) `cudaMallocAsync`/pools: 0** - there are no
   per-window device allocations to pool. **(5) VMM remap: 0 or negative** (the bus is 99 us against
   the host's 44). **(6) managed memory: ruled out** - the Programming Guide's Windows rules forbid the
   engine's exact pattern (concurrent CPU and GPU access to the same host pages, no on-demand
   migration, no oversubscription).
   **The first move is the free one**: the partial pin, as an A/B in the quiet regime
   (`STRATA_PARTIAL_PIN=1 STRATA_PARTIAL_PIN_GIB=8`). It decides whether the round's `submit_ms` falls
   from 6.21 and the join from 2.53; **if they do not fall, every "who waits" option dies and only
   "fewer bytes" survives** (row 2). It needs the box, so it queues behind the DFlash 2 spike.
6. **C - the DFlash 2 prize: measured, and the port is off** (`docs/DFLASH2-PRIZE.md` §6). The spike
   ran on 105 windows of the same target, greedy, with the engine's own prefixes as the control: the
   MTP's joint per-position acceptance **68.6 / 44.8 / 23.8%** (conditional 68.6 / 65.3 / 53.2%,
   tau(2) = 2.133) against DFlash 2's **37.1 / 16.7 / 4.3%** (conditional 37.1 / 44.9 / 25.5%,
   **tau(2) = 1.486**). **That is -30% of the tokens per window - about 92 tok/s where the engine does
   131.6** - and K1 wanted 2.67.
   **The shape answered, the level did not**: the block drafter's conditionals are flat or rising
   (37/45/26) where the MTP's decay (69/65/53). The thesis that a parallel pass does not decay
   **holds**; the level is 30 points below.
   **How much to trust it**: the harness is calibrated (it reproduces the MTP's published ~69% *on this
   target*), the capture's statistics are sane (the hc-mean's norm by depth 3.1/3.6/7.2/25.2/40.2; the
   four hc streams have **diverged**, so the plain mean the draft was trained on is not the engine's
   weighted mix), but the draft sometimes copies the context's tokens, so the spike cannot separate
   "the capture is wrong" from "the model is bad". The reference that would is now cheap (llama.cpp's
   DFlash2 on the same GGUF, same tokens, both captures, per layer).
   **The engine half is done and reusable**: the capture comes out of the **layer split's hand-off**
   (`src/core/verify.cpp:1111` hands `[R][bo][inj2]` in mapped pinned host memory) - **no new kernel
   and no extra VRAM** (the same-device split shares the weights, the session and the expert cache),
   env-gated (`STRATA_CAPTURE_TAPS=6,16,26,36,46`, `STRATA_DUMP_CAPTURE`), off by default.
   **A structural limit on any offline acceptance measurement**: a dump measures **position 0
   cleanly** - the later positions of a window were computed with the *MTP's* drafts in it - so a
   drafter's gamma >= 2 profile needs its own proposals in the window.
   **Next**: the reference-capture comparison, then the port only if the capture is vindicated *and*
   the level reaches 2.67.
7. **D - the GPU-reach wait** (51% of the window): the attribution is done (§2.3) - the hyper-connection
   path is 24% of the GPU's time, the expert math 20% - so the target is a kernel, not a schedule.
8. **E** in its own session, in the sibling engine: its gate is its own host cost per expert (186 us
   assumed there, 44 us measured here).
9. **The test rig: IQ2_XS on C:** the operator's point, and it changes the economics of every
   measurement on this box. The full model at IQ3_S is 50.3 GB of experts against 63.6 GB of RAM, so the
   resident copy has to be capped and the misses come off the disk; **IQ2_XS is ~36 GB**, and the
   n-gram table is the *same file* for every quant (all three repos ship a byte-identical
   28,800,138,432 B shard 2), so the IQ2_XS trunk can be served with the IQ3_S table. The trunk does
   not fit on D: (21 GB free) but fits on C: (97 GB free); it is downloading to `C:\llms\IQ2_XS`.
   The front end needed two changes for it, both made: **`--table PATH`** (the table is not beside the
   model) and the model gate now accepts `-IQ2_XS-` (`frontend/model.hpp`). The pack comes from
   `tools/iq_pack.py` with `--base` the IQ3_S pack (the float tensors are byte-identical across the
   quant files, so `dense.bin` is hard-linked, not copied).
5. **C - the DFlash 2 prize: measured, and the verdict is "not yet"** (`docs/DFLASH2-PRIZE.md`, 559
   lines). The published numbers put DFlash 2's prize at this engine's window at **+1.7% (the mean of the
   five tasks) to +6.4% (the best)**, with a band of **-9.6% to +22%** that only a measurement can
   close - against a **+10%** bar for a new subsystem. The conversion is a *shape* question: the card
   publishes only the aggregate acceptance length (tau(3)), the engine's window is gamma = 2, and the
   shape is the whole conversion. The VRAM fits (the draft replaces the MTP: +362 MiB, the tier absorbs
   it at -0.3% of tok/s, 363-530 MiB free against the 200 MB floor) **provided the draft's own KV is
   windowed** (640 MiB at 32K, 5.0 GiB at 262K otherwise).
   **The order: (1) build the capture** - the engine-side work the port needs anyway, and B2's feature
   source; **(2) run the offline CPU-only spike** - dump the capture with the target's picks, run the
   1.2 GB draft on the CPU over a few hundred windows, count the per-position acceptance, compare with
   the MTP's profile on the same target: no GPU, no target run, an afternoon, and *not* subject to the
   tier confound (§2.7 item 2), because it counts match bits rather than differencing two tok/s;
   **(3) decide** on the measured tau(2) against the +10% bar; **(4) only then the port** - whose
   reference implementation is already in the tree (the vendored llama.cpp's DFlash2, merged 2026-08-29;
   `docs/DFLASH2-INTEGRATION.md` §9), so a full llama.cpp run against the IQ3_S GGUF is the fallback
   spike if the CPU path cannot load the checkpoint.
6. **D - the GPU-reach wait** (51% of the window): the attribution is done (§2.3) - the hyper-connection
   path is 24% of the GPU's time, the expert math 20% - so the target is a kernel, not a schedule.
7. **E** in its own session, in the sibling engine: its gate is its own host cost per expert (186 us
   assumed there, 44 us measured here).

So: two tracks in flight (B2, C), one waiting on the box (D), one in another engine (E), and one
operational item that needs the operator's decision (A0's context).

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
