# DFlash 2: what it would be worth to this engine, and what it would take to host it here

Written 2026-10-22 against the `spectre-moe` branch (HEAD `83ef9cc`). Companion to
[`DFLASH2-INTEGRATION.md`](DFLASH2-INTEGRATION.md), which has the *mechanics* (what DFlash 2 computes, where the
capture goes, what the seams are in this engine). This document is the *decision* document: the prize, the VRAM, the
integration order, the cheapest thing that would settle it, and the go/no-go criteria.

> The engine's `file:line` references are this working tree's as of 2026-10-22 14:05 — HEAD `83ef9cc` plus the
> adaptive tier's adapt-cost work in flight in `src/program/generate.cpp` (a parallel session's), which moved that
> file's line numbers by ~130. `src/core/*` and `include/*` are at HEAD.

**The verdict, in four lines.**

1. **The prize is small and uncertain.** On the published acceptance lengths, DFlash 2 is worth **+1.7%** (the mean
   of the five published tasks) to **+6.4%** (the best task) of this engine's decode *if* its per-position acceptance
   profile is flat — and the shape of that profile is exactly what the model card does not publish. The range over
   shapes and tasks is **-9.6% to +22%** (the per-shape means run -3.5% to +20.3%). The port's business case needs
   **+10%** to justify a new subsystem, and **DFlash 2's best published task does not reach it** under any profile
   shape the card's own aggregate permits, except one that requires a perfect first draft.
2. **The VRAM fits, with one hard requirement.** The draft is 1,150 MiB of BF16; the MTP it replaces is 839 MiB; the
   net is **+362 MiB**, which the expert tier absorbs by losing **186 slots (2.6%)** — about **0.3% of tok/s** — as long
   as the draft is loaded *before* the tier is sized (`src/program/generate.cpp:3207` before `:3397-3416`) and the
   draft's own KV cache is **windowed**. Unwindowed, the draft's KV is 640 MiB at a 32K context and 5.0 GiB at 262K,
   and the plan does not fit at all.
3. **The cheapest thing that would settle it is a CPU-only offline spike**: dump the engine's capture and the
   target's own picks per window, run the 1.2 GB DFlash 2 draft over the dump **on the CPU**, and count the
   per-position acceptance. No GPU, no target run, no 40 GB download. §4.
4. **But the spike needs the capture kernel, which is engine work** — a few hundred lines, and the same work the port
   needs, so it is not wasted. Until that exists, the published numbers are the only evidence, and they do not clear
   the bar.

---

## 1. The prize

### 1.1 The two numbers are the same quantity; the two *shapes* are not

The engine's 2.42 and DFlash 2's 3.26 are both **tokens committed per verification step**:

| | definition | where |
| --- | --- | --- |
| the engine | `1 + a` per window, `a` = the accepted draft prefix; the mean over a request | `src/program/generate.cpp:7483-7484` (the accept loop), `:7496` (the commit of `a + 1` rows) |
| the card | "the per-request mean of completion tokens divided by verification steps at concurrency 1" | <https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2> (Evaluation) |

So the *units* match. What does not match is the **γ** (the drafts offered per step) and the **shape** of the
per-position acceptance.

- The engine runs at **T = 3.08-3.10 rows** (γ = 2.08-2.10) at its baseline: 2.42 tokens/window, 18.34-18.50 ms/window,
  131.6 tok/s (`docs/plans/specter-performance.md` §2.1). The front end's `--spec 4` caps the window at T = 4
  (`frontend/main.cpp:805`; `src/program/generate.cpp:6038-6039`), and the MTP's confidence gate
  (`--spec-min-p 0.5`, `frontend/model.hpp:27-30`) shrinks it further — in the measured runs **13-24% of the windows
  (19% on average) verify a single row and commit one token** (36 requests in
  `logs/ab{2,4}-baseline-*.err.log`, pairing the decode-timing line's window count with the acceptance line's).
- The card's numbers are at **γ = 3**: "Speculation block size: 4 (3 draft tokens per verification step), limited by the
  current QSA implementation" (the card's Evaluation).

### 1.2 The engine's measured profile, and how to read it

The engine now prints a per-position acceptance profile (`src/program/generate.cpp:1376-1412`), and 36 requests' worth
of it is in the box's logs (`C:\Users\yoho\Downloads\UnbiasedStrata\logs\ab{2,4}-baseline-*.err.log`, the A/B regime of
`docs/plans/specter-performance.md` §2.7: `--low-ram on --resident-budget-gib 32 --ctx 32768 --kv int8`, greedy,
`temperature = 0`).

**Read it as a prefix profile, not a per-position rate.** The printed number for position *i* is
`P(a > i)` — the share of windows whose accepted prefix reached position *i* — over the windows that offered at least
one draft (`src/program/generate.cpp:1391-1393` computes the tail sums of the histogram of `a`; `:7488` excludes the
draft-less first window). It is therefore a *joint* rate, and it must decay multiplicatively even for a perfect
drafter. The **conditional** rate — position *i* accepted given that position *i-1* was — is the ratio of two
consecutive tail sums.

| | position 0 | position 1 | position 2 | position 3 |
| --- | ---: | ---: | ---: | ---: |
| **joint** `P(a > i)`, the printed rate | 81-91% (mean 85.4%) | 52-64% (57.3%) | 32-47% (38.6%) | 0-1% (0.4%) |
| **conditional** `P(a > i) / P(a > i-1)` | 85.4% | 60-74% (67.1%) | 58-79% (67.4%) | 0-3% (1.0%) |
| **pooled** per-draft acceptance (all positions) | 65.8-74.0% (mean 71.9%) | | | |

**This is the first thing the decision has to get right.** The tail "collapses" in the *joint* rates (85 → 57 → 39 →
0) but the *conditional* rates are nearly flat from position 1 to 2 (67%, 67%) and then fall off a cliff at position 3
(1%). The MTP is a **sequential** chain (`src/core/mtp.cpp:902-907`: one CUDA graph per draft step, each consuming the
previous step's token), so position 2's quality is conditional on position 1 being right — and that is what the
conditional rates show. But the joint profile's collapse is *also* what any prefix drafter's profile looks like, and it
is what the raw log line invites one to over-read. The thesis to check is therefore **not** "the MTP's profile
collapses" (it does, in the joint rates, necessarily) but "the MTP's *conditional* profile is materially worse than a
parallel block drafter's" (67% → 67% → 1% versus, say, 87% → 87% → 87%).

### 1.3 The conversion: τ(3) → τ(2)

Write `s_i = P(a > i)` (the joint rates) and `q_i = s_i / s_{i-1}` (the conditional rates). The engine's commit rule is
a **prefix** match (`src/program/generate.cpp:7483-7484`), so the tokens committed per step are

```
tau(gamma) = 1 + sum_{i=1..gamma} s_i = 1 + q_1 + q_1 q_2 + q_1 q_2 q_3 + ...
```

The card publishes only `tau(3)`, so it fixes only the **sum** `s_1 + s_2 + s_3 = tau(3) - 1`, never the shape. The
shape is the whole conversion, because the engine's window is γ = 2, not 3.

Three hypotheses, and what each implies for `tau(2)`:

| hypothesis | the shape | `tau(2)` given `tau(3)` |
| --- | --- | --- |
| **H1 flat joint** | `s_1 = s_2 = s_3` (the profile's *rates* are equal) | `1 + 2 (tau(3) - 1) / 3` |
| **H2 flat conditional** | `q_1 = q_2 = q_3 = q`, i.e. `q + q² + q³ = tau(3) - 1` (the profile's *conditionals* are equal — what a parallel block drafter's one-shot prediction should look like) | `1 + q + q²` |
| **H3 the bounds** | the prefix constraints alone: `s_1 = s_2 = s_3` (low) and `s_1 = 1, s_3 = 0` (high) | `1 + 2 (tau(3) - 1)/3` … `2 + min(1, tau(3) - 2)` |

H2 is the thesis, not an assumption to be waved through: a block drafter predicts every position of the block from the
same anchor in one pass, so its conditionals *should* be flat — but that is a claim about a model this engine has not
run, and the card publishes no per-position breakdown.

| task | `tau(3)` published | the card's MTP | H1 flat joint | H2 flat conditional | `q` (H2) | H3 low | H3 high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | **3.26** | 3.08 | 2.507 | 2.613 | 0.865 | 2.507 | 3.000 |
| MATH-500 | **3.17** | 2.93 | 2.447 | 2.563 | 0.847 | 2.447 | 3.000 |
| HumanEval | **2.94** | 2.52 | 2.293 | 2.433 | 0.797 | 2.293 | 2.940 |
| MBPP | **3.06** | 2.65 | 2.373 | 2.502 | 0.823 | 2.373 | 3.000 |
| MT-Bench | **2.83** | 2.50 | 2.220 | 2.369 | 0.772 | 2.220 | 2.830 |

The card's own MTP column is the control: the engine's MTP is within 4% of upstream's MTP at γ = 3 (the engine
measured 2.96 tokens/window at T = 4.07, `docs/plans/specter-performance.md` §2.1, against upstream's 3.08 on GSM8K),
so this engine's drafter is a faithful MTP and the comparison is DFlash 2 against the real thing.

**The engine's own `tau` from its measured profile**: `tau(2) = 1 + 0.854 + 0.573 = 2.427` and
`tau(3) = 2.813`. Note the consequence: **at γ = 2 the engine's tokens per window are already 2.42, and the maximum
possible at T = 3 is 3.00** — a 24% ceiling, not an unbounded one. The plan's "+14% at 85% per-draft acceptance"
(`docs/plans/specter-performance.md` §"the bar for a new drafter") is a *pooled* rate and is reachable at γ = 2 only by
a drafter that is near-perfect at both positions; the same pooled rate at γ = 3 is much easier to reach. Every
comparison below is done on the prefix arithmetic, which is the only one the engine's commit rule implements.

### 1.4 The prize in tok/s

**The window cost.** The sweep measured the window cost as **linear in T**: 18.4 ms at T = 3.10 and 24.11 ms at
T = 4.07 (`docs/plans/specter-performance.md` §2.1), i.e.

```
ms(T) = 0.15 + 5.89 T          (T = 2.89 -> 17.2;  T = 3.08 -> 18.3;  T = 3.83 -> 22.7;  T = 4.07 -> 24.1)
```

**Why it is linear even though the working set is not.** The plan's 28.25 is the count of **VRAM hits per
layer-window** — i.e. *reads*, with multiplicity (`src/program/generate.cpp:7563-7571` prints
`(hits - hits0)/(windows x layers)`), and 28.25 + 1.68 CPU experts ≈ 30 = 10 x T: the router's top-10 per position
(`include/strata/core/layout.hpp:51`, `num_experts_per_tok = 10` in the target's own `config.json`). The **distinct**
experts per layer-window grow **sub-linearly**; from the box's own routing trace
(`D:\llms\strata\routing-trace2.bin`, 318 sweeps, read with `tools/specter_oracle.py`'s segmentation):

| T (rows per window) | sweeps | **distinct** experts per layer-window (mean) | max | reads per layer-window |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 62 | 10.00 | 10 | 10 |
| 2 | 40 | 16.75 | 20 | 20 |
| 3 | 20 | 21.84 | 30 | 30 |
| 4 | 185 | 27.90 | 40 | 40 |
| 5 | 1 | 35.60 | 47 | 50 |
| 6 | 10 | 36.05 | 59 | 60 |

So the *tier's* working set grows sub-linearly (10 → 27.9 for four times the rows) while the *reads* grow exactly
linearly, and the CPU pool's cost is per read. **The sub-linearity is already priced into the measured linear window
cost; a wider window is not cheap, it is cheaper per token than the tokens it returns only if the acceptance holds.**
That is the whole argument for L1, and it is why the acceptance rate — not the window size — is the lever.

**The draft phase's cost.** The MTP's draft reads, per window, its dense weights (110.7 MiB) and the target's head
(~500 MiB) once per forward, and it runs one forward per draft step (`src/core/mtp.cpp:902-907`): 2 forwards at γ = 2
(1,274 MiB) and 3 at γ = 3 (1,911 MiB). Against the 4090's 1.008 TB/s, the measured 1.83 ms at γ = 2 and 2.79 ms at
γ = 3 (`docs/plans/specter-performance.md` §2.1's `draft_ms` column) put the MTP's draft at **72% of peak bandwidth**
(the same arithmetic predicts 2.75 ms for the γ = 3 arm, against the measured 2.79 — the draft phase is bandwidth-bound
and the model is sound). The DFlash 2 block pass reads its 5 layers (844 MiB) + `fc` (62.5 MiB) + the head (500 MiB) =
**1,406 MiB, once**, whatever γ is (`config.json`'s `block_size` is 8; the card's runs used 4). At the same 72%
efficiency that is **2.02 ms**: **+0.19 ms at γ = 2** (the DFlash draft is *more* expensive than the MTP's at two
drafts) and **-0.72 ms at γ = 3** (it is cheaper, because the MTP's per-step forwards multiply). Upstream's own
throughput table shows the same effect on a GB300: 368.2 tok/s for DFlash 2 against 326.0 for its MTP on GSM8K at
concurrency 1 (the card's Throughput table) — +13% where the acceptance length is only +5.8%.

**Case A — the engine keeps its window (T = 3, γ = 2, 18.41 ms/window).** The DFlash drafter is scored against the
engine's own measured `tau(2) = 2.427`; the window cost becomes 18.41 + 0.19 = **18.60 ms** with the DFlash draft.

| task | shape | `tau(2)` | tokens/window | tok/s | vs 131.6 |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | H2 flat conditional | 2.613 | 2.605 | 140.0 | **+6.4%** |
| GSM8K | H1 flat joint | 2.507 | 2.499 | 134.4 | +2.1% |
| GSM8K | H3 high (a perfect first draft) | 3.000 | 2.991 | 160.8 | +22.2% |
| MATH-500 | H2 | 2.563 | 2.556 | 137.4 | +4.4% |
| HumanEval | H2 | 2.433 | 2.426 | 130.4 | -0.9% |
| MBPP | H2 | 2.502 | 2.494 | 134.1 | +1.9% |
| MT-Bench | H2 | 2.369 | 2.362 | 127.0 | -3.5% |
| **the five-task mean** | **H2** | **2.496** | **2.488** | **133.8** | **+1.7%** |
| the five-task mean | H1 | 2.368 | 2.360 | 126.9 | -3.5% |
| the five-task mean | H3 high | 2.954 | 2.946 | 158.4 | +20.3% |

**Case B — the published configuration (T = 4, γ = 3, 24.11 - 0.72 = 23.39 ms/window).** Here the published number is
used directly (the tokens per window *are* `tau(3)`), and the control is the sweep's C1 row — the MTP at T = 4.07:
2.96 tokens/window, 24.11 ms, 122.9 tok/s.

| task | `tau(3)` | tokens/window | tok/s | vs 131.6 | vs the MTP at T = 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 3.26 | 3.260 | 139.4 | **+5.9%** | +13.4% |
| MATH-500 | 3.17 | 3.170 | 135.5 | +3.0% | +10.3% |
| HumanEval | 2.94 | 2.940 | 125.7 | -4.5% | +2.3% |
| MBPP | 3.06 | 3.060 | 130.9 | -0.6% | +6.5% |
| MT-Bench | 2.83 | 2.830 | 121.0 | -8.0% | -1.5% |
| **the mean** | **3.052** | **3.052** | **130.5** | **-0.8%** | **+6.2%** |

Two things to take from the two cases. **At the engine's own window the prize is a low single-digit percentage**; at
the published window it is a wash. **But the DFlash draft is cheaper at γ = 3 than at γ = 2** (one pass for the whole
block), and the engine's window is 2 — so the port's economics improve with the block size, and the card's block size
(4, limited by "the current QSA implementation") is the number to argue with, not the acceptance.

**The bar.** A new model type, a new attention kernel, a capture kernel, a selector: weeks. The decode it has to buy
is **+10%** (about 13 tok/s, ten times the box's ±1.1 tok/s noise floor, `docs/plans/specter-performance.md` §2.1) at
minimum, and the plan's own bar was **+14%** (§2.1). In Case A's arithmetic, +10% needs `tau(2) = 2.670`, i.e. under
H2 a per-position conditional of **0.90 flat** — above anything the card implies (its best task, GSM8K, implies
q = 0.865 and `tau(2) = 2.613`, +7.5% before the tier and the draft's own cost). **DFlash 2's best published task does
not clear the bar under a flat profile; only the H3-high shape (a perfect first draft, s_1 = 1) does, and that shape
is not the one a trained drafter has.**

### 1.5 What moves the number

| lever | direction | size |
| --- | --- | --- |
| **the profile's shape** (H1 → H3-high) | ±20 points of tok/s | the largest single term; *the thing to measure* |
| the task (MT-Bench → GSM8K) | ±8 points | the card's own spread; this engine's traffic is not the card's benchmarks |
| the draft's cost (the 72%-of-peak estimate) | ±3% of the window | measurable in one run once the pass exists |
| the tier's shrinkage for the draft's VRAM | -0.3% | §2 |
| the target's precision (FP8 → IQ3_S) | negative, unquantified | the draft borrows the target's *quantized* embedding and head (`src/core/verify.cpp:594-606`, `src/program/generate.cpp:3221`); the card's numbers are from an FP8 target |
| the sampling regime | **in DFlash 2's favour** | the card evaluates with "temperature 1.0, top-p 0.95, top-k 20"; this engine's measured baseline is **greedy** (`temperature = 0` in the harness), where the exact-match acceptance is higher. The card's numbers are therefore a *harder* setting than the baseline they are compared against — one of the few places the published evidence cuts for the port |
| the T = 1 windows (19% of them today) | positive if the DFlash drafter has no confidence gate | the MTP's `--spec-min-p 0.5` gate truncates the window to a single row when its first draft is unsure (`src/program/generate.cpp:7433-7437`); a block drafter has no per-token probability, so the gate would have to be redesigned — and that redesign is worth up to a fifth of the windows |

---

## 2. The VRAM

### 2.1 The arithmetic

| item | bytes | MiB | source |
| --- | ---: | ---: | --- |
| the DFlash 2 checkpoint (81 BF16 tensors) | 1,206,140,008 | **1,150.3** | the card's `model.safetensors` header |
| the draft's KV, a 2,048-cell sliding window | 44,564,480 | 42.5 | 5 layers × 8 KV heads × 128 dims × (K,V) × 2,176 cells × 2 B |
| the draft's logits (8 rows × 248,320 × f32) | 7,946,240 | 7.6 | `vocab_size` in `config.json` |
| the capture (5 layers × 4 rows × 2560 f32) | 204,800 | 0.2 | `target_layer_ids` in `config.json` |
| the selector's scratch (top-16 × 8 positions) | — | ~1 | `selector_top_k` in `config.json` |
| **the draft's total, windowed** | ~1,259,000,000 | **~1,201** | |
| the MTP it replaces | | **839** (866 in the 262K-context runs) | the engine's own print: `strata mtp: draft layer loaded, 839 MiB of VRAM (experts 675, dense 111)` (`src/core/mtp.cpp:305-309`) |
| **the net** | | **+362** (+335) | |

The free VRAM with the tier in place is **725-865 MiB** measured (`docs/plans/specter-performance.md` §0.1's 747-865;
the A/B arms 725-777; the `ab-forecast4.csv` rows all report `vram_min_mb` = 777). So after the swap:

```
725 MiB free  + 839 (the MTP's VRAM, freed)  - 1,201 (the draft)  =  363 MiB free
865 MiB free  + 866 (the MTP's VRAM, freed)  - 1,201 (the draft)  =  530 MiB free
```

**363-530 MiB free, against the box's 200 MB floor** (`docs/plans/specter-performance.md` §0.1). The verdict is *fits*,
with 1.8-2.7× the floor in hand.

> The arithmetic in `docs/DFLASH2-INTEGRATION.md` §6.2 says "+285 MiB and 465 MiB free". It uses 1,124 MiB for the
> draft; the checkpoint is 1,206,140,008 B = **1,150.3 MiB**, and it ignores the draft's own KV, logits and capture
> (~51 MiB). The corrected net is **+362 MiB** (not +285), and the free VRAM is a **range, 363-530 MiB**, not a single
> 465. The verdict is unchanged; the margin is ~80 MiB tighter and, more importantly, it depends on the draft's KV
> being windowed at all (§2.2).

**The tier absorbs the difference automatically**, and cheaply. The tier is sized from the free VRAM *after* the drafter
is loaded (`src/program/generate.cpp:3397-3416`: `cudaMemGetInfo` → `reserve` → `slots = (free - reserve)/blob`), and
the MTP is loaded before that point on purpose (`src/program/generate.cpp:3207`, and the comment at `:3210-3215` on
why the head and the drafter go first). A DFlash draft must be loaded the same way, and then:

```
362 MiB / 1.944 MiB per slot  =  186 slots of the 7,072-slot tier  (2.6%)
```

Measured on the box's own trace, the marginal 186 slots of a frequency-ranked tier answer **0.53 points** of the reads
(the trace's top-7,072 pairs cover 91.3-91.5% of the reads; the top-6,886 cover 90.8-91.0%). At the measured slope of the
forecast arm — +2.1 points of hit rate bought -1.3% of the window (`docs/plans/specter-performance.md` §2.7 item 2) —
that is **-0.3% of tok/s**. The `C3` arm's -15.5% is a warning, not a cost: that arm changed the tier's *shape*, not
its size.

### 2.2 The draft's own KV: a hard requirement, not an optimisation

The reference implementation (upstream `dflash/model.py`,
<https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py>) keeps a **growing** KV cache of the projected
target features and crops it to the sequence start each step (`dflash/model.py:243-250`, `_crop_to`), while
`config.json` says `use_cache: false`. The
cache's size is what that disagreement costs:

| the draft's KV | bytes | vs the 839 MiB MTP |
| --- | ---: | --- |
| a 2,048-cell sliding window (the config's `sliding_window`) | 42.5 MiB | +362 MiB net — **fits** |
| the 32K context, no eviction | 640 MiB | +1,002 MiB net — the tier loses 515 slots (7.3%), -1.6 points of coverage, ~-1.0% of tok/s |
| the 262K context, no eviction | **5,120 MiB** | +5,482 MiB net — the tier would lose 2,820 slots (40% of it): **not viable** |

**So the port must implement the sliding-window eviction, and the reference implementation's eviction policy is not
specified anywhere** (`docs/DFLASH2-INTEGRATION.md` §8.6). This is the single largest technical unknown in the VRAM
plan, it is arithmetic rather than opinion, and it is checkable in an afternoon: the draft's attention is
non-causal with a 2,048 window (`config.json`: `is_causal: false`, `sliding_window: 2048`), and the reference's mask
keeps `|q_pos - k_pos| < 2048` on both sides (`dflash/model.py:157-171`), so a windowed ring is *semantically* what
the model was trained for — but the cache it is trained with may have been unbounded, and the port has to reproduce
whatever the training used.

### 2.3 The alternatives, and their costs

| lever | what it frees | its cost |
| --- | --- | --- |
| **L1: DFlash 2 replaces the MTP** (the plan) | the MTP's 839 MiB | the MTP's acceptance has to be replaced by DFlash 2's, and the draft's own cost changes (§1.4). This is the point of the port and its risk. |
| **L2: a smaller tier** (if the draft sat *beside* the MTP) | 1,201 MiB = 617 slots (8.7% of 7,072) | measured on the trace: the marginal 617 slots answer 1.85 points of the reads → ~-1.1% of tok/s, at the same slope. Not needed if the MTP is replaced. |
| **L3: the KV at 4 bits** (`--kv q4_0`, the engine's rotated 4-bit codes, `src/program/generate.cpp:611-612`) | the KV's per-cell cost halves (int8's 1,056 B → ~528 B); in the sweep's regime (32K, int8) the KV is ~0.4 GiB, so ~200 MiB = 103 slots | the engine's own help: "slightly lower precision (see bench/results/2026-09-27-kv-q4)". `docs/plans/specter-performance.md` §6.2's lever (via `docs/DFLASH2-INTEGRATION.md` §6.2) quotes an upstream measurement of "+8-12% perplexity" and "~4% slower at 128K" for this mode, which I did not re-verify. **It is not needed**: the swap fits without it. |

> A correction to `docs/DFLASH2-INTEGRATION.md` §6.2: it states that the engine "refuses `q4_0` *together with*
> `--kv-resident` (the help says so)". The code refuses only `k8v4` (`src/program/generate.cpp:1960-1962`); `q4_0` with
> `--kv-resident` is accepted.

---

## 3. The integration, in the order it would be built

### 3.1 The capture (first, because it is also the routing predictor's feature source)

The tap point is already derived (`docs/DFLASH2-INTEGRATION.md` §2.3, and it holds in this tree): the engine's residual
is one `(T, HC, N)` buffer overwritten 48 times per window (`src/core/verify.cpp:560`), and the pending FFN write of
layer *l-1* is applied at the first read of layer *l* (`src/core/verify.cpp:635`, `:696`). So the capture of layer 5
is taken in `pre(6)`, layer 15 in `pre(16)`, layer 25 in `pre(26)`, layer 35 in `pre(36)`, layer 45 in `pre(46)`.

What has to be written:

1. **The reduction kernel.** The upstream patch takes the *plain arithmetic mean* of the target's four
   hyper-connection streams (`patches/qwen38-dflash2-capture.patch`, in the card's repo); the engine's only reduction
   of those streams is the hyper-connection's **learned weighted** mean
   (`include/strata/kernels/fused_gr.hpp:13`), and there is no mean kernel in the tree. Five copies of
   `cap[t][l][:] = (1/4) sum_h R[t][h][:]` inside the verify window's graph.
2. **The five copies**, at the tap points above.
3. **The equivalence question** (`docs/DFLASH2-INTEGRATION.md` §8.1-8.2): the engine folds the pending write into the
   next layer's read (`include/strata/core/layer.hpp:461-465`, `include/strata/kernels/cvec.hpp:55-58`) and its
   summation order differs from the native MMVF's (`include/strata/kernels/fused_gr.hpp:15`). Whether the engine's
   "completed output" equals SGLang's `hidden_states` at the same layer is **not established** and is not settled by
   reading. It is settled by the §4 spike — and the spike does not need it settled first, because the spike feeds the
   draft the *engine's* capture, which is what the port would feed it.

The capture is also B2's feature source for the routing predictor (`docs/plans/specter-performance.md` §2.1's Track
B2), so this work is not exclusive to the drafter.

### 3.2 The draft model's loader (a new model type)

The engine's loader is bespoke: `WeightTable::load` parses a 19-field space-separated `index.txt`
(`src/core/weights.cpp:144-188`), and the header says why there is no JSON parser: "a JSON parser in the engine would
be a new, unaudited component whose failure mode is a wrong byte offset - which decodes to a plausible weight and
produces plausible logits" (`include/strata/core/weights.hpp:20-27`). The MTP has its own second loader
(`dense.txt` + `dense.bin` + `experts.bin`, `src/core/mtp.cpp:163-209`).

So a DFlash draft needs:

1. **A safetensors reader** (an 8-byte length + a JSON header + the tensor map) — the engine has none; the only
   safetensors code in the tree is Python (`tools/gguf_reader.py` is a GGUF reader).
2. **A draft geometry struct.** The draft is 5 layers, hidden 2560, 32 heads / 8 KV / head_dim 128, intermediate
   7680, vocab 248,320, `is_causal: false`, sliding window 2048 (`config.json`) — nothing like the target's
   `ModelGeometry` (`include/strata/core/layout.hpp:27-59`: 24 heads / 2 KV / head_dim 256, QSA + GDN). The draft's
   tensor names (`layers.0.self_attn.q_proj.weight`) are not the engine's (`include/strata/core/layout.hpp:72-83`).
3. **A shape contract** like `check_layer` (`include/strata/core/layout.hpp:91`) over the draft's 81 tensors, so a
   mismatch is a load-time error and not "a plausible weight" — the same argument as the weights header.
4. **The embedding and the head, borrowed from the target** — they are *not* in the checkpoint (the card's 81 tensors
   contain no `embed_tokens` and no `lm_head`), and here both are **quantized**: the embedding is an S2/S4/S8 tensor
   (`src/core/verify.cpp:594-606`) and the head is the engine's `NativeHead` (`src/program/generate.cpp:3219-3223`).
   The published numbers come from an FP8 target, so the draft's inputs here are materially coarser than the ones it
   was trained against. This is a first-order risk to the acceptance, not a detail.
5. **A BF16 arena.** The checkpoint stays BF16 (1,150 MiB of VRAM, no IQ3_S-style option): the engine's quant formats
   are its own IQ family plus Q8_0 (`include/strata/core/weights.hpp:38-46`).

### 3.3 The block-diffusion pass

The draft proposes a block in **one** masked, non-causal, parallel forward over the 5 layers
(`dflash/model.py:233-258`; the paper's §3.2 cost argument: `T_draft = t_parallel`, insensitive to γ for moderate
block sizes). What has to be written, in order:

1. The layer: RMSNorm → conv `prepare` → q/k/v → RoPE → **non-causal** sliding attention over `[context KV ; block]`
   → o_proj → conv `finish` → residual → RMSNorm → conv `prepare` → MLP → conv `finish` → residual, with the two
   two-tap dynamic convolutions (`dflash/model.py:492-512`, `:639-651`).
2. **The KV injection**, the paper's central design (`docs/DFLASH2-INTEGRATION.md` §1.3):
   `H_t = RMSNorm(W_c [H^(5);H^(15);H^(25);H^(35);H^(45)])` with `W_c` 2560×12800 (62.5 MiB), then per layer
   `K = [W_k H_t ; W_k H_d]`, `V = [W_v H_t ; W_v H_d]` — the target's features "bypass the draft model's Q
   projection, output projection, self-attention update, and FFN" (the paper's A.3). The ablation puts the
   difference at GSM8K 3.5 → 4.2 acceptance length.
3. **A non-causal attention kernel.** There is none: the engine's attention is the target's QSA (block-sparse, 24/2/256,
   `include/strata/core/layout.hpp:40-45`) or GDN. The draft needs dense, non-causal, 32/8/128 with a 2,048 window.
4. **The draft's own KV** in the draft's geometry, and its eviction policy (§2.2).
5. **Reusable**: `bf16_gemv_fp32_mmvf_multi` (`include/strata/kernels/bf16_gemv.hpp:53`; 1..8 rows,
   `src/kernels/cuda/native_bf16.cu:141`) covers every projection — the block is 8 positions, so 8 rows fits exactly;
   `rms_norm_weighted` (`include/strata/kernels/elementwise.hpp:82`) covers the norms; the target's head is the
   draft's head.

### 3.4 The selector

DFlash 2's addition over DFlash 1 (`dflash/model.py:515-547`): a top-16 over the draft's logits at every position, a
**predecessor** and a **successor** codebook (`nn.Embedding(248320, 256)` each, 121.25 MiB each), a per-position score
`unary[position][k] + <codebook_pred[predecessor] * hidden_projection(hidden[position]), codebook_succ[candidate[k]]>`,
and a greedy (or sampled) path trace. It is a new kernel: the codebooks are *gathers* (one predecessor row, 16 successor
rows per position), not a full read, so they are cheap but they are a random access pattern the engine has no kernel
for.

Note what the selector is *not*: a wider candidate tree. It proposes **one** token per position and discards the other
15. The engine has no tree verification to reuse and none to break — but it also gets no coverage from the candidates.

### 3.5 The acceptance and verification path

Reusable as they are. The engine's rule is an exact-match prefix (`src/program/generate.cpp:7483-7484`) and the commit
takes the first `a + 1` rows (`:7496`); the K-rows-verified-per-step semantics are already there (T rows, T-1 drafts, an
accepted prefix, the rest discarded). The losslessness contract is already stated the way the card states it — "a draft
is kept only when it EQUALS that row's pick. The output is therefore a function of the seed alone, whatever the drafts
are: drafts only decide how many tokens a window yields" (`include/strata/core/coupled_draft.hpp:4-6`).

What changes is the **proposal's** semantics: one path, a higher hit rate on it, not a tree. And the **coupling**: the
selector's score is not a vocabulary distribution, so the per-position `q` that the coupled rejection sampler needs
(`include/strata/core/coupled_draft.hpp`) is not directly available. Upstream's own selector *does* return a per-position
`q` when it samples (`dflash/model.py:527-533` returns `q_rows`; `_rejection_sample` consumes it at `:104-105`), so the
port can expose it; but under **greedy** decoding — this engine's measured baseline — the selector's argmax path
(`dflash/model.py:535-537`) is what the draft proposes, and the draft's `q` is then 1 on the chosen token, which is
compatible with the coupled sampler's exact-match rule only for the greedy case.

### 3.6 What the MTP does that the new drafter must replace

| what | where | the port's equivalent |
| --- | --- | --- |
| an **autoregressive chain**: one CUDA graph per draft step, each consuming the previous step's token, with a host spin between them | `src/core/mtp.cpp:902-907`, `:910-921` | **one** parallel pass over the block. This is the win, and the reason the block size matters. |
| the **confidence gate** that shrinks the window to the MTP's confident prefix (`--spec-min-p 0.5`) | `src/program/generate.cpp:7433-7437`; `frontend/model.hpp:27-30` | the selector has no per-token probability. A gate has to be redesigned — and 19% of the windows are T = 1 today, so this is worth up to a fifth of the windows, not a detail. |
| the **argmax** proposal | `src/core/mtp.cpp:639-643` (`sample_tokens` with `sp.greedy = true`, `row_top_prob`) | the selector's argmax path, plus the `q` for the coupled mode (§3.5). |
| **512 resident experts (675 MiB)** and the grouped hit kernels | `include/strata/core/mtp.hpp:19`; `src/core/mtp.cpp:189-209`, `:573-579` | **none** — DFlash 2 is dense, with no router and no experts. This is 675 MiB of the 839 that comes back. |
| the **dense weights in Q8_0** through the multi-column MMVQ | `include/strata/core/mtp.hpp:15`; `src/core/mtp.cpp:210-213` | BF16 through `bf16_gemv_fp32_mmvf_multi`. |
| the **K/V ring** in the *target's* QSA geometry, with a host copy for KV streaming | `include/strata/core/mtp.hpp:54-56`; `src/core/mtp.cpp:215-245` (`qsa_state_init` with a ring at `:230`), `kv_restore` | **not reusable**: the draft needs dense, non-causal, 32/8/128 with a 2,048 window, and it has no ring to inherit (§2.2). |
| the **prompt path's** draft K/V (`prefill`) | `src/core/mtp.cpp:673-689`, `include/strata/core/mtp.hpp:91-97` | the draft's own context KV has to be built over the prompt. |
| the **draft head** over a vocabulary subset, and its `bind_bytes` reservation | `src/core/mtp.cpp:313-330`; `src/program/generate.cpp:3407-3409` | the target's head, plus a per-position `q`. |
| the **controller's** drafter choice | `include/strata/spec/controller.hpp:40` (`enum class Source { None, Lookup, Mtp }`), `:23` (`K_MAX = 8`), `:25-38` (the `CostModel`, with `mtp_draft_ms` and `distinct_ratio`) | a fourth source, a DFlash cost term, and a `distinct_ratio` that matches the block's shape. |
| the **suffix drafter's** interplay with the MTP's first draft | `src/program/generate.cpp:7444-7450`; the sweep measured the suffix drafter as idle (1 of 3 drafts accepted) | unchanged, but the MTP's first draft is no longer the reference. |

---

## 4. The cheapest thing that would settle it

### 4.1 What has to be measured

**DFlash 2's per-position acceptance profile (s_1, s_2, s_3) against *this* engine's target**, and comparably the MTP's
on the same target. Everything else in this document is arithmetic on top of that number. §1.3 shows the number's
effect: the shape moves the prize by ±20 points of tok/s, and no published aggregate can supply it.

Two block sizes should be measured, because they are not the same run: the card's **γ = 3** (the configuration it
publishes, and the one that gives the most positions to read) and **γ = 2** (what the engine's own window would use).
A block drafter's unit is the block, so a γ = 2 run is not the first two positions of a γ = 3 run.

### 4.2 The obstacle, stated plainly

- The `dflash` package (<https://github.com/z-lab/dflash>) is a PyTorch implementation; its local backend is
  Transformers on Linux (MLX is Apple Silicon), so a local run needs torch + transformers + a CUDA GPU.
- The target on the box is **EXL3 2.05 bpw** at `D:\llms\Qwen3.8-Flash-Next-exl3-2.05bpw`: 5 safetensors shards
  totalling **36.4 GB** (plus a 26.2 GB n-gram table). EXL3's format is exllamav3's own, so Transformers cannot load
  it, and 36.4 GB does not fit in the 4090's 24 GB of VRAM either.
- The card's own target — "the official FP8 checkpoint" — is **not on the box**, and the model's own geometry
  (48 layers × 512 experts × 3 × 2560 × 640 = **121 G parameters of experts alone**) puts an FP8 copy at ~120 GB: it
  fits neither in 24 GB of VRAM nor in 63.6 GB of RAM, let alone in this session's container. (The IQ3_S pack's
  54.8 GB of experts at ~3.6 bits is the same model, and it is what this engine serves.)
- **This container** sees **22 GB of RAM** (`free -g`), 6 GB of swap, no torch, and the `D:` drive has 21 GB free.
  So a full local run of the target does not fit here, and the box is currently running another agent's measurement —
  whose RAM/VRAM floors are box-level, so the two cannot run together.
- The measurement protocol on the card is 128 samples per task at concurrency 1, five tasks, 4,096 new tokens: that is
  **hours of exclusive GPU**, not an afternoon.

### 4.3 The cheap path: an offline, CPU-only, no-target spike

The acceptance measurement does **not** need the target to run in PyTorch, and it does not need the GPU. It needs:

1. **The engine's capture and the target's own picks, per window.** The engine already records the target's pick for
   every row (`outv[a]`, `src/program/generate.cpp:7483-7484`) and already knows the accepted prefix. What is missing
   is the capture: the 5 layers × 2560 features per position, plus the block's token ids and the target's input
   embedding of the anchor (the draft's `noise_embedding`, `dflash/model.py:239-241`). **The engine-side work is the
   capture kernel of §3.1** — a few hundred lines, and the same work the port needs.
2. **The DFlash 2 checkpoint** (1.2 GB, downloadable) and a **CPU** PyTorch.
3. **A script** that, for each window, runs the draft over the captured block and compares its proposed path with the
   target's recorded picks, position by position.

The arithmetic of the cost: the draft is 0.6 B parameters, so a forward over a block of 8 positions is ~1.2 GB of
weight reads and ~10 GFLOP. On this container's 30 cores that is **well under a second per window** — a few minutes for
a few hundred windows. The RAM needed is the draft's 1.2 GB plus torch: it **fits in 22 GB**. The GPU is **not
needed**: the spike measures *acceptance*, not speed. (The draft's *speed* on this card is a separate, later
measurement — and it is the cheap one, because the block pass is 1,406 MiB of VRAM reads against a measured 72%-of-peak
MTP.)

**What this spike does and does not settle.** It settles the *shape* question on *this* engine's capture and *this*
engine's target — the number §1.3 says the whole decision rests on. It does not settle the capture's equivalence with
SGLang's `hidden_states` (§3.1 item 3), but it does not need to: it feeds the draft the capture the port would feed it,
so a low number would be a true negative for the port, not an artefact. It does not settle the draft's speed on the
4090 (a separate, 10-minute measurement once the pass exists in CUDA).

### 4.3a The second spike path: an end-to-end llama.cpp run on the Windows host

The vendored llama.cpp has DFlash2 (see §4.5), and the target is on the box as a GGUF, so the fuller
spike is available: convert the 1.2 GB draft to GGUF, build a llama.cpp server from the vendored source,
and run the `draft-dflash` type against the IQ3_S target. It measures the acceptance end to end *and*
the draft's speed on the 4090 - at the cost of a build and a conversion, and with the tier confound
absent (it is a different engine). §4.3 remains the first move: it is cheaper, and it measures the
shape, which is the whole decision.

### 4.5 Two things that landed after this document was written (2026-10-21, later)

**1. The port's reference implementation is already in this repo.** `third_party/llama.cpp` is pinned at
`3cf03257` (2026-09-20), after the merge of ggml-org/llama.cpp#27342 (2026-08-29, "spec: add DFlash2
support (local convolution + candidate selector)", +639/-24). The vendored tree has the block-diffusion
drafting path with the draft-side KV injection (`common/speculative.cpp:909+`), the model
(`src/models/dflash.cpp`), the hyperparameters (`src/llama-hparams.h:250-254`), the CLI's `draft-dflash`
type (`common/arg.cpp:370-395`, `common/speculative.cpp:38`) and the GGUF conversion
(`conversion/qwen.py:649+`). So §3.3 and §3.4 (the block pass and the selector) are a *port* of ~600
lines of reference code, not a design from scratch - and the DFlash2 model card's own "does the engine
have it?" framing (§4.3 of `DFLASH2-INTEGRATION.md`) is too strong: it does not exist in *this
engine's* decode path, but the reference is in the tree.

**2. A second spike path, on the Windows host, end to end.** The `dflash` package's two local backends
are MLX (Apple Silicon) and Transformers (Linux), and neither lists Qwen3.8-Flash-Next for DFlash 2
(Transformers: Muse-Glimmer-30B; MLX: Qwen3.8-27B) - for Flash-Next the card documents the SGLang or
vLLM server path, which is Linux-first and is not installed on the box. But the box has the target as a
**GGUF** (`D:\llms\IQ3_S\...-00001-of-00002.gguf`, 54.8 GB) and llama.cpp reads GGUF natively, so:

- convert the DFlash2 draft (1.2 GB of BF16 safetensors) to GGUF with `convert_hf_to_gguf.py`
  (`--target-model-dir` for the tokenizer; the PR warns that GGUF written before 2026-08-27 must be
  reconverted), in a venv;
- build a llama.cpp server from the vendored source (or upstream, which has the same merge);
- run it against the IQ3_S target with the `draft-dflash` speculative type.

That measures the acceptance **end to end on this target and this card** (and the draft's speed on the
4090, which §4.3 defers) - the fuller version of the §4.3 spike, at the cost of a build and a
conversion. The §4.3 spike stays the cheaper first move: it measures the *shape* (the whole decision)
without the GPU, without the target running, and without the tier confound of §5.3.



**No, and the reasons are specific:**

| what the card gives | why it is not enough here |
| --- | --- |
| an **aggregate** `tau(3)` per task | the conversion to this engine's γ = 2 needs the *shape*, and the shape is worth ±20 points of tok/s (§1.3). The aggregate cannot be inverted: many profiles produce the same `tau(3)`. |
| a **different target** (the official FP8) | here the target is IQ3_S and the draft borrows the *quantized* embedding and head (§3.2 item 4). A coarser head and embedding can only lower the acceptance, and the size of the effect is unmeasured. |
| a **different card** (one GB300) | this engine is a 4090 with a CPU expert pool that costs 3.01-3.53 ms of the 18.4 ms window (`docs/plans/specter-performance.md` §0): the acceptance profile of a drafter is a property of the *target's* logits, and a different quantisation moves them in the last bits — the plan measured exactly this effect for the tier (the forecast arm's 2.1 points of hit rate changed the drafter's window by 1.6%, §2.7 item 2). |
| a **sampled** regime (temperature 1.0, top-p 0.95, top-k 20) | this engine's measured baseline is greedy. The card's numbers are from the harder setting, which is *in the port's favour* and is the one thing the published evidence does support. |
| a **different harness** (SGLang, with a capture patch) | the engine's own accept rule, its T=1 windows and its confidence gate are all different; the *quantities* are the same, the *operating points* are not. |

What the published numbers **do** support, stated as a bound: **DFlash 2 beats its own MTP on every published task**
(3.26 vs 3.08, 3.17 vs 2.93, 2.94 vs 2.52, 3.06 vs 2.65, 2.83 vs 2.50 — the card's Acceptance Length table), and the
engine's MTP is within 4% of upstream's MTP. So DFlash 2 is *probably* better than this engine's drafter — by
0.18-0.42 acceptance-length points at γ = 3, which is **+6% to +17% of the tokens per window at that γ** and, at this
engine's γ = 2, **+1.7% to +6.4% of the decode** (§1.4). That is the honest size of the prize on the published
evidence: **a few percent, with an uncertainty band of ±20 points that only the measurement can close.**

---

## 5. Go / no-go

### 5.1 What would justify building the port

The port is a new model type, a new attention kernel, a capture kernel and a selector. The bar is a decode gain that
pays for that: **≥ +10%** (≥ 13 tok/s, ten times the box's ±1.1 tok/s noise floor,
`docs/plans/specter-performance.md` §2.1), and the plan's own bar was **+14%** (§2.1).

| criterion | the threshold | why |
| --- | --- | --- |
| **C1: the acceptance** | the measured `tau(2) ≥ 2.67` at T = 3 (a pooled per-draft acceptance of **83%** at γ = 2, or a per-position conditional of **~0.90 flat**) | +10% at the measured window cost. Under the flat-conditional shape this needs q = 0.902 — above the card's best task (q = 0.865). |
| **C2: the draft's cost** | the measured block pass ≤ **2.0 ms** on the 4090 | the estimate is 2.02 ms at 72% of peak; above ~2.5 ms the draft eats the acceptance gain at γ = 2. |
| **C3: the capture** | the capture kernel exists and its cost is inside the window (< 0.3 ms) | five reductions of 5 × T × 10240 floats per window is ~0.1-0.2 ms; if it is more, the tier's shrinkage and the draft's cost start to compound. |
| **C4: the KV** | a windowed draft KV, or a specified eviction policy that matches the training | §2.2: unwindowed, the plan does not fit at the production context. |
| **C5: the tier** | the draft is loaded before the tier is sized, and the tier's loss is ≤ 200 slots | §2.1: 186 slots, -0.3% of tok/s, measured on the box's own trace. |

### 5.2 What would kill it

| criterion | the threshold | why |
| --- | --- | --- |
| **K1: the acceptance** | the measured `tau(2) ≤ 2.50` at T = 3 | ≤ +3%: below the noise floor of a tok/s A/B, and not worth a subsystem. |
| **K2: the shape** | the measured conditionals are not flat — e.g. q_1 ≈ 0.90 but q_2 ≈ 0.70 | the block drafter's premise is that a parallel pass does not decay; if it decays like the MTP's, DFlash 2 *is* the MTP with a different kernel. |
| **K3: the draft's cost** | > 2.5 ms, or the port's extra buffers push the free VRAM under 400 MiB | the free VRAM is 363-530 MiB after the swap (§2.1); the draft's own buffers, the capture and the selector's scratch have to fit in the same 363 MiB. |
| **K4: the target's precision** | the measured acceptance against IQ3_S is materially below the card's FP8 numbers | the draft borrows the quantized embedding and head; if the loss is > 5 points of `tau(2)`, the prize is gone. This is K1's mechanism, and the spike measures it. |
| **K5: the equivalence** | the engine's capture is *not* what the draft was trained on, and the port cannot fix it | §3.1 item 3. Note the asymmetry: the spike cannot distinguish "the engine's capture is wrong" from "DFlash 2 is not good", because it feeds the draft the engine's capture. If the measured profile is bad, **the port is dead** — a wrong tap point and a bad model look identical from the acceptance alone. So the spike must also dump the residual at the five layers and compare it against a reference implementation's `output_hidden_states` for the same tokens. |

### 5.3 The measurement caveat that applies to any A/B here

A tok/s A/B on this engine **cannot read a change worth less than ~2% of the window**: which experts the GPU computes
and which the CPU does changes the logits in the last bits, which changes the drafter's window (the forecast arm:
+2.1 points of hit rate, -1.6% of the tokens per window, `docs/plans/specter-performance.md` §2.7 item 2). So:

- A **+1 to +2%** effect is not measurable here and must not be claimed from one run.
- A **+5%** effect is ~6x the ±0.8% noise floor but *inside* the 2% tier-confound band: it is only trustworthy in a
  single-process interleaved A/B with the tier's state held equal.
- The **acceptance measurement of §4.3 is not subject to either caveat**: it is a direct count of match bits, not a
  difference of two tok/s. That is the strongest argument for doing the acceptance first and the A/B second.

### 5.4 The order, and the recommendation

1. **Build the capture** (§3.1) and dump it with the target's picks. It is B2's feature source as well, and it is the
   prerequisite for everything else.
2. **Run the offline DFlash 2 spike on the dump** (§4.3): a CPU forward over a few hundred windows, the per-position
   profile, and the same numbers for the MTP on the same target. No GPU, no target run, an afternoon.
3. **Decide on C1** (the measured `tau(2)`) and C2 (the draft's cost, which needs the CUDA pass — but only a
   bandwidth measurement of 1,406 MiB against the 1,274 MiB the MTP reads, not a port).
4. **Only then**, the port: the loader, the block pass, the selector, the acceptance path.

**The recommendation, stated against the evidence available today: do steps 1 and 2, and do not start the port.** The
published numbers put DFlash 2's prize at this engine's window at **+1.7% (the mean of the five tasks) to +6.4% (the
best)**, with an uncertainty band of **-9.6% to +22%** (the per-shape means run -3.5% to +20.3%) that only the
measurement can close, against a bar of **+10%** for a new subsystem. The measurement is cheap, it is the same work the
port needs, and it is the only thing that distinguishes "DFlash 2 is worth +20% here" from "DFlash 2 is worth nothing
here". Everything else in this document is arithmetic waiting for that number.

---

## 6. What the spike measured (2026-10-21): the port's business case does not close

The spike ran. 105 windows of the same target, greedy, the engine's own prefixes as the control.

| per position | joint | conditional |
| --- | ---: | ---: |
| the MTP (the engine's drafter) | 68.6 / 44.8 / 23.8 % | 68.6 / 65.3 / 53.2 % |
| **DFlash 2** (upstream `dflash`, CPU, fed the engine's capture) | **37.1 / 16.7 / 4.3 %** | **37.1 / 44.9 / 25.5 %** |

`tau(2)`: the MTP **2.133**, DFlash 2 **1.486** (and 1.419 at gamma = 2, a block of 3). **That is -30% of the
tokens per window - about 92 tok/s where the engine does 131.6.** K1 wanted 2.67.

**The shape answered, the level did not.** The block drafter's conditionals are flat or rising
(37 / 45 / 26) where the MTP's decay (69 / 65 / 53) - the thesis that a parallel pass does not
decay **holds**. The level is 30 points below.

**How much to trust it.** The harness is calibrated: it reproduces the MTP's published ~69% *on this
target* from the engine's own prefixes, so the measurement of the MTP's side is sound. The capture's
statistics are sane (the hc-mean's norm by depth: 3.1 / 3.6 / 7.2 / 25.2 / 40.2; and the four
hyper-connection streams have *diverged*, so the plain mean the draft was trained on is not the
engine's weighted mix). But the draft sometimes copies the context's tokens, so **the capture may not
be what DFlash 2 was trained on**, and the spike cannot separate "the capture is wrong" from "the
model is bad" (K5). A 38-point loss is far past K4's 5-point allowance either way.

**The reference that would separate them** is now cheap, because the vendored tree has the path:
llama.cpp's DFlash2 on the same IQ3_S GGUF, reading the target's layer inputs
(`common/speculative.cpp:1143`), same tokens, both captures, per layer. If llama.cpp's capture and
the engine's capture agree and DFlash 2 still scores 1.49, the model is the problem and the port is
dead. If they disagree, the capture was the problem and the port is still dead - but differently,
and cheaper.

**A structural limit found on the way, which applies to any offline acceptance measurement:** a dump
measures **position 0 cleanly** - the later positions of a window were computed with the *MTP's*
drafts in it, not the drafter's own proposals. A drafter's gamma >= 2 profile needs its own
proposals in the window, so the numbers above are the *position-0* profile, and DFlash 2's block
structure is not exercised by them.

**The engine half is done and reusable** (`src/program/generate.cpp`, env-gated, off by default:
`STRATA_CAPTURE_TAPS=6,16,26,36,46` and `STRATA_DUMP_CAPTURE`): the capture comes out of the
**layer split's hand-off** (`src/core/verify.cpp:1111` hands `[R][bo][inj2]` in mapped pinned host
memory), so there is **no new kernel and no extra VRAM** - the same-device split shares the weights,
the session and the expert cache. `--dump-layers` could not be used: it is the *session* path's
ladder, and that path refuses a native (IQ) pack outright (`src/core/expert_source.cpp:1890`), which
is what the box's target is.

**The verdict: do not start the port.** Run the reference-capture comparison first (it is now cheap),
and the port only if the capture is vindicated *and* the level reaches 2.67.

