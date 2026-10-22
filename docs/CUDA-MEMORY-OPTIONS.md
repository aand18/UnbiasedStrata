# CUDA memory options for the adaptive tier's churn

Written 2026-10-22 on the `spectre-moe` branch, for B3 of `docs/plans/specter-performance.md` (§3 item 4).
The question: **the adaptive tier's round is bound by its bytes - what does CUDA's own memory
management offer that beats "move fewer bytes"?**

Every claim about CUDA below is from a primary source (§7). Every claim about this engine is
`file:line`.

## How to read the numbers in this document

| term | what it means | where it comes from |
| --- | --- | --- |
| **the window** | one verification step: the engine verifies T tokens, then commits the accepted prefix. 18.6-19.0 ms, 2.47 committed tokens. | the log line `strata decode timing:` (`src/program/generate.cpp:7519-7527`) |
| **the round** | one adaptive-tier round: the ranking, the copies, the bookkeeping. It runs on its own thread beside the commit and the draft (`src/program/generate.cpp:7447-7450`), once every `adapt_every` = 4 windows (`src/program/generate.cpp:508`, `:7449`). | `src/program/generate.cpp:8244-8320` (generate), `:5713-5819` (serve) |
| **a swap** | one expert admitted into a VRAM slot, one evicted from it. 2.66 MB is the layer's blob size (`expert_layout().blob_bytes(layer)`). | `src/program/generate.cpp:256-258`, `:5771-5773`; the 2.66 MB largest blob: `docs/SPECTER-FEASIBILITY.md` §0 |
| **the churn tax** | the round's own cost, per window: the adapt thread's work + `apply_pending`. **3.86 ms/window** at the baseline, **4.75** with the forecast on. | `docs/plans/specter-performance.md` §2.7 item 3 |
| **the quiet regime** | `--resident-budget-gib 32`: the misses come from the RAM copy, so the instrument is not dominated by the disk. 130.5 tok/s, 18.9 ms/window, 92.3% hit rate. | the harness's `ab-churn.ps1`; the floors are hard (2,048 MB available RAM, 200 MB free VRAM) |
| **the noise floor** | 0.65 ms/window (4.60 tok/s, 3.5%). Nothing below it is a difference. | `docs/plans/specter-performance.md` §2.7 |

The measured churn data this document works from is the harness's own output, per request, in the
quiet regime: `logs\ab-churn3.csv` (the shipped build) and `logs\ab-churn.csv` (the chunked-staging
build that was reverted). Both are in `C:\Users\yoho\Downloads\UnbiasedStrata\logs`.

---

## 0. The target, restated as traffic

The plan's summary of the problem (§2.7 item 3) is one sentence: *the round is bound by its bytes,
and no schedule of the copies removes that.* This section says what those bytes are, per swap, so
that each option below can be priced against them.

### 0.1 One swap

| step | bytes | where | measured |
| --- | ---: | --- | --- |
| D2H: the victim's bytes out of the VRAM slot into a pinned exchange buffer | 2.66 MB | `src/program/generate.cpp:256-258` | 1.23 ms/window = 81 us/swap (33 GB/s) |
| host memcpy: the exchange buffer into the RAM copy, over the victim's old slot | 2.66 MB | `src/core/expert_source.cpp:1811` | 1.04 ms/window = 69 us/swap (38.6 GB/s) |
| H2D: the new expert out of the RAM copy into the slot | 2.66 MB | `src/program/generate.cpp:5771-5773` | 1.55 ms/window = 103 us/swap (26 GB/s) |
| **total per swap** | **7.98 MB** | | **0.24 ms of main-thread time per swap** (the join's 168 us + the memcpy's 69 us) |

Per round (60.5 swaps): **160 MB over the link each way** (5.32 MB of PCIe per swap) and 160 MB of
host memcpy. Per window (15.1 swaps): 40.2 MB each way and 40.2 MB of host memcpy.

This is the plan's accounting ("a swap copies the victim's 5.32 MB back into RAM and the new
expert's 2.66 MB into the slot", §2.7 item 3): the victim's bytes touch RAM **twice** - once over
the link into the exchange buffer, once as a host memcpy into the complement - and the plan's
"5.32 MB" is those two together. The link sees 5.32 MB per swap; the host sees 2.66 MB more.

### 0.2 Where the 3.86 ms/window actually sits

From `logs\ab-churn3.csv` (baseline arm, three runs; 16 rounds, 968 swaps, 64 windows per request):

| term | ms/request | ms/round | ms/window | whose time |
| --- | ---: | ---: | ---: | --- |
| `rank` (the candidate ranking) | 1.9 | 0.12 | 0.03 | the adapt thread |
| `stage` (the D2H copies + their `cudaStreamSynchronize`) | 78.7 | 4.92 | **1.23** | the adapt thread, blocked on the link |
| `submit` (the pageable H2D copies) | 99.4 | 6.21 | **1.55** | the adapt thread, blocked in the driver's staging |
| `work` (the round's total) | 180.2 | 11.26 | 2.82 | the adapt thread |
| `join` (the main thread's wait at `adapt_thr.join()`) | 162.0 | 10.13 | **2.53** | **the main thread** (`src/program/generate.cpp:7488-7493`) |
| `apply_pending` rest (the exchange's host memcpy) | 66.7 | 4.17 | **1.04** | **the main thread** (`src/core/expert_source.cpp:1811`) |
| `apply_pending` wait (the copies' landing) | 0.06 (per request) | - | 0.001 | the main thread |
| **the churn tax** (work + apply_pending) | | | **3.86** | matches the plan exactly |

So the tier's cost to the window is **3.57 ms/window of main-thread time** (the join 2.53 + the
memcpy 1.04) - **19% of the 18.9 ms window** - and only 0.001 ms/window of it (0.06 ms per request)
is the copies failing to land. The rest is the main thread *waiting* for the adapt thread, which is
waiting for the link.

Two accountings of the same thing, both useful: the plan's **churn tax** (3.86) counts the adapt
thread's *work* plus `apply_pending`; the **3.57** counts what the *main thread* actually pays (its
wait at the join, plus the exchange's memcpy, which `apply_pending` runs on the main thread). They
differ by the ~0.29 ms/window the main thread's own commit and draft cover while the adapt thread
runs - and the second is the one an optimisation has to move.

That is the shape the options below have to attack: **the tax is not the landing, it is the waiting,
and the waiting is the bytes.**

### 0.3 The two failures this document must not repeat

`docs/plans/specter-performance.md` §2.7 records both:

1. **A shorter memory for the ranking** (the forecast's planes): the ranking change *works* - the
   tier's misses fall 32%, the pool's time 30%, the hit rate +2.1 points - and the tok/s does not
   move, because the round's cost rose with it. The flag stays off.
2. **The chunked staging of the copies** (the next chunk's D2H issued before the current chunk's
   H2D): implemented, measured, **reverted**. The round stayed 11.2 ms and the window 18.9. The
   harness's own two runs show why, and they are the cleanest statement of the problem:

| build | `stage` ms/round | `submit` ms/round | stage + submit | window ms |
| --- | ---: | ---: | ---: | ---: |
| shipped (`ab-churn3.csv`) | 4.92 | 6.21 | 11.13 | 18.72 |
| chunked (`ab-churn.csv`) | 0.76 | 10.31 | 11.07 | 18.89 |

The chunked build moved the waiting from one term to the other and bought **nothing**: the round is
the same 11 ms and the window is the same 18.9. **A schedule cannot change bytes on a link that
does not add its two directions.**

---

## 1. What the engine's memory looks like today

The whole map, so that each option can be priced against the allocation it would change.

| what | where | size | page-locked? | lifetime |
| --- | --- | ---: | --- | --- |
| the VRAM slot arena | `src/core/expert_cache.cpp:440` (one `cudaMalloc`), or `:235-286` (VMM segments, `--vram-elastic`) | 13.4-16 GiB | n/a (device) | the process |
| the RAM copy of the experts the VRAM tier does not hold ("the complement") | `src/core/expert_source.cpp:1498-1591` | **30.89 GiB measured** in the quiet regime (`--resident-budget-gib 32`) | **no** - `cudaHostAlloc` was refused ("page-locking refused (out of memory)"), so it is pageable + a working-set lock; a *registered prefix* under `STRATA_PARTIAL_PIN` is the one pinned thing available | the process |
| the exchange buffers (the D2H's destination) | `src/core/expert_source.cpp:1757-1784` | 96 x 2.66 MB = 255 MB | **yes** (`cudaHostAlloc`) | the process |
| the mapped experts (the file tier's fallback) | `src/core/expert_source.cpp:1836-1845` | the GGUF/pack, ~50 GB | no (mmap) | the process |
| the MTP / verify / session arenas | `src/core/mtp.cpp:177-443`, `src/core/verify.cpp:442-521`, `src/core/session.cpp:428` | hundreds of MiB | mixed | the process |
| the pinned KV | the front end's `--kv` | 0.4 GiB (int8/32K) | yes | the request |

Three facts follow, and they decide most of this document:

1. **The D2H is already a real DMA.** Its destination is the pinned exchange buffer
   (`src/core/expert_source.cpp:1772`), so nothing about the device-to-host direction can be made
   faster - only smaller.
2. **The H2D is the only copy the driver has to stage.** Its source is the complement, which is
   pageable. The CUDA Programming Guide is explicit that an async copy over non-page-locked host
   memory may be synchronous with respect to the host (§7.1: *"Async memory copies might also be
   synchronous if they involve host memory that is not page-locked"*).
3. **There are no per-window device allocations.** Every `cudaMalloc` in the engine is at init
   (`src/core/expert_cache.cpp:440`, `src/core/mtp.cpp:177-443`, `src/core/verify.cpp:442-521`,
   `src/core/session.cpp:428`, `src/core/expert_source.cpp:1503`). The decode path allocates
   nothing on the device. This kills the stream-ordered allocator as a lever on this tax (§2.3).

---

## 2. The options, priced

Each option: what it changes (`file:line`), what it would cost to build, what it would save (from
§0.2's arithmetic), how it would be measured, and the verdict.

### 2.1 Managed memory (`cudaMallocManaged`) - **ruled out on this platform**

**Would the victim's bytes need to be copied back to RAM at all?** No - and that is exactly the
problem. The victim's bytes must be *in host RAM* for the CPU pool to compute the expert at 44 us
(`docs/SPECTER-FEASIBILITY.md` §0). Under managed memory they would be wherever the driver last put
them, and the CPU's next read of them would fault them back. The copy is not removed; it moves from
a DMA into a fault, and it moves from a place the engine controls (the exchange buffer) into a place
it does not (the UM driver's page placement).

**The platform says no, three times over.** The CUDA 13.4 Programming Guide, §4.1.3 *"Unified Memory
on Windows, WSL, and Tegra"* - a section that applies to *"devices with concurrentManagedAccess
property set to 0"* - states for Windows:

- *Data Migration and Coherency:* *"Fine-grained movement of the managed data to GPU on-demand is
  not supported. Whenever a GPU kernel is launched all managed memory generally has to be
  transferred to GPU memory to avoid faulting on memory access. Page faulting is only supported from
  the CPU side."*
- *GPU Memory Oversubscription:* *"They cannot allocate more managed memory than the physical size
  of GPU memory."* The complement is 30.89 GiB; the card is 24 GB.
- *Coherency and Concurrency:* *"Simultaneous access to managed memory is not possible, because
  coherence could not be guaranteed if the CPU accessed a unified memory allocation while a GPU
  kernel is active."*

The last one is the engine's exact pattern: the CPU pool reads the arena's bytes
(`src/core/expert_source.cpp:1826-1835` returns a pointer into the complement) **while the GPU's
kernels run over the same arena**. On Windows that is not a slow path, it is an illegal one.

**Cost:** the whole complement's allocation model changes, and the CPU pool's hot loop becomes a
faulting loop. **Saving:** none that survives the platform's rules.

**Verdict: not a candidate.** It is worth one runtime check - `cudaDevAttrConcurrentManagedAccess`
on this box - because if it ever reads 1 (a future driver, or a Linux box), the option becomes
*possible*; it is still not *useful*, because the copy it would remove is the copy the CPU needs.

### 2.2 Pinning the complement (`cudaHostAlloc` / `cudaHostRegister`) - **the cheapest experiment**

**The state of the complement today - measured, in the engine's own log.** `--resident-budget-gib`
allocates the complement pageable (`src/program/generate.cpp:1682-1692`; `STRATA_RESIDENT_PIN=0` makes
it pageable explicitly), and `pin_cache_complement` tries
`cudaHostAlloc(cudaHostAllocMapped|cudaHostAllocPortable)` first
(`src/core/expert_source.cpp:1502-1524`). In the quiet regime that call **fails**, and the engine
says so (`logs\abchurn-baseline-*.err.log`):

> `FileExpertSource: RAM budget 32.00 GiB: 16168 of the 30.89 GiB of experts the GPU cache does not
> hold, by profile rank; the rest are read from the files`
>
> `FileExpertSource: locked resident cache complement ready: resident 30.89 GiB, pinned 0.00 GiB;
> **page-locking refused (out of memory)**; locked 8192 MiB via working-set minimum + VirtualLock
> (capped by STRATA_ARENA_LOCK_GIB=8)`

So: the complement is **30.89 GiB**, **0.00 GiB of it page-locked**, and it is ordinary pageable
memory plus a working-set lock (`src/core/expert_source.cpp:1526-1589`,
`src/platform/memory.cpp:19-80`). **That is why the H2D's submission blocks** - its source is
pageable, so the driver stages every copy. The refusal is exactly what the engine's own note
predicts for this box (`src/core/pinned.cu:286-291`, issue #243):

> *Page-locked memory the GPU maps is charged to its shared (non-local) WDDM segment; pinned slice
> by slice until the driver refused one (28 GiB of a 63 GB PC), that segment was left full and every
> later cudaMalloc failed "out of memory".*

**What pinning would change.** With a page-locked source, `cudaMemcpyAsync` is asynchronous with
respect to the host (§7.1) and the copy engine does the transfer; the adapt thread's `submit` term
(1.55 ms/window) collapses to the call overhead, and the 6.21 ms/round of DMA moves off the main
thread's path - it runs on `adapt_stream` while the next window's GPU work runs.

**The arithmetic** (from §0.2). The join is the main thread's wait for the adapt thread, whose
per-round work is 11.26 ms; pinning removes the H2D's 6.21 ms of *blocked* submission from that
work, so the join falls:

| the main thread's cost | now | with a pinned complement | saving |
| --- | ---: | ---: | ---: |
| the join (the wait for the adapt thread) | 2.53 ms/window | 0.95-1.55 | **0.98-1.58** |
| the exchange's host memcpy | 1.04 ms/window | 1.04 (unchanged, see below) | 0 |
| **total on the window** | **3.57** | **2.00-2.60** | **0.98-1.58 ms/window = 5-8% of the window** |

(The adapt thread's own terms are the *cause*, not additions: the `submit` term, 1.55 ms/window,
is inside the join. A pinned source turns it into a queued DMA - the adapt thread returns
immediately and the 6.21 ms/round of transfer runs on `adapt_stream` while the next window's GPU
work runs.)

The join's saving is a range, not a number, and the range is honest: the join cannot fall below the
D2H's own wait (1.23 ms/window, less the ~0.28 ms/window the main thread's commit and draft already
cover), so the ceiling is 1.58 ms/window; and the submission's blocked time was not entirely on the
main thread's critical path, so the floor is 0.98. The trace's `submit_ms` and `pw_join` say which
end it lands on - that is the measurement.

The host memcpy cannot go away by pinning: the victim's bytes have to land somewhere that is not
the incoming expert's RAM slot, because the CPU is still computing the incoming expert from that
slot while the swap is in flight (the exchange contract, `include/strata/core/expert_source.hpp:
479-496`). Only §2.6 removes it.

**What it would cost to build: nothing.** The engine already has the mechanism and the knobs:
`STRATA_PARTIAL_PIN=1` registers the complement's hottest prefix
(`src/core/expert_source.cpp:1541-1550`, the registration at `:1563-1571`), and the complement's
layout is in profile-rank order (`:1415-1428`) - the hottest experts the tier does not hold, which
is where the swap's incoming experts come from - so the registered prefix covers the copies that
matter. The knobs are env vars, so the harness needs one line (its env is set once at the top of
`ab-churn.ps1`; the arms differ only by command-line flags), or the engine grows a
`--partial-pin-gib N` flag (5 lines in `src/program/generate.cpp`).

**The risks, both measured in this repo:**

- **The WDDM shared segment - already measured, on this box.** The whole-complement
  `cudaHostAlloc` is *refused* ("page-locking refused (out of memory)", the log line above), and
  the engine's own note says why: page-locked memory the GPU can map is charged to the GPU's
  non-local segment, and above ~28 GiB on a 63 GB PC the driver refuses and later `cudaMalloc`s
  fail with "out of memory" (`src/core/pinned.cu:286-291`). So **the whole complement cannot be
  pinned here - it is already not**; a *prefix* can be (the arena's 4 GiB registration succeeds
  today, `STRATA_ARENA_PIN_GIB=4`, and the partial pin registers in 2 GiB steps down from its cap,
  `src/core/expert_source.cpp:1551-1577`). The engine's startup report prints the free VRAM
  (`src/program/generate.cpp:5900-5921`) and the harness's watchdog watches the floors.
- **The engine's own negative result on another box** (`src/core/expert_source.cpp:1538-1540`):
  with 24 / 16 GiB registered against 7.05-7.74 tok/s unpinned at a 40 GiB budget, the GPU's PCIe
  share of the misses measured no faster (7.30 / 7.44 tok/s) - *"registering adds startup time and
  driver memory pressure"*. Different measurement (the PCIe share, not the swap's H2D), different
  box (an RTX 5070 PC), but it is the engine's own experience of registering this complement.

**How it would be measured.** `ab-churn.ps1` already parses the `strata adapt cost:` line
(`src/program/generate.cpp:7536-7544`) into `stage_ms`, `submit_ms`, `join_ms`, `pw_work`,
`pw_join`, `pw_pend` - the instrument is built. The arms: the quiet regime
(`--low-ram on --ple-io direct --ctx 32768 --kv int8 --resident-budget-gib 32`, `STRATA_TRACE_ADAPT=1`,
`STRATA_ARENA_LOCK_GIB=8`, `STRATA_ARENA_PIN_GIB=4`), with and without
`STRATA_PARTIAL_PIN=1 STRATA_PARTIAL_PIN_GIB=8`, three runs each, the floors enforced by the
watchdog.

**The decision rule** (this is the measurement that decides the whole family):

| observation | conclusion |
| --- | --- |
| `submit` falls from 6.21 ms/round to < 0.5 ms/round **and** `pw_join` falls from 2.53 to under ~1.5 ms/window **and** the window falls by >= 1.0 ms (above the 0.65 ms noise floor) | the pageable staging is real and worth ~1-1.6 ms/window; the next question is how much prefix the WDDM segment allows |
| `submit` does not fall | the driver's staging was already hidden behind the DMA; **every CUDA option that only changes *who waits* is dead**, and only "move fewer bytes" (§2.6) survives |
| the run dies with "out of memory" on a later `cudaMalloc` | the WDDM segment is the binding constraint; the prefix must be smaller, and the option's ceiling is the segment, not the mechanism |

**The A/B as run (2026-10-22): no signal, and the rule's own columns were not captured.** `ab-pin.ps1`
ran the two arms three times each at a **24 GiB resident budget** (pageable: 76.8, one run NOT READY
because the port was still bound from the previous run, 67.6 -> mean 72.2; pinned8: 72.3, 73.1, 76.5 ->
mean 74.0). The difference, **+1.8 tok/s (+2.5%), sits inside the pageable arm's own 9.2 tok/s spread**,
so the tok/s column says nothing - and it could not have said much: the rule asks for a **>= 1.0
ms/window** fall, which is ~5% of the window, and a tok/s A/B cannot read a change worth under ~2% of
the window (the tier's contents shift the drafter's window). The rule's mechanism columns
(`submit_ms`, `join_ms`) came back **empty**: the script set `STRATA_DECODE_TIMING` but not
`STRATA_TRACE_ADAPT=1`, so the adapt round's split was never printed. So the first two rows of the
decision table are still unanswered, and the run was in the **wrong regime** for them: at a 24 GiB
budget the misses come off the *file* (the disk-fed regime, 67-77 tok/s), while the rule's quiet regime
is `--resident-budget-gib 32` (33.55 GiB of experts locked in RAM, no file reads) - which is the regime
where the pageable complement's H2D copy is what the window waits on.
**Next**: re-run `ab-pin.ps1` with `STRATA_TRACE_ADAPT=1` and `--resident-budget-gib 32`, and read the
adapt round's split from the CSV the script already parses.

**Verdict: try this first.** It costs no code, the instrument exists, and its answer decides
whether the whole "make the copies asynchronous" family is worth anything on this box.

### 2.3 Stream-ordered allocation (`cudaMallocAsync`, `cudaMemPool_t`) - **no effect on this tax**

**What the engine's allocation pattern is.** There is none on the decode path. Every device
allocation is at init: the slot arena (`src/core/expert_cache.cpp:440`), the MTP's arenas
(`src/core/mtp.cpp:177-443`), the verify's arenas (`src/core/verify.cpp:442`, `:467`, `:519`,
`:521`, `:1766`), the remote-expert buffers (`src/core/remote_experts.cpp:197-205`). The one
per-call device allocation in the tree is in the layer-split path, `session_replay_stage_prefixes`,
which mallocs a residual buffer and creates 288 events per call
(`src/core/session.cpp:427-440`) - and it is not on the decode path. The per-window buffers the
plan mentions are the *host* staging buffers (the pool's `cudaHostAlloc`s,
`src/core/session.cpp:531`), which the stream-ordered allocator does not manage.

**What the pools would buy.** The allocator's documented value is avoiding the OS calls and the
device-wide synchronisation of `cudaMalloc`/`cudaFree`, with a release threshold and an explicit
trim (§7.3). The engine makes a handful of large allocations, once. There is no fragmentation to
fix and no per-window churn to amortise.

**Where it might still matter (a different problem).** The engine's VRAM budget under WDDM is
delicate: *"under WDDM a GPU filled to the brim does not fail, it pages - and a page-in while the
verify graph spins on a host flag stalls the request for good"*
(`src/program/generate.cpp:5900-5906`). A pool with `cudaMemPoolAttrReleaseThreshold = UINT64_MAX`
and an explicit `cudaMemPoolTrimTo` at the right moment would give the driver a predictable
footprint instead of the allocator's default (release everything at every synchronisation). That is
a **startup/VRAM-reserve** question, not the churn tax, and it is not what this document is about.

**Verdict: no effect on the 3.86 ms/window.** Worth ~0, and it is a half-day of work to prove.

### 2.4 Virtual memory management (`cuMemCreate` / `cuMemMap` / `cuMemAddressReserve`) - **ruled out by the engine's own measurement**

**What the engine already does.** The elastic tier (`--vram-elastic`) reserves one address range and
backs it with physical segments: `cuMemAddressReserve` + `cuMemCreate` with
`CU_MEM_ALLOCATION_TYPE_PINNED` + `CU_MEM_LOCATION_TYPE_DEVICE`, `cuMemMap`, `cuMemSetAccess`
(`src/core/expert_cache.cpp:154-232`, `:235-286`), and `shrink`/`grow` unmap and re-map the tail
(`:322-385`). The engine can already ask the driver for physical memory, map it, unmap it and
release it.

**Could the host copy get a device alias?** It already can, and does, without VMM: the pinned
complement's device alias comes from `cudaHostGetDevicePointer`
(`src/core/expert_source.cpp:1506-1511`), and the partial pin's from the same call after
`cudaHostRegister` (`:1563-1571`). The VMM spelling of the same thing is
`cuMemCreate` with `CU_MEM_LOCATION_TYPE_HOST` (§7.4), which is a driver-allocated page-locked host
buffer that can be mapped into a device address range. It is the same capability with more code:
**the VMM route adds nothing to §2.2, it only re-implements it.**

**Could a slot's pages be remapped instead of copied?** Mechanically, yes: unmap the slot's
backing and map a sub-range of a host allocation at the same virtual address (§7.4). Three things
stop it:

1. **Granularity and structure.** The arena is one `cudaMalloc` by default
   (`src/core/expert_cache.cpp:440`); the VMM path segments it at `seg_bytes`
   (`include/strata/core/expert_cache.hpp:105-112`). A slot-granular remap means 2.66 MB segments
   (the driver's minimum granularity is queried at `src/core/expert_cache.cpp:253`), ~5,000 of them,
   each with a `cuMemCreate`/`cuMemMap` pair.
2. **The bytes are not saved, they are deferred.** A remapped expert is read over PCIe *every time
   it is used*. The engine has measured that trade exactly: `t_bus` = 99 us per offloaded expert
   (2.66 MB / 26.8 GB/s) against `t_host` = 44 us for the CPU to compute it, and the GPU's expert
   kernel runs at 159 GB/s (`docs/SPECTER-FEASIBILITY.md` §0, §2.2). A copy costs 99 us once and
   then 17 us per use from VRAM; a remap costs 99 us per use. For an expert used more than once
   while resident - the normal case, since the tier exists to hold the *hot* experts - the remap is
   strictly worse.
3. **The link does not add its directions** (measured, §2.7). Moving the bytes from the swap to the
   decode does not create bandwidth.

**Verdict: ruled out.** The VMM API's real use in this engine is the elastic tier's
shrink/grow, which it already has. A "remap instead of copy" scheme trades a one-time copy for a
per-use PCIe read, and the engine's `t_bus`/`t_host` measurement says that trade loses.

### 2.5 Batch and 3D copies (`cudaMemcpyBatchAsync`, `cudaMemcpy3DAsync`) - **not worth a build**

`cudaMemcpyBatchAsync` exists in CUDA 13.0+ (§7.2); the box has **CUDA 13.3**
(`logs\build-final-clean.log`: *"CUDA 13.3 at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"*),
so it is available. Its contract: *"Performs a batch of memory copies asynchronously. The batch as a
whole executes in stream order but copies within a batch are not guaranteed to execute in any
specific order"* (§7.2). One detail is interesting for this engine: the source-access-order
attribute `cudaMemcpySrcAccessOrderAny` is *"suited for host pointers allocated outside CUDA (ex.,
via malloc) ... allows the driver to optimize the copy on certain platforms"* (§7.2) - the
complement's source is exactly such a pointer.

**What it can remove:** the per-call submission overhead (60 calls become 1), and possibly a
different staging path on some platforms. **What it cannot remove:** the bytes and the link time.
The `submit` term is 6.21 ms/round for 160 MB - that is the *duration* of the copies, not the cost
of making 60 API calls. The copies stay on `adapt_stream`, behind the same D2H, over the same link
that does not add its two directions.

**Estimate: <= 0.05 ms/window**, i.e. below the noise floor, and it does not change the blocking
(the pageable source is still staged).

`cudaMemcpy3DAsync` is for strided/pitched copies; the engine's copies are contiguous blobs
(`src/program/generate.cpp:256-258`, `:5771-5773`). No gain.

**Verdict: a curiosity.** It becomes worth revisiting only *after* §2.2 has shown that the staging
path is worth attacking - and then only as a way to shave the submission overhead, which is not the
problem.

### 2.6 The lever that is not a CUDA feature: **release the victim to the file tier**

The plan's own next lever (§2.7 item 3): *"A swap copies the victim's 5.32 MB back into RAM and the
new expert's 2.66 MB into the slot. Releasing the victim to the file tier instead would halve the
round's bytes (5.32 -> 2.66 MB) - at the price of the misses coming off the disk."*

**What it changes.** In `resident_stage_swaps` (`src/program/generate.cpp:241-271`) the D2H and its
`cudaStreamSynchronize` are what copy the victim's bytes back; in `commit_exchanges`
(`src/core/expert_source.cpp:1804-1819`) the host memcpy is what parks them in the complement, and
`exchange_cache_complement` (`:173-179`) is what swaps the offsets table so that the copy keeps
holding exactly the experts the GPU does not. Releasing the victim instead means: no D2H, no
exchange buffer, no memcpy - the victim's offsets entry is dropped (the same operation `release()`
already does, `src/core/expert_source.cpp:1736-1755`) and the incoming expert's RAM copy is
released as redundant (it is in VRAM now; `apply_pending` already does exactly that,
`src/program/generate.cpp:5697`). The victim becomes a file-tier expert, and the file tier's
fallback chain already exists and is already counted (`src/core/expert_source.cpp:1836-1845`; the
counters at `include/strata/core/expert_source.hpp:498-508`: `file_reads`, `file_read_bytes`,
`file_ms`).

**The build is small:** skip the D2H and the exchange staging for a victim (one branch in
`resident_stage_swaps`), a `commit_exchanges` variant that drops the victim's offsets entry instead
of copying into it, and a flag. ~60-100 lines in three places, and the file tier's read path is
already there.

**What it saves** (from §0.2). The D2H's wait and the exchange's memcpy both leave the main
thread's path; the H2D's blocked submission stays:

| the main thread's cost | now | after | saving |
| --- | ---: | ---: | ---: |
| the join (the wait for the adapt thread) | 2.53 ms/window | ~1.30 (the D2H's wait is gone; the H2D's blocked submission remains) | 1.23 |
| the exchange's host memcpy | 1.04 ms/window | 0 | 1.04 |
| **total on the window** | **3.57** | **1.30** | **2.27 ms/window = 12% of the window** |

(The bytes themselves: the swap goes from 7.98 MB of traffic - 5.32 MB over the link and 2.66 MB of
host memcpy - to 2.66 MB over the link and no memcpy. The plan's *"5.32 -> 2.66 MB"* is the same
statement counted over the link.)

It is also the only option that **changes the sign of the forecast's A/B**. The forecast's marginal
swap currently costs 240 us to save 276 ± 19 us of pool time - break-even (§2.7 item 3). Releasing
the victim removes 5.32 MB of the swap's 7.98 MB, so the marginal swap's cost drops toward ~120 us
and the forecast's +0.89 ms/window of extra tax disappears while its 1.08 ms/window of pool saving
stays. On the forecast arm the removed traffic is larger (75.3 swaps/round instead of 60.5), so the
forecast's tax would fall from 4.75 to ~2.0 ms/window.

**What it costs, and the honest gap.** The victim's bytes are no longer in RAM, so the next time
that expert is routed it comes off the disk instead of the RAM copy. The engine knows this: the
VRAM-shrink path already prints *"N of M dropped experts have no RAM copy: file reads when they are
routed again"* (`src/program/generate.cpp:6162-6165`). And the counter that would show the bill
starts at **zero**: in the quiet regime the file tier reads *nothing* during the decode - the
engine's own report says *"since the start RAM 48457 blobs, **files 0 blobs**"*
(`src/program/generate.cpp:7854-7859`, and the log line) - because every miss comes from the RAM
copy. **The size of the bill is not known and cannot be estimated from what is measured:**

- the victims are chosen as the *least-routed* resident experts (the victims' ranking is ascending
  by usage, `src/program/generate.cpp:5740-5746`), so they are cold by construction - but "cold" is
  relative to a decayed, thresholded count;
- the file tier's read cost per blob is not in any of the measured tables: it is a page fault on the
  mmap (`src/core/expert_source.cpp:1316-1329`), which the page cache may satisfy at ~10 GB/s
  (266 us for 2.66 MB) or the SSD may not (the box's drive speed is not in this document's data);
- the re-route rate of a victim is exactly the quantity that decides it.

**How it would be measured.** The same `ab-churn.ps1` harness, one new flag, plus the file tier's
own counters, which the engine already prints per request (`src/program/generate.cpp:8607-8614`:
*"decode: RAM N blobs, files N blobs, X MB read from the files (Y MB/round, Z ms/round of reading,
W GB/s per reading thread)"* - the per-round reading time and the per-reading-thread rate are
exactly the bill). The A/B answers two questions at once: the window and the tax (the harness's
columns), and the bill (the file tier's report). The break-even is arithmetic once the counters are
in: the swap saves 150 us of traffic; a file re-read costs 266 us (page cache) to ~1 ms (SSD); so the
lever pays if the victims' re-route rate is below ~0.5 (page cache) or ~0.15 (SSD).

**Verdict: the biggest lever on this list, and the one whose bill is unmeasured.** It is not a CUDA
feature, and it does not need to be: it changes the tier's *policy*, which is where the bytes come
from.

### 2.7 The discard APIs - **nothing here**

The CUDA 13.4 runtime API's memory group lists `cudaMemDiscardBatchAsync` - *"Performs a batch of
memory discards asynchronously"* (§7.2) - alongside `cudaMemPrefetchBatchAsync` and
`cudaMemDiscardAndPrefetchBatchAsync`. These are the one CUDA mechanism whose *purpose* is to
avoid a write-back: the driver is told that a region's contents may be dropped, so it need not
migrate them anywhere first. (The one-line summary is all I could find in the pages fetched; the
detailed description was not in the group page I read, so treat the scope below as my reading of
the mechanism, not as a quote. I did not locate a singular `cuMemDiscardAsync` in the driver
API's memory, virtual-memory or execution groups.)

Either way it does not apply here: the discard applies to memory the driver is allowed to drop -
managed memory and pool memory - and the complement is ordinary pageable host RAM.

Mentioned only so that "is there a CUDA API that skips the write-back?" has an answer: there is
one, and it does not apply here.

---

## 3. The ranking

By (estimated saving) / (build cost), with the certainty stated separately - the two are not the
same thing here.

| # | option | saving (ms/window of an 18.9 ms window) | build cost | certainty | blocked by |
| --- | --- | ---: | --- | --- | --- |
| 1 | **Pin the complement's prefix** (`cudaHostRegister`, `STRATA_PARTIAL_PIN`) | **0.98-1.58** (5-8%) | **0 lines** (the knobs exist) | high - arithmetic from the trace | the WDDM shared segment above ~28 GiB (`src/core/pinned.cu:286-291`) |
| 2 | **Release the victim to the file tier** | **2.27** (12%) | ~60-100 lines, 1 day | medium - the file re-read bill is unmeasured | the file tier's read cost and the victims' re-route rate |
| 3 | Pin the *whole* complement | the same (the memcpy cannot go) | 0 lines | - | the WDDM segment: the whole-complement pin is **already refused** (measured, §2.2) |
| 4 | `cudaMemcpyBatchAsync` / `cudaMemcpy3DAsync` | <= 0.05 | ~30 lines | medium | nothing - it simply cannot pay |
| 5 | `cudaMallocAsync` / memory pools | **0** on this tax | ~50 lines | high - there are no per-window device allocations | nothing - it is the wrong problem |
| 6 | VMM: a host-backed mapping, or a remap instead of a copy | **0, or negative** | 2-3 days | high - `t_bus` 99 us vs `t_host` 44 us (`docs/SPECTER-FEASIBILITY.md` §0) | the engine's own measurement |
| 7 | Managed memory (`cudaMallocManaged`) | **n/a** | large | high - the guide's Windows rules (§7.1) | the platform |

Read the table as: **one option is free and decisive (row 1), one is worth 12% and costs a day
(row 2), and the five CUDA-specific mechanisms are worth nothing on this tax.** That is the honest
result, and it is the same shape as the engine's last two attempts: the tax is not a scheduling
problem, and it is not an allocator problem. It is a *bytes* problem.

**What the CUDA mechanisms would have to fix, and why they cannot.** Every one of them changes
*where the bytes live* or *who waits for them*, not *how many bytes there are*:

| mechanism | what it changes | why the bytes stay |
| --- | --- | --- |
| managed memory | where the bytes live | the CPU still needs the victim's bytes in RAM; they migrate back on the next read (and on Windows they may not migrate at all) |
| pinning | who waits for the copy | the same 2.66 MB crosses the link; only the driver's staging and the blocked submission go |
| `cudaMallocAsync` | where the allocator's pages come from | the decode path allocates no device memory |
| VMM | which physical pages back a slot | the same bytes are read, later and over PCIe per use |
| batch copies | how many API calls the round makes | the same bytes cross the same link |

### 3.1 The one to try first, and the measurement that would decide it

**Try the partial pin first** (§2.2). It costs no code, the instrument is already in the harness,
and its answer decides the whole family:

> **Run `ab-churn.ps1` in the quiet regime with and without `STRATA_PARTIAL_PIN=1
> STRATA_PARTIAL_PIN_GIB=8`, three runs each, the floors enforced.**
>
> **If `submit_ms` per round falls from 6.21 ms to under ~0.5 ms and `pw_join` per window falls from
> 2.53 to under ~1.5 ms and the window falls by at least 1.0 ms (the noise floor is 0.65), then the
> pageable staging is real and worth ~1-1.6 ms/window, and the next question is how large a prefix
> the WDDM segment tolerates.**
>
> **If `submit_ms` does not fall, the driver's staging was already hidden behind the DMA; then every
> CUDA option in this document that only changes *who waits* is dead, the pinning line is closed,
> and the only lever left is §2.6 - move fewer bytes - which is a policy change, not a CUDA
> feature.**

The measurement that would decide §2.6 is the same harness: one flag, and the file tier's own
counters (`file_reads`, `file_read_bytes`, `file_ms`) in the request's report. The number to look
for is the victims' re-route rate; the arithmetic is in §2.6.

### 3.2 What is ruled out, and why

- **Managed memory** - the platform forbids the engine's access pattern (§2.1; Programming Guide
  §4.1.3: no on-demand migration to the GPU, no oversubscription, no concurrent CPU/GPU access).
- **A VMM remap instead of a copy** - it defers the bytes to every use of the expert and the engine
  has measured that per-use PCIe (99 us) is worse than CPU compute (44 us) and much worse than a
  VRAM read (§2.4).
- **`cudaMallocAsync` / pools** - the engine makes no device allocations on the decode path
  (§2.3); the allocator's documented value is against allocations that do not exist here.
- **Batch copies** - the `submit` term is the copies' duration, not the API calls' overhead; the
  ceiling is ~0.05 ms/window, below the noise floor (§2.5).
- **Any schedule of the copies** - measured and reverted: the chunked staging moved the waiting
  from one term to the other and bought nothing (§0.3).

---

## 4. What this costs the engine, in one paragraph

Nothing, if the partial pin is the answer: the mechanism is already in the tree
(`src/core/expert_source.cpp:1541-1571`), the knobs exist, and the harness parses the counters. If
the partial pin shows nothing, the CUDA family is closed and the engine's next move is a policy
change - release the victim instead of copying it back (`src/program/generate.cpp:241-271` and
`src/core/expert_source.cpp:1804-1819`), worth an estimated 2.27 ms/window (12%) at a build cost of
about a day and a bill - the file tier's re-reads, which today are zero - that the same harness
measures. Either way the engine's own experience holds: the adaptive tier's churn is a *bytes*
problem, and the last two attempts to fix it by rearranging them cost more than they saved.

---

## 5. Open questions this document cannot close

1. **How much prefix can this box's WDDM segment hold?** The whole-complement registration is
   already refused (the log line in §2.2), and the repo's own measurement puts the refusal at
   ~28 GiB on a 63 GB PC (`src/core/pinned.cu:286-291`), but the complement's prefix would compete
   with the arena's 4 GiB pin and the KV's. Cheap to find: sweep `STRATA_PARTIAL_PIN_GIB` and
   watch the engine's own line (*"N.NN GiB of it registered for the GPU (the hottest)"*,
   `src/core/expert_source.cpp:1578-1581`), the free VRAM, and the floor.
2. **What does the file tier cost per blob on this box?** Not in any measured table; the engine's
   counters (`file_ms`, `file_read_bytes`) would give it in one run.
3. **How often are the evicted victims re-routed?** The single number that decides §2.6, and it is
   measurable only by making the change (or by replaying the routing trace against the victim
   ranking offline, as `tools/specter_oracle.py` does for the tier's misses).
4. **Is `cudaDevAttrConcurrentManagedAccess` 0 on this box?** It should be (Windows), and it costs one
   line to confirm. It is the check that closes §2.1 for good on this platform.

---

## 6. What the engine's own numbers say about the tier (the context the options are priced against)

| fact | number | source |
| --- | --- | --- |
| the static profile alone covers | 28.4% of the reads | `docs/plans/specter-performance.md` §2.2 |
| the running tier (profile + swaps) covers | 92.3-94.1% | the log line `decode expert cache hit rate` |
| the CPU pool's cost per offloaded expert | 44 us | `docs/SPECTER-FEASIBILITY.md` §0 |
| the bus's cost of the same expert | 99 us (2.66 MB / 26.8 GB/s) | the same |
| the marginal swap | 240 us to save 276 ± 19 us of pool time | `docs/plans/specter-performance.md` §2.7 item 3 |
| the link | PCIe 4.0 x16, 26.8 GB/s probed; 160 MB each way concurrently = 12.2 ms, the same as the two one after the other | the same |
| the host's DRAM copy rate | 38.6 GB/s (the exchange's memcpy) | `logs\ab-churn3.csv` |

The tier earns its keep: without it the CPU pool would compute 48 x 10 x (1 - 0.284) = 343 experts
per window instead of 48 x 2.16 = 104, i.e. ~11.9 ms instead of 3.6 ms - so the swaps buy ~8.3 ms
and cost 3.57 ms. The tier is at break-even *at the margin*, which is exactly why a lever that
halves the swap's traffic (§2.6) is worth more than any lever that rearranges it.

---

## 7. Sources

### 7.1 CUDA Programming Guide (13.0 archive and 13.4 current), docs.nvidia.com

- *Unified Memory on Windows, WSL, and Tegra* - §4.1.3 in the current guide; the same section is
  §24.3.2 in the 13.0 archive (<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html>,
  <https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html>): *"Fine-grained
  movement of the managed data to GPU on-demand is not supported... Page faulting is only supported
  from the CPU side"*; *"They cannot allocate more managed memory than the physical size of GPU
  memory"*; *"Simultaneous access to managed memory is not possible."* The table *Overview of
  Unified Memory Paradigms* (13.0 archive, Table 31) places Windows in the row with
  `concurrentManagedAccess = 0`.
- *Asynchronous Concurrent Execution* (13.0 archive, §6.2.8.1): *"Async memory copies might also be
  synchronous if they involve host memory that is not page-locked."*
- *Concurrent Data Transfers* (13.0 archive): *"In order to be overlapped, any host memory involved
  in the transfers must be page-locked."*
- *Page-Locked Host Memory* (current guide, §2.6.3): *"Page-locked host memory is required for
  asynchronous copies between the CPU and GPU. Page-locked host memory also improves performance of
  synchronous copies."* And on mapped memory: *"Mapped memory should not be considered a performant
  alternative to unified memory or explicit memory management for the majority of a kernel's memory
  needs."*

### 7.2 CUDA Runtime API reference (13.4), docs.nvidia.com

- `cudaMallocManaged`: *"Device memory oversubscription is possible for GPUs that have a non-zero
  value for the device attribute cudaDevAttrConcurrentManagedAccess... managed memory may be evicted
  from device memory to host memory at any time by the Unified Memory driver."*
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>
- `cudaHostRegister`: *"Page-locking excessive amounts of memory may degrade system performance,
  since it reduces the amount of memory available to the system for paging. As a result, this
  function is best used sparingly to register staging areas for data exchange between host and
  device."*
- `cudaMemcpyBatchAsync` (13.0+): *"The batch as a whole executes in stream order but copies within a
  batch are not guaranteed to execute in any specific order"*; and
  `cudaMemcpySrcAccessOrderAny`: *"suited for host pointers allocated outside CUDA (ex., via
  malloc)... allows the driver to optimize the copy on certain platforms."*
- Stream Ordered Memory Allocator (§6.33): `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemPoolCreate`,
  `cudaMemPoolSetAttribute`, `cudaMemPoolTrimTo`.
- `cudaMemDiscardBatchAsync`: *"Performs a batch of memory discards asynchronously"* (the same
  memory group; the singular spelling is not in the pages fetched).

### 7.3 The Programming Guide's stream-ordered allocator chapter (13.0 archive, ch. 15)

*"Managing memory allocations using cudaMalloc and cudaFree causes the GPU to synchronize across all
executing CUDA streams"*; the release threshold (*"the amount of memory in bytes a pool should hold
onto before trying to release memory back to the OS"*) and `cudaMemPoolTrimTo`
(*"Tries to release memory back to the OS"*).

### 7.4 CUDA Driver API reference (13.4), docs.nvidia.com

- `cuMemCreate`: *"To create a CPU allocation that doesn't target any specific NUMA nodes,
  applications must set CUmemAllocationProp::CUmemLocation::type to CU_MEM_LOCATION_TYPE_HOST."*
- `cuMemMap` / `cuMemUnmap` / `cuMemRelease`: the address and the physical memory are decoupled;
  *"The memory allocation will be freed when all outstanding mappings to the memory are unmapped and
  when all outstanding references to the handle... are also released."*
- `cuMemHostRegister` / `cuMemHostAlloc` / `cuMemHostGetDevicePointer`: the runtime-API equivalents
  the engine already uses.
- `cuMemSetAccess`: *"Users cannot specify CU_MEM_LOCATION_TYPE_HOST_NUMA accessibility for
  allocations created with other location types."*

### 7.5 NVIDIA technical blogs

- *Using the NVIDIA CUDA Stream-Ordered Memory Allocator, Part 1* (2021-07-27):
  <https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/> - the
  allocator's value is *"to avoid expensive calls into the OS through memory pools maintained by the
  CUDA driver"*; the default release threshold is zero, so *"all unused memory in the pool is
  released back to the OS during every synchronization operation."*
- *Simplifying GPU Application Development with Heterogeneous Memory Management* (2023-08-22):
  <https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/>
  - HMM (the Linux equivalent of full unified memory) is *"currently supported on Linux only"*.

### 7.6 The engine's own measurements and code

- `docs/plans/specter-performance.md` §0.1 (the floors), §2.5 (one pool of RAM), §2.7 (the churn
  tax, the link probe, the chunked staging's reversion).
- `docs/SPECTER-FEASIBILITY.md` §0 (the baseline table, `t_bus` 99 us vs `t_host` 44 us), §2.2 (the
  tiers and the GPU's 159 GB/s).
- The harness: `C:\Users\yoho\Downloads\UnbiasedStrata\{sweep.ps1, ab-forecast.ps1, a0-measure.ps1,
  ab-churn.ps1}` and their output in `logs\`:
  - `ab-churn3.csv` - the shipped build's per-request churn split (the numbers in §0.2);
  - `ab-churn.csv` - the chunked-staging build's, the one that was reverted (§0.3);
  - `abchurn-*.err.log` - the engine's own report of the complement (30.89 GiB, *page-locking
    refused*), the decode timing, and the file tier's *"files 0 blobs"* (§2.2, §2.6);
  - `build-final-clean.log` - the box's toolkit: CUDA 13.3 (§2.5).
