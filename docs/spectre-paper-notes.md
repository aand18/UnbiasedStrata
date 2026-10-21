# SPECTER (arXiv 2511.14102) — Primary-Source Notes

> Feasibility study for reimplementing SPECTER inside a C++/CUDA MoE inference engine.
> All claims below are traced to a primary source (arXiv HTML v1/v2, the official GitHub repo, or the HF paper page).
> Nothing here is inferred from third-party summaries.

## 0. Version map / identity (read this first — the paper was renamed)

| Version | Date | Title | Notes |
|---|---|---|---|
| v1 | 18 Nov 2025 | **"MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts"** | 3-component design: Expert Scheduler / Speculative Governor / Execution Engine. Baseline = Mixtral-Offloading. |
| v2 | 20 Oct 2026 | **"Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading"** | Current version. Renamed, restructured, new baseline (SpecMoEOff), 3 components renamed SIC/PIO/SEE. ASPLOS '27. |

- Abs page: <https://arxiv.org/abs/2511.14102> — "Accepted to ASPLOS 2027. Includes the artifact appendix"; DOI `10.1145/3845814.3850026` (v2 front matter, <https://arxiv.org/html/2511.14102v2>).
- v1 HTML: <https://arxiv.org/html/2511.14102v1> · v2 HTML: <https://arxiv.org/html/2511.14102v2> · HF page: <https://huggingface.co/papers/2511.14102>
- **There is no v3.** The abs page states "last revised 20 Oct 2026 (this version, v2)".
- Code: <https://github.com/wwwwwf/specter_asplos> (public, 3 stars, 4 commits on `main`). Artifact archive: Zenodo <https://doi.org/10.5281/zenodo.22937478>, Git commit `c0325dc` (v2 Appendix A.3.1).
- ⚠️ **Anything quoting "MoE-SpeQ" and "2.34× over Mixtral-Offloading" is v1, not the current paper.** v2's headline is "average 1.50× (up to 1.88×) TPOT speedup over the state-of-the-art system."

---

## A. Problem setting and exact baseline

### A.1 The setting

"Mixture-of-experts (MoE) models often exceed a single accelerator's memory capacity. Offloading experts to host memory creates a severe PCIe I/O bottleneck because data-dependent routing keeps transfers on the critical path."
— v2 Abstract, <https://arxiv.org/html/2511.14102v2#abstract1>

Target regime: **single GPU, single request (batch size 1), model weights partially offloaded to host memory.** v2 §1: *"We target memory-constrained MoE inference on a single GPU, with model weights partially offloaded to host memory."*

The core diagnosis: existing prefetching work is **intra-token** (predict later-layer routes for the *current* token from earlier-layer hidden states), which yields only a *microsecond-level* window — useless against *millisecond-level* PCIe transfer of gigabyte-scale expert working sets. v2 §1 and §2.1:

> "existing methods are largely restricted to intra-token prediction, i.e., leveraging hidden states from early layers to predict later-layer routes for the current token. This approach only yields a microsecond-level computation window, which is fundamentally insufficient to overlap with the millisecond-level PCIe transfer latency of gigabyte-sized experts. Consequently, the GPU remains starved, and significant stalls remain unresolved. To truly hide I/O latency, the system requires multi-step, inter-token lookahead."

v1 §2.4 quantified this: for Mixtral-8x7B, "Memory — dominated by fetching experts over PCIe — consume[s] a staggering 98.9% of the total time" (v1 Fig. 4, A100-PCIE-40G, bfloat16, 256 GSM8K tokens).

### A.2 THE baseline it beats: **SpecMoEOff** (NOT Mixtral-Offloading, NOT DeepSpeed, NOT llama.cpp)

**v2 §1, verbatim:**

> "Compared with **SpecMoEOff** ([Wang et al., 2025b]), our **primary MoE speculative-offloading baseline**, Specter achieves an average 1.50× TPOT speedup and up to 1.88× under single-request inference."

**v2 §4.2.1, verbatim:**

> "We use **SpecMoEOff as the primary MoE speculative-offloading baseline** and Specter achieves an average 1.50× TPOT speedup and up to 1.88×."

SpecMoEOff = *Wang et al., 2025b* — "SpecMoEOff" (cited in v2 §5.2 as: *"SpecMoEOff (Wang et al., 2025b) emphasizes speculative I/O pipelining"*). The repo ships the authors' implementation at `baselines/specmoeoff/` (with `PROVENANCE.md`): *"the author's retained Transformers implementation, with an INT4 draft, an FP16 target, route-frequency prefetching, and independent weights and KV caches."* — <https://github.com/wwwwwf/specter_asplos/blob/main/baselines/README.md>

**Full baseline set (6 total), v2 §4.1.3:**

| Group | Baselines | Notes from §4.1.3 |
|---|---|---|
| Serving frameworks | **vLLM**, **Transformers**, **llama.cpp** | "their default offloading behaviors differ substantially": vLLM = on-demand weight offloading; Transformers = coarse-grained layer-wise offload; llama.cpp = "CPU–GPU collaborative execution style, where a configurable number of layers are offloaded to the GPU while the remaining layers stay on the CPU" |
| Research offloading | **Mixtral-Offloading-SC (MO-SC)**, **Mixtral-Offloading-SM (MO-SM)**, **SpecMoEOff** | SC = same-cache regime, SM = same-memory regime |

Critical detail for anyone comparing numbers: **Mixtral-Offloading is also a substrate, not just a baseline** — v2 §4.1.3: *"our cache mechanism is also implemented on top of its cache-management foundation."* So MO-SC/MO-SM numbers partly share Specter's cache code.

**Baseline handicap fairness settings** (repo `baselines/README.md`, `benchmarks/protocol.json`):

| Setting | Specter residents / K | SpecMoEOff residents / K | Mixtral-Offloading residents |
|---|---|---|---|
| High | 16 / 16 | 12 / 3 | 32 |
| Low | 4 / 16 | 2 / 3 | 20 |

(Residents = per-MoE-layer cache slots. Specter/SpecMoEOff each use 32 global buffer allocations; Mixtral-Offloading uses 4.) Repo `README.md`, "Speedup comparison" section.

### A.3 Hardware / software platform

**v2 §4.1.1 (verbatim):** *"All experiments are conducted on a workstation with one NVIDIA A100 GPU with 40 GB of HBM memory and a 24-core Intel Xeon Silver 4310 CPU with 256 GB RAM. The GPU is connected to the CPU via a PCIe 4.0 x16 interface, providing a theoretical bandwidth of 31.5 GB/s for H2D data transfers."*

- **Single GPU only. No multi-GPU anywhere in the paper.**
- v1 §4.1.1 gave "16 GB/s per direction, totaling 32 GB/s of aggregate bandwidth" — v2 quotes 31.5 GB/s H2D.
- Software (v2 §4.1.1 + Appendix A.2/A.3.3): HuggingFace Transformers **4.53.3**, PyTorch **2.7.0** (CUDA runtime 12.6), CUDA Toolkit **12.8**, **GPTQModel 4.0.0.dev0** @ commit `037c5c0` + integrated **Marlin**, FlashAttention 2.8.1, vLLM 0.9.2, flashinfer-python 0.2.11.post3. Build arch: `TORCH_CUDA_ARCH_LIST=8.0` (sm_80).
- Repo README warns: *"Upstream wheels with matching version numbers are not interchangeable."* — a custom-patched GPTQModel is required.
- Host RAM is a hard requirement: 256 GB, because the whole offloaded model is resident in host DRAM (see C.4).

### A.4 Models, quantization, sizes

**v2 Table 2** (<https://arxiv.org/html/2511.14102v2#S3.T2>):

| | Phi-3.5-MoE | Qwen1.5-MoE | DeepSeekV2-Lite |
|---|---|---|---|
| Total Param. | 41.9B | 14.3B | 15.7B |
| Activated Param. | 6.6B | 2.7B | 2.4B |
| Total Weight | 78 GB | 29 GB | 34 GB |
| MoE Layer Number | 32 | 24 | 26 |
| First Dense Layer | 0 | 0 | 1 |
| Experts (Per Layer) | **16** (v1 said 8) | 60 | 64 |
| Top-k | 2 | 4 | 6 |
| Shared Experts | 0 | 1 (4× Dim + gate) | 1 (2× Dim) |
| Hidden Size | 4096 | 2048 | 2048 |
| MoE Inter. Size | 6400 | 1408 | 1408 |

**Quantization format (v2 §4.1.2):** *"We use the official FP16 weights for the target and a GPTQ-quantized draft. Specifically, all expert-module linear layers use symmetric INT4 quantization with a group size of 128."*
→ **GPTQ, symmetric, INT4, group size 128** (i.e. W4A16 — activations FP16, weights INT4). Not AWQ, not GGUF.
v1 §4.1.3 is identical. v2 §3.4.1: *"We adopt GPTQ (Frantar et al., 2022) for draft model quantization due to its favorable balance of implementation simplicity and accuracy retention."*

Repo README: *"The target stores the prepared source tensors and is loaded as FP16 by Specter; the draft quantizes routed experts to GPTQ INT4, group size 128."*

### A.5 Performance numbers (v2)

**v2 §4.2.1, high-memory setting:**

| Model | Specter | SpecMoEOff | MO-SM | vLLM | Transformers |
|---|---|---|---|---|---|
| Phi-3.5-MoE | **163.3 ms/tok** | 292.1 | 295.1 | 297.5 | 960.6 (5.9× vs Specter) |
| Qwen1.5-MoE | **35.3 ms/tok** | 66.4 | 74.2 | 176.6 | 258.2 |
| DeepSeekV2-Lite | **72.2 ms/tok** | — | — | — | 344.9 |

**GPU memory budgets (v2 Fig. 14 caption), "under shared-weight accounting":** high/low = **24/18 GB** (DeepSeekV2-Lite), **20/16 GB** (Qwen1.5-MoE), **40/32 GB** (Phi-3.5-MoE).

**B=32 (v2 Table 3), DeepSeekV2-Lite AVG:** Transformers 1493.6 ms → MO-SM 1262.9 → SpecMoEOff 718.0 → **Specter 119.2 ms** (throughput 21.4 → 25.3 → 40.2 → **130.3 tok/s**). Qwen1.5-MoE AVG: SpecMoEOff 633.8 → Specter 260.3 ms. v2 §4.2.2: *"Against SpecMoEOff, the maximum gains are a 6.02× mean-TPOT speedup at B=32 and 3.89× the mean throughput at B=16, both on DeepSeekV2-Lite."*

**Cache hit rate (v2 §4.4, Fig. 17, DeepSeekV2-Lite):** Specter **99.85% / 98.62% / 96.25%** at 16 / 24 / 32 GB. LRU = 29.2/50.94/76.56; LRU-S (scaled cache) = 61.13/76.56/95.77; SP-E / SP-L single-prefetch ≈ 92–96%.
(v1 Table 3 reports the same numbers; v1 Fig. 4 "Impact of Prefetch Strategy" is the v1 form of this.)

**Per-iteration latency breakdown (v2 Table 5a), ms:**

| Metric | DeepSeekV2-Lite | Qwen1.5-MoE | Phi-3.5-MoE |
|---|---|---|---|
| Draft | 674.0 | 223.3 | 952.5 |
| Verification | 136.1 | 88.2 | 82.2 |
| Total H2D Load | 243.4 | 89.7 | 2089.2 |
| Exposed H2D Stall | 108.3 | 35.4 | 1621.3 |
| Hidden H2D | 135.1 | 54.3 | 467.9 |
| **Hidden Ratio** | **55.5%** | **60.5%** | **22.4%** |

**Cumulative component ablation (v2 Table 5b, DeepSeekV2-Lite):** Full 69.2 ms → w/o PIO 83.5 (1.21×) → w/o PIO+SIC 94.4 (1.36×) → w/o all three 130.4 (**1.88×**).
v1 Table 4 ablation is different (fused kernel is the dominant term: 13.02 → 8.88 tok/s, i.e. −31.8%, when the fused kernel is removed; async prefetch is worth only −4.99%).

**Robustness to routing prediction error (v2 Table 6a, DeepSeekV2-Lite):** prefetch accuracy 1.000 / 0.785 / 0.575 / 0.410 → TPOT 63.0 / 66.5 / 82.6 / 84.2 ms (1.00×/1.06×/1.31×/1.34×).
**Natural recall (v2 Table 6b):** recall (fraction of distinct target-selected experts also selected by the draft, per speculative iteration, averaged over layers) = **0.930–0.961** across task domain (WT/C4/GK), context length (8/16/32) and decoding strategy (greedy, T=0.7, T=1.0).

**Datasets (v2 §4.1.2):** C4, WikiText-2-v1, HumanEval, GSM8K, GPQA (abbrev. C4/WT/HE/GK/GP). 16-token prefix cap, 128 output tokens, greedy, 5 inputs/dataset × 3 repeats = 75 records per method (repo `benchmarks/protocol.json`).

---

## B. Full design, decomposed

### B.0 ⚠️ CRITICAL DISTINCTION: SPECTER IS **STANDARD SPECULATIVE DECODING THAT PRODUCES TOKENS**

**It is NOT "expert prediction only" with no token drafting.** The draft model generates real candidate tokens; the target verifies them with standard accept/reject speculative sampling. Expert prediction is a *by-product* read out of the draft router during drafting.

Verbatim quotes:

- v2 §1: *"Meanwhile, the rigorous draft-then-verify mechanism mathematically guarantees zero degradation in generation quality and output distribution."*
- v2 §2.2: *"In SD, a lightweight draft model rapidly generates a sequence of K future tokens, which the target model then verifies in parallel. This accept/reject mechanism mathematically guarantees lossless generation quality, even if the draft predictions are imperfect. For MoE offloading, the SD framework naturally provides a multi-step time window to orchestrate PCIe prefetching. The drafting stage, running entirely in GPU VRAM, races ahead to generate future tokens. We can extract future expert routing information during this stage to orchestrate PCIe prefetching."*
- v2 §3.2.3: *"Standard speculative sampling determines the accepted prefix and produces either a corrected token after rejection or a bonus token when all candidates are accepted, thereby preserving the target output distribution."*
- v2 §5.2: *"Specter couples speculative decoding with expert offloading, using a quantized draft's multi-token routing lookahead for prefetching while standard target verification preserves the output distribution."*
- v2 §4.6: *"Target verification preserves the output distribution, limiting the impact of routing mispredictions to prefetching and execution performance."*
- v1 §2.2 (v1 spelled out the mechanism in the abstract too): *"the system gains an accurate, multi-step lookahead, allowing it to prefetch necessary experts for a subsequent, highly parallel verification step."*

**Consequence for a port:** to get SPECTER's I/O benefit you must implement *full* speculative decoding (draft tokens + target verification + accept/reject + KV rollback). You cannot extract only the expert lookahead without also paying for the token drafting — but you also cannot skip verification if you want the paper's "lossless" claim.

### B.1 The draft model

**Same architecture family? — It is literally the SAME MODEL, quantized. Not a smaller model, not a sibling.**

- v2 §1: *"we discover that a low-bit quantized version (e.g., 4-bit integer (INT4)) of the target MoE model itself serves as an ideal, **zero-training-cost** predictor. By sharing the identical routing architecture and parameter lineage, this quantized draft guarantees high-fidelity expert activation predictions."*
- v2 §2.2: *"a quantized version of the target MoE itself (e.g., INT4) serves as an ideal draft predictor. Because it shares the identical routing architecture and parameter lineage, it provides high-fidelity routing predictions **without any training overhead**."*
- Repo `docs/design.md`: *"A quantized draft proposes tokens; an FP16 target verifies them while expert transfers overlap computation."* and *"Routed draft experts retain INT4 weights. Matching non-expert parameters, including routers and shared experts, reuse target tensors."*
- Repo `speculative_inference_controller/model_init.py` / `initialization/model_loader.py`: the draft is loaded via `GPTQModel.load(quant_path, backend="marlin")` — i.e. a full-size model checkpoint, with per-model weight loaders `weight_loader_deepseekv2` / `weight_loader_qwenmoe` / `weight_loader_phimoe`.

**How is it produced? — Post-training quantization only. NO distillation, NO finetuning, NO auxiliary loss.**
There is **no objective/equation for a routing-fidelity loss anywhere in the paper**. The "high-fidelity expert activation patterns" is an *empirical observation* about INT4 PTQ, not a trained property. (v2 §2.2, §3.1 principle 2: *"high-fidelity routing predictions from a 4-bit draft"*.) The only equation in the design section relating to the draft is the depth-selection model, not a training loss.

**Hybrid precision layout (v2 §3.2.1) — the key memory trick:**

| Component | Precision | Why (v2 §3.2.1) |
|---|---|---|
| Routers / gates | **FP16** | *"MoE routing is sensitive to quantization noise, which can alter expert selection… the router logits directly determine which experts are selected for each token, and even small perturbations can change the top-k routing outcome."* |
| Embeddings, attention, norms | **FP16** | shared with target; *"The same applies to attention and other components outside the MLP."* |
| Shared experts | **FP16** | *"shared experts are activated for every token and remain resident in VRAM throughout inference… keeps these shared experts in FP16."* |
| Routed expert MLPs (draft) | **INT4, symmetric, g128 (GPTQ)** | *"These parameters dominate the overall model size and are the primary target for reducing the memory footprint."* |
| Routed expert MLPs (target) | **FP16** | offloaded to host, demand/prefetch loaded |

**How many layers/params?** All layers of the target model, same count (26 / 24 / 32 for DS / Qwen / Phi). The draft is NOT depth-reduced.

### B.2 "Preserving high-fidelity expert activation patterns" — the evidence

- **v2 §2.2, Figure 2:** INT4 Qwen1.5-MoE draft vs FP16 target, **top-4 expert-ID match rate**, over **1,000 requests × 24 MoE layers = 24,000 request-layer pairs**. All three aggregations (instance-level, layer-wise, request-wise) peak **>90%**, **global mean 90.8%**.
- **v2 Table 1** (expert activation similarity %, native top-k, various quant methods + GGUF precisions on DeepSeekV2-Lite):

| | Phi@GPTQ&Q4 | DS@GPTQ&Q4 | Qwen@GPTQ&Q4 | DS@GGUF | DS@AWQ | DS@Q2 | DS@Q4 | DS@Q8 | DS@GGUF&DS |
|---|---|---|---|---|---|---|---|---|---|
| Max | 99.79 | 99.45 | 93.48 | 99.99 | 99.45 | 99.45 | 99.98 | 99.99 | 99.99 |
| Min | 95.15 | 94.49 | 86.30 | 95.09 | 94.49 | 93.17 | 90.10 | 95.09 | 98.66 |
| **Mean** | **96.50** | **96.33** | **90.82** | **96.53** | **96.33** | **95.72** | **92.96** | **96.53** | **99.38** |

  → INT4 GPTQ is not special: AWQ ≈ GPTQ ≈ 96.3% mean, even **GGUF Q2 gets 95.72%**. This is a property of *4-bit PTQ of the router's inputs*, not of the GPTQ format per se.
- **v1 §2.5, Figure 6** (the finer-grained decomposition, dropped in v2): total 90.9% = **44.1% "hard"** (identical set AND identical importance order) + **46.8% "soft"** (same set, different order); **mismatch only 9.1%**. v1 also notes this beats a *specialized one-layer-ahead predictor* which reaches only **84.7%** (Song et al., 2024), and *"this quantized predictor can predict all layers simultaneously in a single pass."*
- v1 §2.5: *"a mismatch does not imply a total failure; it simply means at least one of the top-4 experts was not anticipated."*

### B.3 "Multi-step" speculative offloading + layer correspondence

- **Layer correspondence is 1:1 and trivial**, precisely because the draft *is* the target: same layers, same router, same expert count, same top-k. v2 §2.2 spells out the three problems this avoids:
  > "First, **lack of routing correspondence**: a smaller dense model lacks an expert routing mechanism entirely, and even a smaller MoE model possesses a different routing space (e.g., different hidden dimensions or expert counts), making it impossible to directly map its expert activations to the target model. Second, **poor universality and availability**: not all open-source MoE families provide a perfectly matched, lightweight counterpart, and training a dedicated draft model incurs prohibitive costs. Third, **memory redundancy**: an independent draft model requires maintaining separate weights and a redundant key–value (KV) cache, which severely exacerbates the memory wall and leaves less VRAM for caching experts."
- **Multi-step = K speculative tokens.** K is chosen by **offline profiling** (v2 §3.2.2):
  - Estimate mean token-acceptance rate ᾱ from representative prompts; **ᾱ = 96.63%** on DeepSeekV2-Lite.
  - Expected consecutive accepts **E[A] = ᾱ/(1−ᾱ) = 28.7** → conservative upper bound **K_max = 32**.
  - Sweep **K ∈ {2,4,6,8,12,16,20,24,32}**, 25 representative prompts × 3 repeats, pick argmin TPOT: **K = 16 (DeepSeekV2-Lite), 8 (Qwen1.5-MoE), 16 (Phi-3.5-MoE)**. (v2 Fig. 7.)
  - Repo `benchmarks/protocol.json` encodes the same: `ds_high` gamma=16, `qwen_high` gamma=8, `phi_high` gamma=16; `depth` sweep candidates `[2,4,6,8,12,16,20,24,32]`.
  - Repo `speculative_inference_controller/depth_profiler.py`: *"selects draft depth by comparing mean decode TPOT across candidate depths. Warmup is excluded, and each measurement begins with the same resident expert set and cache order."*
- ⚠️ **v1 vs v2 differ substantially here.** v1 had an *online* "Speculative Governor" solving an "Amortization Roofline Model" (v1 §3.2, Eqs. 1–5: Θ(k), I_amort(k), T_cycle(k), k* = argmax Θ(k), plus an SLO-bounded k_SLO). **v2 deleted the online optimizer entirely** in favor of offline profiling (v2 §3.2 opening: *"Rather than acting as a heavyweight online optimizer, it is responsible for organizing the speculative workflow…"*). **For a port, v2's approach is dramatically simpler and is what the released code implements.**
- v1 also used a *different ELB representation*: `ELB[i][j] = (expert_id, confidence_score)` with the confidence "derived from the gating network's logits" (v1 §3.3.1, Eq. 6), k×L entries. **v2 replaced it with per-layer counters** (below). v1 noted: *"our current implementation primarily uses the 'expert_id'."*

### B.4 How expert selection is predicted

- **Source: the DRAFT's router** (the draft runs on GPU; its top-k output is captured by forward hooks on the draft's MoE modules). Repo `predictive_io_orchestrator/routing_hooks.py` hooks `layer.mlp` for Qwen2-MoE and DeepSeek-V2, `layer.block_sparse_moe` for Phi-MoE; for DeepSeek it **skips layer 0** (`if layer_idx != 0`) because layer 0 is dense. `prefetch_controller.bind_draft_routes(...)`.
- **Granularity: per layer, per token, top-k expert IDs.** The draft's top-k selection for draft token *i* at layer *j* is exactly the target's top-k (fidelity ≈ 91–97%).
- **v2 ELB = GPU-resident, layer-indexed activation counters** (v2 §3.3.1, Eq. 2):
  > `ELB_cnt^(t)[j,e] = Σ_{i=1..t} 1[ e ∈ ELB_route[i,j] ]`, t ≤ K
  > *"Only the counter representation is materialized and maintained on the GPU."*
  At each prefetch checkpoint, experts are ranked per layer by cumulative count and the **top-M** are selected, *"where M matches the expert-cache capacity assigned to that layer"*. The selected (layer, expert) pairs are packed into a compact plan and copied **D2H once per checkpoint** (not per token).
  The ELB is **cleared at the end of each draft window** — *"preventing stale routing evidence from carrying into the next draft window."*
- **The TARGET's router is also used, but only at verification time** — the "Verification Expert Cache Residency Manager" uses the target router's *authoritative* decisions to classify cache-hit vs cache-miss experts (v2 §3.3.3). So: **draft router → prefetch planning; target router → verification-time demand loads.**
- **No auxiliary loss, no trained predictor.** The v1 "confidence_score" from gating logits was dropped in v2.

### B.5 Runtime scheduling of expert transfers

**v2 §3.3.2 + Figure 9 — three phases:**

1. **Cold Start** — a single *Initial Chunk* of draft tokens. The Pattern Analyzer collects routing metadata. **"No predictive transfer is submitted before b₁ because the available ELB lookahead remains limited."**
2. **Prefetch Overlap** — multiple *Adaptive Chunks*. At each boundary bᵢ, the analyzer snapshots the ELB counters, produces a per-layer top-M ranking, computes the *incremental* set of experts not yet cached, forms the next H2D batch, and **chooses the next boundary b_{i+1} so that the expected draft-compute time of the chunk ≈ the expected H2D time of the batch**:
   > "The planner estimates the transfer time of this batch together with the draft-computation cost and selects b_{i+1} such that the expected computation time of T_{bᵢ:b_{i+1}} approximately matches the expected H2D time. The paired chunk and batch execute concurrently on the CUDA compute and I/O streams."
   Loop continues *"until the next incremental batch is sufficient to fill the remaining expert-cache slots. Under the memory-constrained offloading configuration, this condition is reached before drafting completes."*
3. **Final Cleanup** — the last incremental batch overlaps the remaining draft compute and fills the cache. *"routing metadata produced by the final chunk is discarded and the ELB is cleared because the full cache cannot admit another predictive batch and target verification follows immediately."*

**Cache management (v2 §3.3.3 + repo `expert_cache.py`):**
- Residency is a **per-layer slot budget** (repo `docs/design.md`: *"Expert residency is a per-layer slot budget. Total GPU memory also includes model state, KV, and transfer buffers; a slot count is not a process memory cap."*).
- Eviction = **LRU within a layer**: `EvictionGroupInfo.choose_expert_to_evict` returns the least-recently-used entry of `main_infos` (repo `predictive_io_orchestrator/expert_cache.py:61-64`).
- **Slot protection:** *"Each cache-hit expert remains protected while it is waiting in the compute queue or processing its assigned token activations. Once all tokens routed to that expert have completed computation, its slot is unprotected and becomes eligible for replacement."* Implemented with per-slot `last_use` CUDA events and a slot *generation* counter to invalidate stale handles.
- **Two policies for demand misses** (`specter_async_demand_policy`): `"staging"` (default — demand misses load into a separate staging pool, preserving the prediction-managed resident set) vs `"resident"` (demand misses evict into resident slots). The PIO sets `"resident"` (`prefetch_planner.py:18`).

**Concurrency primitives (v2 §3.4.3, verbatim):**
> "It first improves Draft Kernel Fusion by preallocating and reusing intermediate activation buffers and tuning the shared-memory allocation of expert MLP kernels. Within the Pattern Analyzer, it incrementally accumulates ELB counters on the GPU and transfers only a compact expert plan through a batched device-to-host (D2H) operation at each prefetch point, avoiding frequent host synchronization after individual routing updates. Finally, it coordinates the Prefetch Planner and Residency Manager using **dedicated CUDA streams** to separate computation from data movement and **CUDA events** to enforce expert-readiness and safe cache-slot-reuse dependencies. It further combines **pinned host memory with non-blocking copies** to support efficient asynchronous H2D transfers."

**v1 §4.1.2 "Implementation Details"** (a good standalone checklist of the sync primitives):
> "we employ CUDA multi-stream scheduling with distinct CUDA events to manage synchronization and mutual exclusion across four critical dimensions: multi-stage prefetching, coordination between prefetching and on-demand parameter loading, overlapping of communication and computation, and bidirectional host-device data transfers."
plus: unified preallocated cache params for token-wise activations (no runtime-alloc stalls); static shared-memory config for attention/MLP kernels; batched expert-selection patterns to minimize D2H transfers; "Pipeline-based asynchronous loading mechanism with a prefetch window, leveraging pinned memory and non-blocking CUDA memory copy."

**No "cutoff layer" concept** exists in either version (the draft is a full model, so there is no layer beyond which speculation stops).

### B.6 Acceptance / verification mechanism

- **v2 §3.2.3 "Target-Consistent Decoding State Management"** (Figure 8):
  > "each iteration begins from a target-consistent prefix. The draft model extends this prefix with K candidate tokens, while the target model verifies them in a single causally masked forward pass. Standard speculative sampling determines the accepted prefix and produces either a corrected token after rejection or a bonus token when all candidates are accepted, thereby preserving the target output distribution. The State Manager commits these tokens to the authoritative sequence and discards the rejected suffix."
  > "The draft and target paths **share the target-authoritative KV cache** of the committed prefix while maintaining separate temporary KV extensions during speculation. The draft path records its speculative KV states in a scratch extension, whereas target verification generates a target-side extension for the candidate sequence. After verification, only target-generated KV entries corresponding to accepted candidates extend the shared cache. The draft extension and target-side entries for the rejected suffix are discarded."
- **Exact algorithm** — repo `speculative_inference_controller/state.py::spec_inf`:
  - Greedy: accept draft token if `draft_token == argmax(target_probs)`; else break.
  - Stochastic: accept with probability `min(1, p_target/q_draft)`; on rejection sample from the residual `max(0, p − q)` distribution.
  - On acceptance of all K: a **bonus token** is emitted.
  - `state.commit(token_len - 1)` rolls the target KV back to the accepted boundary and rebases the draft onto it.
  - Prefill: *"Leave one uncached token for both forward paths. Prefix KV comes exclusively from the full precision target, including first round."*
- **Effect of acceptance rate on prefetch accuracy — they are DECOUPLED.** The token acceptance rate drives the *choice of K* (higher ᾱ → deeper K is worth it: ᾱ = 96.63% → E[A] = 28.7). It does **not** affect routing fidelity, which is a property of the INT4 draft vs FP16 target router. v2 §4.6: *"Recall is the fraction of distinct target-selected experts also selected by the draft per speculative iteration… It remains within 0.930–0.961 across prompt domains, context lengths, and decoding strategies. Target verification preserves the output distribution, limiting the impact of routing mispredictions to prefetching and execution performance."* And v2 Table 6a: degrading prefetch accuracy from 1.000 → 0.410 costs only 1.34× TPOT, and even the *oracle* (accuracy 1.0) only reaches 63.0 ms vs Specter's own 69.2 ms — i.e. **prediction quality is not the bottleneck; the I/O overlap and verification scheduling are.**
- v1 §4.5: *"our speculative decoding achieves a significantly higher acceptance rate (over 90%) compared to conventional speculative decoding methods (e.g., Eagle's 80%)… We attribute this exceptional compatibility to the extreme similarity between the draft and target models—not only in architecture, but also in parameters and output distributions."*

### B.7 Execution engine (v2 §3.4)

- **§3.4.1 Fine-Grained MoE Draft Kernel Fusion:** Marlin is *"primarily optimized for large, regular GEMMs (e.g., 72K×18K)"; modern fine-grained MoE uses narrow expert matrices (2,048×1,408 or 4,096×6,400)*, so *"frequent kernel-launch overheads and poor Tensor Core utilization across fragmented expert-local GEMMs counteract the benefits of low-bit arithmetic, occasionally rendering quantized execution slower than native FP16."* Fix: *"gathers routed tokens by expert assignment and then schedules the disparate expert sub-computations into a single fused GPU launch"*, converting sparse token activations into dense tiles. v1 §3.4.3 called this a *"single, monolithic fuseMoE CUDA kernel."* **This is the single largest contributor in the v1 ablation (−31.8% throughput without it).**
- **§3.4.2 Cache-Aware Expert Verification Reordering** (Figure 12): instead of token-wise verification (fragmented memory-bound GEMVs), gather tokens by expert into contiguous sub-batches → **dense compute-bound GEMMs** per expert → scatter routing-weighted outputs back to token positions. Cache-resident experts are processed first while missing experts' H2D transfers are dispatched concurrently. *"This transformation alters only the execution schedule of independent MoE FFN operations. It preserves the original target routing decisions. Attention mechanisms still evaluate using the standard causal mask. Finally, a layer-wise synchronization ensures all tokens complete before advancing. Consequently, this expert-wise reordering guarantees strict mathematical equivalence."*
- Repo `streamlined_execution_engine/expert_reorder.py` implements per-architecture wrappers: `SparseMoeWrapperShared` (Qwen), `SparseMoeWrapperPhimoe`, `SparseMoeWrapperDeepseekv2`, `QwenmoeBlockSparseTop2MLP`, `PhiMoEBlockSparseTop2MLP`, `Deepseekv2BlockSparseTop2MLP`.
- **§3.4.3 Overlap-Oriented Parallelism Optimizer** — see B.5.
- Repo `streamlined_execution_engine/kernels/` contains **vendored CUDA C++ sources**: `csrc/vllm/moe/marlin_moe_wna16/*` (Marlin W4A16 MoE GEMM, ~8 kernel variants incl. `kernel_fp16_ku4.cu`, `marlin_template.h`, `ops.cu`), `csrc/vllm/moe/moe_align_sum_kernels.cu` (the `moe_align_block_size` sort/sum), `csrc/vllm/quantization/gptq_marlin/dequant.h`, `csrc/gptq/marlin_repack.cu`. Built via `python -m streamlined_execution_engine.kernels.build` → `specter_cuda.so`. **Licenses: vLLM + GPTQModel (Apache-2.0) vendored under `kernels/licenses/`.**

---

## C. Hard requirements / preconditions for a port

### C.1 Does the draft have to be on-device with its own quantized weights? **YES — and all of it.**

- v2 §2.2: *"The drafting stage, running entirely in GPU VRAM, races ahead to generate future tokens."*
- v2 §4.3: *"GPU memory stores **all INT4 draft experts** and 4 FP16 target experts per layer."*
- The draft is never offloaded. Only the **target's** FP16 routed experts live in host DRAM.

**Extra VRAM cost (v2 Table 4, Qwen1.5-MoE, 12K-token context):**

| Scenario | Draft | Target | Shared | Total |
|---|---|---|---|---|
| Independent (no sharing) | 6.13 GB | 7.27 GB | — | **13.40 GB** |
| + Shared Experts | 4.58 | 5.72 | 1.55 | 11.85 |
| + Other Params | 2.66 | 3.80 | 3.47 | 9.93 |
| + KV Cache | 0.41 | 1.55 | 5.72 | **7.68 GB (↓43%)** |

So: **+6.13 GB for an independent INT4 draft**, reduced to **+2.66 GB** after sharing non-expert params and shared experts, and **+0.41 GB** after also sharing the KV cache. (Figures are for Qwen1.5-MoE at 12K context; scale ≈ model_size/4 for other models.)

### C.2 Required CUDA/HIP features

**CUDA (NVIDIA only — there is NO HIP/ROCm support anywhere in the paper or repo):**

| Feature | Required? | Evidence |
|---|---|---|
| Dedicated CUDA streams for compute vs. I/O | **Yes** | v2 §3.4.3; repo `expert_cache.py:93-95` creates `stream_rx`, `stream_tx`, `stream_prefetch`; `prefetch_planner.py` has its own `torch.cuda.Stream`; `pattern_analyzer.py:24` another |
| `cudaEventRecord` / `cudaStreamWaitEvent` | **Yes** | v2 §3.4.3; repo `overlap_optimizer.py` — `ready_event`, `last_use_event`, `reuse_event`, all `torch.cuda.Event(enable_timing=False, blocking=False)` |
| **Pinned host memory** | **Yes, hard requirement** | repo `overlap_optimizer.py::_async_copy_storage` raises `RuntimeError("Specter async loading requires pinned host expert storage")` if `not source.is_pinned()`. Created via `torch.UntypedStorage(...).pin_memory()` (`expert_cache.py:166-172`, with an unpinned fallback that then *will* fail the async path) |
| **Non-blocking H2D copy** (`cudaMemcpyAsync`) | **Yes** | `destination.copy_(source, non_blocking=True)` on `stream_tx` |
| Batched D2H for the prefetch plan | **Yes** | v2 §3.4.3 — "transfers only a compact expert plan through a batched device-to-host (D2H) operation at each prefetch point" |
| NVTX ranges | Used, optional | `expert_cache.py:838,854` `torch.cuda.nvtx.range_push("[LOAD] Expert ...")` |
| CUDA managed memory (`cudaMallocManaged`) | **No** | Not used anywhere |
| `cudaHostRegister` | **No** | Not used; `pin_memory()` is used instead |
| CUDA Graphs | **Not mentioned** | no reference in paper or repo |
| GPU arch | **sm_80 (A100)**; `TORCH_CUDA_ARCH_LIST=8.0` | Marlin W4A16 requires sm_80+; repo builds only for 8.0 |
| Multi-GPU / NVLink | **No** | single GPU throughout |

**What a C++ port needs:** `cudaStreamCreate` (≥2–3), `cudaEventCreate/Record/Wait`, `cudaHostAlloc(..., cudaHostAllocPortable)` for the pinned host expert pool, `cudaMemcpyAsync(..., cudaMemcpyHostToDevice)` on a dedicated stream, `cudaEventSynchronize`/`cudaStreamWaitEvent` for slot readiness + last-use protection, and a batched `cudaMemcpyAsync` D2H for the per-checkpoint plan.

### C.3 On-disk format of the target's experts

- **safetensors shards**, HF layout. Repo `initialization/target_builder.py`:
  - `from safetensors.torch import load_file`
  - `index_path = os.path.join(states_dir, "model.safetensors.index.json")` → per-shard `load_file(...)` → `expert.load_state_dict(state_dict, strict=True)`.
  - The "trunk" (non-expert weights) is loaded from the index separately (`target_builder.py:378-389`).
- **Experts are flattened into one contiguous byte blob per expert.** Repo `streamlined_execution_engine/expert_storage.py::ExpertWrapper.replace_layer_storage` flattens all tensors of an expert's `state_dict` into a single `torch.UntypedStorage` (byte offsets recorded), so an expert is **one contiguous H2D copy**. The state-dict hook stores it back as `state_dict[prefix + 'storage'] = torch.as_tensor(self.storage, dtype=torch.uint8)`.
- **The draft checkpoint is GPTQModel format** (GPTQ-packed `qweight/qzeros/scales` + `quantization_config`), loaded with `GPTQModel.load(..., backend="marlin")`; the repo then re-packs for Marlin via `initialization/marlin_pack.py` and `streamlined_execution_engine/draft_fusion.py` (`weight_loader_deepseekv2` etc.).
- **GGUF is not used by the implementation.** GGUF appears only as a *fidelity comparison* in v2 Table 1 (column "Precision@(GGUF&DS)") — i.e. GGUF Q4/Q8 as alternative quantization, still on DeepSeekV2-Lite.
- **No disk I/O at inference time.** The entire target model is loaded into **pinned host RAM** at startup (256 GB RAM; the full 34–78 GB of FP16 experts are resident). `docs/design.md` / `expert_cache.py:115-120`: every offloaded expert gets a pinned `UntypedStorage` and `add_expert_storage` copies into it. **A port that streams experts from disk per-request would have a different (worse) I/O profile than the paper.**

### C.4 Does it need the target's router weights on device? **YES.**

- Routers are **FP16, on device, and shared between draft and target** (v2 §3.2.1: *"Specter keeps gating modules in FP16"*; repo `docs/design.md`: *"Matching non-expert parameters, including routers and shared experts, reuse target tensors."*).
- Two distinct uses: (a) the **draft's** router output drives prefetch planning (captured via forward hooks on the draft); (b) the **target's** router output drives verification-time cache residency (v2 §3.3.3).
- The draft's KV cache is built on the target's FP16 KV (v1 §3.4.1: *"The draft model operates on the high-precision KV cache generated by the target model from previous steps. This provides a more accurate and stable context for generating speculative tokens, directly improving the prediction quality of the ELB and increasing the overall token acceptance rate."*).

### C.5 Which MoE architectures does this work for? / what structural traits are required?

**Required traits (derived from the design + the three supported models):**

1. **The draft must be the same model, quantized.** You must be able to produce (or obtain) an INT4 GPTQ checkpoint of the *exact* target architecture, with the **same router topology, same expert count, same top-k**. Otherwise the "layer correspondence" collapses to the unsolved problem in §2.2. **This is the single biggest blocker** for a generic port.
2. **A router whose top-k output can be intercepted** at each MoE layer, for *every* MoE layer, *including the first dense layer exclusion* (DeepSeek layer 0 is dense → skipped).
3. **Per-layer resident expert slots** in VRAM, sized to the layer's expert working set.
4. **Layer-wise homogeneous expert shapes** (all experts in a layer have the same shape) so a single fused/contiguous buffer per layer works.
5. **Shared experts are optional** — Phi-3.5-MoE has none; Qwen has 1 (4× dim + gate); DeepSeek has 1 (2× dim). When present they are kept FP16 and resident.
6. **Top-k varies by model** (2 / 4 / 6) but is not a hard-coded constant in the mechanism — the ELB is indexed by whatever the router's top-k is.
7. **The number of MoE layers is not a constraint** (26 / 24 / 32 all work).

**Model-specific porting work in the repo** (evidence of what varies per architecture): `models/draft/{dsv2lite,qwen2moe,phimoe}/` each with their own `configuration_*.py` + `modeling_*.py` (DeepSeek needs `mlp.fusedexperts` — `model_loader.py` raises *"DeepSeek draft has no mlp.fusedexperts. Use the supported fused GPTQ checkpoint/package."*); per-model expert wrappers in `expert_reorder.py`; per-model router hook selectors in `routing_hooks.py`.

---

## D. What is NOT in the paper / the gaps

**Genuinely absent or weak in the paper:**

1. **No draft-model recipe.** The paper never says how to *produce* the INT4 GPTQ draft. §4.1.2 just states the format. The repo ships a **pre-quantized** DeepSeek draft and explicitly says *"No local quantization is needed"* (Appendix A.6) and *"Qwen/Phi checkpoints are not bundled"* (repo README). → **A from-scratch port must run GPTQ itself (or use an existing GPTQ export) for the target architecture.**
2. **No public artifact for the quantization**, and the required GPTQModel wheel is a **custom patched build** — *"Upstream wheels with matching version numbers are not interchangeable."* (repo README)
3. **No multi-GPU.** Single A100 throughout. No TP/NVLink/PCIe-switch discussion.
4. **No long-context evaluation.** The only 12K-token data point is the *memory accounting* table (v2 Table 4). Reported TPOT is at a **16-token prefix cap**, 128 output tokens (repo `benchmarks/protocol.json`). The repo has a `memory` experiment with 12,000-token prompts, but that is a VRAM measurement, not a TPOT eval.
5. **No apples-to-apples baseline in the same codebase for SpecMoEOff** — SpecMoEOff is the authors' *retained Transformers implementation* (repo `baselines/specmoeoff/PROVENANCE.md`), i.e. a re-implementation/adapter, not the original binary.
6. **Baseline resource asymmetry is not normalized**: Specter uses 16 resident/layer, SpecMoEOff 12, Mixtral-Offloading 32 (high); 4 / 2 / 20 (low). Specter & SpecMoEOff get 32 global buffers, Mixtral-Offloading 4. The paper calls this "fixed presets without calibration" and reports each method's measured GPU footprint (repo README) — but it is a confound.
7. **v2 removed the online governor** (v1 §3.2 Amortization Roofline Model) in favor of per-model offline profiling. **There is no published, general rule for picking K** — you must profile per model/hardware.
8. **v1 vs v2 factual discrepancies** that are not reconciled in v2: expert count for Phi-3.5-MoE (v1 Table 1: 8 per layer; v2 Table 2: 16); headline speedup (v1 abstract: "at most 2.34× speedup over the state-of-the-art offloading framework", v1 Fig. 13 shows 3.3× over MO-SC and 2.5× on Qwen vs MO-SC; v2: 1.50× avg over SpecMoEOff). The two papers have **different baselines and different system designs** and should not be conflated.
9. **The artifact is not a bit-exact reproduction of the paper.** Repo `benchmarks/protocol.json` states: `"workload_basis": "25 inputs, three repeats; historical input selection is not fully recovered."` and for the `ds_low8` case `"original_paper_configuration_confirmed": false`; also `"reference_budget_is_enforced_process_cap": false` and `"decoding_implementation": "Current corrected greedy argmax; existing native-precision normalization retained."`
10. **No ablation of the acceptance-rate/prefetch-accuracy coupling in v2.** v2's Table 6a varies prefetch accuracy but not the token acceptance rate; the two are asserted to be independent without a joint experiment.
11. **No `cutoff layer`, no `no-prefetch` warm-start analysis, no energy/carbon numbers.**
12. **No formal error bound / guarantee on the ELB.** The fidelity numbers are empirical measurements on 1,000 requests × 24 layers for Qwen and a small cross-model table; there is no theoretical argument that INT4 preserves top-k.
13. **The implementation is Python/PyTorch**, with the only C++ being vendored vLLM Marlin kernels. There is **no standalone C++ engine**, no HIP/ROCm path, and no HIP kernels despite the "asplos" repo name.
14. **Licensing on the kernels**: the vendored Marlin sources carry vLLM + GPTQModel Apache-2.0 licenses (`kernels/licenses/`). A from-scratch C++ port that reimplements Marlin-equivalent W4A16 GEMM must respect those.
15. **Practical footprint:** ~200 GB disk, 68 GB of downloads, 256 GB host RAM, A100 40 GB, ~1 h setup (Appendix A.2–A.3).

---

## E. Minimum viable port checklist

Assumed starting point for the target engine: a C++/CUDA MoE inference engine that **already has** (i) host→VRAM expert offloading with an expert cache, (ii) a working MoE forward with router/top-k, (iii) optionally a draft model.

### E.1 Phase 0 — prerequisites (blocking, decide before writing code)

- [ ] **P1. Produce/obtain an INT4 (GPTQ, sym, g128) checkpoint of the *exact* target model**, with identical router/expert topology. Without this, SPECTER reduces to a non-starter. (Paper: v2 §3.4.1, §4.1.2; artifact gap: D.1.)
- [ ] **P2. Have a W4A16 GEMM path** (Marlin-equivalent) for the draft's routed experts. If the engine only has FP16, the draft's *memory* benefit disappears (the whole point of P1) — a FP16 draft of a 34 GB model is 34 GB of VRAM. Requires sm_80+ for Marlin-class kernels; otherwise a custom dequant-fused kernel is needed (paper v2 §3.4.1 argues Marlin *alone* is not enough at fine-grained MoE shapes — the fusion is mandatory).
- [ ] **P3. Confirm the target model's router can be read out per layer** (top-k expert IDs + scores) without a full extra forward pass.

### E.2 Phase 1 — core mechanisms (required for the headline number)

- [ ] **M1. Quantized draft inference path** (W4A16 routed-expert MLP, FP16 attention/norms/router). Implement the **fused multi-expert draft kernel**: gather tokens by expert → single kernel launch → dense tiles → reduce (paper v2 §3.4.1; repo `streamlined_execution_engine/draft_fusion.py` + `kernels/`). *This is the biggest single win in the v1 ablation.*
- [ ] **M2. Full speculative decoding loop**: draft K tokens autoregressively; target verifies K+1 in one causally-masked pass; greedy accept (`argmax` equality) or stochastic accept (`min(1, p/q)`); residual-distribution corrected token on rejection; bonus token when all accepted. (Paper v2 §3.2.3, §5.2; repo `speculative_inference_controller/state.py`.) **This is mandatory — see §B.0.**
- [ ] **M3. Target-authoritative KV with shared prefix**: target owns the committed KV; the draft reads the target's committed prefix and appends a *scratch* extension; after verification, crop target KV to the accepted boundary and rebase the draft onto it; discard rejected-suffix KV. (Paper v2 §3.2.3, Fig. 8; repo `shared_prefix_cache.py`, `state.py`.) *Saves 43% of the speculative footprint (v2 Table 4).*
- [ ] **M4. Expert Lookahead Buffer (ELB) as GPU counters**: per-layer activation counters over the draft's top-k selections; top-M ranking per layer at checkpoints; compact D2H plan transfer once per checkpoint; clear at end of window. (Paper v2 §3.3.1, Eq. 2; repo `pattern_analyzer.py`.)
- [ ] **M5. Layer-Aware Prefetch Planner with 3 phases and adaptive chunk boundaries**: Cold Start → Prefetch Overlap (chunk compute time ≈ batch H2D time) → Final Cleanup. (Paper v2 §3.3.2, Fig. 9; repo `prefetch_planner.py`.)
- [ ] **M6. Async H2D infrastructure**: pinned host expert pool (`cudaHostAlloc` / `cudaHostRegister` on the existing host buffers), dedicated I/O stream, `cudaMemcpyAsync` non-blocking, per-slot `ready_event` / `last_use_event` / `reuse_event`, per-slot generation counters to invalidate stale handles. (Paper v2 §3.4.3; repo `overlap_optimizer.py`, `expert_cache.py`.)
- [ ] **M7. Verification Expert Cache Residency Manager**: at verification time use the *target's* router output; cache-hit experts → compute queue, cache-miss experts → I/O queue concurrently; protect a slot until its last routed token has been computed; support both "staging" and "resident" demand-miss policies; per-layer LRU eviction. (Paper v2 §3.3.3; repo `expert_cache.py`.)
- [ ] **M8. Expert-wise verification reordering**: group the K candidate tokens by expert into contiguous sub-batches → one dense GEMM per expert → scatter routing-weighted outputs back to token positions; layer-wise barrier. (Paper v2 §3.4.2, Fig. 12; repo `expert_reorder.py`.) *Without this, the "expert explosion" cost of SD eats the gain (v2 §2.3, Fig. 3).*
- [ ] **M9. K selection**: offline profile over K ∈ {2,4,6,8,12,16,20,24,32}, pick argmin TPOT. (Paper v2 §3.2.2; repo `depth_profiler.py`.) Start with K=16 for DeepSeekV2-Lite-scale models, K=8 for Qwen1.5-MoE-scale.

### E.3 Phase 2 — fidelity & sharing (needed to hit the paper's numbers, not just "works")

- [ ] **S1. Keep routers (and attention/norms/embeddings) in FP16** in the draft; quantize only routed expert MLPs. (Paper v2 §3.2.1.) *Quantizing the router is the single easiest way to destroy the 90% routing fidelity.*
- [ ] **S2. Share non-expert parameters and the KV cache between draft and target** (same tensors, not copies). (Paper v2 §3.2.1, §4.3, Table 4.)
- [ ] **S3. Keep shared experts in FP16 and resident** where the architecture has them.
- [ ] **S4. Measure routing fidelity** as a first-class metric: per-layer top-k ID match rate draft-vs-target, and per-iteration recall of distinct target experts. (Paper v2 Fig. 2, Table 6b; repo `--experiment routing`, `fidelity`.) *If fidelity < ~90%, the whole prefetch premise degrades (Table 6a: accuracy 0.575 → 1.31× TPOT).*
- [ ] **S5. Ensure the target's experts are stored as one contiguous byte blob per expert** so a prefetch is a single `cudaMemcpyAsync`. (Repo `expert_storage.py`.) In a GGUF-backed engine this means: either a byte-range copy per expert from the mmap'd file, or a repacked "expert blob" file.

### E.4 Phase 3 — optional / later

- [ ] **O1.** Batched execution (B up to 32) with per-request KV/acceptance/rollback independence (v2 §4.2.2, Table 3).
- [ ] **O2.** Long-context (>12K) validation — not covered by the paper.
- [ ] **O3.** Online K governor (the v1 Amortization Roofline Model) — dropped in v2; only worth it if K must adapt to changing hardware/workload.
- [ ] **O4.** Multi-GPU — not in the paper at all.

### E.5 Risk register for the port

| Risk | Evidence | Mitigation |
|---|---|---|
| No INT4 checkpoint for the target arch | D.1 | Build/obtain GPTQ export early; validate fidelity (S4) before building the scheduler |
| W4A16 Marlin-class GEMM unavailable / not sm_80+ | v2 §3.4.1 | Budget for a custom dequant-fused kernel; the *fusion* matters more than the format |
| No separate draft weights available → VRAM blowup | v2 Table 4 (+6.13 GB independent) | Implement S1–S3 sharing aggressively |
| Routing fidelity collapses (quantized router, wrong top-k, different expert set) | v2 Fig. 2, Table 1, Table 6a | FP16 routers (S1); measure (S4) |
| Expert explosion: K too deep → PCIe thrash + cache overflow | v2 §2.3 Fig. 3–4 (K=1→16 grows per-layer demand 5.29× / 7.18× / 4.65×) | Profile K (M9); per-layer slot budgeting |
| Host RAM too small (256 GB in the paper) | v2 §4.1.1 | Port to a disk-backed or mmap'd expert store — but then I/O behaviour diverges from the paper |
| Non-pinned host memory → async path disabled | repo `overlap_optimizer.py` | `cudaHostAlloc`/`cudaHostRegister` the expert pool |
| Baseline comparison invalid | D.5, D.6 | Normalize resident slots and buffer counts when reporting your own numbers |

---

## Appendix — quick source index

| Source | URL | What to go to |
|---|---|---|
| Paper v2 (current) | <https://arxiv.org/html/2511.14102v2> | §1 baseline claim; §2.2 fidelity; §3.2 hybrid precision + K; §3.3 PIO; §3.4 SEE; §4 eval; Table 2/3/4/5/6; Fig. 7/9/12/14/17; Appendix A |
| Paper v1 | <https://arxiv.org/html/2511.14102v1> | §2.5 hard/soft match decomposition; §3.2 Amortization Roofline Model; §3.3 ELB w/ confidence; §3.4 fuseMoE kernel; §4.5 ablation |
| Abs / metadata | <https://arxiv.org/abs/2511.14102> | versions, ASPLOS'27 acceptance, DOI |
| HF paper page | <https://huggingface.co/papers/2511.14102> | title, authors, upvote count (17), community questions |
| Repo README | <https://github.com/wwwwwf/specter_asplos/blob/main/README.md> | AE install, Zenodo/DOI, resident-vs-K table, experiment families |
| Repo design doc | <https://github.com/wwwwwf/specter_asplos/blob/main/docs/design.md> | module-by-module mapping (SIC/PIO/SEE), W4A16 semantics, per-layer slot semantics |
| Repo baselines | <https://github.com/wwwwwf/specter_asplos/blob/main/baselines/README.md> | SpecMoEOff & Mixtral-Offloading provenance, TPOT definition |
| Repo `benchmarks/protocol.json` | <https://github.com/wwwwwf/specter_asplos/blob/main/benchmarks/protocol.json> | per-case gamma/resident/offload/buffer, prefix cap 16, 128 tokens, "historical input selection is not fully recovered" |
| Repo `speculative_inference_controller/state.py` | <https://github.com/wwwwwf/specter_asplos/blob/main/speculative_inference_controller/state.py> | exact accept/reject algorithm, KV commit/rebase |
| Repo `predictive_io_orchestrator/expert_cache.py` | <https://github.com/wwwwwf/specter_asplos/blob/main/predictive_io_orchestrator/expert_cache.py> | streams, pinned host pool, LRU eviction, slot events/generations, staging vs resident policy |
| Repo `streamlined_execution_engine/overlap_optimizer.py` | <https://github.com/wwwwwf/specter_asplos/blob/main/streamlined_execution_engine/overlap_optimizer.py> | pinned-memory requirement, async copy, event ordering |
| Repo `initialization/target_builder.py` / `model_loader.py` / `expert_storage.py` | same repo | safetensors index + shard loading; GPTQModel/marlin draft load; per-expert contiguous byte blob |
