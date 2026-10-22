# DFlash 2 as this engine's speculative drafter

Written 2026-10-21 against this tree's `spectre-moe` branch (HEAD `1137903`). The question: what would it take to use
the **DFlash 2** block-diffusion draft model as this engine's speculative drafter, replacing or supplementing the
model's own MTP layer? Companion to [`SPECTER-FEASIBILITY.md`](SPECTER-FEASIBILITY.md) §5.6 (the DFlash 2 summary)
and [`plans/specter-performance.md`](plans/specter-performance.md) Track C (the plan).

**The short answer.** DFlash 2 is the right *shape* of drafter for this engine — 5 dense layers, no recurrence, no
MoE, one masked parallel pass over a block, and a 12,800-feature capture from the target at five layers — and it
attacks the lever the sweep measured first: the **per-draft acceptance rate**, which is **68.3-68.8%** today and
would have to reach **85%** for the port to be worth building (that would take the window from 2.42 to 2.77
committed tokens and the decode from 131.6 to ~150 tok/s, **+14%**). DFlash 2's published acceptance lengths, converted
to this engine's window, average **68.4%** — the engine's own number — and its best published case reaches 75.3%
(**+6%**). But the engine has none of the machinery, and each missing piece is small and specific:

| Piece | Exists here? |
| --- | --- |
| A block-diffusion draft pass (masked, non-causal, parallel over the block) | no — the MTP drafts one row at a time (`src/core/mtp.cpp:469`) |
| A candidate selector over top-k per position | no — the MTP takes an argmax (`src/core/mtp.cpp:639-643`) |
| Two-tap dynamic convolutions | no |
| The target's per-layer "completed output" tap at 5 layers | no — the residual is updated in place, one buffer (`src/core/verify.cpp:560`) |
| A loader for a 1.2 GB safetensors BF16 draft | no — the pack loader is a bespoke `index.txt` format (`src/core/weights.cpp:144`), the MTP has its own (`src/core/mtp.cpp:164`) |
| A BF16 multi-row GEMM | **yes** (`bf16_gemv_fp32_mmvf_multi`, 1..8 rows) |
| A 32-head / 8-KV-head / head_dim-128 sliding attention | no — the engine's attention is the *target's* QSA (24/2/256) or GDN |
| The sampling machinery and the exact-match acceptance | **yes** (`include/strata/core/coupled_draft.hpp:4-6`) |
| The K/V ring handling the MTP needs | not needed — DFlash 2 has `use_cache: false` and no recurrence |

The honest summary: this is **a new subsystem**, not a config change. The cheapest first step is not a port — it is
§8's measurement, which decides whether the port is worth anything at all.

---

## 0. The measured baseline this has to beat

From [`SPECTER-FEASIBILITY.md`](SPECTER-FEASIBILITY.md) §0 (RTX 4090 24 GB, Ryzen 9 7950X3D, PCIe 4.0 x16, Qwen3.8-
Flash-Next at IQ3_S, the production serve path `--low-ram on --ple-io direct`, `STRATA_DECODE_TIMING=1`):

| term | ms/window | share |
| --- | ---: | ---: |
| window cost (2.47 tokens/window, T = 3.00) | 19.56 | |
| **GPU-reach wait** (host waits for the GPU) | **9.90** | **51%** |
| per-layer host, of which the CPU expert pool | 4.01 / **3.53** | 21% / **18%** |
| draft (MTP + suffix) | 2.05 | 10% |
| commit/emit | 0.07 | 0.4% |
| GPU stages: VRAM hits / waitCPU / PCIe grp | 2.66 / 1.70 / **0.19** | |
| per layer-window: 28.25 VRAM hits, 1.68 CPU experts, **0.00 PCIe** | | |

126 tok/s. The tier holds 7,998 resident experts of 24,576; the drafts are accepted ~73% of the time (94 of 128).
The window is dominated by *per-window* work, so tokens/window is the first lever; the draft phase is only 10% of the
window, which bounds how much a faster drafter can pay (§6).

---

## 1. What DFlash 2 actually computes

Sources, all primary: the model's `config.json` and `model.safetensors` header
(<https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2>), its model card
(<https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/raw/main/README.md>), the method repository's
implementation (`dflash/model.py`, <https://raw.githubusercontent.com/z-lab/dflash/main/dflash/model.py>, MIT), and the
method paper (<https://arxiv.org/html/2602.06036>, v2, 2026-05-28).

### 1.1 The model

From `config.json` (fetched 2026-10-21):

| field | value | note |
| --- | --- | --- |
| `architectures` | `["DFlash2DraftModel"]` | not a language model |
| `num_hidden_layers` | 5 | all `layer_types: ["sliding_attention" x 5]` |
| `hidden_size` | 2560 | matches the target's `n_embd` |
| `intermediate_size` | 7680 | |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 32 / 8 / 128 | Qwen3 geometry, **not** the target's QSA (24/2/256) |
| `sliding_window` | 2048 | |
| `is_causal` | **false** | bidirectional inside the block |
| `use_cache` | **false** | no KV cache for the draft |
| `dtype` | `bfloat16` | |
| `vocab_size` | 248320 | the target's vocabulary |
| `dflash_config.block_size` | 8 | |
| `dflash_config.selector_rank` | 256 | |
| `dflash_config.selector_top_k` | 16 | candidates kept per position |
| `dflash_config.conv_kernel_size` | 2 | the "two-tap" convolution |
| `dflash_config.conv_group_size` | 16 | |
| `dflash_config.target_layer_ids` | `[5, 15, 25, 35, 45]` | the capture layers |
| `dflash_config.mask_token_id` | 248077 | |
| `num_target_layers` | 48 | the target's depth |

The checkpoint is a single `model.safetensors`, **1,206,140,008 bytes** (1.12 GiB): an 8-byte little-endian header
length, an 8,800-byte JSON header, and 1,206,131,200 B of tensor data in **81 tensors, all BF16** (header read
directly from the file, 2026-10-22). They are 5 decoder layers of **168.8 MiB each** (`q_proj` 4096x2560, `k_proj`/`v_proj`
1024x2560, `o_proj` 2560x4096, `mlp.gate/up` 7680x2560, `mlp.down` 2560x7680, the two convolutions, two RMSNorms,
`q_norm`/`k_norm` 128), the two convolutions per layer (`base_kernel` [2,2,2560], `kernel_projection` [640,2560]),
`fc.weight` 2560x12800 (62.5 MiB), `norm.weight`, `hidden_norm.weight`, and the selector
(`hidden_projection` 256x2560, `predecessor_codebook` 248320x256 and `successor_codebook` 248320x256 — **121.25 MiB
each**). 603.07 M parameters, i.e. the "0.6 B" of the model card.

**There is no `embed_tokens` and no `lm_head` in the checkpoint.** That is the paper's "shared embedding and LM head"
(§4.2 of arXiv:2602.06036: the draft shares the target's embedding and head, frozen). Consequence for a port: the
draft borrows the engine's *target* embedding table and head, and the head is a different code path from the target's
(§4.3).

### 1.2 The draft pass: one masked, non-causal, parallel forward

The draft proposes the next block by running the 5 layers **once** over a block of positions, all of which are masked
except the first (the anchor). The upstream reference loop (`dflash/model.py:233-258`):

```
verify_size = min(block_size, max_length - start)
block_output_ids = output_ids[:, start : start + verify_size]      # anchor + mask tokens
noise_embedding = _raw_input_embeddings(target, block_output_ids) # the anchor's real embedding, the rest MASK
draft_hidden = model(target_hidden=target_hidden, noise_embedding=noise_embedding, ...)[:, 1-verify_size:, :]
```

What is **parallel**: all `block_size - 1` masked positions of the block are predicted in a *single* forward pass
(`dflash/model.py:573-597`: one loop over the 5 layers, the whole block as the batch dimension). The paper's §3.2
makes the cost argument explicit: `T_draft = t_parallel` instead of `T_draft = gamma * t_step`, so the drafting cost
is "largely insensitive to gamma" for moderate block sizes.

What is **sequential**: the block is a *prefix* of the response. The next block's anchor is a token the target
produced (the bonus token of the previous verification step), so block *k* cannot start before block *k-1* has been
verified. The training procedure matches: "the draft model always conditions on a clean token produced by the target
model (i.e. the bonus token from the previous verification step)" (§4.2 of the paper, "Random sampling of masked
blocks").

The attention inside a layer is `is_causal: false` with a sliding window of 2048, and the *target context feature* is
appended to the K/V sequence ahead of the block's own tokens (`dflash/model.py:384-391`: `k = cat([k_ctx, k_noise])`).
So the draft attends to (a) the injected target context and (b) the block's own tokens, bidirectionally.

### 1.3 Conditioning: KV injection, not input fusion

The paper's central design choice (§4.1, §A.3): the target's hidden features from 5 layers are concatenated, projected
once by `fc` (2560x12800) and RMSNorm'd, and then **injected into the K and V projections of every draft layer**, not
just fused into the input:

```
H_t = RMSNorm(W_c [H^(l1) ; ... ; H^(l5)])          # A.3
K_i = [W_i^K H_t ; W_i^K H_d],   V_i = [W_i^V H_t ; W_i^V H_d]      # A.3
```

"Target features only serve as additional KV entries for the masked-block draft tokens. They bypass the draft model's
Q projection, output projection, self-attention update, and FFN." (§A.3). The ablation (§5.5.5, Table 9) puts the
difference at GSM8K 3.5 -> 4.2 acceptance length and MT-Bench 2.6 -> 3.0 for block-diffusion drafting, so the
difference is not cosmetic.

### 1.4 The selector: top-k per position, one coherent path

DFlash 2's addition over DFlash 1 is a **candidate selector** (`dflash/model.py:515-547`):

1. `torch.topk(logits, self.top_k, dim=-1)` over the draft's logits at every position -> 16 candidates per position
   with their unary scores;
2. a **predecessor** and a **successor** codebook, each `nn.Embedding(vocab_size, rank=256)`, i.e. a 256-dim vector
   per vocabulary token;
3. at each position, the score of candidate *k* is
   `unary[position][k] + <codebook_pred[predecessor] * hidden_projection(hidden[position]), codebook_succ[candidate[k]]>`;
4. the path is traced **greedily** (argmax) or sampled, and the chosen candidate becomes the *predecessor* for the
   next position.

So the selector is a first-order Markov chain over positions with a learned token-pair affinity, plus a per-position
unary term. The other 15 candidates at each position are **discarded**: DFlash 2 proposes exactly one token per
position, one path, `block_size - 1` tokens.

### 1.5 The two-tap dynamic convolutions

Two `GroupedDynamicCausalConv` modules per layer, one after the attention and one after the MLP
(`dflash/model.py:492-512`, `639-651`). Each has a `base_kernel` of shape `[2, kernel_size, hidden_size]` and a
`kernel_projection` (2560 -> 2*kernel_size*groups = 640, with groups = 2560/16 = 160) that produces a *dynamic*
per-position, per-group kernel. The convolution is split in two: `prepare()` runs the tap-0 half before the
projection and returns the dynamic kernel, `finish()` runs the tap-1 half after it, so the two taps straddle the
attention/MLP. The model card's reason: "Two-tap dynamic convolutions in the backbone keep the draft from decaying
toward the end of the block."

### 1.6 Losslessness

The model card: "Decoding is lossless: greedy output matches the target model exactly, and sampling preserves its
distribution." The paper's framing is the same (§1, §6): the diffusion model is confined to the drafting stage, and
the autoregressive target verifies, so the output quality is the target's. The paper reports "over 6x lossless
acceleration" (§Abstract) and the SGLang harness measures the *acceptance length* as "the per-request mean of
completion tokens divided by verification steps at concurrency 1" (model card, Evaluation).

The engine already has the same contract, stated the same way: "a draft is kept only when it EQUALS that row's pick.
The output is therefore a function of the seed alone, whatever the drafts are: drafts only decide how many tokens a
window yields" (`include/strata/core/coupled_draft.hpp:4-6`). So the losslessness claim needs no new machinery here —
but it does need the *proposal* to be well-formed (§4.3).

---

## 2. What it consumes from the target: the capture

### 2.1 The upstream patch, and what it says the capture is

The capture is the patch in the model repo:
`patches/qwen38-dflash2-capture.patch`
(<https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/raw/main/patches/qwen38-dflash2-capture.patch>), against
SGLang's `python/sglang/srt/models/qwen4_exp.py`:

```python
if i in getattr(self, "_dflash_completed_layer_ids", ()):
    # DFlash2 was trained on completed, zero-based layer outputs,
    # averaged over the four hyper-connection streams.
    assert not shards_rows, "DFlash2 capture is validated for TP1/CP1"
    assert hidden_states.shape[-1] == self.hc_count * self.hidden_size
    aux_hidden_states.append(
        hidden_states.unflatten(-1, (self.hc_count, self.hidden_size)).mean(dim=-2)
    )
```

and the setup:

```python
def set_dflash_layers_to_capture(self, layer_ids):
    assert list(layer_ids) == [5, 15, 25, 35, 45], layer_ids
    self.capture_aux_hidden_states = True
    self.model._dflash_completed_layer_ids = tuple(layer_ids)
    print("Q38_DFLASH_CAPTURE completed_layers=[5,15,25,35,45] hc_mean=True features=12800", flush=True)
```

**"Completed, zero-based layer outputs" in the target's terms, precisely:**

- *"zero-based"* — the layer index is 0-based, so `5` is the **sixth** layer of the 48 (`blk.5` in this engine's
  naming, `src/core/verify.cpp:623` builds `LayerView v(wt, l)` over 0-based `l`). The upstream reference agrees:
  `extract_context_feature` takes `hidden_states[layer_id + offset]` with `offset = 1`
  (`dflash/model.py:37-43`), i.e. HF's `output_hidden_states` index 6 is 0-based layer 5's output. The patch's
  assertion `layer_ids == [5, 15, 25, 35, 45]` is the same convention.
- *"completed"* — the layer's **finished** output, not an intermediate: the whole layer (attention half *and* the
  MoE half) has been applied. In SGLang the capture is taken at the end of the layer loop body, after
  `norms.output`.
- *"averaged over the four hyper-connection streams"* — the target's residual is **not one vector**: it is a stack of
  `hc = 4` streams of `n_embd = 2560` each (this engine: `ModelGeometry::hc = 4`, `include/strata/core/layout.hpp:48`;
  `hc_dim() = hc * n_embd = 10240`, `layout.hpp:55`). The capture takes the **plain arithmetic mean of the 4
  streams**, giving one 2560-vector per layer. `unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)` is exactly
  "reshape the last axis into (4, 2560) and average over the 4".
- `features=12800` — 5 layers x 2560. The number the patch prints is the width of the concatenated capture.

Note what the capture is **not**: it is not the target's logits, not the MTP's residual, and not a routing decision.
It is a plain mean over the residual streams, taken at five depths.

### 2.2 When the capture is available in a decode step

The upstream loop takes the capture from the **previous** verification step (`dflash/model.py:307-308`):

```python
if verify_size > 1:
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :produced, :]
```

i.e. after the target has verified a window, the capture is the *last accepted* position's features, and the next
draft pass is conditioned on them. In the engine's round that is exactly the seam between `ver.run(...)` and the
draft call (§3.6): the capture must be ready when the drafter is asked to propose, and the drafter's conditioning
feature is the target's state *after* the window it will be verified against.

### 2.3 The engine's equivalent of "the target's completed layer output"

This is the part that has to be re-derived, because the engine's residual is not kept per layer. In
`src/core/verify.cpp`:

- `R_` is the window's residual buffer, `(T, HC, N)` floats; row *t* is `Rt(t) = R_ + t*HC*N`
  (`src/core/verify.cpp:560`, `verify.hpp:337`).
- It is **updated in place**: the attention half's write is folded into the FFN half's read
  (`gr_read_group(1, true, inj_, inj2_)`, `src/core/verify.cpp:902`), and the FFN half's write is *pending* until the
  next layer's read applies it (`pending = l > 0 && !cvec().covers(l - 1)`, `src/core/verify.cpp:635`; the pending
  write's arithmetic is `R += bo * 2 sigmoid(inj/hc)`, `include/strata/kernels/cvec.hpp:55-58`).
- Only the **final** residual survives the window: `Verifier::final_R(t)` returns `R_ + t*hc*n_embd`
  (`src/core/verify.cpp:533`), and that is what the MTP drafter is given (`ver.final_R_all()`,
  `src/program/generate.cpp:5148`).

So **there is no per-layer "completed output" to tap today**: `R` is one buffer, overwritten 48 times per window, and
by the time layer 5's output existed it was gone.

The correct tap point follows from the write ordering. After `pre(l)`'s first `gr_read_group(0, pending, inj2_, inj_)`
(`src/core/verify.cpp:696`), the pending FFN write of layer *l-1* has been applied, so `Rt(t)` holds **layer l-1's
completed output** — the attention half *and* the MoE half, in the engine's own arithmetic. The capture of layer 5 is
therefore taken in `pre(6)`, layer 15 in `pre(16)`, layer 25 in `pre(26)`, layer 35 in `pre(36)`, layer 45 in
`pre(46)`.

Two consequences, both real:

1. The capture is a **copy**, not a reference: 5 layers x `T` rows x `HC*N` floats, reduced to 5 x `T` x 2560. For
   `T = 4` that is 5 x 4 x 10240 f32 read (819 KB) and 5 x 4 x 2560 f32 kept (205 KB) per window.
2. The engine's "completed output" is the *pending-write-applied* residual. Whether that is bit-identical to
   SGLang's `hidden_states` at the same layer is **not established here** — the engine's fused-GR path folds the write
   into the next read (`include/strata/core/layer.hpp:461-465` documents the one-half lag), and llama.cpp's
   `build_hc_combine` arithmetic is the same formula but not the same instruction order. This is a correctness
   question that has to be answered by measurement, not by reading (see §8).

### 2.4 The capture is also free routing features

The feasibility study's §5.6 point 3: the capture is a 12,800-feature vector per position from layers the target
computes anyway, which is exactly the input a lightweight estimator of `P(routing_l | captured features)` wants.
That is Track B2 in the plan, and it is **independent of the DFlash 2 port** - the capture is worth building for the
routing predictor even if the DFlash draft never lands.

---

## 3. The seams in THIS engine

### 3.1 The current drafter: the model's own MTP layer

`MtpDrafter` (`include/strata/core/mtp.hpp:36`). It is a **full MoE decoder layer**, not a small head:

| property | value | where |
| --- | --- | --- |
| shape | `fc_embedding` + `fc_hidden` (per stream) -> attention hyper-connection -> QSA attention (own K/V) -> MLP hyper-connection -> MoE (512 experts, top-10, shared expert) -> `R += y * 2 sigmoid(inject/hc)` -> final mixer -> head | `mtp.hpp:9-11` |
| the projections' format | **Q8_0** through the multi-column MMVQ (the offline tool quantizes them) | `mtp.hpp:15` |
| the attention | **dense** over every cell the layer has seen - but in the *target's* QSA geometry (24 heads / 2 KV / head_dim 256), via `qsa_decode_attn_batch` | `mtp.hpp:16-18`, `mtp.cpp:550-552` |
| the experts | **all 512 routed experts resident in VRAM, 708 MB** | `mtp.hpp:19`, `mtp.cpp:189-209` |
| the dense weights | 111 MiB | `mtp.cpp:92` |
| the K/V | the target's `QsaState`, optionally a ring over the drafter's window with a host copy for KV streaming | `mtp.cpp:215-245`, `mtp.hpp:54-56` |
| the drafting | **autoregressive**: the round runs the layer for row `a`, then **one CUDA graph per draft step**, each consuming the previous step's token | `mtp.cpp:854-976`, esp. `902-907` |
| the proposed token | the **argmax** of the draft head's logits (`sp.greedy = true`), plus its softmax probability | `mtp.cpp:639-643` |
| the VRAM | ~0.9 GB, and it **must be allocated before the expert tier is sized** | `mtp.hpp:43-46`, `generate.cpp:3029` |

Three of those matter for a port. The drafter is **autoregressive** (one sequential forward per draft token); it is
**quantized** (Q8_0); and it is **an MoE layer** (708 MB of VRAM experts, plus a router, plus the grouped hit kernels,
`mtp.cpp:573-579`).

### 3.2 Where the target's per-layer hidden states are produced

- The window's residual is one buffer: `R_` is `(T, HC, N)` f32 (`src/core/verify.cpp:408`), and row *t* is
  `Rt(t) = R_ + t*HC*N` (`src/core/verify.cpp:560`).
- The layer loop is `pre(l, grp)` (`src/core/verify.cpp:620`) and `post(l, grp)`, driven by
  `src/core/verify.cpp:1105-1111`.
- **The tap point derived in §2.3**: after `gr_read_group(0, pending, inj2_, inj_)` (`src/core/verify.cpp:696`) the
  pending FFN write of layer *l-1* has been applied, so `Rt(t)` holds layer *l-1*'s completed output. The captures are
  therefore taken in `pre(6)`, `pre(16)`, `pre(26)`, `pre(36)`, `pre(46)`. The pending flag is
  `l > 0 && !cvec().covers(l - 1)` (`src/core/verify.cpp:635`); when a control vector *does* cover layer *l-1*, its
  write was already applied by `cvec_apply` at the end of `post(l-1)` (`src/core/verify.cpp:1099-1101`), so the tap
  point is the same either way.
- **What does not exist**: no per-layer copy of `R`, no per-layer output buffer. `R` is overwritten 48 times per
  window; only the *final* residual survives (`Verifier::final_R`, `src/core/verify.cpp:533`; `final_R_all`,
  `include/strata/core/verify.hpp:194`), and that is what the MTP is given.
- The capture is **not** "the layer's output plus its state". The GDN layers' recurrent state lives outside `R`
  (`ss.gdn_state`, `src/core/verify.cpp:712`) and the QSA layers' K/V live in the session's pools. What the capture is
  is the residual only, which is what the upstream patch takes too.

### 3.3 The hyper-connection stream layout

| | | where |
| --- | --- | --- |
| the count | `hc = 4` | `include/strata/core/layout.hpp:48` |
| the width | `hc_dim() = hc * n_embd = 10240` | `include/strata/core/layout.hpp:55` |
| where the streams live | `R[t][h][:]`, *h* in [0,4) - the **middle** axis of the `(T, HC, N)` buffer | `src/core/verify.cpp:560` |
| where they come from | `broadcast_streams(emb_, R_, N, HC, T, cs)` at the window's start: all four streams begin as the *same* embedding | `src/core/verify.cpp:592`, `606`; `include/strata/kernels/verify_kernels.hpp:64-65` |
| how they diverge | each layer's attention half and FFN half compute a per-stream injection (`inj` / `inj2`, `T x HC` floats, `src/core/verify.cpp:410`) and the residual write is `R += bo * 2 sigmoid(inj / hc)`, per stream | `include/strata/kernels/cvec.hpp:55-58` |
| how the engine reduces them | the hyper-connection read reduces them with **learned weights**: `mixed[d] = mean_c xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)` - the same mixer serves the attention half, the FFN half and the head | `include/strata/kernels/fused_gr.hpp:13` |

**The capture's reduction is not the engine's reduction.** The upstream patch takes the *plain arithmetic mean* of the
four streams (`hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)`). The engine's only reduction of the
hc streams into an `n_embd` vector is the hyper-connection read's **learned weighted mean**. So the capture needs its
own reduction kernel - `cap[t][l][:] = (1/4) * sum_h R[t][h][:]` at five layers - and **there is no such kernel in the
tree**.

### 3.4 The weight/model loader

- The engine's loader is bespoke: `WeightTable::load` parses `<pack>/index.txt`, a **19-field space-separated**
  manifest written by `tools/pack_index.py` (`src/core/weights.cpp:144-188`). `include/strata/core/weights.hpp:20-27`
  says why there is no JSON parser in the engine: "a JSON parser in the engine would be a new, unaudited component
  whose failure mode is a wrong byte offset - which decodes to a plausible weight and produces plausible logits."
- The MTP drafter has a **second, separate** loader: `dense.txt` + `dense.bin` + `experts.bin`
  (`src/core/mtp.cpp:163-209`), produced by `tools/mtp_rt.py` (`src/core/mtp.cpp:165`).
- The DFlash 2 checkpoint is **neither**: one `model.safetensors` (1,206,140,008 B; an 8-byte little-endian header
  length, an 8,800-byte JSON header, 1,206,131,200 B of tensor data, **81 BF16 tensors** - read from the file's own
  header, 2026-10-22).

What a new model type would need:

1. **A safetensors reader** (JSON header + a tensor map + the offsets). The engine has none; `tools/gguf_reader.py`
   is a Python tool, not engine code.
2. **A draft geometry struct.** The draft's (5 layers, hidden 2560, 32 heads / 8 KV / head_dim 128, intermediate 7680,
   vocab 248320) is *not* the target's `ModelGeometry` (`include/strata/core/layout.hpp:27-59`), and the draft's tensor
   names (`layers.0.self_attn.q_proj.weight`) are not the engine's (`blk.<l>.*`, `include/strata/core/layout.hpp:72-83`).
3. **A shape contract** like `check_layer` (`include/strata/core/layout.hpp:91`) over the draft's 81 tensors, so a
   mismatch is a load-time error and not "a plausible weight" (the same argument as the weights header).
4. **The embedding and the head, borrowed from the target** - they are *not* in the checkpoint (verified: no
   `embed_tokens`, no `lm_head` among the 81). Upstream takes the target's embedding table and its `lm_head`
   (`dflash/model.py:135-144`, `599-605`). **Here both are quantized**: the embedding is an S2/S4/S8 tensor
   (`src/core/verify.cpp:594-606`) and the head is the engine's quantized `NativeHead`
   (`src/program/generate.cpp:3041-3044`). The published numbers come from an **FP8** target (model card,
   Evaluation), so the draft's inputs here would be materially coarser than the ones it was trained against. That is a
   first-order risk to the acceptance length, not a detail.

### 3.5 The drafter's VRAM reservation and the expert cache sizing around it

- The MTP is loaded **before** the cache is sized - `generate.cpp:3017-3031` versus the cache block at
  `generate.cpp:3215-3238` - because it takes ~0.9 GB that has to be subtracted. The comment at
  `generate.cpp:3015-3016` says why it is first: "Secure
  MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts. In particular WDDM can
  refuse the draft weights after mapping tens of GiB of host pages."
- The sizing itself (`src/program/generate.cpp:3219-3238`): `cudaMemGetInfo` (`3221`), then
  `reserve = ((vram_reserve_mib + prefill_mib) << 20) + mtp_bind` (`3231`), where
  `mtp_bind = mtp.bind_bytes(native_head.row_bytes(), n_vocab)` (`3229-3230`; `src/core/mtp.cpp:313-330`), then
  `slots = (free_b - reserve) / blob` (`3233`) against
  `strata::kernels::cpu::expert_layout().max_blob`.
- So a DFlash draft's **1.12 GiB** would have to be either loaded before this point or reserved explicitly. If it were
  simply added afterwards, the tier would be sized 1.12 GiB too large and the run would end below the box's **200 MB**
  free-VRAM floor (plan §0.1).
- The tier is `ExpertCache` (`include/strata/core/expert_cache.hpp`); its slots are the expert blobs, per-layer sizes on a
  native pack (`include/strata/kernels/cpu/expert_layout.hpp:35`), the single 1,382,400 B Q2_0 form otherwise
  (`include/strata/kernels/cpu/expert.hpp:45`).

### 3.6 The drafting call site and the window/commit logic it feeds

The decode round (the serve path; the batch path is the same shape at `src/program/generate.cpp:7231`):

| step | code |
| --- | --- |
| the window's rows are built | `window[0] = x`; `window[i] = drafts[i-1]` (`src/program/generate.cpp:8087-8094`) |
| the window is verified | `ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)` (`src/program/generate.cpp:8102`; `include/strata/core/verify.hpp:84`) |
| **acceptance** | `int a = 0; while (a < T - 1 && window[a + 1] == outv[a]) ++a;` (`src/program/generate.cpp:8113`) - **exact match, prefix only** |
| the commit | `ver.commit(a + 1, err)` (`src/program/generate.cpp:8134`; `include/strata/core/verify.hpp:129`) - the first `a+1` rows are committed, the rest discarded |
| the next draft | `mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) o.spec_min_p)` (`src/program/generate.cpp:8156`) |
| the policy | `DraftPolicy policy(o.spec)` (`src/program/generate.cpp:8054`); `policy.choose(T, k, sfx_match)` (`8077`) |

The controller's vocabulary would need a fourth source: `enum class Source { None, Lookup, Mtp }`
(`include/strata/spec/controller.hpp:40`), `K_MAX = 8` (`controller.hpp:23`), and a `CostModel` term for the DFlash
draft's per-round cost (`controller.hpp:25-38`).

**What fits without changing the contract.** The K-rows-verified-per-step semantics are already there (T rows, T-1
drafts, an accepted prefix, the rest discarded), and the exact-match acceptance rule accepts a *single* proposed path
exactly as it accepts the MTP's chain. **What changes is the acceptance semantics**: the selector proposes **one** path
- one token per position, the other 15 candidates at each position discarded
(`dflash/model.py:524-547`) - so the drafter's contribution is a *higher hit rate on one path*, not a wider candidate
tree. The engine has no tree-verification path to reuse, and none to break.

---

## 4. The integration surface, honestly

### 4.1 What has to be written

| # | piece | the job |
| --- | --- | --- |
| 1 | a safetensors reader, a BF16 arena, the draft's 81-tensor map | small but new; plus the shape contract of §3.4 |
| 2 | the masked parallel block pass: 5 layers x (RMSNorm -> conv `prepare` -> q/k/v -> RoPE -> **non-causal** sliding attention over `[ctx KV ; block]` -> o_proj -> conv `finish` -> residual -> RMSNorm -> conv `prepare` -> MLP -> conv `finish` -> residual) -> final norm | the whole model; the five layers are a loop (`dflash/model.py:586-597`) but every kernel is new |
| 3 | the **KV injection**: `H_t = RMSNorm(W_c [H^(5);H^(15);H^(25);H^(35);H^(45)])` (paper A.3), then per layer `K = [W_k H_t ; W_k H_d]`, `V = [W_v H_t ; W_v H_d]` | one 2560x12800 GEMM plus a per-layer K/V concatenation; the paper is explicit that the target features "bypass the draft model's Q projection, output projection, self-attention update, and FFN" (A.3) |
| 4 | the selector: top-16 over the draft's logits at every position, two 248320x256 codebooks, the greedy path trace (`dflash/model.py:515-547`) | a new kernel; the codebooks are *gathers* (1 row of the predecessor, 16 of the successor per position), not a full read |
| 5 | the capture hooks: the mean over the 4 hc streams at layers 5/15/25/35/45, inside the verify window's graph | a new reduction kernel (§3.3) plus five copies |
| 6 | the draft's own KV, in the draft's geometry: 5 layers x 8 KV heads x 128 dims, non-causal, a 2048 sliding window, and the **context entries accumulated across steps** (`dflash/model.py:202`, `247-250`, `396`) | a second ring, a second format; the paper says the projected features are "stored in the draft model's KV cache and reused across drafting iterations" (paper §4.1) |
| 7 | the target's embedding and head, borrowed | free in principle, quantized here (§3.4) |

### 4.2 What can be reused

- **A BF16 multi-row GEMM**: `bf16_gemv_fp32_mmvf_multi` (`include/strata/kernels/bf16_gemv.hpp:53`) - FP32
  activations, BF16 weights, **1..8 rows** (`src/kernels/cuda/native_bf16.cu:141`). It covers q/k/v/o/gate/up/down,
  `fc` (2560x12800) and the codebook projections. The block is 8 positions, so 8 rows fits exactly.
- **RMSNorm**: `rms_norm_weighted` (`include/strata/kernels/elementwise.hpp:82`), `native_qsa_rms_norm_weighted`
  (`include/strata/kernels/native_gr_norm.hpp:15`) for the per-head q/k norms.
- **RoPE**: `rope_neox_apply` / `native_rope_apply` (`src/core/mtp.cpp:515-516`) - but the draft's is a plain Qwen3 rope
  over all 128 dims with theta 10000 (`config.json`), while the target's is QSA's 64-rotary split; a separate check is
  needed.
- **The sampler**: `sample_tokens` (`src/core/mtp.cpp:642`), `row_top_prob` (`src/core/mtp.cpp:643`). But the selector
  is *not* a sampler - it is an argmax over a learned path score - so only the head-side of this is reusable.
- **The acceptance and the window/commit**: reusable as they are (§3.6).
- **The coupled draft sampling** (`include/strata/core/coupled_draft.hpp`): the losslessness machinery. The MTP can be
  sampled with the target's chain and the target's Philox draw; the DFlash selector's proposal is a *path* chosen by an
  argmax (or a sample) over a learned score (`dflash/model.py:535-540`), so exposing a per-position `q` for the
  rejection sampler - as upstream's `_rejection_sample` does (`dflash/model.py:104-105`) - is a design decision, not a
  reuse.
- **NOT reusable: the MTP's K/V ring.** `qsa_state_init` with a ring (`src/core/mtp.cpp:228-240`), `kv_restore`
  (`include/strata/core/mtp.hpp:56`) and the whole `QsaState` are the *target's* QSA geometry (24/2/256, block-sparse
  selection, `qsa_decode_attn_batch`, `src/core/mtp.cpp:552`). The draft needs dense, **non-causal**, 32/8/128 attention
  with a 2048 sliding window. **There is no such kernel here.**
- **NOT reusable, and this is the good news: the MTP's expert tier.** 708 MB of VRAM-resident experts
  (`src/core/mtp.cpp:189-209`) is the price of the MTP being an MoE layer. DFlash 2 is dense - it has no experts and no
  router at all - so if it *replaces* the MTP, that 708 MB comes back.

### 4.3 What does not exist here at all

- **No Python, no torch, no Marlin.** The engine is CUDA C++ (`src/kernels/cuda/`); Python appears only under `tools/`
  as offline packers (`tools/mtp_rt.py`, `tools/pack_index.py`). There is no Marlin kernel anywhere in the tree.
- **The engine's quant formats are its own IQ family plus Q8_0.** The pack's tensors are the IQ2_0/IQ3_S/IQ4_NL/IQ4_XS
  and Q4_0/Q5_0/Q8_0 forms, with the activation contract Q8_0 or Q8_K
  (`include/strata/core/weights.hpp:38-46`, `80-84`). The DFlash checkpoint is **BF16** and has no quantized
  counterpart here. It can be kept in BF16 - `bf16_gemv_fp32_mmvf_multi` reads BF16 weights - but then it is 1.12 GiB
  of VRAM with no IQ3_S-style option.
- **No safetensors reader.**
- **No non-causal attention** and **no hc-stream mean**.
- **No draft geometry**: `ModelGeometry` (`include/strata/core/layout.hpp:27-59`) describes the target only.

---

## 5. The acceptance-length question

### 5.1 The two numbers are not the same quantity

Upstream (model card, Evaluation): "Acceptance length is the per-request mean of completion tokens divided by
verification steps at concurrency 1", and "Both speculative methods propose three draft tokens per verification step."
So with gamma = 3, `tau = 1 + (mean accepted drafts)` and the **per-draft acceptance rate** is `p = (tau - 1) / gamma`:

| task | DFlash 2 tau | -> p | MTP tau | -> p | delta p |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 3.26 | 75.3% | 3.08 | 69.3% | +6.0 |
| MATH-500 | 3.17 | 72.3% | 2.93 | 64.3% | +8.0 |
| HumanEval | 2.94 | 64.7% | 2.52 | 50.7% | +14.0 |
| MBPP | 3.06 | 68.7% | 2.65 | 55.0% | +13.7 |
| MT-Bench | 2.83 | 61.0% | 2.50 | 50.0% | +11.0 |

The engine's number is the same kind of quantity at a different gamma and on a different harness. The sweep's
baseline rows (`<box>/UnbiasedStrata/logs/sweep-summary.csv`; aggregated in `docs/plans/specter-performance.md` §2.1):

| | C0-1 | C0-2 | C0-3 | C0 aggregate |
| --- | ---: | ---: | ---: | ---: |
| tokens/window | 2.42 | 2.42 | 2.42 | 2.42 |
| avg T | 3.13 | 3.08 | 3.08 | 3.10 |
| drafts ok / offered | 226/337 | 223/324 | 223/324 | 224/328 |
| **per-draft acceptance** | **67.1%** | **68.8%** | **68.8%** | **68.3%** |
| ms/window | 18.41 | 18.34 | 18.50 | 18.4 |
| tok/s | 131.7 | 132.1 | 131.0 | **131.6** |

So the engine's drafter, at T ~ 3.08 (gamma = 2.08), reaches a **68.3-68.8% per-draft acceptance** - within a point of
the upstream MTP's 69.3% on GSM8K at gamma = 3. **The engine's drafter is not broken.** And it is the MTP head that
does the work: the sweep's C7 row (`--suffix-draft 0`) is 132.5 tok/s, inside the noise floor, and its log says
"suffix drafts: 1 window, 1 of 3 drafts accepted" (plan §2.1). So this engine's drafter *is* the upstream MTP, at
the upstream MTP's acceptance level. The comparison DFlash 2 has to win is against a drafter that is already at the
upstream MTP's level.

The sweep's fourth configuration, `C3-per-layer`, is the one to keep in mind for §6: it changed the routing
configuration, the hit rate fell from 94.1% to **88.6%**, the CPU pool grew from 3.01 to **5.25 ms**, and tok/s fell
from 131.6 to **111.2** (-15.5%) - a loss from a change that did not touch the drafter at all. That is the shape of the
trap in §6.

### 5.2 The conversion, done carefully

The engine's arithmetic, from the sweep: `tokens/window = 1 + (T-1) * p` and
`tok/s = 1000 * tokens/window / ms_per_window`. Check C0-2: `1 + 2.08 * 0.688 = 2.43` (~2.42 measured);
`2.42 / 18.34 * 1000 = 132.0` (~132.1 measured). The model is right.

At the baseline's T = 3.08 and 18.4 ms:

| per-draft acceptance p | tokens/window | tok/s | vs 131.6 |
| ---: | ---: | ---: | ---: |
| 68.8% (today) | 2.43 | 132 | +0.4% |
| **75.3% (DFlash 2's best published, GSM8K)** | **2.57** | **139** | **+6%** |
| 68.4% (DFlash 2's mean over the five tasks) | 2.42 | 132 | +0% |
| 80% | 2.66 | 145 | +10% |
| 85% | 2.77 | 150 | +14% |
| 100% | 3.08 | 167 | +27% |

**DFlash 2's best published rate buys about +6% here; its mean published rate (68.4% over the five tasks) buys nothing at
all - it is the engine's own number.** The bar for a drafter to be worth building here is **85% per-draft
acceptance**, which would take the window from 2.42 to 2.77 committed tokens and the decode from 131.6 to ~150
tok/s (**+14%**) - ten points above anything DFlash 2 publishes.

Two caveats cut the other way, and both are large:

1. **The published tau is at gamma = 3; the engine's is at gamma = 2.08.** DFlash 2's whole claim is that it drafts a
   *block* in one pass, and the block positions after the first are exactly where a block-diffusion drafter is supposed
   to beat an autoregressive chain - the engine is currently only asking for two drafts. Running DFlash 2 at T = 4
   (gamma = 3) is the configuration upstream measured. The sweep says that at T = 4.07 *with the MTP* tok/s falls to
   122.9, but that is the MTP's per-draft acceptance decaying (69% -> 66%) as the chain lengthens, plus the CPU pool
   growing 3.01 -> 5.04 ms and the hit rate falling 94.1% -> 92.9% (plan §2.1). A drafter whose per-position acceptance does *not*
   decay would not pay the acceptance part of that. **The number to beat at T = 4 is 122.9 tok/s, not 131.6.**
2. **The published numbers come from a different target.** Model card: "Target: the official FP8 checkpoint", "Runtime:
   SGLang on one NVIDIA GB300 (TP1)", "Speculation block size: 4 (3 draft tokens per verification step), limited by the
   current QSA implementation". Here the target is IQ3_S, and the draft would borrow the *quantized* embedding and head
   (§3.4). A coarser head and embedding can only lower the draft's acceptance.

### 5.3 What would have to be measured before believing any of it

1. **The per-position acceptance profile, not the pooled rate.** The engine's acceptance is a *prefix* rule
   (`src/program/generate.cpp:8113`), so what matters is p_1, p_2, p_3 separately; the published `tau` collapses them.
   The good news: **the engine already computes the per-position match bits and throws them away** - the loop at
   `src/program/generate.cpp:8113` evaluates `window[a+1] == outv[a]` for every `a` and keeps only the first failure.
   Recording them is a one-line change.
2. **The draft's cost per round, measured on this box, not assumed.** The sweep's draft phase is 1.83-2.87 ms/window
   (C0 1.83-1.87; C1 2.79). A DFlash pass reads 5 x 168.8 MiB of layers + 62.5 MiB of `fc` + the target's head, **once**,
   for the whole block - against the MTP's 111 MiB of dense weights *plus* 708 MB of experts **per chain step**. Whether
   that is a win is a bandwidth question on this card, and it is measurable.
3. **The acceptance rate with a quantized head and embedding** - one line of change in the upstream code (BF16 head vs
   this engine's IQ3_S head) to see what it costs.
4. **The noise floor.** +-1.1 tok/s (0.8%) across processes on this box - the sweep's own baseline spread
   (131.7 / 132.1 / 131.0, plan §2.1). A +6% effect (~8 tok/s) is ~7x the noise floor and measurable in three runs; a
   +1-2% effect is not, and must not be claimed from one run.

---

## 6. The VRAM and time budget - and how the test is made to FIT

### 6.1 The budget

| item | size | note |
| --- | ---: | --- |
| the DFlash 2 checkpoint | **1.12 GiB** (1,206,140,008 B) | 81 BF16 tensors; 1.12 GiB of VRAM if kept in BF16 |
| of which the 5 layers | 844 MiB (168.8 MiB each) | |
| of which `fc` | 62.5 MiB | |
| of which the selector codebooks | 243.75 MiB | resident, but read by gather (a few KB per round), not streamed |
| the expert tier | **13.41 GiB**, 7,998 resident experts of 24,576 | measured at boot (the feasibility study's tier) |
| the MTP head | **839 MiB** (measured: 824 MiB) | of which 708 MB is the 512 resident experts (`include/strata/core/mtp.hpp:19`) |
| free VRAM at boot with the tier in place | **~750 MiB** (747-865 MiB measured during a run) | plan 0.1 |
| the box's VRAM floor | **200 MB** | plan 0.1 |

### 6.2 Making the test fit: three levers, and what each costs

The operator's instruction was to make the test fit by (1) offloading more experts to RAM (a smaller
VRAM tier) and (2) a lower KV quantisation. Both are available, and a third - which the first draft of
this document missed as a *lever* - is simply that **DFlash 2 replaces the MTP** rather than sitting
beside it.

| lever | what it frees | its cost |
| --- | --- | --- |
| **L1: DFlash 2 replaces the MTP** | the MTP's **839 MiB** (its 708 MB of resident experts + 111 MiB of dense) | the MTP's drafting goes away with it: 1.83-1.87 ms/window of the window's cost, and its acceptance (68.3-68.8% measured) has to be replaced by DFlash 2's. This is the point of the port, and also its risk. |
| **L2: a smaller VRAM tier** (more experts offloaded to RAM) | 1.12 GiB = **670 of the 7,998 slots** (the tier's average blob is 1.72 MiB) | the hit rate falls. The measured slope: the sweep's `C3-per-layer` row lost 5.5 points of hit rate (94.1% -> 88.6%) and 15.5% of tok/s; 670 slots is 8.4% of the tier, i.e. **~0.6 points** of hit rate (94.1% -> ~93.5%) and ~**-1.5%** of tok/s. |
| **L3: the KV at 4 bits** (`--kv q4_0`, the engine's rotated 4-bit codes) | the KV's per-cell cost halves again (int8 1,056 B -> q4 ~528 B); at `--kv-resident 32768` and 12 QSA layers that is **~620 MiB of VRAM** | upstream's own measurement for this mode: **~4% slower at 128K** and **+8-12% perplexity** (needle tests still pass) - a quality cost, not just a speed one. The engine refuses `q4_0` *together with* `--kv-resident` (the help says so), so L3 and the streaming KV are mutually exclusive. |

**The fit, with the arithmetic:**

```
free VRAM at boot                                      750 MiB
+ L1: the MTP's head, if DFlash 2 replaces it         +839 MiB
+ L3: the KV at q4 (if --kv-resident is given up)     +620 MiB
- the DFlash 2 draft                                 -1,124 MiB
-------------------------------------------------------
                                                     1,085 MiB free
                                                     = 885 MiB above the 200 MB floor
```

**So the test fits, with room, in three configurations**, in ascending order of cost:

1. **DFlash 2 replaces the MTP and the tier loses 165 slots** (2.1% of it, ~-0.3% of tok/s), the KV
   stays fp16/int8: 750 + 839 - 1,124 = **465 MiB free**, 265 MiB above the floor. *The cheapest
   configuration, and the one to try first.*
2. **The same plus the tier at 6,500 slots** (the sweep's `C9` configuration, for margin): ~0.9 GiB free.
3. **The same plus the KV at q4** (the sweep's `C10`): ~1.1 GiB free, at the price of the KV's quality
   cost.

**The sweep configurations that price L2 and L3 are `C8-kv4`, `C9-tier6500` and `C10-both`**
(`sweep.ps1`); their direct measurement is currently blocked - see 6.4.

### 6.3 The time budget

The sweep's window costs 18.34-18.50 ms at T = 3.08, of which the draft phase is 1.83-1.87 ms (10%).
A DFlash draft replaces that 1.83 ms with an unknown; at 3-4 ms it would be 16-22% of the window, and
it has to buy that back in tokens. The paper's argument that a block drafter's cost is largely
insensitive to gamma for moderate block sizes (paper 3.2) is what makes the swap plausible at all -
but it is a claim about an FP8 target on a GB300, not about this engine on this card.

### 6.4 The blocker on the measurement itself (not on the port)

The direct measurement of L2 and L3 is blocked by the **load's memory transient**, not by the port.
Measured on the box (available RAM every 5 s during one load; the floor is 2,048 MB):

```
09:00:33  52,387 MB
09:00:43  40,470 MB
09:00:48   9,563 MB
09:00:53   1,591 MB   <- below the floor; the watchdog kills the run here
09:00:58  19,620 MB
09:01:09  52,464 MB
```

The load dips ~1.6 GB below the floor for a few seconds and recovers, so the watchdog fires and the
engine dies mid-load. The cause is the arena's lock: `strata::platform::lock_resident`
(`src/platform/memory.cpp:18-55`) raises the *process's* minimum working set and `VirtualLock`s the
arena region, and **nothing bounds it** - `STRATA_ARENA_PIN_GIB` caps only `cudaHostRegister`
(`src/program/generate.cpp:3166-3174`), not this lock. The dip's depth also depends on the page
cache: the runs that succeeded had the 51 GB of the GGUF already cached, and the ones that failed
did not.

**So the measurement infrastructure needs one small engine change**: a cap on the locked bytes
(`STRATA_ARENA_LOCK_GIB`, the same shape as the existing pin cap), or a load that releases the lock as
it goes. Until then, every measurement on this box is at the mercy of the cache's state - which is
itself worth fixing, because the same transient is what leaves the box at 0 GB available in the
production configuration.

---

## 7. Risks, and the cheapest first step

### 7.1 The risks

| # | risk | where it bites |
| --- | --- | --- |
| 1 | **The capture has to be re-implemented.** The upstream patch targets SGLang's `python/sglang/srt/models/qwen4_exp.py` and is "validated at TP1" (model card, Quick Start); nothing in it applies to this engine. The equivalent is a new reduction kernel plus five copies in the verify window's graph. | §3.3, §4.1 |
| 2 | **The capture may not be what the draft was trained on.** The engine's "completed output" is the residual with the pending FFN write folded into the next read (`src/core/verify.cpp:635`, `696`; `include/strata/core/layer.hpp:461-465`; `include/strata/kernels/cvec.hpp:55-58`) - llama.cpp's formula, possibly not its instruction order. The draft is conditioned on **nothing else** but the capture, so a small arithmetic difference can move the acceptance. | §2.3, §8.1 |
| 3 | **The acceptance semantics change.** The selector proposes one path and discards the other 15 candidates per position (`dflash/model.py:524-547`). The engine's exact-match prefix acceptance is compatible with that, but the drafter contributes a *higher hit rate on one path*, not coverage over a tree. | §3.6 |
| 4 | **The MTP + a tuned `--spec-min-p` may get most of the win for free.** That is issue #2, and Track A is **closed**: ten configurations, no free win. `--spec-min-p 0.2` **lost** 8.9% (131.6 -> 119.9) and `--spec 6` **lost** 6.6% (plan §2.1). The cheap lever is not the window size and not the acceptance gate; what is untried is a drafter whose per-position acceptance does *not* decay - which is exactly DFlash 2's claim. | issue #2, plan §2.1 |
| 5 | **The quantized head and embedding.** The draft borrows the target's embedding and head (`dflash/model.py:135-144`); here both are quantized (`src/core/verify.cpp:594-606`, `src/program/generate.cpp:3041-3044`), and the published numbers are from an FP8 target. | §3.4, §5.2 |
| 6 | **The draft's KV injection into a recurrent target.** 36 of the target's 48 layers are Gated DeltaNet - recurrent (`include/strata/core/layout.hpp:32-38`). The capture is the *residual*, taken after the recurrent state has been updated, so it is well defined; but the draft's own attention (non-causal, 2048 window, 32/8/128) has no counterpart in the engine and no ring to inherit. | §4.2 |
| 7 | **1.12 GiB of VRAM that does not fit in the 750 MiB free.** | §6 |
| 8 | **The block size.** The checkpoint's `block_size` is 8 but the published runs used 4 ("limited by the current QSA implementation"). The engine's `kVerifyMaxT` and `K_MAX` are 8 (`include/strata/spec/controller.hpp:23`), so 8 would fit - but the *cost* of a wider window is what the sweep measured, and it is linear in T. | §5.2, §6 |

### 7.2 The cheapest first step

**Not the port. The measurement that decides whether the port is worth anything.**

The gate is issue #2 / A1: *can the existing drafter reach DFlash 2's acceptance?* Track A is **done** - ten
configurations, no free win, and L1 narrowed from "+40-60%" to "**+14% at 85% acceptance**" (plan §2.1). The sweep has
answered the two cheap versions of the gate (a wider window: no; a lower gate: no). It has **not** answered the version
that matters: the **per-position acceptance profile** of the existing drafter at gamma = 3, and what a drafter with a
*flat* profile would be worth here. Concretely:

1. **Record the per-position acceptance.** The match bits are already computed and discarded at
   `src/program/generate.cpp:8113`; keeping them costs one array and one increment per window. Run it at `--spec 4`
   (gamma = 3). If p_1, p_2, p_3 decay, the MTP's problem is structural and DFlash 2's premise holds here; if they do
   not, the premise is about the *block* and not about this engine. **This is an afternoon and it decides the port.**
2. **Re-run the same points to fix the noise floor on this configuration** - the +-1.1 tok/s is across processes on
   tok/s, and the acceptance histogram is a different statistic.
3. **Then, and only then, a no-engine feasibility spike of the acceptance itself.** Take the upstream `dflash`
   implementation, point it at the *same target* this engine runs (IQ3_S through llama.cpp, or FP8 to reproduce the
   published baseline), and measure p_1..p_3 for DFlash 2 and for the MTP under identical conditions. One Python
   process on the box, an afternoon, and it produces the one number the port's business case rests on: **DFlash 2's
   per-position acceptance at gamma = 3 against this target.** If it is at or below 75% (the published GSM8K best), the
   port buys about +6% here, and the answer is no.

The cheapest *code* step, if the measurement says yes, is not the draft model: it is the **capture** (§3.3), because it is
a few hundred lines, it is independent of the draft, and it is B2's feature source for the routing predictor (plan
Track B2) - a second use for the same work.

---

## 8. The open correctness questions

1. **Is the engine's "completed output" the same number as SGLang's `hidden_states` at the same layer?** The tap point
   is fixed (§2.3, §3.2: the capture of layer 5 is taken in `pre(6)`, after `gr_read_group(0, pending, ...)` at
   `src/core/verify.cpp:696`). What is *not* established is the arithmetic: the engine folds the pending FFN write into
   the next layer's read (`include/strata/core/layer.hpp:461-465`), and the fused-GR header says the summation order
   "differs from it (G-C judges the result)" (`include/strata/kernels/fused_gr.hpp:15`). llama.cpp's `build_hc_combine`
   computes the same formula; whether it computes it in the same order is **not established here**. This must be
   answered by measurement: dump the engine's residual at layers 5/15/25/35/45 and compare against llama.cpp's
   `output_hidden_states` for the same tokens, to a tolerance, not bitwise.
2. **Is the capture taken at the same point in the layer?** Upstream takes it at the end of the layer body, after
   `norms.output` (the patch's context). The engine's equivalent point is after the FFN half's write has been applied,
   which is `pre(l+1)`'s read - one layer later in program order but the same value. The GDN state update happens inside
   `post(l)` (`src/core/verify.cpp:700-720`), so the ordering matches; **this has not been checked line-by-line against
   SGLang's `qwen4_exp.py` loop body.**
3. **The draft's own attention geometry against the engine's.** 32 heads / 8 KV / head_dim 128 / a 2048 sliding window /
   non-causal / `rope_type: default` with theta 10000 (`config.json`), against the engine's QSA 24/2/256 with a 64-rotary
   split and block-sparse selection (`include/strata/core/layout.hpp:40-45`). They are different models' geometry; the
   draft's rotary has to be implemented separately, and its non-causal mask (`dflash/model.py:157-171`) has no counterpart
   here.
4. **The selector against the engine's coupled sampling.** Under sampling the engine keeps a draft only when it *equals*
   the target's pick for that row (`include/strata/core/coupled_draft.hpp:4-6`). The MTP proposes one token per position
   and can be sampled with the target's chain. DFlash 2's selector produces a *path* chosen by an argmax (or a sample)
   over a learned score (`dflash/model.py:535-540`), and the candidates it discarded are gone - so the draft's
   per-position probability is not a vocabulary distribution in the usual sense. Either the DFlash draft stays greedy
   (coupled mode off for it), or the selector's sampling distribution is exposed as the draft's `q`, as upstream's
   `_rejection_sample` does (`dflash/model.py:104-105`). **Undecided here.**
5. **The losslessness claim.** Model card: "Decoding is lossless: greedy output matches the target model exactly, and
   sampling preserves its distribution." The engine states the same contract the same way
   (`include/strata/core/coupled_draft.hpp:4-6`), so **no new machinery is needed for losslessness** - a well-formed
   proposal plus the engine's exact-match verification gives the guarantee the MTP has today. What is *not* claimed is
   that the proposal is *good*; that is the acceptance rate, and it is the thing to measure.
6. **The draft's KV cache across steps.** The reference implementation keeps a growing KV cache of the *context* entries
   (the projected target features) and crops the block's own entries every step (`_crop_to(past_key_values_draft, start)`,
   `dflash/model.py:250`), while the config says `use_cache: false`. The paper says the projected features are "stored
   in the draft model's KV cache and reused across drafting iterations" (paper §4.1). The reference implementation is the
   authority for what a port must do, and it says the draft keeps a KV cache that grows with the sequence - a second
   ring, in the draft's geometry. **Its eviction policy at 262,144 positions is not specified in anything I could
   find.**

