# COCO Retrieval Experiments — Research Log

Task: image-text retrieval on COCO using a quantum-inspired tensor network VLM.
Text encoder: DisCoCat/lambeq tensor networks (CCG-parsed einsum contractions).
Image encoder: TTN (Tree Tensor Network).
Loss: symmetric InfoNCE (in-batch negatives), fixed temperature 0.07.
Metric: R@1/R@5/R@10 image-to-text and text-to-image retrieval.

Random baseline: ln(batch_size) ≈ 6.2 (batch=512).

---

## Vocabulary / sparsity context

~68,922 unique symbols across 389k training pairs.
Average ~5-6 training occurrences per symbol.
Symbol names follow `{lemma}_{index}__{CCG_type}` (e.g. `lick_1__B.r@s@B`).
Tensor shapes (native, before remap): nouns [512], verb cores [10, 512, 10], bond connectors [10, 512] etc.
Remap at training time: `{constants.embedding_dim → cfg.embedding_dim, constants.bond_dim → cfg.bond_dim}`.

---

## Experiments

### 1. coco_single_caption — linear baseline
**Script:** `submit_coco_train.sh`, `submit_coco_single_caption.run`
**Model:** `EinsumModel`, linear contractions, no NLC gate.
**Config:** embedding_dim=256, bond_dim=20, batch_size=512.
**Dataset:** `coco_single_caption` (no precomputed paths).
**What it does:** Straight einsum contraction via cotengra; each word tensor is an independent parameter.
**Observations:** —

### 2. coco_single_caption — frozen image tower (early experiment)
**Script:** `submit_coco_train_frozen.sh`
**Model:** `EinsumModel` linear, frozen TTN image encoder.
**Config:** bond_dim=30, NLC=false.
**What it does:** Tests whether the text encoder alone can learn without joint image training. Frozen image tower reduces co-adaptation.
**Observations:** —

### 3. coco_single_caption — NLC (non-linear contractions)
**Script:** `submit_coco_train_nlc.sh`
**Model:** `EinsumModel`, NLC enabled, global scalar gate.
**Config:** embedding_dim=256, bond_dim=20, batch_size=256.
**Dataset:** `coco_single_caption_nlc` (with precomputed opt_einsum paths).
**What it does:** Adds a residual non-linearity after each pairwise contraction: `output = contraction + gate * GELU(contraction)`. Gate is clamped ≥ 0.1 and initialised at 0.1.
**Observations:** Gate stuck near zero; single global gate faces conflicting gradients from sentences of varying length/structure.

### 4. coco_multi_caption — NLC (job 6977205, best known run)
**Script:** `submit_coco_multi_caption.sh`
**Model:** `EinsumModel`, NLC enabled, global gate.
**Config:** embedding_dim=256, bond_dim=20, batch_size=256, NLC=true.
**Dataset:** `coco_single_caption_nlc` (multi-caption training = all 5 captions per image as separate rows).
**What it does:** Same NLC model but trained with 5x more rows; deduplication applied at val/test time for correct retrieval eval.
**Observations:** Overfitting from epoch 9 onward (val loss rises while train loss drops). Best checkpoint at epoch 4 had zero retrieval. text_weight_decay=0.001 too weak. Gate saturated. Modality gap grew. Root cause: 68k independent tensors memorise training pairs. Suggested fix: increase text_weight_decay to 0.05.

### 5. coco_multi_caption — frozen image tower
**Script:** `submit_coco_multi_caption_frozen.sh`
**Model:** `EinsumModel`, NLC, frozen TTN.
**Config:** embedding_dim=512, batch_size=256, NLC=true.
**What it does:** Frozen image encoder variant of experiment 4.
**Observations:** FAILED (job 6979930). NLC gate became NaN during epoch 1 backward passes (gradient explosion at embedding_dim=512). `nonlinear_gate: nan` reported at end of epoch 1 train. Epoch 1 val was empty `{}` (gate already NaN → 100% of val samples dropped). Epoch 2 onward: all train metrics empty, completed in 20 min instead of ~70 min, confirming every batch was instantly skipped. Root cause: larger tensors at embedding_dim=512 produce larger contraction magnitudes → gradient explosion on the scalar NLC gate. Fixed in submit script: `ML_MAX_GRAD_NORM=0.1` added. Ready to resubmit.

### 6. coco_multi_caption_lr — left-to-right contraction path
**Script:** `submit_coco_multi_caption_lr.sh`
**Model:** `EinsumModelLR` (inherits EinsumModel), always NLC=true, overrides path to fold-left order.
**Config:** bond_dim=20, batch_size=256.
**What it does:** Replaces opt_einsum path with linguistically-motivated left-to-right fold: `[(0,1)] + [(0, n-k) for k in range(2,n)]`. The hypothesis was that left-to-right matches human reading order.
**Observations:** ABANDONED. Left-to-right order does not avoid large intermediates — ~35% of sentences exceeded MAX_INTERMEDIATE_ELEMENTS (50M elements) and were skipped as NaN. The opt_einsum path was specifically chosen to minimise intermediate size; LR has no such guarantee.

### 7. coco_multi_caption — MLP head
**Script:** (via `ML_USE_MLP_HEAD=true`)
**Model:** `EinsumModel` NLC + two-layer MLP projection head.
**Config:** bond_dim=20.
**What it does:** Stacks a non-linear MLP on top of the NLC text model to add expressivity after contraction.
**Observations:** FAILED. Stacking two non-linearities (NLC gate + MLP) produced conflicting gradient signal. Gate oscillated ±0.06 for 11 epochs. Best epoch was epoch 1. Zero retrieval. Abandoned.

### 8. coco_multi_caption_expressive — per-symbol gates + word bypass
**Script:** `submit_coco_multi_caption_expressive.sh`
**Model:** `EinsumModelExpressive` (inherits EinsumModel).
**Config:** bond_dim=20, batch_size=256, NLC=true, MLP=false.
**What it does:** Two additions over baseline NLC:
  1. Per-symbol NLC gates (instead of one global gate). Each symbol has a scalar gate; when two tensors contract the effective gate is their arithmetic mean; intermediates inherit recursively.
  2. Word bypass: per-symbol embedding mean-pooled and added to contraction output before L2 normalisation, giving every word a direct path to the sentence embedding.
  Both initialised at zero so model starts equivalent to base NLC.
**Observations:** 58 min/epoch (vs 25 min baseline) due to `torch.tensor(gate_indices, device=x.device)` call per sample in Python loop. Only 2 epochs of data seen as of 2026-06-20; cosine_similarity rising. Results pending.

### 9. coco_single_caption_cp — CP decomposition
**Script:** `submit_coco_single_caption_cp.sh`
**Model:** `EinsumModelCP` (standalone nn.Module, no NLC).
**Config:** embedding_dim=512, bond_dim=10, cp_rank=4, batch_size=512.
**Dataset:** `coco_single_caption` (linear, no stored paths).
**What it does:** Replaces each word's full parameter tensor with a CP (Canonical Polyadic) decomposition of rank R. A tensor of shape [d0, d1, ..., dk] is approximated as a sum of R outer products of k vectors — one per dimension. For a verb core [10, 512, 10] at R=4: 2,128 params instead of 51,200 (24x reduction). Vectors materialised at each forward pass via einsum.
**Motivation:** Reduce parameters per symbol to combat sparsity (68k symbols seen ~5 times each).
**Limitation:** Still 68k independent symbols — CP reduces params per symbol but does not share information across related symbols (e.g. "lick" noun vs "lick" verb are still independent).
**Observations:** Not yet run (as of 2026-06-20).

### 10. coco_single_caption_tied_noun — tied noun embeddings
**Script:** `submit_coco_single_caption_tied_noun.sh`
**Model:** `EinsumModelTiedNoun` (standalone nn.Module, no NLC).
**Config:** embedding_dim=512, bond_dim=10, bond_rank=4, batch_size=512.
**Dataset:** `coco_single_caption` (linear, no stored paths).
**What it does:** Directly addresses sparsity by sharing a single 512D noun embedding across ALL grammatical forms of the same lemma. Lemma parsed from symbol name via regex `{lemma}_{index}__{type}`. Each symbol's tensor materialised as:
  - 1D noun `[512]`: IS the shared noun embedding (no extra params).
  - 2D `[512, 10]` or `[10, 512]`: `outer(noun, v)` or `outer(v, noun)` where v ∈ R^10.
  - 3D `[10, 512, 10]`: `noun[s] * M[a,c]` where `M = sum_r u_r ⊗ v_r` is a rank-4 bond matrix.
  Bond factors (u, v) are learned per-symbol; the 512D semantic core is shared across all forms of the lemma.
**Motivation:** Every sentence containing "lick" in any role (noun, verb, adjective) contributes gradient to the same noun embedding, multiplying effective sample count ~4x. Bond factors encode syntactic role independently.
**Parameter comparison per verb lemma (bond_dim=10, embedding_dim=512):**
  - Full tensors: ~112,640 params
  - CP rank-4: ~2,800 params (independent)
  - Tied noun: ~600 params (512 shared + ~20 bond params per symbol form)
**Observations:** Not yet run (as of 2026-06-20).

### 11. coco_single_caption_frozen — frozen CLIP tower, NLC (emb=512, bond=10)
**Script:** `submit_coco_single_caption_frozen.sh`
**Model:** `EinsumModel`, NLC=true, frozen CLIP ViT-B/32 image encoder (in-memory cache).
**Config:** embedding_dim=512, bond_dim=10, max_grad_norm=0.1, batch_size=256.
**Dataset:** `coco_single_caption_nlc` (selected automatically when NLC=true).
**What it does:** Only the text model and a linear text head are trained. Images encoded once with CLIP ViT-B/32 and cached in memory — subsequent epochs are instant lookups. max_grad_norm tightened to 0.1 to prevent the NLC gate gradient explosion seen in job 6979930.
**Observations:** FAILED — random performance on all benchmarks. Best checkpoint was epoch 2 (early stopping). true/false cosine similarities were ~-0.0016 throughout, meaning text embeddings are completely uncorrelated with CLIP image embeddings — the text model learned nothing.
  - ARO overall: 0.4991 (attribution 0.5007, relation 0.4972) — chance is 0.50
  - SugarCREPE swap_obj: 0.4805 — chance is 0.50
  - Winoground: text 0.2555, image 0.2372, group 0.1423 (60 pairs skipped)
  Root cause: early stopping at epoch 2 — the text model (68,922 symbols × 512-dim) has too many parameters for the sparse COCO vocabulary (~5 examples per symbol) to learn anything meaningful in 2 epochs. With embedding_dim=512 the NLC contractions produce larger tensors and more sentences are skipped as NaN, making training signal even noisier.

### 12. coco_single_caption — final linear run (emb=512, bond=10, lr=0.003)
**Script:** `submit_coco_single_caption_final.sh`
**Model:** `EinsumModel`, linear contractions, no NLC.
**Config:** embedding_dim=512, bond_dim=10, text_lr=0.003, batch_size=256.
**Dataset:** `coco_single_caption`.
**What it does:** Baseline linear run at native dataset dimensions (no remapping) with a higher text learning rate. embedding_dim=512 and bond_dim=10 match the dataset's native shapes exactly, so no tensor resizing occurs. Higher text_lr (0.003 vs 0.001 default) to accelerate symbol tensor learning given the sparsity constraint (~5-6 examples per symbol).
**Observations:** Not yet run (as of 2026-06-20).

### 13. coco_aro_style — ARO pipeline with TF-IDF hard negatives (2026-06-21)
**Scripts:** `qnlp/scripts/coco_aro_style/`, `scripts/submit_coco_aro_style.sh`
**Model:** `EinsumModel` NLC + `ContrastiveVLM`, same architecture as all previous runs.
**Config:** embedding_dim=512, bond_dim=10, batch_size=128, NLC=true.
**Dataset:** `coco_aro_style_{train,val,test}.parquet` — created by `create_dataset.py`.
**What it does:**
  Root cause of all prior COCO failures: random-derangement negatives gave contradictory
  gradients because "negative" captions from other images often described very similar scenes.
  This run fixes the negative quality problem by using TF-IDF cosine similarity to find
  captions that share vocabulary with the true caption but come from different images —
  i.e., genuinely confusable but definitively wrong. Exactly the ARO signal applied to COCO.

  Key changes vs. all prior COCO runs:
  1. Loss: InfoNCE + triplet (weight=40k, margin=0.2) instead of InfoNCE alone.
     The triplet term directly pushes sim(img, true) > sim(img, false) with a margin.
  2. Negatives: TF-IDF top-10 most similar from different images (not random derangement).
  3. Monitor: hard_neg_acc — binary ranking signal, same as ARO.
  4. Image augmentation: RandomCrop + ColorJitter + flip (ARO-style, not fixed Resize).

  Dataset creation:
    `python -m qnlp.scripts.coco_aro_style.create_dataset` (local, ~20 min for 184k rows)
    Strategies: --strategy bm25_hard (top-10), bm25_medium (top-50), random (ablation)

  Evaluation (after training):
    `python -m qnlp.scripts.coco_multi_caption.evaluate <checkpoint>` (Winoground/ARO/SugarCREPE)
    COCO val retrieval (R@1/R@5/R@10) runs automatically at end of training.
**Observations:** Not yet run (as of 2026-06-21).

---

## What has been ruled out

- **Wider bond dimension**: tried, did not improve results. Do not suggest again.
- **Hard negatives (retrieval-model-based)**: previously rejected — too complex. Now implemented differently via TF-IDF (see experiment 13).
- **Small batch size (2–4)**: rejected — InfoNCE with batch_size=2 gives only 1 negative per sample; loss signal is too weak.
- **MLP head on top of NLC**: failed (see experiment 7).
- **Left-to-right contraction path**: abandoned (35% sentence skip rate).
- **Tree grammar / dependency parsing**: architecturally cleaner but requires full rewrite of data pipeline; deferred.

---

## Open questions

- Will tied-noun embeddings produce measurable retrieval improvement over CP baseline?
- Does the expressive model (per-symbol gates + word bypass) eventually converge or is 58 min/epoch a blocker?
- Should text_weight_decay be increased to 0.05 to combat overfitting in NLC models?
- Can lemmatisation (grouping inflected forms: run/runs/ran/running → run) further reduce effective vocabulary beyond what tied-noun achieves?

---

## 2026-07-19 — Training slowness root cause found: sequential tail-batch contractions

**Question:** Why did the TopologyBucketSampler + batched-contraction fast path not reduce epoch time (still 40–70 min)?

**Answer (evidence-based, from cluster logs + local measurement):** The ~20% "tail" rows (88,335 rows across 28,290 rare diagrams, per the sampler's own log) were still contracted **one opt_einsum call per sample** (~4–10 ms each, dispatch-bound), costing ~35–40 min/epoch — the entire epoch. Key evidence: frozen **val** (deduped ~11K mostly-tail rows, no backward, images cached) took only ~40 s, i.e. ~4 ms/sample, which scaled to the 88K-row train tail with backward reproduces the full epoch time. Fast-path batches cost seconds in total. Ruled out: data loading/deserialization (`num_workers=0` frozen loader measured at 0.0075 ms/row → 0.06 min/epoch), fp64 (prior benchmark), diagram-string canonicalization (expressions already canonical: 28,996 unique before and after).

**Fix (landed on `coco-training-nlc`, lossless):**
1. `TopologyBucketSampler` now lays out tail rows grouped by diagram (group-level shuffle) so same-diagram rows are contiguous within mixed batches.
2. `EinsumModel.forward` now contracts heterogeneous batches **per same-diagram group** (one batched call per unique diagram in the batch) instead of per sample; singletons keep the single path. New counters: `fast_path_samples`/`fallback_samples`, logged by all 4 run scripts.

Verified: forward bit-identical (0.0 diff) and grads at fp32 epsilon vs the per-sample loop; sampler coverage/contiguity/no-size-1-batch invariants hold. Expected: tail contraction calls per epoch 88,335 → ~28,290 (8,809 multi-groups, mean multiplicity 7.8, + 19,481 singletons) ≈ 3.1× fewer; predicted epoch ~40 min → ~15 min (frozen). Remaining floor is the 19,481 singleton rows (~8 min); the designed `tail_fraction < 1.0` rotation knob is the next lever if needed. Awaiting a short cluster run to confirm.

**Follow-up optimization (same day):** `_prepare_rescaled_weights` — each `forward()` now rescales + fp64-casts every *unique* symbol in the batch once, via a few shape-grouped batched ops, instead of per occurrence inside every contraction call (~2/3 of each dispatch-bound call's ops were this preprocessing). Threaded through both the batched and single paths as a `prepared` dict; valid only within one forward (never cached across optimizer steps); NLC untouched. Verified bit-identical forward (0.0 diff on mixed/pure/B=1 batches, fp32 output dtype preserved), grads at fp32 accumulation noise (1.5e-5). Measured on CPU tail batches: 54.0 -> 23.4 ms/batch (**2.3x** vs pre-change per-sample path, grouping + prepared combined); GPU gain expected larger since the removed ops were pure dispatch. Revised prediction: frozen epoch ~40 min -> **~6-10 min**.
