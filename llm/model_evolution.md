# ML Model Evolution

Tracks every meaningful change to the model architecture, loss function, and training setup, with the motivation and observed outcome for each.

---

## v1 — ARO Contrastive Baseline
**Script:** `scripts/aro_contrastive/`
**Loss:** InfoNCE + weighted TripletMarginLoss (`triplet_weight=40000`)
**Architecture:** EinsumModel (text) + TTNImageModel (image), no shared head
**Data:** COCO contrastive pairs (true/false caption per image)

The legacy `train_aro_clean.py` result ported to the new framework.
Triplet loss at weight 40000 is the dominant signal — InfoNCE with batch_size=128
provides structure but the triplet term drives discrimination.

**Result:** 78% hard_neg_accuracy on ARO test set (target benchmark).

---

## v2 — COCO Single-Caption, VICReg
**Script:** `scripts/coco_single_caption/`
**Loss:** VICReg (invariance=25, variance=25, covariance=1) + VICRegProjector heads
**Architecture:** EinsumModel + TTNImageModel → separate 3-layer MLP projectors (1024→256, BatchNorm, no final norm)
**Data:** COCO single-caption (one image–caption pair per row, no explicit negatives)
**Monitor metric:** `cosine_similarity` (computed on projector outputs — later identified as a bug)

**Motivation:** InfoNCE with explicit negatives requires hard-negative pairs. Single-caption
training has no negatives. VICReg is batch-size invariant and provides dense per-sample
alignment signal.

**Issues identified:**
- Projectors never entered eval mode (not registered in `ContrastiveVLM`).
- `cosine_similarity` metric was computed on projector outputs, not backbone — misleading.
- Projectors not checkpointed; best-epoch weights not restored for test evaluation.
- Val loss eventually failed with `RuntimeError: stack expects each tensor to be equal size`
  caused by 2D einsum outputs from old LMDB entries that bypassed `UnifyEinsumRankStep`.

**Result:** Peaked at `cosine_similarity≈0.26` on val at epoch 11, then collapsed.

---

## v3 — COCO Single-Caption, SymmetricInfoNCE (small dataset)
**Loss:** SymmetricInfoNCE (temperature=0.07, fixed)
**Architecture:** EinsumModel + TTNImageModel, no projectors, no shared head
**Data:** ~1k COCO samples
**Monitor metric:** `accuracy` (top-1 in-batch)

**Motivation:** With B=512 in-batch negatives, InfoNCE provides rich contrastive signal
without needing explicit hard negatives. VICReg was switched out.

**Issues identified:**
- Dataset was too small (~1k samples → ~2 val batches), metrics too noisy.
- Early stopping fired too aggressively on noisy `accuracy`.

**Result:** Near-random accuracy. Abandoned in favour of scaling the dataset.

---

## v4 — COCO Single-Caption, SymmetricInfoNCE (28k dataset)
**Loss:** SymmetricInfoNCE (temperature=0.07, fixed)
**Architecture:** EinsumModel + TTNImageModel, no projectors, no shared head
**Data:** ~28k COCO samples (batch_size=512, ~56 train batches / epoch)
**Monitor metric:** `accuracy` → switched to `loss` after early stopping issues

**Changes from v3:**
- Atlas scaled to 28k samples; derived parquets regenerated with 2D output filter.
- `monitor_metric` changed to `loss` (minimize=True) — val accuracy too noisy (only ~7 val batches).
- `image_lr` raised from 0.00005 → 0.0002 (4× increase to reduce LR imbalance).

**Issues identified:**
- Val loss decreasing but val accuracy/cosine_similarity showed no clear trend.
- Diagnosis: InfoNCE loss can decrease by improving log-probability of correct pair
  without making it the argmax — accuracy only improves when the correct pair beats
  all 511 competitors. Model was in the early regime of this gap.
- Temperature fixed at 0.07 provides too-sharp a distribution for early training.

**Result:** Train cosine_similarity slowly climbing (0 → 0.016 over 24 epochs),
val cosine_similarity near 0 and noisy. Model learning but very slowly.

---

## v5 — Add Learnable Temperature + Alignment Loss + New Metrics
**Loss:** SymmetricInfoNCE (learnable temperature) + per-sample cosine alignment loss (`alignment_weight=1.0`)
**Architecture:** EinsumModel + TTNImageModel, no shared head
**New metrics:** `alignment_gap`, `hard_neg_accuracy`, `sim_ratio`, `modality_gap`, `temperature`

**Changes from v4:**
- Temperature made learnable (`logit_scale = nn.Parameter(log(1/T))`), clamped to temp ≥ 0.01.
- Added per-sample alignment loss: `(1 - cos_sim(matched pairs)).mean()` — direct gradient
  independent of batch hardness, to bootstrap early alignment.
- Added batch-size-agnostic metrics to `SingleCaptionLoss`.

**Issues identified:**
- Temperature increased from 0.07 → 0.14 over 32 epochs (wrong direction). Model
  was choosing a softer distribution to avoid hard discrimination.
- Train modality_gap rose to 0.33 while val modality_gap stayed near 0 — alignment
  loss was memorising the training mean, not generalising.
- InfoNCE loss barely moved (6.22 → 6.21). Alignment loss dominated gradient signal.
- `alignment_loss` and `InfoNCE` are competing objectives on the backbone without a
  projector to isolate them — alignment loss pushes collapse, InfoNCE prevents it.

**Result:** No meaningful improvement. Alignment loss actively harmful without projectors.

---

## v6 — Add Per-Modality Alignment Heads
**Loss:** SymmetricInfoNCE (learnable temperature) + alignment loss (`alignment_weight=1.0`)
**Architecture:** EinsumModel + TTNImageModel + `AlignmentHead` per modality
**AlignmentHead:** `Linear(D, D) → L2Norm`, initialised as identity (no-op at epoch 0)
**Data:** ~28k COCO (batch_size=512)

**Changes from v5:**
- `ContrastiveVLM` now holds `image_head` and `text_head` as registered submodules,
  checkpointed and used at both train and eval time.
- Heads initialised as identity matrices — training starts from the same geometry as v5.
- Head parameters added to optimizer as a separate group (`head_lr=0.001`).
- `load_state_dict` and `clip_gradients` updated to include heads.

**Motivation:** Bridge the modality gap with a learned rotation per modality. Unlike
VICReg projectors: no BatchNorm, L2-normalised output, used at eval time, checkpointed.

**Issues identified:**
- `cosine_similarity` shot to ~0.95 and `modality_gap` to ~0.999 within 5 epochs —
  alignment head collapsed all embeddings to a single direction.
- `alignment_gap` near 0 and `hard_neg_accuracy` at random (0.0025) — no discrimination.
- `InfoNCE` stuck at 6.20. Root cause: `alignment_weight=1.0` dominates the gradient,
  alignment loss has no uniformity constraint and drives all embeddings to a single point.
- Alignment loss should be removed; heads trained on InfoNCE alone will bridge modality
  gap without collapse (InfoNCE inherently prevents uniform collapse).

**Result:** Representation collapse. All pairs have ~0.95 cosine similarity.

---

## v7 — Alignment Heads + Pure InfoNCE (current)
**Loss:** SymmetricInfoNCE (learnable temperature, bounded to [0.01, 0.3])
**Architecture:** EinsumModel + TTNImageModel + AlignmentHead per modality
**alignment_weight:** 0.0 (alignment loss removed)
**Temperature clamp:** scale ∈ [1/0.3, 100] → temperature ∈ [0.01, 0.3]

**Changes from v6:**
- `alignment_weight=0.0` — alignment loss disabled. InfoNCE alone trains the heads.
- Temperature upper-bounded at 0.3 (scale clamped from below at 3.33) to prevent
  the softening failure seen in v5 where temperature drifted to 0.14+.

**Hypothesis:** InfoNCE gradient naturally encourages matched pairs to be close AND
unmatched pairs to be far — the correct version of what the alignment head should learn.
Alignment loss was a shortcut that collapsed the solution.

**ARO evaluation (v6 checkpoint, epoch 7):**
- `hard_neg_accuracy: 0.500` (random baseline — confirmed collapse)
- `true_cosine_similarity: 0.963`, `false_cosine_similarity: 0.963`, `margin: -0.0004`
- 32% of test samples skipped (unknown symbols — EinsumModel vocabulary gap)

---

## Known Issues / Open Questions

- **EinsumModel vocabulary generalisation**: Val-only symbols receive no gradient during
  training. Their tensors remain near random initialisation, contributing to the
  train/val generalisation gap.
- **2D diagram filter**: Old LMDB entries bypass `UnifyEinsumRankStep`. Fixed by
  `filter_2d_outputs=True` in `enrich_atoms`. Parquets must be regenerated after adding
  new Atlas data.
- **Epoch time ~130s**: Dominated by EinsumModel forward (512 sequential cotengra einsum
  calls per batch). `num_workers=4` + `persistent_workers=True` applied (v4 onward).
  Next: cotengra path caching, epoch-level text embedding cache, `torch.autocast(bfloat16)`.
