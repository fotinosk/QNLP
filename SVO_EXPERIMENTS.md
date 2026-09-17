# SVO-Probes / SVO-Swap Experiments — Research Log

Task: SVO-Probes benchmark (DeepMind) — a caption paired with a correct
("positive") image and a negative image that differs in subject, verb, or
object. Also SVO-Swap: subject/object-swapped captions for pairs where both
referents are human/animal, evaluated against the fixed positive image.

Same architecture as the COCO campaign (see `COCO_EXPERIMENTS.md`):
DisCoCat/lambeq text encoder (`EinsumModel`), TTN image encoder
(`TTNImageModel`), symmetric InfoNCE loss, fixed temperature 0.07. Trained
and evaluated **independently of the COCO pipeline** — own dataset, own
checkpoint dir (`runs/checkpoints/svo_probes/`), own script
(`qnlp/scripts/svo/run.py`).

Random baseline for SVO-Probes accuracy: 0.50 (binary pos/neg image choice).

**Target numbers to beat: ~83% SVO-Probes, ~94% SVO-Swap.**

## ⚠ Data-versioning bug that invalidated two runs (found & fixed 2026-09-17)

`split_by_groups` (`qnlp/core/data_engine/dataset_creator/dataset_generator.py`)
shuffled `atoms[group_column].unique().to_list()` directly. Polars'
`.unique()` does not guarantee stable output order, so the "same seed"
shuffle was silently **non-reproducible across reruns** — confirmed in
practice: rerunning `prepare_datasets.py` with the same `seed=42` changed
the SVO test split's row count (1790→1786) and swap-pair count (52→74).

This caused real train/test contamination for **experiments 3 and 4**
(jobs 7426186, 7426189): both started training against one split
realization (call it split A), then `prepare_datasets.py` was rerun
mid-training (to build `svo_train_probes.parquet` for experiment 6) and
produced a *different* split (B) — overwriting `svo_test_probes.parquet`
and `svo_swap_eval.parquet` on disk. Both jobs' final evaluation ran
*after* that rerun, so they were scored against split B's test set while
having trained on split A's train set — some "test" images were very
likely images the model had already trained on as positives. This is the
actual explanation for their inflated scores (77-90%, in the target
ballpark) — **not a real result**. Their in-training val metrics (flat at
chance the entire time, before any resplit) are the reliable signal for
those two runs, not the final eval numbers below.

**Fix:** sort `unique_ids` before shuffling, making the split fully
deterministic for a given seed regardless of `.unique()`'s internal
ordering. This is a shared utility (also used by COCO/ARO/Winoground) —
low-risk, only changes *which* valid partition a seed produces.

Job 7426202 (experiment 6) is unaffected — it started training entirely
after the resplit and both trains and evaluates against the same split B
throughout. No pipeline rerun happened while it was in flight. It is the
first trustworthy read on the ARO-matched architecture. The data pipeline
will be rerun once more (with the fix, producing a final canonical split C)
after 7426202 finishes, so it isn't disturbed mid-run — any further
comparisons should wait for runs trained AND evaluated post-fix.

---

## Data pipeline

- Source: `data/svo/raw/svo_probes_corrected.csv` (Llama-3.2-3B corrected
  captions) + `data/svo/raw/images/` + `images_old/` (15,351 unique images
  downloaded, of 14,097 referenced — more than the original paper's 8,984).
- 36,841 raw rows → 26,189 with both pos/neg images available → 9,107 after
  the word-frequency filter (<50 occurrences dropped, matching the paper).
- CCG compilation parallelized via an SGE job array
  (`scripts/submit_svo_compile_array.sh`, 16 tasks × 2 workers) that
  pre-warms the LMDB cache before the main pipeline job runs — this
  cluster's nodes are mostly 4-8 cores, so a job array schedules far faster
  than one large multi-core request.
- Final split: **train 5,421 (59.5%) / val 1,896 (20.8%) / test 1,790
  (19.7%)**. subj/verb/obj proportions in val and test both track the full
  corpus (~20% / 56% / 24%).
- SVO-Swap: 52 pairs, built only from the test split's captions.

### Bugs found and fixed during pipeline construction

1. **Split collapsed to one giant component.** Grouping images by full
   pos+neg connectivity (via union-find, to guarantee zero image overlap
   between splits) chained 65% of all rows into a single component — SVO's
   subject/object-swap negatives are typically *borrowed* from some other
   row's positive image, so full connectivity fuses most of the dataset
   together. Produced an 87/7/6 split with only 1-5 subj_neg/obj_neg
   examples in val/test (unusable for a per-subset breakdown). **Fix:**
   group by `pos_image_id` only; allow negative images to repeat across
   splits (mild, one-directional leakage — a train positive can reappear as
   an eval negative elsewhere, which can only help the model correctly
   reject an already-well-modeled negative).
2. **SVO-Swap leaked training images.** It was built from the *full*
   26,189-row manifest rather than the test split — 59 of 408 unique swap
   images (14.5%) overlapped with training positives. **Fix:** restrict to
   `svo_test_probes.parquet`'s sample_ids before building.

---

## Experiments

### 1. svo_final — job 7426179 (baseline config)
**Script:** `submit_svo.sh`, defaults (`embedding_dim=256, bond_dim=10,
batch_size=128, text_lr=0.003, image_lr=0.0002, text_weight_decay=0.001,
image_weight_decay=0.05, head_weight_decay=0.001, max_epochs=50,
patience=10`).
**Checkpoint:** `runs/checkpoints/svo_probes/2026-09-17_00-45-44/best_model.pt`
(epoch 6).
**Observations:** Early-stopped at epoch 16. Train accuracy climbed
0.161 → 0.310 (epochs 5→7) while val accuracy stayed flat at chance
(0.009-0.012) with val loss *rising* the whole time (6.08 → 8.58 by epoch
13) — clear overfitting from the very first few epochs. Same failure mode
already documented for COCO (experiment 4 there): a large-vocabulary,
high-capacity model memorising a comparatively tiny training set (5,421
rows vs. COCO's 389k).

**Results:**
| SVO-Probes subset | N | acc |
|---|---|---|
| subj_neg | 329 | 0.5167 |
| verb_neg | 1054 | 0.5142 |
| obj_neg | 407 | 0.5086 |
| **overall** | 1790 | **0.5134** |

SVO-Swap: **0.7692** (52 pairs — small n, wide error bar; also a
structurally easier task since a swapped caption produces a completely
different word-order composition rather than a subtle image difference).

### 2. svo_final — job 7426184 (longer patience)
**Script:** `submit_svo.sh` with `SVO_ML_MAX_EPOCHS=100,SVO_ML_PATIENCE=20`,
otherwise same config as experiment 1.
**Observations (through epoch 14, still running):** Same pattern, more
extreme — train accuracy reached 0.706 by epoch 14 while val accuracy never
moved off chance (0.007-0.013) from epoch 1 onward, and val loss climbed
monotonically (5.25 → 8.50). Confirms more patience alone does not help;
the model is not lacking training time, it's lacking generalisation
capacity/regularisation for this dataset size. Best checkpoint still epoch
6 (val acc 0.0126, essentially identical to experiment 1's best).
**Final results:** Early-stopped at epoch 26. SVO-Probes overall 0.5190
(subj 0.5046 / verb 0.5123 / obj 0.5479) — essentially unchanged from
experiment 1, confirming more patience alone does not help. SVO-Swap 0.5962
(down from experiment 1's 0.7692 — noisy given n=52, attributable to which
epoch's checkpoint happened to be selected as best, not a real trend).

### 3. svo_final — job 7426186 (reduced capacity + more regularization)
**Script:** `submit_svo.sh` with `SVO_ML_EMBEDDING_DIM=128,
SVO_ML_TEXT_LR=0.001, SVO_ML_IMAGE_LR=0.0001, SVO_ML_TEXT_WEIGHT_DECAY=0.01,
SVO_ML_IMAGE_WEIGHT_DECAY=0.15, SVO_ML_HEAD_WEIGHT_DECAY=0.01,
SVO_ML_MAX_EPOCHS=100, SVO_ML_PATIENCE=20`.
**Rationale:** Experiments 1-2 show textbook overfitting (train accuracy to
70%+, val flat at chance from epoch 1). The image tower alone has ~2.4M
parameters against only 5,421 training rows. Halving embedding_dim cuts
model capacity, lower learning rates slow memorisation, and higher weight
decay (particularly on the image tower, whose weight_decay was the
weakest-regularised parameter group) directly penalises it. Not sweeping
these individually — with a training set this small, one run per
variable is expensive relative to the signal, and the three effects are
expected to compound rather than conflict.
**Note:** did *not* try widening `bond_dim` — already ruled out in the
COCO campaign, did not help there.
**Observations (through epoch 29, still running):** Reduced capacity and
higher regularization did NOT help — if anything, train accuracy reached
89.1% by epoch 29 (highest of any run so far) while val stayed flat at
chance (0.005-0.007) the entire time. This was the strongest evidence yet
that the problem isn't capacity or regularization: it's the total absence
of hard negatives during training (only in-batch random negatives), which
motivated the architecture switch in experiment 6.
**Results:** ⚠ **INVALID — see the data-versioning bug above.** Final eval
reported SVO-Probes overall 0.7968 / SVO-Swap 0.7973, but this run trained
against split A and was evaluated (after `prepare_datasets.py` was rerun
mid-run) against split B — likely train/test contamination, not a real
result. The trustworthy signal from this run is its own in-training val
metrics: flat at chance (0.005-0.007) through epoch 29, consistent with
every other pre-fix run.

### 4. svo_final — job 7426189 (alignment loss, isolated)
**Script:** `submit_svo.sh` with `SVO_ML_ALIGNMENT_WEIGHT=0.5`, otherwise
identical to experiment 1's baseline config (embedding_dim=256, default
LRs/weight_decay, max_epochs=50, patience=10).
**Rationale:** `alignment_weight` (weight of the per-sample cosine
alignment loss relative to InfoNCE) was hardcoded to 0.0 in `run.py` —
newly exposed as a config field (`SVO_ML_ALIGNMENT_WEIGHT`) so it could be
tried. Deliberately isolated against the unmodified baseline rather than
combined with experiment 3's capacity/regularization changes, to get a
clean read on what this term does on its own before combining levers.
**Caution:** a *warmup* schedule for this (weight 0.5 for the first 5
epochs, then presumably dropped) caused embedding collapse in the COCO
campaign (modality_gap → 1.0, R@1 random). A constant weight held
throughout training is a different, untested experiment — watch for the
same collapse signature (modality_gap climbing to ~1.0) early on.
**Results:** ⚠ **INVALID — see the data-versioning bug above.** Same issue
as experiment 3: trained against split A, evaluated against split B after
the mid-run resplit (final eval reported 0.7671 / 0.8919). No collapse
signature was observed in-training, but the final numbers cannot be
trusted either way.

### 5. svo_final — job 7426191 (alignment loss, extreme weight)
**Script:** `submit_svo.sh` with `SVO_ML_ALIGNMENT_WEIGHT=1000`, otherwise
identical to experiment 4 (baseline config).
**Rationale:** At weight=1000, `loss = infonce_loss + 1000 * alignment_loss`
is almost entirely dominated by the alignment term (range [0,2]) — InfoNCE's
contrastive/hard-negative signal becomes negligible by comparison. Tests
"pure per-sample alignment, no contrastive pressure" as a distinct point
from experiment 4's moderate weight, on the hypothesis that in-batch
negative mining itself (not just capacity) may be a driver of the
memorization seen in experiments 1-2.
**Results:** Early-stopped at epoch 12. SVO-Probes overall 0.5156 (subj
0.5106 / verb 0.5161 / obj 0.5184) — no better than baseline, extreme
alignment weighting doesn't fix the image-discrimination task. SVO-Swap
**0.8269** — best SVO-Swap result so far (though n=52, noisy). Consistent
with alignment loss being a per-sample *caption-image* similarity signal:
it can plausibly sharpen basic caption-image plausibility (helping reject
a wildly different swapped caption) without adding anything that helps
rank two visually similar candidate images against the same caption.

### 6. svo_final — job 7426202 (match legacy ARO architecture)
**Script:** `submit_svo.sh`, no env overrides — `SVOExperimentConfig`
rewritten to match `qnlp/scripts/aro_contrastive/config.py` field-for-field:
`embedding_dim=512, bond_dim=10, batch_size=128, text_lr=0.001,
image_lr=0.00005, text_weight_decay=0.001, image_weight_decay=0.05,
head_lr=0.001, head_weight_decay=0.001, max_epochs=100, patience=10,
temperature=0.07 (fixed), triplet_weight=40000.0, triplet_margin=0.2,
distance=cosine`.
**Rationale:** Experiments 1-3 all show the same overfitting signature
regardless of capacity, regularization, or LR — strong evidence the actual
problem is architectural, not tuning: training so far has only ever used
plain in-batch InfoNCE, i.e. *zero explicit hard negatives during
training* despite SVO-Probes being a hard-negative benchmark by
construction. The legacy ARO pipeline (`qnlp/scripts/aro_contrastive/`,
78% hard_neg_accuracy reported in `llm/model_evolution.md`) trains directly
on `(anchor, positive, negative)` triples via InfoNCE + a heavily-weighted
(`triplet_weight=40000`) triplet margin loss — COCO's config was never
built for this since COCO has no hard negatives.
**Implementation:** new `ImageContrastiveLoss`
(`qnlp/core/training/losses/image_contrastive.py`) mirrors ARO's
`ContrastiveLoss` with image/caption roles swapped (SVO's hard negative is
the image, not the caption); new `SVOHardNegStep`
(`qnlp/scripts/svo/step.py`) mirrors `AROContrastiveStep`, running two
forward passes (one per candidate image) since `ContrastiveVLM.forward`
only embeds one image against up to two captions. Training now happens
directly on `svo_train_probes.parquet` (true/false image triples, 5,437
rows) instead of positive-only pairs — the same shape already used for
val/test evaluation. Monitor metric switched from `accuracy` to
`hard_neg_acc` to match ARO's convention. Model architecture itself
(EinsumModel + TTNImageModel + ContrastiveVLM) is unchanged — confirmed
identical to what `aro_contrastive/run.py` uses; only the loss, step, data
shape, and hyperparameters changed.
**Valid** — trained and evaluated entirely on split B, no cross-split
contamination (unlike experiments 3-4).
**Observations:** Train `hard_neg_acc` climbed fast and cleanly (0.52 →
0.84 by epoch 10), but val `hard_neg_acc` never moved off chance the
entire run (0.522 at epoch 1 — also the best epoch — drifting between
0.48-0.53 through epoch 10). Early-stopped at epoch 11 (best=epoch 1,
patience=10).
**Results:** SVO-Probes overall **0.4826** (subj 0.4569 / verb 0.4971 /
obj 0.4677) — the *worst* clean SVO-Probes result of any run so far,
slightly below chance. SVO-Swap 0.6216.
**Conclusion:** giving the model explicit hard negatives during training
(matching ARO exactly) did not fix generalization — it just gave the model
a very effective way to memorize the *specific* negative shown per
training example (hence the clean, fast train-accuracy climb) without any
of that transferring to held-out images. This suggests the core problem is
more fundamental than "missing hard negatives": most likely insufficient
training data (5,437 rows) for this model's capacity (~5M combined
text+image params) to generalize, regardless of loss formulation. Next
candidates: the `AlignmentHead`'s own learnable parameters (~525K total,
identical in ARO — not yet tested as an ablation), or accepting that this
architecture/dataset-size combination may not close the gap to the
83%/94% targets without either more data or a smaller model.

### 7. svo_final — job 7426415 (triplet_weight=100)
**Script:** `submit_svo.sh` with `SVO_ML_TRIPLET_WEIGHT=100`, otherwise
identical to experiment 6.
**Rationale:** At `triplet_weight=40000`, the triplet term (~0.1-0.3) totally
swamps `infonce_loss` (~5-6) — effectively the model is only ever trained
against ONE FIXED (caption, false_image) pair per training row, every
single epoch, since the InfoNCE term's diverse in-batch random negatives
contribute negligible gradient by comparison. That's a plausible
memorization mechanism distinct from "not enough data": the model never
sees varied negatives for a given caption. `triplet_weight=100` puts the
two terms on comparable scale (100 * 0.2 ≈ 20 vs infonce ~5-6), so the
diverse in-batch signal should actually contribute alongside the explicit
hard negative.
**Results:** _(pending)_
