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

## Baseline protocol (adopted 2026-09-17)

Going forward: one fixed **current baseline** config, every new experiment
is a single, named deviation from it, and the baseline only moves when a
deviation demonstrably beats it. This replaces the earlier more ad-hoc
practice of comparing against whichever prior run seemed most relevant.

**Current baseline (updated 2026-09-17 — see correction below): the
ARO-matched hard-negative architecture with corrected defaults**
(`embedding_dim=512, bond_dim=10, batch_size=128, text_lr=0.001,
image_lr=0.00005, text_weight_decay=0.001, image_weight_decay=0.05,
max_epochs=100, patience=10, temperature=0.07, triplet_weight=40000.0,
triplet_margin=0.2, distance=cosine, use_non_linear_contractions=false,
use_alignment_head=false`), not the older in-batch-only one from
experiment 1, and not experiment 6's original config (which had
`use_alignment_head=true` — that was matching a drifted, unvalidated
config, corrected in the section below). Chosen because:
- It's the only config with independent precedent of working at all — ARO
  itself reportedly reached ~78% hard_neg_acc with this recipe on a
  similarly-scaled hard-negative benchmark. The in-batch-only config was
  never validated anywhere; it was just a reasonable-sounding starting
  guess for SVO specifically.
- The in-batch-only architecture already absorbed four single-variable
  attempts (longer patience, capacity+regularization, two alignment-weight
  values) without closing the gap to target — plausibly near its ceiling.
- Experiments 7-9 are already single deviations from this exact config, so
  this formalizes rather than restarts the current investigation.

Honest caveat: by raw current numbers, experiment 6 itself (0.4826 probes /
0.6216 swap) is worse than the in-batch architecture's best result
(experiment 5: 0.5156 / 0.8269). Baseline choice is about future headroom
and prior validation, not "currently winning" — probes accuracy is at
chance either way, so there's no real loss, and swap has room to move as
triplet_weight/alignment-head get tuned.

## ⚠ Baseline correction: matched the wrong ARO reference (found 2026-09-17)

Experiments 6-10 all used `qnlp/scripts/aro_contrastive/config.py`'s
*current* values as "the legacy ARO config" — but that pipeline has drifted
from the actual result. Git archaeology (`git log --follow` on
`aro_contrastive/config.py` and `run.py`) traced the documented 78%
hard_neg_accuracy to the **true** legacy script,
`qnlp/discoviz/trainers/unfrozen/train_aro_clean.py`, which differs in two
ways that matter:

- **No learnable projection head at all.** Raw `image_model(images)` /
  `text_model(captions)` outputs go straight into the loss — no
  `ContrastiveVLM`, no `AlignmentHead`, no `head_lr`/`head_weight_decay`.
  Those were added later when the pipeline was ported/refactored into
  `aro_contrastive/`.
- **Trains and evaluates directly on ARO's own data** (via
  `get_aro_dataloader`) — confirming this, not COCO. (An earlier version of
  this doc briefly floated a COCO-training theory based on an initial port
  draft of `aro_contrastive/` having a misleadingly COCO-named parquet
  path — that was wrong; corrected here.)
- Predates `non_linear_contractions` entirely, consistent with the
  earlier NLC fix below.

One historical difference that does NOT need fixing: `train_aro_clean.py`
never ran the modern CCG-compilation `Pipeline` (with its
`filter_2d_outputs`/`UnifyEinsumRankStep` rank-unification), but checking
our own job logs confirms this filter has dropped **zero** SVO rows across
every run — `LemmatizeStep` already guarantees rank-1 diagram outputs, so
this difference isn't costing us anything in practice.

**Fix:** `use_alignment_head` default flipped to `False` (previously an
*ablation hypothesis* tested in experiment 8 — now a confirmed match to
the actual validated setup), alongside the earlier
`use_non_linear_contractions=False` fix. **Baseline updated accordingly —
see below.**

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

## Diagnostics added (2026-09-17)

Two precisely-scoped additions to `SVOHardNegStep`'s per-batch metrics,
chosen to distinguish specific generalization-failure mechanisms rather
than log broadly:
- `true_cosine_std` / `false_cosine_std` — near-zero means the model isn't
  discriminating AT ALL for that batch, distinct from "discriminating the
  wrong way" (which the existing mean-only metrics can't tell apart).
- `image_pairwise_cos_mean` / `caption_pairwise_cos_mean` — mean
  off-diagonal cosine similarity among a batch's own embeddings. Climbing
  toward 1.0 on val while staying lower on train would indicate
  anisotropic embedding collapse (embeddings crowd into a narrow cone that
  still satisfies the one specific triplet trained on per example, without
  preserving general discriminative structure) as the mechanism, rather
  than plain data/capacity insufficiency.

Also added: `use_alignment_head` config flag (`ContrastiveVLM`'s new
`NoOpHead` — plain `F.normalize`, no learnable params — as a third head
option alongside `AlignmentHead`/`MLPProjectionHead`), to ablate the
head's own ~dim²+dim learnable parameters per modality (~525K total at
embedding_dim=512) as a capacity source neither the COCO nor ARO campaigns
ever tested removing.

### 8. svo_final — job 7426422 (no AlignmentHead)
**Script:** `submit_svo.sh` with `SVO_ML_USE_ALIGNMENT_HEAD=false`,
otherwise identical to experiment 6 (triplet_weight=40000).
**Rationale:** Isolates the AlignmentHead's own learnable parameters as a
capacity/overfitting source, independent of backbone size (already
explored in experiment 3) or loss formulation (experiments 4-7).
**Results:** Early-stopped at epoch 22. SVO-Probes overall 0.5157 (subj
0.5431 / verb 0.5010 / obj 0.5299) — marginally above baseline (0.4826) but
still within the chance-level noise band we've seen across configs.
SVO-Swap 0.4730 — worse than baseline (0.6216). Net: removing the
AlignmentHead's ~525K params doesn't meaningfully help Probes and hurts
Swap. Doesn't move the baseline.

### 9. svo_final — job 7426423 (diagnostics baseline rerun)
**Script:** `submit_svo.sh`, no env overrides — identical config to
experiment 6, rerun purely to capture the new diagnostic metrics above for
a like-for-like comparison against experiment 8.
**Results:** Early-stopped at epoch 16. SVO-Probes overall 0.4849 (subj
0.4540 / verb 0.4836 / obj 0.5149), SVO-Swap 0.5135 — reproduces
experiment 6's chance-level result almost exactly (0.4826/0.6216), so that
outcome wasn't a fluke of one run's random init.

**Diagnostic finding (specific mechanism identified).** At epoch 16:

| | train | val |
|---|---|---|
| hard_neg_acc | 0.906 | 0.508 (chance) |
| true_cosine_mean | 0.568 | 0.304 |
| false_cosine_mean | 0.272 | 0.301 |
| true/false_cosine_std | 0.13 / 0.18 | 0.20 / 0.20 |
| image_pairwise_cos_mean | 0.19 | 0.20 |
| caption_pairwise_cos_mean | 0.50 | 0.53 |

On train, true/false separate clearly (0.568 vs 0.272). On val they are
statistically indistinguishable (0.304 vs 0.301) at every epoch from ~3
onward — climbing together early, then plateauing together, never
diverging. This is "no discrimination between true/false for unseen
pairs" — distinct from "discriminating the wrong way." Val cosine std
(~0.20, not near zero) shows the model does distinguish different
examples from each other, ruling out total embedding collapse to a single
point. Pairwise similarity (0.19-0.20 images, 0.50-0.53 captions) isn't
near 1.0 either, ruling out severe anisotropic collapse (captions are
more tightly clustered than images, worth noting, but neither is fully
degenerate).

**Conclusion:** the triplet loss is doing exactly what it's told — push
the *specific* false image shown per training row away, pull the
*specific* true image closer — but every training row sees the *same
fixed* false image every single epoch, so the model has no incentive to
learn a transferable "what makes an image match this caption" feature. It
only needs to remember "this one particular alternative image is wrong
for this one particular caption." Strong empirical support for the
rationale behind experiment 7 (`triplet_weight=100`, diluting this
fixed-negative-per-row signal relative to InfoNCE's diverse in-batch
negatives).

### 10. svo_final — job 7426649 (triplet_weight=10)
**Script:** `submit_svo.sh` with `SVO_ML_TRIPLET_WEIGHT=10`, otherwise
identical to experiment 6.
**Rationale:** Extends experiment 7's dose-response test one step further
down (100 → 10) rather than assuming one value is enough to characterize
the effect — mapping the curve, not just checking a single point.
**Status:** Killed before completion — was running with the pre-correction
config (`use_alignment_head=true`, and briefly also NLC=true before that
fix). Not worth letting finish on a config we no longer believe is right;
superseded by experiment 11.

### 11. svo_final — job 7426702, superseded by 7426745 (corrected baseline)
**Script:** `submit_svo.sh`, no env overrides — the corrected default
config (`use_non_linear_contractions=false, use_alignment_head=false`,
otherwise identical to experiment 6). This *is* the new current baseline
per the correction above, not a deviation from it.
**Status:** Job 7426702 was killed — `submit_svo.sh` itself had a hardcoded
`export SVO_ML_USE_NON_LINEAR_CONTRACTIONS=true` left over from before the
correction, silently overriding config.py's fixed default via the
environment. Removed that line from the script (commit `adff7ca`).
Relaunched clean as **job 7426745** — verified in its logged config banner
(`use_non_linear_contractions: False, use_alignment_head: False`; no
`nonlinear_gate` metric appears anywhere in its logs, confirming NLC is
genuinely off this time).
**Results:** Early-stopped at epoch 17 (best epoch 5, val hard_neg_acc
0.533). SVO-Probes overall **0.4983** (subj 0.4914 / verb 0.4884 / obj
0.5299), SVO-Swap **0.5270**. At chance on both. This is the properly
corrected, legacy-faithful config (no head, no NLC, trained/evaluated
directly on SVO's own hard negatives) — and it performs no better than
any prior variant. The overfitting signature is identical: train
hard_neg_acc climbed steadily past 0.68 by epoch 7 while val never
sustained improvement past epoch 5.

### 12. svo_final — job 7426704, superseded by 7426746 (corrected baseline + triplet_weight=100)
**Script:** `submit_svo.sh` with `SVO_ML_TRIPLET_WEIGHT=100`, otherwise
identical to experiment 11 (the corrected baseline).
**Rationale:** Re-runs experiment 7's dose-response test (diluting the
fixed-negative-per-row signal relative to InfoNCE) now on top of the
corrected config, rather than the pre-correction one experiment 7 used.
**Status:** Job 7426704 killed for the same reason as experiment 11.
Relaunched clean as **job 7426746** — verified config banner shows
`use_non_linear_contractions: False, triplet_weight: 100.0,
use_alignment_head: False`.
**Results:** Early-stopped at epoch 12. SVO-Probes overall **0.5073**
(subj 0.5201 / verb 0.5154 / obj 0.4751), SVO-Swap **0.5541**. Still at
chance; diluting triplet_weight doesn't help on the corrected config
either, consistent with experiment 7's finding on the pre-correction one.

## Status as of 2026-09-17

Every clean (uncontaminated) run to date, sorted:

| Exp | Config vs. baseline | SVO-Probes | SVO-Swap |
|---|---|---|---|
| 11 | corrected baseline (legacy-faithful) | 0.4983 | 0.5270 |
| 1  | old in-batch arch, no hard negatives | 0.5134 | 0.7692 |
| 2  | + longer patience | 0.5190 | 0.5962 |
| 5  | + alignment_weight=1000 | 0.5156 | **0.8269** |
| 12 | corrected baseline + triplet_weight=100 | 0.5073 | 0.5541 |
| 7  | pre-correction + triplet_weight=100 | *(pending — not yet run on corrected config)* | |
| 10 | pre-correction + triplet_weight=10 | *(killed pre-completion, superseded)* | |
| 6  | pre-correction ARO-matched (head=true, NLC=true) | 0.4826 | 0.6216 |
| 9  | reproduction of 6 | 0.4849 | 0.5135 |
| 8  | pre-correction, no head, NLC=true | 0.5157 | 0.4730 |

Target: ~0.83 / ~0.94. **Every single configuration tried — old and new
architecture, with and without hard negatives, with and without a
learnable head, with and without NLC, across a wide triplet_weight range,
alignment-weight range, capacity range, and regularization range — lands
within noise of chance (0.48-0.52) on SVO-Probes.** SVO-Swap has shown more
movement (0.47-0.83) but no config has approached target there either.
This breadth of negative results across truly orthogonal levers is itself
informative: it argues against any single hyperparameter or architectural
choice being the bottleneck, and increasingly points at either (a)
insufficient training data (5,421-5,437 rows) for this model family to
generalize on this task, or (b) something structural we haven't yet
isolated (e.g. a data-quality issue in the SVO pairs themselves, or a
mismatch between what the compiled CCG diagrams can represent and what
the task needs). Next investigative step should probably target
data-quality/data-sufficiency questions directly rather than further
hyperparameter search.

## Data investigation (2026-09-17)

Three concrete checks, moving off pure hyperparameter search per the
conclusion above.

**1. Training-set scale vs. ARO — large, direct explanation.** Checked
ARO's actual compiled dataset: `aro_train.parquet` has **36,585 rows**
(`aro_val.parquet` 8,000, `aro_test.parquet` 8,088) — roughly **7x** larger
than SVO's ~5,400-5,437 training rows. The 78% hard_neg_accuracy recipe was
validated at a scale we cannot match, since SVO-Probes' full benchmark is a
fixed size after the word-frequency filter and split. This is a strong,
simple candidate explanation for the gap that doesn't require any further
architecture/loss tuning to test — it's just a hard data ceiling.

**2. Image corruption — ruled out.** Every training log is full of
"Corrupt JPEG data" warnings, which had been treated as background noise.
Sampled 500 images across `images/`+`images_old/` and decoded each fully
via PIL (`.load()`, not just header inspection): **zero decode failures**,
zero tiny/degenerate images, healthy resolution distribution (median
1067×900, min 173×160). The warnings are harmless libjpeg trailing-byte
complaints tolerated by PIL; not a real data-quality issue.

**3. Manual pair inspection — negative difficulty is highly inconsistent.**
Pulled and viewed several real (caption, pos_image, neg_image) triples:
- *"The player fumbles the ball."* (verb_neg): positive shows a basketball
  scramble for a loose ball; negative shows an unrelated tennis serve —
  completely different sport, trivially easy to distinguish.
- *"A woman sits on a balcony."* (obj_neg): both positive and negative show
  a woman sitting outdoors in a similar stock-photo pose, differing only in
  setting (balcony railing vs. open field) — genuinely subtle, a real test
  of scene grounding.

That the model has never sustained val hard_neg_acc above ~0.53 across any
run — including on trivially-easy pairs like the tennis/basketball example
— suggests the failure isn't really about the *hardest* negatives being
too hard. It looks more like the model isn't learning any reliable general
image-caption grounding at all, consistent with the memorization signature
already characterized (train separates cleanly, val does not).

**Working conclusion:** the ~7x data-scale gap vs. ARO is probably the
single largest, most defensible explanation for why this recipe hasn't
closed the gap to target on SVO-Probes specifically, and it's not a gap
further hyperparameter search can close. Worth deciding explicitly whether
to (a) report this as a data-scale-limited result, (b) explore ways to
expand effective training signal (e.g. data augmentation, synthetic
negative generation, or relaxing the word-frequency filter threshold to
recover more rows), or (c) treat the SVO-Swap task (which has shown real
movement, up to 0.83) as the more tractable target given its much smaller
demands.

## Round 2: recover more training data (2026-09-17)

Acted on option (b) above rather than stopping. Lowered
`WORD_FREQ_THRESHOLD` 50 → 10 in `qnlp/scripts/svo/prepare_datasets.py`
(job 7427077 regenerated the data): **svo_train 8,609 rows** (up from
5,421-5,437), **svo_val 2,908**, **svo_test 2,767** — total 14,284,
matching the predicted count exactly. Splits still land at ~60/20/20
(60.3/20.4/19.4%), subj/verb/obj proportions still track the corpus in
both val (18.0/58.7/23.5%) and test (17.5/60.3/22.2%), and zero
positive-image overlap across any split pair. SVO-Swap grew to 105 pairs
(from 74). Still ~4.25x smaller than ARO's 36,585, but a real 57% increase
over the previous training set.

### 13. svo_final — job 7427093 (corrected baseline, larger dataset)
**Script:** `submit_svo.sh`, no env overrides, trained on the
threshold=10 dataset (8,609 train rows).
**Rationale:** Re-tests the corrected baseline (experiment 11) on the
recovered data, isolating the data-volume variable specifically.
**Status:** Job 7427093 crashed immediately on model construction with
`RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or
unavailable` — transient GPU contention on a shared node, not a code or
data bug (never got past `EinsumModel(...).to(device)`, before any
training happened). Note: the submit script's "Job finished successfully"
message is misleading here — it doesn't check the Python process's exit
code, so a crashed run still prints that banner. Relaunched clean as
**job 7427395**.
**Results:** Early-stopped at epoch 27. SVO-Probes overall **0.5103**
(subj 0.4392 / verb 0.5312 / obj 0.5098), SVO-Swap **0.4952**. This is the
clean isolation point: same 8,609-row dataset as experiment 14, standard
config. Experiment 14's numbers (0.5305/0.5810) are modestly but
consistently higher on both metrics — suggesting the capacity/
regularization change contributes something beyond the extra data alone,
though the gap is small enough that it warrants multiple seeds before
treating it as conclusive rather than noise.

### 14. svo_final — job 7427094 (reduced capacity + regularization, larger dataset)
**Script:** `submit_svo.sh` with `SVO_ML_EMBEDDING_DIM=128,
SVO_ML_IMAGE_LR=0.0001, SVO_ML_TEXT_WEIGHT_DECAY=0.01,
SVO_ML_IMAGE_WEIGHT_DECAY=0.15`, on the larger dataset.
**Rationale:** Retests experiment 3's "reduce capacity + more
regularization" hypothesis — previously only tested on the old,
now-removed in-batch-only architecture, and only with contaminated final
results — cleanly, on the current corrected architecture and the larger
dataset. Two variables (architecture correctness + data volume) changed at
once relative to experiment 3, but that experiment's own result was never
valid to begin with, so there's no clean prior number to hold fixed.
**Results:** Early-stopped at epoch 20. SVO-Probes overall **0.5305**
(subj 0.5423 / verb 0.5180 / obj 0.5554), SVO-Swap **0.5810**. Best
SVO-Probes result of any clean run to date (previous best: 0.5190,
experiment 2) — still far below target and within a plausible noise band,
but a small step in the right direction. Can't yet isolate whether this
came from the extra data, the capacity/regularization change, or both —
needs experiment 13's result (same data, standard config) for comparison.
Best checkpoint was epoch 10, out of 20 run (patience=10 stopped it there).

### 15. svo_final — job 7427418 (experiment 14 config, patience=40)
**Script:** `submit_svo.sh` with the same overrides as experiment 14
(`SVO_ML_EMBEDDING_DIM=128, SVO_ML_IMAGE_LR=0.0001,
SVO_ML_TEXT_WEIGHT_DECAY=0.01, SVO_ML_IMAGE_WEIGHT_DECAY=0.15`) plus
`SVO_ML_PATIENCE=40`.
**Rationale:** Experiment 14 stopped at epoch 20 (best was epoch 10,
patience=10). Longer patience alone didn't help the *old* config back in
experiment 2, but that's a different architecture/regularization regime —
worth checking whether this specific (regularized, smaller-capacity,
larger-dataset) config has a slower-but-better convergence curve that
patience=10 cut off prematurely, given `max_epochs=100` leaves plenty of
room.
**Results:** Ran the full 100 epochs (never early-stopped — val monitor
metric kept finding new bests at epochs 1, 2, 11, 14, 40, 57, and finally
67). Final checkpoint (epoch 67): SVO-Probes overall **0.5150** (subj
0.5031 / verb 0.5078 / obj 0.5440), SVO-Swap **0.4000** — *worse* than
experiment 14's epoch-10 checkpoint (0.5305/0.5810) on the exact same
config, just trained longer.

**Important finding:** the val monitor metric (`hard_neg_acc` on
`svo_val_probes.parquet`) improving at later epochs did **not** translate
to a better held-out test result — if anything, the opposite. This is
strong evidence that at the current performance level, "which checkpoint
looks best by val metric" is dominated by noise rather than tracking real
generalization improvement: both epoch 10 and epoch 67 are wandering
within the same chance-level band (0.40-0.58 across both metrics observed
so far), and early stopping picking a *later* epoch doesn't reliably pick
a *better* one. Reinforces (independently, on a new architecture) the
same lesson from experiment 2: **longer patience does not help here** —
it just changes which noisy point in the band gets selected.

## Fallback plan (agreed 2026-09-17): sanity-check on ARO itself

If experiments 13-15 don't show anything promising, next step is to run
our current SVO pipeline's implementation (`ImageContrastiveLoss`,
`SVOHardNegStep`, corrected config) **against ARO's own data**
(`aro_train/val/test.parquet`, already on the cluster) rather than
continuing to guess on SVO. This checks a different thing than everything
above: whether our *implementation* is faithful to the legacy recipe,
independent of whether SVO's specific data/scale can ever reach the
target. If our code reproduces something close to 78% hard_neg_accuracy on
ARO, that rules out an implementation bug and confirms the SVO gap really
is about SVO's data (scale, quality, or task difficulty). If it doesn't
reproduce 78% on ARO either, that points at a remaining implementation
divergence we haven't found yet, independent of SVO entirely. Not started
yet — no action needed until the current round of experiments concludes.

**Trigger condition reached (2026-09-17).** Experiments 13-15 are complete:
recovering ~57% more training data (13/14 comparison) gave a small,
plausibly-real edge from the capacity/regularization change, but nothing
approaching target; letting that same config run the full 100 epochs
(15) produced a *worse* result than stopping early, confirming the
remaining gap isn't a convergence/patience issue either. Best SVO-Probes
result across all 15 experiments remains ~0.53, essentially chance. Next
step per the agreed fallback plan: run this implementation against ARO's
own data to check whether it's a code-fidelity issue or genuinely an
SVO-specific data/scale ceiling.

## ARO sanity check (2026-09-17)

**Purpose:** isolate implementation correctness from SVO-specific
data/scale/quality issues, per the fallback plan above. Runs the same
loss/step design our SVO pipeline uses (`ImageContrastiveLoss`/
`SVOHardNegStep` mirror `qnlp/scripts/aro_contrastive/`'s `ContrastiveLoss`/
`AROContrastiveStep`, which is already correctly shaped for ARO's
caption-side hard negatives) directly on ARO's own 36,585-row dataset, with
the true legacy config (no NLC, no learnable head), to check whether it
reproduces something near the documented 78% hard_neg_accuracy. If yes,
the SVO gap is genuinely about SVO's data. If no, there's a remaining
implementation divergence independent of SVO.

**Setup fixes required first:** `qnlp/scripts/aro_contrastive/` had the
same two bugs already found and fixed in `qnlp/scripts/svo/`:
`use_non_linear_contractions` defaults to `True` (not the legacy value),
and `submit_aro_contrastive.sh` hardcoded `export
ML_USE_NON_LINEAR_CONTRACTIONS=true`, which silently blocks any `-v`
override (`export` always wins). Fixed conservatively — since this
pipeline has its own ongoing use (dataset-suffix path ablations) —by
restoring the ability to override rather than changing the default, and
adding a new `use_alignment_head` field (default `True`, preserving
existing behavior) that can be set `False` via `-v
ML_USE_ALIGNMENT_HEAD=false` for a legacy-faithful run. Also fixed the
same empty-optimizer-param-group issue (`NoOpHead` has no parameters) and
the same misleading always-prints-success submit script bug.

### 16. aro_contrastive — job 7428516 (legacy-faithful config on ARO's own data)
**Script:** `submit_aro_contrastive.sh` with `-v
ML_USE_NON_LINEAR_CONTRACTIONS=false,ML_USE_ALIGNMENT_HEAD=false`.
**Verified:** both settings confirmed to resolve correctly with these
exact env vars (`use_non_linear_contractions: False, use_alignment_head:
False`); job log confirms `Train: 36585 | Val: 8000 | Test: 8088`, 2,719
unique symbols.
**Training trajectory:** climbed steadily and smoothly — val hard_neg_acc
0.623 (epoch 5) → 0.676 (epoch 19) → plateaued around 0.67-0.68 from
epoch ~20 onward, best at epoch 28 (0.678). Early-stopped at epoch 38
(patience=10). Train hard_neg_acc reached 0.918 by the end — some
overfitting gap present, but nothing like SVO's complete train/val
divergence (val never stopped tracking train reasonably here).
**Results (ARO held-out test set, N=8,088), compared against the
documented per-task targets (attribution 78%, relation 59% — the
"78% hard_neg_accuracy" figure in `llm/model_evolution.md` is the
attribution-task number, not an aggregate; the earlier version of this
entry incorrectly compared our aggregate against it):**

| ARO task | N | acc | target | diff | true_cos | false_cos |
|---|---|---|---|---|---|---|
| attribution | 4,438 | 0.7573 | 0.78 | -0.023 | 0.2933 | 0.0176 |
| relation | 3,650 | 0.6036 | 0.59 | **+0.014** | 0.1404 | -0.0254 |
| overall | 8,088 | 0.6879 | — | — | 0.2243 | -0.0018 |

Per-task, this is essentially a match: attribution lands 2.3 points under
target, relation lands 1.4 points *over* target. This is a much stronger
reproduction of the legacy result than the aggregate number suggested.
(Winoground/SugarCREPE were mostly skipped due to vocabulary gap —
SVO/ARO-specific vocabulary doesn't cover those benchmarks' symbols; not
meaningful here.)

**Conclusion:** ⚠ *superseded in part — see "The image tower is the
bottleneck" at the end of this doc: the ARO score below turns out to be
image-invariant, so it validates the text tower and the port, not the
recipe's visual grounding.* Our
`ImageContrastiveLoss`/`SVOHardNegStep` implementation (mirroring
`ContrastiveLoss`/`AROContrastiveStep` used here) **reproduces the
documented legacy result on ARO to within ~2 points per task** —
dramatically above anything achieved across all 15 SVO experiments (best:
~0.53, effectively chance). This rules out a fundamental implementation
bug as the explanation for the SVO gap: **SVO's chance-level results are
genuinely about SVO's data (scale: ~8,600 vs ARO's 36,585 rows; and/or
task difficulty/quality, per the earlier manual pair inspection), not a
bug in how we ported the recipe.**

SVO's own granular breakdown (subj_neg/verb_neg/obj_neg, the direct
analogue of ARO's attribution/relation split) is already reported in
every SVO experiment above via `evaluate_svo_probes` — this was built
specifically to match the original SVO-Probes paper's per-category
reporting.

## Transfer learning from ARO (2026-09-18)

**Rationale:** the sanity check above proved the implementation is
correct and the SVO gap is about data scale (~8,600 rows vs ARO's
36,585). Rather than accepting a scale-limited result outright, tried
warm-starting SVO training from the validated ARO checkpoint (job
7428516) before training on SVO — hypothesis: a model that already
learned ARO's hard-negative image/text discrimination needs less of
SVO's small training set to adapt, versus learning discrimination from
scratch.

**Implementation** (`qnlp/scripts/svo/config.py`'s new
`pretrained_checkpoint` field, wired into `qnlp/scripts/svo/run.py`'s
`_warm_start_from_aro`): image tower (`TTNImageModel`) loads in full —
identical architecture/shapes to ARO's. Text tower (`EinsumModel`) is
per-symbol (per-word), so only symbols present in both vocabularies
with matching shape transfer via `EinsumModel.set_weights`; the rest
stay randomly initialised.

### 17. svo_probes — job 7429309 (ARO-pretrained warm start)
**Setup:** `-v SVO_ML_PRETRAINED_CHECKPOINT=.../aro_contrastive/2026-09-17_18-37-31_optimal_pid33414/best_model.pt`,
otherwise the same legacy-faithful config as experiment 16's baseline
(no NLC, no alignment head). Log confirmed: image tower loaded in
full; text tower transferred 504/1356 SVO symbols (~37%) found in
ARO's 2,719-symbol vocab with matching shape.
**Training:** early-stopped at epoch 20 (best epoch 11, patience=10).

**Results:**

| Metric | This run (ARO warm start) | Best prior SVO run (from scratch) |
|---|---|---|
| SVO-Probes overall | 0.5096 (subj 0.5052 / verb 0.5024 / obj 0.5326) | 0.5305 (experiment 14) |
| SVO-Swap | **0.6095** (N=105) | 0.5270 (experiment 14) |

**Conclusion:** SVO-Probes stayed at chance — the warm start did not
help the image-side hard-negative task, consistent with it being a
genuine data-scale/task-difficulty limitation, not something a better
starting point can fix on its own. **SVO-Swap improved meaningfully**
(+8 points over the prior best, +11 over chance) — plausible
explanation: SVO-Swap's hard negative is a *caption-side* swap
(subject/object exchanged), the same shape as ARO's task, so the
transferred text-tower symbols directly carry over subject/verb/object
compositional structure learned from ARO's much larger caption corpus.
SVO-Probes' hard negative is *image-side* (a different photo), which
depends on the image tower discriminating fine-grained visual
differences under SVO's own (smaller, lower-quality per the earlier
manual pair inspection) image data — not something ARO's photos-are-
plausible-or-not signal transfers to. Still far from the 94%
SVO-Swap target, but the first result clearly and reproducibly above
chance on this benchmark.

## ⚠ The image tower is the bottleneck — ARO never tested it (2026-09-18)

The ARO sanity check above concluded "the implementation is correct, so the
SVO gap is about SVO's data." That conclusion was too generous to the ARO
result. Direct ablation shows **the ARO score does not use the image at
all**, which changes what every experiment in this log was measuring.

**Method** (`qnlp/discoviz/diagnostic/image_ablation.py`, new): take a
trained `aro_contrastive` checkpoint and re-score ARO test three ways —
with the real image, with each row scored against *some other row's* image
("shuffled"), and with an all-zero image ("zeros"). Separately, measure the
mean pairwise cosine between different images' embeddings.

**Results** (checkpoint `runs/checkpoints/aro_contrastive/2026-06-11_10-25-56/`,
epoch 53, NLC=true + AlignmentHead; 512-row sample of `aro_test.parquet`):

| variant | N | hard_neg_acc | true_cos | false_cos |
|---|---|---|---|---|
| real | 512 | 0.7168 | 0.7593 | 0.5695 |
| shuffled | 512 | 0.7129 | 0.7594 | 0.5696 |
| zeros (no image at all) | 512 | **0.7168** | 0.7603 | 0.5708 |

Feeding a blank image changes accuracy by 0.0000. The mechanism, measured
on the same checkpoint:

```
pairwise cosine between DIFFERENT images: mean 0.9984  min 0.9489  std 0.0031
per-dimension std across images:          0.00164
```

The image tower has collapsed to a single constant vector — it emits
essentially the same embedding whatever it is shown. The ~70% ARO score is
a **text-only caption-plausibility classifier**: the text tower learned to
point well-formed captions toward one fixed direction and swapped ones away
from it. ARO permits this because both candidate captions are scored
against the *same* image, so the image never has to break a tie; the
benchmark is known to be largely solvable blind (the SugarCREPE critique of
ARO makes the same point about text-only baselines).

**Why this explains SVO-Probes exactly.** SVO-Probes inverts the roles: the
caption is identical for both candidates, so the decision is
`cos(t, I_pos) > cos(t, I_neg)` and the *entire* discriminative burden falls
on the image tower. With an image-invariant model that is exactly 0.50, for
any loss, any triplet_weight, any capacity, any dataset size. SVO-Probes is
the first task in this project that requires the image tower to work at
all — and the rest of the project's history is consistent with it never
having worked: COCO retrieval was always "zero retrieval"
(`COCO_EXPERIMENTS.md`), and experiment 17's ARO warm start helped
SVO-**Swap** (caption-side, +8pts) while doing nothing for SVO-Probes
(image-side).

Fifteen experiments tuned the loss, the head, the capacity, the
regularization and the data volume around a bottleneck none of those levers
touch.

**Provenance caveat.** This ablation ran on the June checkpoint above, not
on job 7428516's (the September legacy-faithful run); the June one is the
newest `aro_contrastive` checkpoint held locally. It scores 0.72 on this
sample vs. 7428516's 0.688 on the full test set — the same regime — and the
collapse mechanism is independent of the NLC/head flags that differ between
them. Re-running the script on 7428516's checkpoint (and on an SVO
checkpoint, `--task svo`) is the first item below.

**Ruled out while investigating this** (so they aren't re-chased):
- Image loading/normalization is correct: `read_image` → `.float().div(255)`
  → `Normalize`, in `qnlp/domain/datasets/dataset.py`.
- Untrained symbols at eval time: only ~8% of val/test rows contain a word
  unseen in train on the *unfiltered* corpus, less after the freq≥10 filter.
  Not enough to pin val at chance.
- An eval-only bug in `evaluate_svo_probes`: val `hard_neg_acc` goes through
  the same `SVOHardNegStep` as train and is at chance, so the failure is in
  training, not in the final eval path.

**The target was also never the right one.** ~83% SVO-Probes comes from
models with an ImageNet-pretrained visual backbone trained on ~3M
Conceptual Captions pairs, where SVO-Probes is a *zero-shot probe* rather
than a training set. The paper's "SVO is easier than ARO" ordering
presupposes a working vision encoder; with a from-scratch tensor-network
tower the ordering inverts, because ARO can be passed blind and SVO-Probes
cannot.

### Next steps

1. ✅ Re-ran `image_ablation.py` on job 7428516's checkpoint (2026-09-18) —
   confirms the collapse on the actual validated-against-target run, not
   just the older June checkpoint used above: real/shuffled/zeros accuracy
   0.6855/0.6914/0.6797 (statistically identical), pairwise image cosine
   **0.9342** (near-collapsed, closes the provenance caveat).

   Also ran on the SVO checkpoint (job 7429309, ARO warm start) —
   **different failure mode than ARO's, as predicted**: pairwise image
   cosine is **0.2108**, not collapsed — SVO's own true/false-image
   training objective does push the tower to vary per image. But accuracy
   stays at chance (real caption 0.5508 vs shuffled-caption 0.4844 — only
   a small gap), meaning that variation doesn't carry caption-relevant
   discriminative signal. So SVO's image tower isn't frozen at a constant
   output like ARO's — it has learned *something* image-specific — but
   whatever it learned doesn't help decide which of two images matches a
   given caption. Confirms experiment 9's `image_pairwise_cos_mean ≈ 0.19`
   reading rather than overturning it: two distinct failure modes (ARO:
   collapsed and unused; SVO: varying but not caption-grounded), same
   practical consequence (image tower contributes ~nothing to accuracy on
   both benchmarks).
2. Make the image tower the object of study: linear probe on
   `TTNImageModel` features for SVO's object/verb class at 64×64. Converts
   "SVO-Probes failed" into a quantified statement about the encoder, and
   connects to the 16×16 training ceiling already documented on the quantum
   side (`llm/research_log.md`).
3. ✅ Implemented (2026-09-18), launch pending: `qnlp/scripts/svo/run_frozen.py`
   + `scripts/submit_svo_frozen.sh`. Frozen CLIP ViT-B/32 image tower (reuses
   `CLIPImageCache` from `coco_multi_caption/run_frozen.py`, 512-dim, matching
   `SVOExperimentConfig`'s default `embedding_dim`) in place of TTNImageModel;
   DisCoCat `EinsumModel` text tower unchanged, still trained from scratch
   with the same `ImageContrastiveLoss` + hard-negative triplet on SVO's own
   data. A learnable linear text head is added (unlike the legacy no-head
   config) since the text tower now has to land in CLIP's fixed embedding
   space rather than co-adapt with a from-scratch image tower. Bespoke eval
   functions mirror `evaluate_svo_probes`/`evaluate_sugarcrepe`'s logic
   (subj/verb/obj breakdown for Probes; SVO-Swap reuses the ARO/SugarCREPE
   hard-neg shape directly, since `svo_swap_eval.parquet`'s schema already
   matches it). If Probes jumps to 70-80%, the bottleneck is isolated beyond
   argument. (COCO experiment 11 tried frozen CLIP but died at epoch 2
   against a 389k-way retrieval objective; a binary hard-negative task is a
   far easier target.) This is a diagnostic control, not a proposal to put
   classical capacity in the quantum pipeline.
4. Collapse-adjacent settings worth revisiting on their own: `image_lr=5e-5`
   is very low for a tower trained from scratch, there is no image
   augmentation anywhere (train transform == val transform), and
   `triplet_weight=40000` with a constant image direction is trivially
   satisfiable by moving captions alone — it actively rewards ignoring the
   image. Log `image_pairwise_cos_mean` in ARO training too.
5. Reframing: SVO-Swap is caption-side and is where this architecture can
   legitimately show results (0.61 with the ARO warm start). SVO-Probes
   becomes a vision-bottleneck negative result backed by the ablation above.

## Direct image-tower fix attempt (2026-09-18)

Rather than spending compute on the frozen-CLIP diagnostic control (item 3
above — deferred, not run), went straight at fixing the from-scratch
TTNImageModel itself, per the three concrete defects item 4 already
identified: `image_lr=5e-5` is very low for a tower training from scratch,
there was **zero image augmentation** (`qnlp/scripts/svo/run.py` used the
identical transform for train and val — confirmed by reading the code, not
assumed), and `triplet_weight=40000` with a near-static image tower lets
the loss get satisfied almost entirely by moving the caption embedding away
from the false image, never requiring the image tower to encode anything
useful.

**Fixes applied:**
- `qnlp/scripts/svo/run.py`: train transform now
  `RandomResizedCrop(scale=(0.8,1.0)) → ColorJitter → RandomHorizontalFlip →
  Normalize` (val stays `Resize → Normalize`, unchanged). `RandomResizedCrop`
  handles SVO's widely varying native image sizes (173×160 to 1067×900+, per
  the earlier data investigation) directly. This is a real code fix, not a
  tunable default — applies to every future SVO run regardless of other
  config.
- Launched via `-v` overrides (not new config.py defaults, pending
  validation): `SVO_ML_IMAGE_LR=0.001` (was 0.00005) and
  `SVO_ML_TRIPLET_WEIGHT=100` (was 40000). Note triplet_weight=100 was
  already tried alone in experiments 7/12 without effect — but always on top
  of the zero-augmentation, near-frozen-image-tower setup, so it never got a
  real test of "does the image tower learn something once it's actually
  allowed to move meaningfully." This run tests the combination, not a
  repeat.

**Results — job 7429715 (`image_lr=0.001, triplet_weight=100`, + augmentation):**
Early-stopped at epoch 13 (best epoch 3, val hard_neg_acc 0.5277 — picked by
noise, not a real trend: val bounced 0.49-0.53 the whole run with no
direction). **SVO-Probes overall 0.4977** (obj 0.4919 / subj 0.5320 / verb
0.4898), **SVO-Swap 0.4476** — both *worse* than the legacy baseline
(0.4983/0.5270) and worse than the best prior result (experiment 14:
0.5305/0.5810). Worth stating plainly: this combination made things worse.

**The unexpected part — train itself never left chance.** Every prior run
(with or without hard negatives) showed train `hard_neg_acc` climbing fast,
often to 0.68-0.9+, while val stayed flat — the standard
memorise-train/fail-val signature. Here train sat at 0.495-0.531 for all 13
epochs, with `true_cosine_mean`/`false_cosine_mean` both hovering within
±0.01 of zero throughout (i.e. statistically indistinguishable, not just
"not separating well") and `image_pairwise_cos_mean` oscillating noisily in
0.003-0.06 — nowhere near the 1.0 collapse seen on ARO, but also not
settling into any stable structure. Confirmed this isn't silent data
dropping: `n_dropped` was 0 in every logged epoch.

**Follow-up run — job 7430070 (`image_lr=0.0003`, same triplet_weight=100 +
augmentation), launched to check whether 0.001 was simply too large a
step:** same qualitative picture through 6 epochs — train `hard_neg_acc`
0.510-0.531 with no trend, cosines still near-zero. Lowering the LR 3x
changed nothing structural, which argues against "LR magnitude" as the
specific cause.

**Interpretation, read together with the init-spread trace below:** that
trace shows a freshly-initialised tower already separates real images well
(pairwise cos 0.146) — the representational capacity is there. The
previous (pre-augmentation) runs' fast train-accuracy climb was most likely
the model exploiting a shortcut unrelated to real visual grounding:
memorising the *one exact, unaugmented* false image shown per training row
every single epoch. Augmentation correctly removes that shortcut (the crop
differs each epoch) — but nothing has yet replaced it with real learning at
either LR tried. This is a third, previously-unseen failure mode (neither
"memorise-then-fail-to-generalise" nor "collapse to a constant vector") and
is not yet understood. Recommendation before trying further LR values in
this same three-way bundle: isolate variables (augmentation alone at the
original `image_lr=5e-5`/`triplet_weight=40000`, vs. `triplet_weight`
lowered alone with no augmentation) rather than continuing to vary one
knob inside a combination that is itself not yet behaving as expected.

## Is the collapse architectural? No — init-spread trace (2026-09-18)

Before redesigning the image tower, checked whether a collapsed tower is
something the architecture *is* or something training *does to it*.
`qnlp/discoviz/diagnostic/tower_spread_trace.py` (new) runs real photos
through `TTNImageModel` and measures mean pairwise cosine between different
images at every internal stage. With no `--checkpoint` it traces a freshly
initialised tower.

**Random init, 128 real ARO photos:**

```
raw pixels (ImageNet-normalised)   pairwise cos mean 0.1754
raw pixels, dataset-mean-centred   0.0069
after colour projection (linear)   0.1757
after pixel projection (linear)    0.1489
after bilinear product c*p         0.5406   <- bilinear patch map costs spread
+ positional embedding             0.5406
after quadtree layer 0             0.3649
after quadtree layer 1             0.1510   <- quadtree recovers it
after quadtree layer 2             0.1441
after quadtree layer 3             0.1487
after final_norm + head (output)   0.1459   <- trained ARO checkpoint: 0.9984
```

**At initialisation the tower separates images perfectly well** (0.146).
The bilinear `c_feat * p_feat` patch map does cost real spread (0.15 →
0.54), and that's worth remembering as a second-order concern, but the
quadtree recovers it by layer 1 and the output is healthy. So the 0.9984
collapse measured on the trained ARO checkpoint is **learned, not
structural** — the representational capacity to distinguish images is
there at init and training destroys it.

Conclusion for the architecture question: **don't redesign the tower yet.**
The objective, not the multilinear quadtree, is the first thing to fix.

### The mechanism: `triplet_weight` deletes the only anti-collapse term

`ContrastiveLoss` (and its `ImageContrastiveLoss` mirror) computes
`total = infonce + triplet_weight * triplet`. On ARO the triplet term is
caption-side — `d(I, t_true) < d(I, t_false) - margin` — and that is
**fully satisfiable with a constant image embedding**, by moving only the
captions. The one term that *requires* different images to embed
differently is InfoNCE, whose in-batch matching is impossible when every
image embeds identically. At a weight ratio of 1 : 40,000 its gradient is
negligible, so the collapsed solution is the easy minimum and training
finds it.

This sharpens experiment 7's original hypothesis. That experiment framed
`triplet_weight=40000` as "swamping InfoNCE's diverse in-batch negatives";
the more precise statement is that it removes the objective's only
anti-collapse pressure. It also explains why experiments 7 and 12
(triplet_weight 100 and 10) showed nothing: both ran on SVO, whose tower
was never collapsed in the first place (`image_pairwise_cos ≈ 0.19`), so
neither run ever tested this mechanism. The combination launched in
"Direct image-tower fix attempt" above — augmentation + `image_lr=0.001` +
`triplet_weight=100` — is the first run that does.

`image_pairwise_cos_mean` should be logged as a first-class training metric
on the ARO pipeline too, not just SVO's: it is the metric that would have
caught this at the time.

### Revised architecture position

Ranked, given the above (items 1-4 of the previous list are largely
actioned; this replaces the architectural part of item 4):

1. **Objective and signal first** — the launched fix run, plus
   `image_pairwise_cos_mean` logging on ARO. No architecture change is
   interpretable while the objective still admits a degenerate solution.
2. **Then one architectural change, if the fix run doesn't move Probes:**
   the image tower is purely multilinear — no non-linearity anywhere. The
   text side got NLC (`contraction + gate * GELU(contraction)`, gate init
   0) and demonstrably learned to use it (gate rose 0.044 → 0.393 in the
   COCO campaign). The same gated device inside `CPQuadRankLayer` is a
   strict generalisation — gate=0 recovers today's model exactly — reuses a
   mechanism already validated in this codebase on the other tower, and
   makes a clean thesis question: does the image tower need the same
   non-linearity the text tower needed? Adding capacity *before* fixing the
   objective would only give the model a better way to collapse.
3. **Second-order, only if the trace above becomes the binding
   constraint:** the bilinear patch map's 0.15 → 0.54 spread loss, and
   dataset-mean-centring the input (0.1754 → 0.0069 at the pixel stage).
4. **Not resolution.** 64×64 / 16×16 patches is not the binding constraint
   while the objective admits a constant-image solution.

Still ruled out, unchanged: wider `bond_dim`, and further sweeps of the
kind experiments 1-15 already covered.
