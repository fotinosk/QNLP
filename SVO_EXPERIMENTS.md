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
**Results:** _(pending)_

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
**Results:** _(pending)_
