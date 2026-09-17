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

**Target numbers to beat: ~83% SVO-Probes, ~94% SVO-Swap.** Current best
(experiment 1, job 7426179) is 51.3% / 76.9% — a large gap, consistent with
the severe overfitting documented below rather than a ceiling on the
architecture itself.

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
**Final results:** _(pending — job still running as of this entry)_

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
**Results:** _(pending)_
