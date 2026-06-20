# Experiment Plan: COCO Training Ablation

## Goal

Exhaustive search over {contraction type} × {image model}:

| Experiment | Contraction | Image Model          | Training Script                     |
| ---------- | ----------- | -------------------- | ----------------------------------- |
| A          | Linear      | Learnable TTN        | `coco_single_caption/run.py`        |
| B          | Non-linear  | Learnable TTN        | `coco_single_caption/run.py`        |
| C          | Linear      | Frozen CLIP ViT-B/32 | `coco_single_caption/run_frozen.py` |
| D          | Non-linear  | Frozen CLIP ViT-B/32 | `coco_single_caption/run_frozen.py` |

All 4 evaluated on: **ARO**, **SugarCREPE** (swap_obj), **Winoground**

---

## Status

### 1. Preprocessing Pipeline

- [x] COCO pipeline fixed and running on cluster (wordnet, spawn, CLIP model on SAN)
- [x] Pipeline run complete and verified healthy (ok% > 0 in LMDB)

### 2. Datasets

- [ ] `coco_single_caption_{train,val,test}.parquet` — linear, recreating on cluster
- [ ] `coco_single_caption_nlc_{train,val,test}.parquet` — NLC (needs contraction paths, slow)
- [x] `aro_{train,val,test}.parquet`
- [x] `winoground_{train,val,test}.parquet`
- [x] `sugarcrepe_swap_obj_test.parquet`

### 3. Training Scripts

- [x] `coco_single_caption/run.py` — TTN image, linear/NLC via `ML_USE_NON_LINEAR_CONTRACTIONS`
- [x] `coco_single_caption/run_frozen.py` — Frozen CLIP, linear/NLC via `ML_USE_NON_LINEAR_CONTRACTIONS`

### 4. Evaluation Scripts (TTN checkpoint: `model_state_dict`)

- [x] ARO: `coco_single_caption/evaluate_aro.py`
- [x] Winoground: `coco_single_caption/evaluate_winoground.py`
- [x] SugarCREPE: `sugarcrepe/evaluate.py`

### 5. Evaluation Scripts (Frozen checkpoint: `text_model_state_dict` + `text_head_state_dict`)

- [ ] ARO frozen: `coco_single_caption/evaluate_aro_frozen.py`
- [ ] Winoground frozen: `coco_single_caption/evaluate_winoground_frozen.py`
- [ ] SugarCREPE frozen: `sugarcrepe/evaluate_frozen.py`

---

## Execution Order

### Step 1 — Verify pipeline output is healthy

Run the health check and confirm ok% > 0 in LMDB. If 0%, pipeline failed — check logs and resubmit.

Module: `qnlp.preprocessing_pipelines.coco.pipeline`

### Step 2 — Create linear dataset

Required for experiments A and C.

Module: `qnlp.scripts.coco_single_caption.create_dataset`

Output: `data/datasets/coco_single_caption_{train,val,test}.parquet`

### Step 3 — Train experiment A (Linear + TTN)

Module: `qnlp.scripts.coco_single_caption.run` with `ML_USE_NON_LINEAR_CONTRACTIONS=false`

### Step 4 — Train experiment C (Linear + Frozen CLIP)

Can run in parallel with Step 3.

Module: `qnlp.scripts.coco_single_caption.run_frozen` with `ML_USE_NON_LINEAR_CONTRACTIONS=false`

### Step 5 — Create NLC dataset

Can run in parallel with Steps 3 and 4. Slow due to contraction path computation.

Module: `qnlp.scripts.coco_single_caption.create_dataset --paths`

Output: `data/datasets/coco_single_caption_nlc_{train,val,test}.parquet`

### Step 6 — Train experiment B (Non-linear + TTN)

Requires Step 5.

Module: `qnlp.scripts.coco_single_caption.run` with `ML_USE_NON_LINEAR_CONTRACTIONS=true`

### Step 7 — Train experiment D (Non-linear + Frozen CLIP)

Requires Step 5.

Module: `qnlp.scripts.coco_single_caption.run_frozen` with `ML_USE_NON_LINEAR_CONTRACTIONS=true`

### Step 8 — Write frozen evaluation scripts

Prerequisite for evaluating experiments C and D. See §5 above.

### Step 9 — Evaluate all experiments

#### Experiments A and B (TTN checkpoint)

- ARO: `qnlp.scripts.coco_single_caption.evaluate_aro <checkpoint>`
- Winoground: `qnlp.scripts.coco_single_caption.evaluate_winoground <checkpoint>`
- SugarCREPE: `qnlp.scripts.sugarcrepe.evaluate <checkpoint>`

#### Experiments C and D (Frozen checkpoint)

- ARO: `qnlp.scripts.coco_single_caption.evaluate_aro_frozen <checkpoint>`
- Winoground: `qnlp.scripts.coco_single_caption.evaluate_winoground_frozen <checkpoint>`
- SugarCREPE: `qnlp.scripts.sugarcrepe.evaluate_frozen <checkpoint>`

---

## Notes

- Linear and NLC training use different dataset files — ensure the correct one exists before training
- Frozen eval scripts load `text_model_state_dict` + `text_head_state_dict`, not `model_state_dict`
- SugarCREPE currently only has `swap_att` subset processed — extend if needed
- `ML_BOND_DIM=30` used across all experiments; ensure it matches at evaluation time
