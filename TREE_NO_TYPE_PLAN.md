# Tree-No-Type Pipeline — Run Checklist

Living checklist for building the `tree_no_type` (lambeq TreeReader NO_TYPE) datasets and
training/evaluating both linear and non-linear COCO models. Update the boxes as you go.

`[x]` done · `[~]` in progress · `[ ]` not started

---

## Step 0 — Prep (once)
- [x] `git pull` on the cluster (`coco-training-nlc`)
- [x] Wipe poisoned artifacts:
  ```bash
  cd /SAN/intelsys/discoviz/fotinos/QNLP
  rm -rf .cache/lambeq/bobcat_tree_no_type      # corrupt tree diskcache
  rm -rf data/sentence_mapping_tree_no_type      # poisoned LMDB (sticky errors)
  rm -rf data/atlases/*/derived_tree_no_type     # partial derived chunks
  rm -f  data/datasets/*_tree_no_type*.parquet   # datasets from bad data
  ```

## Step 1 — Processing (⚠️ ONE JOB AT A TIME — shared diskcache; concurrent cross-host writes corrupt it)
Verify each with `python scripts/check_tree_lmdb.py` (expect ~100% ok) and watch `qstat -j <id> | grep -i maxvmem`.
- [x] ARO — `qsub scripts/submit_aro_process_tree_no_type.sh`
- [x] Winoground — `qsub scripts/submit_winoground_process_tree_no_type.sh`
- [x] SugarCREPE full — `qsub scripts/submit_sugarcrepe_full_process_tree_no_type.sh`
- [x] SugarCREPE++ — `qsub scripts/submit_sugarcrepepp_process_tree_no_type.sh`
- [x] COCO (the long one) — `qsub scripts/submit_coco_pipeline_tree_no_type.sh`
- [ ] Final `check_tree_lmdb.py` shows all datasets ~100% ok

## Step 2 — Dataset creation (may run in parallel; read-only on the cache)
All non-linear (`compute_contraction_paths=True`) — the `path` column is ignored by linear, so one build serves both modes.
- [ ] COCO train/val/test — `qsub scripts/submit_coco_create_dataset_tree_no_type.sh` → `coco_single_caption_nlc_tree_no_type_{train,val,test}`
- [ ] ARO eval — `qsub scripts/submit_aro_create_eval_dataset_tree_no_type.sh` → `aro_eval_tree_no_type`
- [ ] Winoground eval — `qsub scripts/submit_winoground_create_eval_dataset_tree_no_type.sh` → `winoground_eval_tree_no_type`
- [ ] SugarCREPE full eval — `qsub scripts/submit_sugarcrepe_full_create_eval_dataset_tree_no_type.sh` → `sugarcrepe_full_eval_tree_no_type`
- [ ] SugarCREPE++ eval — `qsub scripts/submit_sugarcrepepp_create_eval_dataset_tree_no_type.sh` → `sugarcrepepp_eval_tree_no_type`

## Step 3 — Training (linear ∥ non-linear; both read the same nlc dataset)
- [ ] Non-linear (NLC) — `qsub scripts/submit_coco_multi_caption_tree_no_type.sh`
- [ ] Linear — `qsub scripts/submit_coco_multi_caption_tree_no_type_linear.sh`

## Step 4 — Evaluation
Each training job auto-runs the full benchmark battery at the end (retrieval + Winoground + ARO + SugarCREPE full/++),
reading the `*_eval_tree_no_type` sets — results land in the training log's `FINAL TRAINING REPORT`.
- [ ] NLC end-of-training report captured
- [ ] Linear end-of-training report captured
- [ ] (optional) standalone re-eval: `ML_CHECKPOINT=/path/best_model.pt qsub scripts/submit_evaluate_coco_single_caption_tree_no_type.sh`

---

## Reference — processing config per dataset
| dataset | workers | worker_batch_size | -pe smp | tmem/slot | total |
|---|---|---|---|---|---|
| COCO | 8 | 500 | 10 | 16G | 160G |
| ARO | 4 | 500 | 6 | 8G | 48G |
| Winoground | 4 | 500 | 5 | 8G | 40G |
| SugarCREPE full | 4 | 500 | 5 | 8G | 40G |
| SugarCREPE++ | 4 | 500 | 5 | 8G | 40G |

## Gotchas (why the config is what it is)
- **lambeq leaks memory cumulatively** in tree mode → a worker OOMs → the pool hangs silently. Bounded by recycling every `max_tasks_per_child(5) × worker_batch_size(500) = 2500` sentences/worker. `chunk_size` does NOT help (pool persists across chunks).
- **`tmem` is PER SLOT** on this cluster — a job with no `-pe smp` gets one slot's memory and OOM-kills a worker. Slots also = cores.
- **Run processing sequentially** — all tree jobs share `bobcat_tree_no_type/diskcache`; concurrent jobs on different nodes corrupt the SQLite cache (NFS cross-host locking).
- **polars needs AVX** — node `arbuckle` lacks it → SIGILL. Fixed via `polars[rtcompat]` in the env.
- Datasets are all non-linear; linear training/eval just ignores the `path` column.
