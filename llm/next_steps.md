# Next Steps

## ✅ Done: Dataset Creator Infrastructure & Torch Dataset

- `composition_strategy.py`: `CompositionStrategy` protocol defined.
- `dataset_generator.py`: Refactored around `enrich_atoms`, `split_by_groups`, `create_dataset`, `create_train_val_test_datasets`. `dataset_constructor.py` stub deleted.
- `strategies/single_caption.py`: `SingleCaptionStrategy` — pass-through, one atom per sample.
- `strategies/contrastive_pair.py`: `ContrastivePairStrategy` — labeled path (ARO-native) + unlabeled path (synthetic negatives via derangement).
- `scripts/coco_single_caption/create_dataset.py`: COCO in COCO format.
- `scripts/coco_contrastive/create_dataset.py`: COCO in ARO (contrastive) format with synthesized negatives.
- `domain/datasets/dataset.py`: `VLMDataset` — generic parquet-backed torch Dataset. `compiled_columns` bundles `(diagram, symbols)` into a `(str, [Symbol, ...])` tuple under a named output key, matching the existing model interface.
- `domain/datasets/dataloader.py`: `get_dataloaders` + `vlm_collate_fn` — mirrors `get_aro_dataloader` return shape.

---

## ✅ Done: Training Loop Infrastructure (`qnlp/core/training/`)

- `protocols.py`: `TrainingStep` and `LossFunction` protocols.
- `trainer.py`: Generic `Trainer` — epoch loop, early stopping, checkpointing, MLflow epoch logging, per-model grad clipping.
- `losses/contrastive.py`: `ContrastiveLoss` — InfoNCE + optional triplet loss for contrastive pairs.
- `losses/symmetric_infonce.py`: `SymmetricInfoNCE` — bidirectional CLIP-style loss for single-caption training.
- `losses/single_caption.py`: `SingleCaptionLoss` — `LossFunction` wrapper around `SymmetricInfoNCE`.
- `domain/models/vlm/contrastive_vlm.py`: `ContrastiveVLM` — model wrapper holding text + image sub-models with `clip_gradients` and checkpoint-safe `load_state_dict`.
- `scripts/aro_contrastive/`: `step.py`, `config.py`, `run.py` — fully ported `train_aro_clean.py` on COCO contrastive data.

---

## 1. COCO Single-Caption Training (`qnlp/scripts/coco_single_caption/`)

Train on flat COCO using symmetric InfoNCE. Infrastructure is ready; only the experiment wiring is needed.

- `step.py`: Unpack `local_image_path` + `caption` from batch. Call `model(images, captions)`. Use `SingleCaptionLoss`. No cosine-sim diagnostics needed beyond `accuracy`.
- `config.py`: Hyperparams — `temperature`, `batch_size`, learning rates. `monitor_metric = "accuracy"`.
- `run.py`: Same structure as `aro_contrastive/run.py` but with single-caption parquets and `compiled_columns=[("diagram", "symbols", "caption")]`.

---

## 2. ARO Preprocessing Pipeline (`qnlp/preprocessing_pipelines/aro/`)

No HF dataset located yet — stub the Atlas and implement the pipeline against the assumed schema (mirroring the legacy `aro_dataset.py`): one row per sample with `image_path`, `true_caption`, `false_caption`.

- `steps.py`:
  - `AROFlattenStep`: Melts `true_caption` / `false_caption` into two rows per `sample_id`, adding a `label: bool` column (`True` for positive, `False` for negative).
  - `SchemaMappingStep`: Renames the melted caption column to `processed_text`.
- `pipeline.py`: Wires flatten → schema map → remove dots → lemmatize → CCG compile → unify rank.

Once an HF source is found, only the Atlas creation script needs updating.

---

## 3. Atlas from Local Files

Support ingesting local image/caption datasets (not from HuggingFace) into an Atlas. This unblocks ARO and any other dataset held as local files.

- Add a `LocalFileAtlas` or a `local` ingestion mode to the existing Atlas that reads from a directory or manifest CSV/JSON instead of HF streaming.
- Should produce the same `chunk_*.parquet` derived manifest format so all downstream steps (pipeline, dataset creator) are unchanged.
- Key design question: how to handle `sample_id` generation for local files (auto-increment vs filename-derived).

---

## 4. Mixed Dataset Scripts (`qnlp/scripts/`)

Once ARO derived data exists:

- `scripts/coco_aro_single_caption/create_dataset.py`: `SingleCaptionStrategy` on COCO + ARO derived dirs.
- `scripts/coco_aro_contrastive/create_dataset.py`: `ContrastivePairStrategy` on COCO + ARO derived dirs (ARO groups use labeled path, COCO groups use synthesized negatives).

---

## 5. Migrate ARO & SVO to New Stack

Replace legacy JSONL/CSV-backed dataloaders with the new parquet-backed generic dataloader:
- ARO: driven by `ContrastivePairStrategy` output — `image_columns=["local_image_path"]`.
- SVO: `CompositionStrategy` produces rows with `pos_image_path` and `neg_image_path` — `image_columns=["pos_image_path", "neg_image_path"]`.
- No new dataset classes needed. The generic dataloader handles both.
- Keep legacy dataloaders until end-to-end validation passes.

---

## 6. End-to-End Integration Test

Verify the full pipeline for COCO:
1. Atlas ingest → `data_manifest.parquet` populated.
2. Pipeline → `derived_v1/chunk_*.parquet` created, LMDB populated.
3. Dataset creator → `data/datasets/coco_single_caption_{train,val,test}.parquet` with `diagram` and `symbols`.
4. Torch Dataset → loads parquet, `__getitem__` returns correct structure.
5. Incremental run → only new Atlas rows processed; LMDB skips already-compiled sentences.

---

## 7. Future: Hard Negative Sampling

Implement a new `CompositionStrategy` (e.g. `HardNegativeStrategy`) that accepts a similarity function or embedding index. Drop-in replacement for `ContrastivePairStrategy` with no infrastructure changes. The similarity function is the research boundary.

---

## 8. Future: Multi-Image Flattening (Winoground)

Winoground samples (2 images × 2 captions) require a `MultiImageFlattenStep` that produces 4 atoms per group: `(img0, cap0)`, `(img0, cap1)`, `(img1, cap0)`, `(img1, cap1)`. The `CompositionStrategy` for Winoground reconstruction groups these by `sample_id` and reshapes into `(img0, img1, cap0, cap1)` schema. No changes to the pipeline orchestrator or dataset creator infrastructure.
