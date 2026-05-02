# Project Context: Scalable VLM Data Pipeline & Feature Store

## Motivation

The goal is to train a VLM in many different ways and evaluate it in even more. This framework enables fast iteration by allowing quick composition of datasets for various tasks, with research logic (sampling, pairing, negative selection) cleanly separated from infrastructure.

---

## Codebase Layout

### Active Directories (write new code here)

- **`qnlp/core/`**: Domain-agnostic infrastructure
  - `data_engine/atlas/`: Ingestion & manifest management
  - `data_engine/processing/`: Common pipeline steps & orchestrator
  - `data_engine/dataset_creator/`: Enrichment, composition, splitting
  - `parsing/`: CCG/NLP utilities (`cached_bobcat.py`, `ansatz.py`)
  - `utils/`: Shared utilities

- **`qnlp/domain/`**: Domain-specific implementations
  - `datasets/`: Torch Datasets & dataloaders
  - `models/`: Vision, language, and tensor network models
  - `experiments/`: Trainers and evaluators

- **`qnlp/preprocessing_pipelines/<experiment>/`**: Experiment-specific pipelines
  - `pipeline.py`: Wires together steps for this experiment
  - `steps.py`: Dataset-specific preprocessing steps (flattening, schema mapping, etc.)
  - `dataset_creator.py`: Instantiates `CompositionStrategy` and calls dataset creator infrastructure

- **`qnlp/scripts/<experiment>/`**: Experiment-specific run scripts

### Legacy Directories (kept for comparison — do not add new code)

- `qnlp/discoviz/`: Original implementation. Migration is ongoing and gradual.
- `qnlp/image_tower/`: Duplicate of `qnlp/domain/models/vision/`
- `qnlp/utils/`: Duplicate of `qnlp/core/utils/`

---

## Implementation Status

### 1. Ingestion Layer (`qnlp/core/data_engine/atlas/`) — Complete

- `atlas.py`: `Atlas` class with state via `metadata.json`, incremental ingestion, `sample_id = <name>_<index>`, appends to `data_manifest.parquet`.
- `hf_utils.py`: Lazy HF fetch via `pl.scan_parquet`, concurrent image download via `ThreadPoolExecutor`.
- **Data**: `data/atlases/coco/` exists, cursor at 1200 rows.

### 2. Processing Layer (`qnlp/core/data_engine/processing/`) — Complete

- `pipeline.py`: Anti-join delta detection on `sample_id`, chunked processing, writes to `derived_vX/chunk_*.parquet`.
- `lemmatize_step.py`: spaCy-based normalization. Ensures Rank-1 CCG tensors.
- `conform_rank_step.py`: Polars-native regex to truncate einsum output indices to 1D.
- `compiler_step.py`: SHA-256 hashing, LMDB delta check, multiprocessing CCG compilation, LMDB write.

### 3. Experiment Pipeline (`qnlp/preprocessing_pipelines/coco/`) — Complete

- `pipeline.py`: Full COCO pipeline wired: flatten → schema map → remove dots → lemmatize → CCG compile → unify rank.
- `steps.py`: `COCOFlattenStep` (explodes `sentences_raw`), `SchemaMappingStep`, `RemoveTrailingDotsStep`.

### 4. Parsing & Compilation (`qnlp/core/parsing/`) — Complete

- `cached_bobcat.py`: `CachedBobcatParser` with `diskcache` backing.
- `ansatz.py`: `CustomMPSAnsatz` with MPS decomposition (`bond_dim=10`).

### 5. Dataset Creator (`qnlp/core/data_engine/dataset_creator/`) — Complete

- `composition_strategy.py`: `CompositionStrategy` protocol.
- `dataset_generator.py`: `enrich_atoms`, `split_by_groups`, `create_dataset`, `create_train_val_test_datasets`. Wired around `CompositionStrategy`.
- `strategies/single_caption.py`: `SingleCaptionStrategy` — pass-through, one atom per sample.
- `strategies/contrastive_pair.py`: `ContrastivePairStrategy` — labeled path (ARO-native join) + unlabeled path (synthetic negatives via random derangement over groups, vectorised with Polars joins).
- `dataset_constructor.py`: Deleted.

### 6. Dataset Creation Scripts (`qnlp/scripts/`) — Partial

- `scripts/coco_single_caption/create_dataset.py`: COCO in COCO format.
- `scripts/coco_contrastive/create_dataset.py`: COCO in contrastive format (synthetic negatives).
- COCO+ARO mixed scripts pending ARO pipeline.

### 7. Torch Dataset (`qnlp/domain/datasets/`) — Not yet implemented (next step)

- ARO and SVO dataloaders exist but use legacy JSONL/CSV files and import from `qnlp/discoviz/`. To be migrated.
- New generic dataloader: not yet implemented.

---

## Key Design Decisions

### `sample_id` as Group Key

`sample_id` (e.g. `coco_42`) is generated at ingestion time — one per source sample. After flattening in the pipeline, multiple derived rows share the same `sample_id` (one per caption). This makes `sample_id` the natural group key for splitting and reconstruction.

**Unique atom reference:** `(sample_id, text_hash)` — the composite key that uniquely identifies one flattened atom.

### `CompositionStrategy` Protocol

Research logic for constructing task-specific composite samples is isolated behind a `CompositionStrategy` protocol. Two modes:

- **Reconstruction**: Group atoms by `sample_id`, reshape into target schema. Matches dataset's native structure.
- **Synthesis**: Sample across `sample_id` groups to construct artificial samples (e.g. COCO → Winoground format). Currently random; hard negatives are a future strategy implementation.

Composition always happens **within each split** (after train/val/test group split) to prevent data leakage.

---

## Feature Store

- **Location**: `data/sentence_mapping/` (LMDB)
- **Key**: `text_hash` (SHA-256 of `processed_text`, as bytes)
- **Value**: JSON bytes with `"diagram"` (einsum string) and `"symbols"` (serialized tensor network symbols)
- **Written by**: `CCGCompilerStep` during processing
- **Read by**: Dataset Creator during dataset assembly
- **Not accessed at training time**

---

## Constants (`qnlp/constants.py`)

```python
class Constants(BaseSettings):
    embedding_dim: int = 512
    bond_dim: int = 10
    atlases_path: Path = Path("data/atlases/")
    lmdb_path: Path = Path("data/sentence_mapping/")
    datasets_path: Path = Path("data/datasets/")
    logs_path: Path = Path("runs/logs/")
    checkpoints_path: Path = Path("runs/checkpoints/")
```

---

## Technical Environment

- **Data handling**: `polars` (strict — no pandas)
- **Parallelism**: `concurrent.futures` (I/O), `ProcessPoolExecutor` (CCG compilation)
- **Key dependencies**: `lambeq`, `diskcache`, `polars`, `torch`, `torchvision`, `lmdb`, `spacy`
