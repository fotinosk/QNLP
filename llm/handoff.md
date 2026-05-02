# Researcher Handoff Document

## Quick Reference — Key Documents

| Document | Purpose |
|---|---|
| `llm/spec.md` | Full architecture specification and design decisions |
| `llm/context.md` | Codebase layout, implementation status, active vs legacy directories |
| `llm/model_evolution.md` | Complete history of every model version (v1–v7+) with results |
| `llm/next_steps.md` | Prioritised list of future work |
| `llm/improvements.md` | Recent fixes and incremental improvements |

Read `llm/spec.md` first. Everything in this document assumes you have read it.

---

## 1. Project Goal

Train a Vision-Language Model (VLM) using quantum-inspired tensor network text representations paired with a classical image encoder. The text model (EinsumModel) is built on Combinatory Categorial Grammar (CCG) compiled into tensor network diagrams. The image model (TTNImageModel) is a tree tensor network image encoder.

The framework is designed for fast iteration over many training and evaluation configurations by keeping research logic (loss functions, negative sampling strategies, model heads) cleanly separated from infrastructure (data ingestion, preprocessing pipelines, dataset creation, training loop).

---

## 2. Environment Setup (From Scratch)

### Prerequisites

- Python 3.11+
- `pip install -e .` from the project root (installs `qnlp` as an editable package)
- `pip install -r requirements.txt`
- spaCy model: `python -m spacy download en_core_web_sm`
- lambeq Bobcat parser: first run will auto-download the parser weights (~500MB) to `~/.cache/lambeq/`

### Key dependencies

```
lambeq==0.5.0          # CCG parsing and tensor network compilation
torch==2.9.0           # Deep learning
torchvision==0.24.0    # Image loading and transforms
lmdb==2.2.0            # Feature store (text_hash → compiled diagram)
diskcache==5.6.3       # Bobcat parser cache (avoids re-parsing)
polars                 # All tabular operations — no pandas
mlflow==3.8.1          # Experiment tracking
pyinflect==0.5.1       # MUST be imported to register spaCy token._.inflect extension
```

### All paths are relative to the project root

Running scripts with `python -m qnlp.scripts.<experiment>.run` from the project root resolves all paths correctly via `qnlp/constants.py`.

---

## 3. Local Data Setup (Recreating From Scratch)

All data is local. There is no shared database. New researchers must build their own. This section walks through every step in order.

### Step 1 — Create the COCO Atlas

The Atlas downloads images from HuggingFace and builds the raw manifest. Images are written once and never duplicated.

```python
# scripts/load_coco_to_atlas.py
from qnlp.core.data_engine.atlas.atlas import Atlas

atlas = Atlas.create_atlas(
    name="coco",
    source_url="nlphuji/flickr30k",   # or the COCO HF dataset URL
    base_path="data/atlases/",
)
atlas.ingest_data_from_remote(n_rows=28000)
```

If the atlas already exists and you want to add more rows (incremental):
```python
atlas = Atlas.load_atlas("data/atlases/coco/metadata.json")
atlas.ingest_data_from_remote(n_rows=10000)   # appends next 10k rows
```

**Output:** `data/atlases/coco/raw_images/`, `data/atlases/coco/data_manifest.parquet`, `data/atlases/coco/metadata.json`

**Note:** Ingestion is idempotent and resumable. Cursor state is maintained in `metadata.json`.

### Step 2 — Run the Preprocessing Pipeline

The pipeline transforms raw text from the manifest into CCG-compiled tensor diagrams, stores compiled output in LMDB, and writes derived atom chunks.

```python
# qnlp/preprocessing_pipelines/coco/pipeline.py
# Run directly:
python -m qnlp.preprocessing_pipelines.coco.pipeline
```

This runs: Flatten → Schema Map → Remove Trailing Dots → Lemmatize → CCG Compile → Unify Rank.

**Output:** `data/atlases/coco/derived_test/chunk_*.parquet`, `data/sentence_mapping/` (LMDB)

**Notes:**
- CCG compilation is slow on first run (~1-2 sentences/second). Subsequent runs skip already-compiled sentences via the LMDB delta check.
- The Bobcat parser cache lives in `~/.cache/lambeq/`. If you move machines, this cache does not transfer (it will re-compile on first run).
- Pipeline is idempotent — re-running only processes new manifest rows.

### Step 3 — Create the Training Datasets

Dataset creation reads the derived atoms, enriches them from LMDB, and composes them into task-specific train/val/test parquets.

```python
# Single-caption dataset (one image, one caption per row)
python -m qnlp.scripts.coco_single_caption.create_dataset

# Contrastive dataset (one image, true caption, synthetic false caption per row)
python -m qnlp.scripts.coco_contrastive.create_dataset
```

**Output:** `data/datasets/coco_single_caption_{train,val,test}.parquet`, `data/datasets/coco_contrastive_{train,val,test}.parquet`

**Notes:**
- The split is deterministic and seeded. Re-running produces identical splits.
- `filter_2d_outputs=True` is set by default in `enrich_atoms()`. This discards old LMDB entries with 2D einsum outputs (a known bug from earlier pipeline versions). Do not disable this flag.
- Composition happens independently within each split — synthetic negatives only sample from the same split. This prevents data leakage.

### Step 4 — Start MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 8080
# Visit http://localhost:8080
```

### Step 5 — Run Training

```bash
python -m qnlp.scripts.aro_contrastive.run       # v1 baseline (proven 78%)
python -m qnlp.scripts.coco_single_caption.run   # single-caption InfoNCE training
```

Each training script prompts for a run description before starting. This is logged to MLflow as the run name.

---

## 4. Data Flow (Summary)

```
HuggingFace
    ↓  Atlas.ingest_data_from_remote()
data/atlases/coco/
    raw_images/             ← image files, written once
    data_manifest.parquet   ← one row per image, flexible schema
    ↓  preprocessing_pipelines/coco/pipeline.py
    derived_test/chunk_*.parquet   ← one row per (image, text) atom
    ↓  (side-effect of pipeline)
data/sentence_mapping/      ← LMDB: text_hash → {diagram, symbols}
    ↓  scripts/*/create_dataset.py
data/datasets/
    coco_single_caption_{train,val,test}.parquet
    coco_contrastive_{train,val,test}.parquet
    ↓  VLMDataset + DataLoader
Training
```

---

## 5. What Can and Cannot Be Changed Per Experiment

This is the most important section for new researchers. The framework is deliberately split into infrastructure (never change per experiment) and research boundary (change freely per experiment).

### NEVER change per experiment

These are stable infrastructure components. Changing them breaks everything downstream.

| Component | Location | Why frozen |
|---|---|---|
| Atlas ingestion | `qnlp/core/data_engine/atlas/` | Manages cursor state, image deduplication, `sample_id` generation |
| Pipeline orchestrator | `qnlp/core/data_engine/processing/pipeline.py` | Delta detection, LMDB writes, derived chunk management |
| `LemmatizeStep` | `qnlp/core/data_engine/processing/lemmatize_step.py` | Ensures CCG produces Rank-1 tensors — changing this invalidates all existing LMDB entries |
| `CCGCompilerStep` | `qnlp/core/data_engine/processing/compiler_step.py` | Writes LMDB. Changing compilation parameters requires purging and rebuilding the entire LMDB |
| `UnifyEinsumRankStep` | `qnlp/core/data_engine/processing/conform_rank_step.py` | Corrects 2D einsum outputs — must always be the last pipeline step |
| Dataset Creator infrastructure | `qnlp/core/data_engine/dataset_creator/dataset_generator.py` | Enrichment from LMDB, group splitting — the split logic must stay stable across experiments |
| `VLMDataset` + `vlm_collate_fn` | `qnlp/domain/datasets/dataset.py`, `dataloader.py` | Generic — serves all experiments unchanged |
| `Trainer` | `qnlp/core/training/trainer.py` | Generic training loop — should not be experiment-specific |
| `constants.py` | `qnlp/constants.py` | Single source of truth for all paths |
| Data schemas | Defined in `spec.md` section 4 | All components depend on these contracts |

### Change freely per experiment

These are the research boundary — everything you iterate on lives here.

| Component | Location | What to change |
|---|---|---|
| **CompositionStrategy** | `qnlp/core/data_engine/dataset_creator/strategies/` | How atoms are paired and negatives selected. Implement a new strategy class for each new task format (single caption, contrastive, hard negatives, Winoground, etc.) |
| **Preprocessing steps** | `qnlp/preprocessing_pipelines/<experiment>/steps.py` | Dataset-specific flattening, schema mapping, text cleaning |
| **Pipeline wiring** | `qnlp/preprocessing_pipelines/<experiment>/pipeline.py` | Which steps to apply and in what order (must still end with `UnifyEinsumRankStep`) |
| **ExperimentConfig** | `qnlp/scripts/<experiment>/config.py` | All hyperparameters: learning rates, batch size, temperature, warmup epochs, patience, etc. |
| **Loss function** | `qnlp/core/training/losses/` | The loss and all training metrics. Current losses: `ContrastiveLoss`, `SymmetricInfoNCE`, `SingleCaptionLoss`, `VICReg` |
| **TrainingStep** | `qnlp/scripts/<experiment>/step.py` | Batch unpacking, model forward call, loss invocation |
| **Run script** | `qnlp/scripts/<experiment>/run.py` | Optimizer definition, scheduler, monitor metric, checkpoint path |
| **Model architecture** | `qnlp/domain/models/vlm/contrastive_vlm.py` | Heads, projectors, normalisation layers. The text and image backbones (`EinsumModel`, `TTNImageModel`) are in legacy and should be treated as fixed for now |

### The key rule

If a change requires regenerating parquet files or the LMDB, it is a pipeline-level change. If it only affects what happens after `data/datasets/*.parquet` exists, it is a research-level change.

---

## 6. Model Architecture

### Text Model — EinsumModel (`qnlp/discoviz/models/einsum_model.py`)

A quantum-inspired tensor network model. Each word symbol has a learnable tensor parameter. Sentences are represented as CCG diagram strings (einsum notation) which contract those tensors into a single embedding vector.

- **Input:** `(diagram_str, [Symbol, ...])` tuple — the `caption` batch key
- **Output:** `[B, D]` embedding tensor (D=512 by default)
- **Parameters:** One tensor per unique symbol in the vocabulary. Symbols are learned from scratch — there is no pretrained initialisation.
- **Key limitation:** Symbols never seen during training have random parameters and produce meaningless embeddings. Val/test captions with out-of-vocabulary symbols must be filtered or skipped. This is handled in evaluation scripts via `known_symbols = set(model.text_model.sym2weight.keys())`.

### Image Model — TTNImageModel (`qnlp/discoviz/models/image_model.py`)

A Tree Tensor Network image encoder. Takes a preprocessed image tensor `[B, C, H, W]` and produces a `[B, D]` embedding.

### ContrastiveVLM (`qnlp/domain/models/vlm/contrastive_vlm.py`)

Wrapper that holds both backbones plus per-modality alignment heads.

```
images → TTNImageModel → AlignmentHead (Linear → L2Norm) → image_emb [B, D]
captions → EinsumModel → AlignmentHead (Linear → L2Norm) → text_emb [B, D]
```

`AlignmentHead` is initialised as identity (no-op at epoch 0) so training starts from the same geometry as having no head. It is checkpointed and used at eval time.

**`forward(images, true_captions, false_captions=None)`** — pass `false_captions` only for contrastive tasks (ARO-style).

---

## 7. Current Training Experiments

### v1 — ARO Contrastive (`qnlp/scripts/aro_contrastive/`)

**Status: Production baseline. Use this as the reference.**

- Loss: InfoNCE + TripletMarginLoss (`triplet_weight=40000`)
- Data: `coco_contrastive_{train,val,test}.parquet` (explicit true/false caption pairs per image)
- Monitor: `hard_neg_acc`
- Result: **78% hard_neg_accuracy on ARO test set**

### v7+ — COCO Single Caption (`qnlp/scripts/coco_single_caption/`)

**Status: Active research. Not yet matching v1.**

- Loss: SymmetricInfoNCE (learnable temperature, clamped to T ∈ [0.01, 0.3]) + alignment warmup
- Data: `coco_single_caption_{train,val,test}.parquet` (one image, one caption per row, no explicit negatives)
- Monitor: `hard_neg_accuracy` (batch-size-agnostic, maximize)
- Warmup: `alignment_weight=0.5` for first 5 epochs to push embeddings into same hemisphere, then 0.0
- Best result so far: ~3.3% hard_neg_accuracy on test (random baseline: 0.2% with B=512)

For a complete history of every architecture change, failure mode, and result, see `llm/model_evolution.md`.

---

## 8. Evaluation Scripts

Two evaluation scripts exist in `qnlp/scripts/coco_single_caption/`. Both load a checkpoint and run inference on the test set — no training.

### `evaluate_aro.py`

Binary hard-negative accuracy. For each test sample: does the model rank the true caption higher than the false caption?

- Data: `coco_contrastive_test.parquet` (requires explicit true/false pairs)
- Primary metric: `hard_neg_accuracy` (50% = random)
- Also reports: `true_cosine_similarity`, `false_cosine_similarity`, `margin`
- Skips samples with out-of-vocabulary symbols

### `evaluate_retrieval.py`

Full-corpus retrieval evaluation. Encodes all test samples, builds an [N × N] similarity matrix, ranks captions per image.

- Data: `coco_single_caption_test.parquet`
- Metrics: R@1, R@5, R@10, Median Rank, Mean Rank, MRR (both i2t and t2i)
- Also computes batch-level accuracy (sliding window, comparable to training InfoNCE accuracy)
- Skips samples with out-of-vocabulary symbols or 2D diagram outputs

To run either script, update the `checkpoint_path` in the `__main__` block at the bottom of the file and run:

```bash
python -m qnlp.scripts.coco_single_caption.evaluate_aro
python -m qnlp.scripts.coco_single_caption.evaluate_retrieval
```

---

## 9. Known Issues and Open Questions

### EinsumModel vocabulary gap

Val and test captions contain symbols never seen during training. These symbols receive no gradient and stay near random initialisation. In practice, ~32% of ARO test samples are skipped during evaluation due to unknown symbols. This is a fundamental ceiling on generalisation and is the primary open research problem.

### 2D diagram outputs

Older LMDB entries (compiled before `UnifyEinsumRankStep` was added) have 2D einsum outputs that cause `RuntimeError: stack expects each tensor to be equal size`. This is handled by:
- `filter_2d_outputs=True` in `enrich_atoms()` during dataset creation (discards bad atoms)
- `_is_1d_diagram()` check in evaluation scripts (skips bad samples at inference time)

If you see this error, the sample passed through one of the missing filters.

### Val metrics noise

The val set has ~7 batches of 512 samples. Batch-level metrics like `hard_neg_accuracy` are sensitive to which samples share a batch, causing high epoch-to-epoch variance. Val loss is more stable but also noisy. Retrieval metrics computed over the full val set (as in `evaluate_retrieval.py`) are more reliable but expensive.

### Training speed

Each epoch takes ~65 seconds. The bottleneck is the EinsumModel forward pass — it runs 512 sequential cotengra einsum contractions per batch. This does not parallelise within a batch. Mitigation options: cotengra path caching across epochs, epoch-level text embedding cache, `torch.autocast(bfloat16)`.

---

## 10. Adding a New Experiment

This is the standard pattern for adding a new training experiment.

1. **Create a preprocessing pipeline** (if the data format is new): `qnlp/preprocessing_pipelines/<name>/pipeline.py`, `steps.py`
2. **Create a CompositionStrategy** (if the task format is new): `qnlp/core/data_engine/dataset_creator/strategies/<name>.py`
3. **Create the dataset**: `qnlp/scripts/<name>/create_dataset.py` — calls `create_train_val_test_datasets` with your strategy
4. **Create the experiment script**: `qnlp/scripts/<name>/config.py`, `step.py`, `run.py`
5. **Create a loss** if needed: `qnlp/core/training/losses/<name>.py`

Steps 1-3 touch infrastructure lightly (add files, don't modify existing ones). Steps 4-5 are purely additive.

Do not add code to `qnlp/discoviz/`, `qnlp/image_tower/`, or `qnlp/utils/`. These are legacy directories kept for reference only.
