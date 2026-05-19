# Overnight changes — 2026-05-07 → 2026-05-08

Context: full-COCO end-to-end run on a UCL CS lab box (`klo`, RTX 3080
Laptop, 16 GB VRAM, 31 GB RAM, Linux/bash). The goal was to take the
113 287-image COCO atlas through the CCG preprocessing pipeline and train
the ARO contrastive model.

This document covers every modification I made: source-code edits in
this repo, data-side workarounds applied on `klo`, and a list of bugs I
diagnosed but did not fix.

---

## 1. Source-code edits (in this repo)

### 1.1 `qnlp/core/data_engine/processing/compiler_step.py` — hybrid GPU/CPU rewrite

**Before:** every multiprocessing worker held its own `BobcatParser` and
ran the full pipeline (parse → diagram → ansatz → tn_to_einsum). With
`device="cuda"` this hit the *fork-pool / CUDA-context* incompatibility
and every parse silently errored ("Cannot re-initialize CUDA in forked
subprocess").

**After:** a hybrid architecture:

- **Main process** owns one `BobcatParser` on the chosen `device`
  (CUDA in our case). It calls `processor.sentences2trees(batch)` once
  per batch — a single batched BERT forward pass on the GPU.
- A **CPU `mp.Pool`** of `max_workers` workers does the per-tree
  post-parse work (`tree.to_diagram` → `rewriter` → `ansatz` →
  `tn_to_einsum`). Workers never touch CUDA.
- Trees stream from the parser into `pool.imap_unordered` so the GPU
  computes batch *N+1* while workers are still chewing through trees from
  batch *N*.

Other things in the same file:

- **`_post_parse_one`** wraps the per-tree work in a `signal.alarm`-based
  wall-clock timeout (default 30 s). Safe because workers are CPU-only —
  signals don't reliably interrupt CUDA kernels, but they do interrupt
  pure-Python lambeq operations.
- **Per-batch heartbeat log** (`Parsed batch N/M`) so any external
  watchdog has steady signal during long chunks.
- **Default `device` changed from `"mps"` to `"cpu"`** so the constructor
  is portable (Mac users override to `"mps"`, Linux users to `"cuda"` or
  leave as-is).
- **New constructor knobs:** `parser_batch_size` (BERT batch size — flows
  through `**kwargs` into `lambeq.bobcat.tagger.Tagger.batch_size`), and
  the existing `max_workers`.

### 1.2a `qnlp/scripts/aro_contrastive/config.py` — bigger batch for InfoNCE

`batch_size: int = 128` → `batch_size: int = 1024`. Contrastive losses benefit
super-linearly from in-batch negatives (each anchor sees `B - 1` negatives in
InfoNCE), and on a 16 GB RTX 3080 Laptop the larger batch fits comfortably for
this model. Reduced epochs from 3521 → 440 batches without losing data, with a
much stronger discriminative gradient.

**File-descriptor gotcha at bs=1024.** On a default Linux shell `ulimit -n` is
1024 — exactly batch size, which means the DataLoader's worker processes
deadlock when they try to open all 1024 image files of a batch in parallel
(symptom: every process at <3% CPU, GPU at 0%, no log progress, no traceback).
Launch training with `ulimit -n 8192` (or higher) before invoking python:

```bash
ulimit -n 8192
python -m qnlp.scripts.aro_contrastive.run
```

Or wrap in bash: `nohup bash -c 'ulimit -n 8192; ... python -m ...'`. After
this fix, all workers came back to 97–99 % CPU and training proceeded
normally. The same caveat applies to any future `batch_size ≥ ulimit -n`.

**OOM gotcha at bs=1024.** Even after the fd fix, the run got OOM-killed at
~1.5 h in. The cause is `bs=1024 × num_workers=4 × prefetch_factor=2` ⇒
8 batches × 1024 image tensors in flight, plus cotengra's path cache
accumulating across the corpus, plus pinned-memory copies. On a 31 GiB
RAM box this pushes into swap and trips the OOM killer (confirmed via
`journalctl -k`). Mitigations if you want to retry bs=1024 properly:
drop `num_workers` from 4 to 2, set `prefetch_factor=1`, drop pinned
memory, or just use `bs=512` (still 4× the original batch).

**Empirical result for the bs=1024 experiment.** Val hard_neg_acc at
epoch 1 was **0.501** vs **0.500** for the bs=128 run on the same
dataset. The 8× larger contrastive batch moved nothing. Combined with
the OOV analysis (96 % of val captions are nominally in-vocab but 58 %
of train symbols are seen ≤ 2 times), this confirms the bottleneck is
the long-tail of barely-trained symbol tensors, not the contrastive
signal strength. Future progress likely requires structural changes
(lemma sharing across typed variants, pretrained word-embedding init)
rather than more data or bigger batches.

### 1.4 `qnlp/scripts/clip_baseline/` — CLIP-from-scratch control experiment

A new script package mirroring `aro_contrastive` in every detail except
the encoders. Same dataset (`coco_single_caption_*.parquet`), same
`ContrastiveLoss` (InfoNCE + triplet, identical hyperparameters), same
`Trainer`, same monitor metric, same early stopping. Only the encoders
change:

  - **`MiniCLIPText`** — a 4-layer, 256-hidden BERT-style transformer
    built via `transformers.BertConfig` + `BertModel(cfg)`. Random init,
    no pretrained weights. ~11.2 M params, vocab 30 522 (BERT WordPiece
    tokenizer downloaded purely for its vocab).
  - **`MiniCLIPImage`** — `torchvision.models.resnet18(weights=None)`
    with the final fc replaced to project into `embedding_dim`. ~11.4 M
    params. Random init.
  - Synthetic negatives are generated **per-batch via random
    derangement** inside `ClipBaselineStep`, rather than pre-computed at
    dataset-creation time (functionally equivalent to
    `ContrastivePairStrategy`'s in-split derangement, just resampled
    every step).

Files:

  - `qnlp/scripts/clip_baseline/__init__.py`
  - `qnlp/scripts/clip_baseline/config.py` — `ExperimentConfig` mirroring
    `aro_contrastive`, defaults tuned for ResNet-18 + small BERT on the
    3080 Laptop (`batch_size=256`, `text_lr=image_lr=head_lr=1e-4`,
    `weight_decay=0.01`, same loss/early-stop knobs).
  - `qnlp/scripts/clip_baseline/step.py` — `ClipBaselineStep`, the
    `TrainingStep` that performs the per-batch derangement and returns
    `hard_neg_acc` against those in-batch negatives.
  - `qnlp/scripts/clip_baseline/run.py` — instantiates everything and
    runs through the same generic `Trainer`. Self-contained: encoders,
    `Dataset`, collate, optimiser, and orchestration all in one file.

Run knobs that turned out to matter on this hardware:

  - `batch_size=1024` (which I tried first) **deadlocks** the DataLoader
    on default `ulimit -n=1024` (each worker tries to open ≥1024 image
    files in parallel; the locks fight, every process sits at <3 % CPU
    forever with no traceback). Fix: `ulimit -n 8192` before launching
    python, OR drop to `batch_size=256` (what I shipped — workers
    cooperate at this size, GPU sits at 100 % util).
  - At `batch_size=1024` with ResNet-18 in fp32, peak activation memory
    blows past the 16 GB VRAM ceiling (~3 GB just for conv1's first
    feature map). 256 fits comfortably in 12.5 GB.
  - `torchvision.io.read_image(path, mode=2)` no longer accepts an int
    in torchvision 0.24; use `ImageReadMode.RGB` instead. (Tripped me
    once.)

### 1.5 Final result of the clip_baseline run

Training ran for **55 epochs** (~29 h wall-clock at ~32 min/epoch on
the RTX 3080 Laptop, GPU at 100 % util the whole time, no cotengra
bottleneck). Early stopped via the standard `patience=10` rule. Best
checkpoint was **epoch 45** with val `hard_neg_acc = 0.9627`.

**Test metrics on the epoch-45 checkpoint:**

```
hard_neg_acc      : 0.9632
true_cosine_mean  : 0.679
false_cosine_mean : 0.049
infonce_loss      : 2.97
triplet_loss      : 0.027
accuracy          : 0.117  (in-batch 1-of-256; random = 0.004)
```

**Side-by-side, same data, same loss, same trainer:**

| Run | Architecture | Train rows | Test hard_neg_acc |
|---|---|---|---|
| EinsumModel smoke | TTN + EinsumModel | 4,791 | 0.473 |
| EinsumModel 1st full | TTN + EinsumModel | 5,584 | 0.518 |
| EinsumModel 13-chunk | TTN + EinsumModel | 9,515 | 0.497 |
| EinsumModel full bs=128 | TTN + EinsumModel | 450,719 | 0.500 (val ep1; OOM-killed before test) |
| EinsumModel full bs=1024 | TTN + EinsumModel | 450,719 | 0.501 (val ep1; OOM-killed before test) |
| **clip_baseline** | **ResNet-18 + small BERT** | **450,719** | **0.9632** |

**Conclusion.** The architectural difference is the entire story.
Both encoders are 11–12 M random-init parameters; both train under
identical conditions; the dataset is the same; the only difference is
that ResNet-18 + BERT-style transformer *share* parameters across
spatial positions and across tokens, while EinsumModel allocates one
free tensor per `(lemma, CCG type)` pair. With ~80 k such pairs
distributed Zipf-like across 450 k train rows, ~58 % of EinsumModel's
parameters never see enough gradient signal to converge — and any val
caption containing one of those parameters degenerates to noise.

This is consistent with the original `llm/handoff.md` §9 prediction
that "the primary open research problem" is the vocabulary gap. The
data confirms it isn't the data, the task, the optimiser, the loss, or
the contrastive batch size — it's the per-symbol-per-typed-variant
parameterisation itself. To make the EinsumModel competitive without
abandoning its inductive bias, weight tying across typed variants of
the same lemma (and ideally pretrained initialisation from a word
embedding) appears unavoidable.

Files written / checkpoints on klo:
- `runs/checkpoints/clip_baseline/<ts>/best_model.pt` — epoch 45 model
- MLflow run `clip-baseline-v4-bs256_2026-05-09_11-29-57` at
  `http://localhost:8080/#/experiments/2/runs/<run-id>` (browse via
  `ssh -L 8080:localhost:8080 klo`).

### 1.6 clip_baseline on the real ARO benchmark — bag-of-words failure mode

To make a clean head-to-head with v1's historical `0.78` ARO number, I wrote
`qnlp/scripts/clip_baseline/evaluate_aro.py` and ran the epoch-45
clip_baseline checkpoint on the existing
`data/aro/processed/visual_genome_{relation,attribution}/test.json` test
sets. (Pre-step: downloaded the 991 unique Visual-Genome images those
test entries reference, from `cs.stanford.edu/people/rak248/VG_100K{_2,}`,
~94 s, stored at `klo:data/aro/images/`.)

Result:

```
                                                hard_neg_acc
visual_genome_relation    (n=3,650)             0.5038
visual_genome_attribution (n=4,438)             0.5038

cosine diagnostics:
                          true_cos    false_cos    gap
relation                  0.489       0.488        +0.001
attribution               0.467       0.468        -0.001
```

**Exactly random on both.** The model assigns essentially identical
similarity to `"the dog is beside the man"` and `"the man is beside the
dog"` for an image of a dog beside a man. This is the well-known
"bag-of-words" failure mode of CLIP-style contrastive training
(Yuksekgonul et al. 2022): with random in-batch negatives, the model is
never penalised for ignoring word order, so the optimum converges to a
token-multiset representation that's structurally blind.

**Combined experimental picture across both metrics:**

| Run | Architecture | Easy synth-neg | ARO hard-neg |
|---|---|---|---|
| EinsumModel (our 5 runs, 80 k symbols) | TTN + EinsumModel | ≈ 0.50 | (not run, expected ≈ 0.50 — vocab tail) |
| EinsumModel — historical v1 | TTN + EinsumModel | (not run) | **0.78** |
| **clip_baseline** | ResNet-18 + small BERT | **0.96** | **0.50** |
| CLIP-large (literature ref) | ViT-L + Transformer | ~0.95+ | ~0.59–0.63 |
| Random baseline | — | 0.50 | 0.50 |

**Interpretation.** The two architectures have inverse failure modes —
not relative quality. CLIP/BERT share parameters across tokens and
positions: this makes them great at distributional / topical semantics
but structurally blind to word order. EinsumModel contracts a *typed*
CCG diagram, so `dog_n` and `dog_n.r@B` are different tensors when
"dog" is in different syntactic positions — structurally compositional
by construction, which is why v1's 0.78 on ARO is genuinely strong (it
beats vanilla CLIP-large). The cost was the long-tail vocabulary
problem that made our 80 k-symbol full-COCO runs collapse to 0.50 on
the easy metric.

The honest framing isn't "X is better than Y"; it's:

- For **topical retrieval** (image search, caption ranking),
  clip_baseline wins.
- For **compositional understanding** (relations, attribute binding,
  structural reasoning), EinsumModel-style typed composition is the
  right inductive bias — *but only if its vocabulary tail is solved*
  (weight tying across typed variants of the same lemma, or
  pretrained word-embedding init for the symbol tensors).

A hybrid model that uses BERT-style shared embeddings at the lexical
layer and EinsumModel-style typed composition above would in principle
get both — open question whether it's tractable to train.

Files added in this round:

- `qnlp/scripts/clip_baseline/evaluate_aro.py` — ARO-relation /
  ARO-attribution eval against the same `hard_neg_acc` metric, batched
  through the existing DataLoader / collate path.
- (One-off, not in repo) `klo:data/aro/images/` — 991 VG images
  downloaded for evaluation.

### 1.7 `qnlp/scripts/einsum_frozen_clip/` — image-side ablation for EinsumModel

The 1.6 result above said clip_baseline only learned bag-of-words. The
v1 EinsumModel+TTN got 0.78 on ARO — so EinsumModel has compositional
capacity in principle, but our full-COCO EinsumModel runs (with 80 622
typed symbols and a long-tail vocab) collapsed to ~0.50 on the *easy*
metric, never reaching the regime where ARO eval is meaningful.

To isolate whether EinsumModel itself is the bottleneck, this experiment
removes image-side training entirely. The image encoder is a **frozen
pretrained CLIP-ViT-B/32** plus one trainable Linear projection to
`embedding_dim`. The text side is the full trainable EinsumModel as
before. If EinsumModel can express a faithful alignment given a perfect
image signal, this run will fit. If it still fails, the bottleneck is
EinsumModel itself (long-tail symbol parameters, or InfoNCE giving it no
incentive to encode structure given the random-negative training).

Files added:

- `qnlp/scripts/einsum_frozen_clip/__init__.py` — package marker.
- `qnlp/scripts/einsum_frozen_clip/config.py` — `ExperimentConfig`
  with `clip_model_name` and `clip_image_size`. Same loss / triplet
  weight / temperature as `aro_contrastive` so the comparison stays
  apples-to-apples. Env prefix `EFC_`.
- `qnlp/scripts/einsum_frozen_clip/image_model.py` — `FrozenClipVisionModel`:
  wraps `transformers.CLIPVisionModel`, sets `requires_grad=False` on the
  backbone, overrides `.train()` so the backbone always stays in
  `.eval()` mode regardless of the outer model's `.train()` /
  `.eval()` switch, runs the forward pass under `torch.no_grad()` (no
  activation-memory cost from the frozen backbone), then projects the
  pooled CLS embedding to `embedding_dim` via a trainable Linear. Output
  is L2-normalised so the downstream `AlignmentHead` behaves identically
  to the TTN case.
- `qnlp/scripts/einsum_frozen_clip/run.py` — training entrypoint.
  Re-uses `ContrastiveVLM`, `AROContrastiveStep`, `Trainer`,
  `ContrastiveLoss`, and the same `coco_contrastive_*.parquet` data as
  `aro_contrastive`. Image transforms use CLIP's standard 224×224
  resize + center-crop + CLIP-specific mean/std normalisation. Optimiser
  has three param groups: EinsumModel (lr 1e-3), CLIP `proj` only (lr
  1e-4, *not* the frozen backbone), and the two `AlignmentHead`s (lr
  1e-3). Stdin is piped at launch so `setup_mlflow_run`'s `input()` call
  doesn't block.
- `qnlp/scripts/einsum_frozen_clip/evaluate_aro.py` — final eval on
  `data/aro/processed/visual_genome_{relation,attribution}/test.json`.
  Reuses `ProcessedARODataset` + `aro_tn_collate_fn`, so the same
  CCG-compiled sidecar files (`{stem}_processed_512.jsonl`) that the
  historical v1 eval used. Image preprocessing matches training
  (CLIP normalisation).

Parameter budget:

- Text (EinsumModel, 80 622 typed symbols): **1.04 B params**, all
  trainable. Dominated by the long tail (~30 % of symbols seen exactly
  once).
- Image (FrozenClipVisionModel): 87.8 M total / **394 k trainable**
  (just the Linear → 512 projection). The CLIP backbone weights are
  loaded once from HuggingFace and never updated.
- Heads: 2 × `AlignmentHead(512)` ≈ 0.5 M trainable.

Klo state at launch (2026-05-12 ~00:55 BST): GPU 1 MB used /
15.98 GB free pre-run, jumps to ~10 GB during training; the EinsumModel
weights + AdamW moments are the bulk of the resident memory. Run name
in MLflow: `einsum-frozen-clip-vit-b32-bs1024`. Log file:
`klo:projects/discoviz/logs/einsum_frozen_clip_2026-05-12_01-56.log`.

Result will be added here once the run finishes — see end of doc.

### 1.2 `qnlp/preprocessing_pipelines/coco/pipeline.py` — explicit device / workers

```python
ccg_parsing_step = CCGCompilerStep(
    lmdb_path=constants.lmdb_path,
    bond_dim=constants.bond_dim,
    embedding_dim=constants.embedding_dim,
    device="cuda",
    parser_batch_size=64,
    max_workers=8,
)
```

`max_workers=8` was the empirical sweet spot on the RTX 3080 Laptop:
4 workers → CPU bottleneck, 8 saturates BERT, 12+ adds nothing.
`parser_batch_size=64` is comfortable on 16 GB VRAM (peak ≈ 3 GB) but
the kwarg's effect plateaus past 8 — the chart-parser CPU work after
BERT is roughly half the per-sentence cost on this dataset.

### 1.3 The two `index.html` edits in `~/projects/index.html`

Unrelated to discoviz but worth noting if anyone's reading this in
sequence: the FQC 2026 site had its third hero meta entry changed
from `Format: In-person workshop` to
`Host: Quantum Learning Labs, Computer Science, UCL`, the palette
swapped to the UCL brand colours (Dark Purple `#361a54`, Heritage Blue
`#30d6ff`, Mid Purple `#ba82ff`, Light Purple `#ddbdff`), the theme
picker UI removed, density locked to `compact`, and `Student rate`
removed from the Fees list.

---

## 2. Bugs I diagnosed but did NOT patch in source

These are hidden landmines for any future user. Worth their own PR but
I didn't have time during the overnight to do it cleanly.

### 2.1 `requirements.txt` is broken on a fresh checkout

| Line | Problem |
|---|---|
| `clip==1.0` | doesn't exist on PyPI; the OpenAI CLIP isn't pip-installable from PyPI. Used only by the legacy `encode_aro_images_with_clip.py`, so safe to drop on the active path. |
| `lemminflext==0.2.3` | typo for `lemminflect`; only used by Tilen's deleted `coco_processing.py`. Safe to drop. |
| `pandas==3.0.0` | doesn't exist; mlflow caps at <3 anyway. Drop the pin. |
| `numpy==2.4.1`, `Pillow==12.1.0` | aspirational pins; resolver gave 2.4.4 / 12.2.0. Drop pins. |
| missing | `pydantic-settings` (imported by `qnlp/constants.py`) |
| missing | `torchmetrics` (imported by `qnlp/core/training/trainer.py`) |
| missing | `transformers<5` — see 2.2 below; this is the load-bearing one |
| missing | `sqlalchemy<3` (only needed for the mlflow sqlite backend) |

### 2.2 lambeq 0.5.0 ↔ transformers 5.x breakage

`transformers 5.x` removed `BertForChartClassification.all_tied_weights_keys`,
which lambeq's Bobcat parser relies on. **Every parse silently errors**
inside the worker; the orchestrator catches the exception and writes a
`{"diagram": null}` payload to LMDB. The pipeline finishes "successfully"
and the dataset creator drops every row as `null`. Symptom: empty
training parquets, no error message anywhere obvious.

Pin **`transformers<5`** in `requirements.txt`.

### 2.3 `compiler_step.py:139` — empty-directory crash on cold LMDB

```python
if self.lmdb_path.exists():
    env = lmdb.open(str(self.lmdb_path), readonly=True, ...)
```

If the directory exists but doesn't contain a real LMDB env (which
happens after a `mkdir -p data/sentence_mapping/`), `lmdb.open(readonly=True)`
raises `lmdb.Error: data/sentence_mapping: No such file or directory`.

Fix: replace with

```python
try:
    env = lmdb.open(str(self.lmdb_path), readonly=True, ...)
except lmdb.Error:
    pass  # treat as empty cache
```

(I applied this in the rewritten `compiler_step.py` shipped here.)

### 2.4 `dataset_generator.py:_is_1d_diagram` — ASCII-only regex misses Unicode outputs

```python
return len(re.findall(r"[a-zA-Z]", output_part)) == 1
```

`opt_einsum.get_symbol(i)` returns `a..z` for `i < 26`, `A..Z` for
`i < 52`, then **Unicode characters beyond ASCII** for `i >= 52`. Long
sentences with many indices use those Unicode letters. A diagram ending
in `->aÀ` (one ASCII letter, one Unicode) returns `len(...)==1` so the
filter accepts it as 1D — but the actual contracted output is rank-2.
At training time the collate function tries to stack `[512]` tensors with
`[512, 512]` and crashes.

Fix: change the regex to `r"\S"` (any non-whitespace). I applied this
manually to the parquet files on `klo` (see §3.4) but the source is still
buggy.

Confirmed in our run: 4 such rows out of ~566 k captions made it into
the parquets after the existing filter dropped 2 201 obvious ones. They
were enough to crash the first 1.5 h of training.

### 2.5 Pipeline orchestrator overwrites chunks on every run

`qnlp/core/data_engine/processing/pipeline.py` writes chunks as
`chunk_{i:06d}.parquet` where `i` is the *iteration counter within this
run*, not the sample-id offset. So a re-run starts numbering from
`chunk_000000.parquet`, **overwriting** any chunks from a previous run
that happened to fail before completing.

In our case this caused us to lose ~4 400 sample_ids of CCG-compiled
content during a kill-and-restart cycle until I noticed and renamed
existing chunks to `chunk_old_*.parquet` (see §3.3).

The fix is to name chunks by the first sample_id they contain
(content-addressed) rather than by iteration index. That way restarts
never overwrite, and `glob("chunk_*.parquet")` still works for the
anti-join.

### 2.6 BERT `sentences2trees` can hang inside CUDA code

Particular COCO captions (the ones that hit lambeq's
`KeyError: Ty(conj)` for unsupported CCG conjunction rules, or possibly
other shapes) cause `parser.sentences2trees(batch)` to hang in a CUDA
kernel. SIGALRM doesn't reliably interrupt CUDA kernels — Python checks
signals between bytecode ops, and CUDA C extensions don't yield while
the GPU op is in flight.

This bit us repeatedly during the overnight run. The proper fix is a
**subprocess-level wall-clock timeout** around the BERT call: run
`sentences2trees` in a child process with `kill -9` after N seconds.

Workaround (which I used, see §3.3): write a sentinel parquet for the
hung chunk's sample_id range so the anti-join skips past it.

### 2.7 `setup_mlflow_run` uses `input()`, hardcodes port

`qnlp/utils/mlflow_utils.py`:

```python
mlflow.set_tracking_uri(f"http://localhost:{port}")
...
run_name = input("Please provide a short description for the run:\n> ")
```

- Port is positional and hardcoded (8080) at the call site in
  `qnlp/scripts/aro_contrastive/run.py`. Should be env-configurable.
- `input()` is fine interactively but for batch runs you have to pipe
  the description via stdin: `echo "smoke" | python -m qnlp.scripts...`.

### 2.8 `device="mps"` was the default in `compiler_step.py`

Mac-only. I changed it to `"cpu"` (see §1.1).

---

## 3. Data-side workarounds applied on `klo`

These are NOT in the repo. They live on `klo:/home/kinianlo/projects/discoviz/`.
Anyone reproducing this should know about them.

### 3.1 Independent venv at `~/venvs/discoviz/`

Built with `uv venv ~/venvs/discoviz --python 3.12`, then a manually
filtered `uv pip install -r /tmp/req-filtered.txt` (the original
`requirements.txt` was unusable; see §2.1). Plus `pydantic-settings`,
`torchmetrics`, `transformers<5`, `sqlalchemy<3` installed explicitly.

### 3.2 MLflow server running as a daemon

```bash
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 8080 \
  > logs/mlflow.log 2>&1 &
```

The training script connects to it. Browse runs by SSH-tunnelling the
port: `ssh -L 8080:localhost:8080 klo`.

### 3.3 Sentinel chunks for hung sample_id ranges

When the BERT call hung mid-batch, I killed the pipeline and wrote
"sentinel" parquets that claim the sample_ids in the hung range so the
orchestrator's anti-join skips them on restart:

```python
sub = manifest.filter(/* sample_id in [start, start+chunk_size) */)
df = pl.DataFrame({
    "sample_id": sub["sample_id"],
    "local_image_path": sub["local_image_path"],
    "processed_text": [None] * len(sub),
    "text_hash": [None] * len(sub),
})
df.write_parquet(f"data/atlases/coco/derived_test/chunk_{start:06d}.parquet")
```

The dataset creator drops these rows as null (because `text_hash` is
null). Cost: ~500 captions per sentinel.

Ranges I sentineled this run:
`1800, 2000, 2200, 2400, 83000`. So chunks `chunk_001800.parquet`,
`chunk_002000.parquet`, etc., on `klo:data/atlases/coco/derived_test/`
contain null text rather than real CCG content.

### 3.4 Strict re-filter of dataset parquets

After §2.4 surfaced, I re-ran a stricter filter on
`data/datasets/coco_*.parquet` in place:

```python
strict_re = re.compile(r'\S')

def is_1d(d):
    if d is None or '->' not in d:
        return True
    op = d.split('->')[1].strip()
    return len(strict_re.findall(op)) == 1
```

Dropped 4 rows total across the 6 parquets. The source-code filter is
still bugged.

### 3.5 `chunk_*.parquet` files were renamed to `chunk_old_*.parquet`

Because of §2.5, before the final preprocess restart I renamed all
existing chunk files so the new run wouldn't overwrite them:

```bash
cd data/atlases/coco/derived_test
for f in chunk_*.parquet; do
    case "$f" in
        chunk_old_*) ;;
        *) mv "$f" "chunk_old_${f#chunk_}" ;;
    esac
done
```

The orchestrator's anti-join uses `glob("*.parquet")` so renamed files
are still consulted for sample_id coverage. The new run wrote `chunk_*.parquet`
without colliding.

After completion, `data/atlases/coco/derived_test/` contains:

- `chunk_old_*.parquet` (831 files, sample_ids from older partial runs)
- `chunk_*.parquet` (303 files from the final clean run)

Both sets are valid input to the dataset creator.

---

## 4. Final state on `klo`

```
~/projects/discoviz/
├── data/atlases/coco/
│   ├── data_manifest.parquet     113 287 images
│   ├── raw_images/               ~10 GB
│   ├── metadata.json             cursor=113287
│   └── derived_test/             1 134 chunks (831 old + 303 new)
├── data/sentence_mapping/        LMDB ~1.2 GB, ~563 k diagrams
├── data/datasets/                450 719 / 56 339 / 56 371 splits
└── runs/checkpoints/             best_model.pt from each run
```

MLflow runs:
- `full-coco-final-v3_2026-05-08_16-56-23` — crashed in epoch 1 from §2.4
- `full-coco-final-v4_*` — relaunched after parquet filter; running at the time of writing

Earlier runs comparison (all on the same architecture, varying data
size):

| Run | Train rows | Test hard_neg_acc |
|---|---|---|
| Smoke (n_rows=2000) | 4 791 | 0.473 |
| 1st incomplete | 5 584 | 0.518 |
| 13-chunk (sentineled) | 9 515 | 0.497 |
| Full (this run) | 450 719 | (pending) |

All previous runs sit within statistical noise of 50 %. The full run is
the first one with enough symbol coverage (~80 624 unique CCG symbols
over 450 k train rows ≈ 5.6 examples per symbol, vs ~0.8 in the 9 515-row
run) that the EinsumModel might actually learn something
generalisable. Result will determine whether scale alone closes the
vocabulary gap or whether a structural change (pretrained text
embeddings, symbol sharing) is needed.
