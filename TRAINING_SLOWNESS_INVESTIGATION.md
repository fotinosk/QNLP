# Investigation brief: COCO linear training is still slow (~40–70 min/epoch) despite a batching fast path that should have made it much faster

**Date**: 2026-07-19
**Repo**: `/Users/fotinoskyriakides/Desktop/Dev/qnlp` (or the cluster mirror at `/SAN/intelsys/discoviz/fotinos/QNLP`)
**Branch**: `coco-training-nlc`
**Conda env**: always run Python via `conda run -n qnlp <command>` — base env lacks torch/polars/lambeq.

## TL;DR — what you're being asked to do

We implemented a batching optimization (`TopologyBucketSampler` + `EinsumModel`'s batched-same-topology fast path) intended to fix a severe training bottleneck: `EinsumModel.forward` is a Python `for`-loop calling one `opt_einsum` contraction per sample. On the real COCO dataset, ~80% of rows share a diagram topology with at least 63 other rows, so grouping same-topology rows into batches and doing ONE contraction call per batch (instead of one per sample) should have been a huge win.

**It wasn't.** Epochs still take 40–70 minutes, essentially unchanged from before the optimization. We benchmarked the contraction itself in isolation on 4 different cluster GPU nodes and confirmed the batched contraction is 300x–1000x faster than doing it sequentially, in both float32 and float64 — so **the contraction is not the bottleneck**. Something else in the training pipeline is dominating wall-clock time, and we haven't found it yet. That's your job: find where the 40–70 minutes is actually going, and fix it.

**Do not** re-investigate or re-fix the correctness bugs listed under "Already fixed, do not re-litigate" below — those are settled and verified. This is now purely a *performance* investigation.

---

## Architecture context

This is a vision-language contrastive model (COCO retrieval + Winoground/ARO/SugarCREPE compositional benchmarks). Two components:

- **Text encoder**: `EinsumModel` (`qnlp/discoviz/models/einsum_model.py`) — a "quantum-inspired" tensor-network text encoder. Each word/symbol in the vocabulary has a learned tensor (`nn.Parameter`). A sentence is represented as a CCG-derived tensor-network diagram (an `opt_einsum` expression string + list of symbols), and the sentence embedding is the contraction of all its word-tensors together, followed by L2-normalize.
- **Image encoder**: either `TTNImageModel` (trainable tensor-network image tower) or a **frozen CLIP ViT-B/32** tower (cached embeddings) — hence the "frozen" vs "non-frozen" script variants.
- **Training entry points** (4 total, all should behave similarly since they share the same underlying code):
  - `qnlp/scripts/coco_single_caption/run.py` (non-frozen)
  - `qnlp/scripts/coco_single_caption/run_frozen.py` (frozen)
  - `qnlp/scripts/coco_multi_caption/run.py` (non-frozen)
  - `qnlp/scripts/coco_multi_caption/run_frozen.py` (frozen)

Non-frozen scripts use a generic `Trainer` class (`qnlp/core/training/trainer.py`) with a standard epoch loop. Frozen scripts have their own hand-rolled, near-duplicate training loop (`_run_epoch` function defined locally in each `run_frozen.py`) — **not** using the `Trainer` class. Keep this duplication in mind: whatever the bottleneck turns out to be, check both loop implementations, since they're separate code paths that happen to look similar.

Dataset: `coco_single_caption_nlc_tree_no_type_{train,val,test}.parquet` in `data/datasets/` — 452,783 train rows, 56,601 val rows, 56,608 test rows. Each row has a `diagram` (opt_einsum expression string), `symbols` (list of `[symbol_dict, shape]` pairs, JSON-serialized), a `caption` string, `local_image_path`, and `sample_id`.

Training config: `batch_size=256` (single-caption) or `512` (multi-caption), `embedding_dim=512`, `bond_dim=10`, run via `ML_*` env vars (pydantic-settings, `qnlp/scripts/coco_*/config.py`) and submitted to an SGE cluster via `qsub scripts/submit_coco_*_tree_no_type_linear*.sh`.

---

## What was implemented this session (the speed optimization)

### 1. `TopologyBucketSampler` (`qnlp/domain/datasets/topology_bucket_sampler.py`, new file)

A custom `torch.utils.data.Sampler` (used as `batch_sampler=`) that groups dataset row-indices by their `diagram` string. Diagrams occurring `>= min_bucket_size` times (default 64) get "pure" batches — every row in the batch shares the exact same diagram/topology. Diagrams below that threshold are pooled into ordinary heterogeneous "tail" batches (same as today's random batching).

Verified on the real training parquet (452,783 rows, 28,996 unique diagrams): top 500 diagrams cover 76.8% of rows, top 1000 cover 83.7%. No truncation — `tail_fraction=1.0` hardcoded for val/test (enforced structurally, not just by default), full dataset coverage every epoch, verified via unit test (every index appears in exactly one batch).

### 2. `EinsumModel`'s batched fast path (`qnlp/discoviz/models/einsum_model.py`)

`forward()` now checks: if linear mode (not NLC) AND batch size > 1 AND every sample in the batch shares the same diagram string → dispatch to `_forward_batch_same_topology()`, which:
- Stacks each structural position across the batch (`torch.stack` of the relevant symbol's weight tensor for each of the B samples at that position) into a `[B, *shape]` tensor.
- Defensively re-verifies per-position shape alignment across the batch (falls back to the old per-sample loop on any mismatch, rather than risk silently contracting mismatched legs).
- Runs **ONE** cached `opt_einsum.contract_expression` call for the whole batch (with an added shared "batch" index in the einsum string, e.g. `"ab,bc->ac"` → `"zab,zbc->zac"`), instead of one call per sample.

Verified numerically: fast path output is bit-identical (0.0 max diff) to the old per-sample loop, and gradients match to float32 machine epsilon.

Wired into all 4 training scripts via `get_dataloaders(topology_bucketing=not cfg.use_non_linear_contractions)` (non-frozen, `qnlp/domain/datasets/dataloader.py`) and equivalent manual wiring in both `run_frozen.py` scripts' `_build_loaders`/`_dedup_loader`.

Diagnostic counters `text_model.fast_path_batches` / `text_model.fallback_batches` exist on `EinsumModel` and are logged at the end of non-frozen training runs (`logger.info(f"Forward path statistics: ...")`) — **check these first** in the next run's logs to confirm the fast path really is engaging at the expected ~80% rate. (It was confirmed once already in an earlier, now-superseded run: `18181 batches on batched fast path, 4553 batches on sequential fallback path` ≈ 80%.)

### 3. Numerical fixes (unrelated to speed, but landed in the same session — don't re-investigate)

These are **already fixed and verified**, do not re-diagnose:
- `EinsumModel`'s weight-norm layer was pinning every symbol to unit Frobenius norm, causing exponential magnitude decay for realistic sentence lengths (~1e-18 by 20 symbols) → vanishing gradients → models stuck at chance accuracy. Fixed: reverted to per-symbol target-norm scaling (`_target_norm`/`_rescale_single`/`_rescale_batched`) + the contraction now runs in **float64** (cast back to float32 only after the final `F.normalize`) for headroom against overflow. Verified via unit tests: both the underflow scenario (typical random directions, chain lengths 5–20) and the overflow scenario (correlated/aligned directions) now stay finite and correctly normalized.
- `EinsumModel.load_state_dict` was silently resetting the model to CPU on every checkpoint reload (`torch.empty()` with no `device=`) regardless of an earlier `.to(device)` call — caused a `cuda:0`/`cpu` device-mismatch crash in `run_frozen.py`'s final test evaluation. Fixed: now captures device/dtype before rebuilding weights.
- `TopologyBucketSampler` could produce a batch of exactly size 1 as a bucket's trailing remainder, which made `SingleCaptionLoss`'s `S[~eye].mean()` (off-diagonal similarity mean) literally NaN for a 1x1 similarity matrix — poisoning that epoch's entire accumulated metric under naive summation. Fixed: merges size-1 remainders into the previous chunk.

All three were verified fixed in the most recent job run: no crashes, no `nan` anywhere in the logs, `sim_ratio` values are real finite numbers now.

---

## The actual mystery: why didn't epoch time improve?

### The benchmark that ruled out float64 (already done — do not repeat)

Hypothesis tested: maybe the float64 contraction (needed for the correctness fix above) is disproportionately slow on this cluster's GPUs (some non-datacenter GPUs have crippled fp64 throughput, e.g. 1/32 of fp32), cancelling out the batching win.

**Ruled out.** Benchmark script `scripts/benchmark_fp32_vs_fp64_contraction.py` (still in the repo, already run) measures single-sample vs batched (B=256) contraction time for a realistic diagram shape (bond_dim=10, embedding_dim=512, 15-position chain), in both float32 and float64, run via `scripts/submit_benchmark_fp32_vs_fp64.sh`. Results across 4 different cluster nodes:

| Node | fp32 single | fp32 batched | fp32 speedup | fp64 single | fp64 batched | fp64 speedup |
|---|---|---|---|---|---|---|
| gonzo-605-13 (card 1) | 8.59ms | 35.10ms | 62.7x | 10.53ms | **2.64ms** | 1019.9x |
| gonzo-605-15 (card 3) | 4.85ms | 2.51ms | 493.6x | 5.71ms | 3.14ms | 465.1x |
| gonzo-605-15 (card 2) | 4.14ms | 2.31ms | 458.0x | 5.82ms | 2.84ms | 524.1x |
| gonzo-605-15 (card 1) | 3.98ms | 2.86ms | 355.8x | 5.89ms | 3.56ms | 424.0x |

Batching gives 350x–1000x+ speedup in **both** dtypes, on **every** node tested. A full epoch's worth of contractions (~1700 batches × ~3ms each) should take on the order of **seconds**, not 40–70 minutes. **The text contraction is not the bottleneck** — something else in the pipeline dominates, before or after the contraction, and it was already dominating before this optimization too (which is *why* the optimization didn't move the needle — it sped up a part of the pipeline that was never the critical path).

### Suspects not yet investigated (start here)

**1. `num_workers=0` hardcoded in every frozen DataLoader.** In both `run_frozen.py` scripts, `_dedup_loader` (val/test) and `_build_loaders`'s train loader construction pass `num_workers=0` to every `DataLoader`. This means `FrozenCOCODataset.__getitem__` (which deserializes the `symbols` JSON column via `orjson.loads` + reconstructs `Symbol` dataclasses, per row) runs **synchronously in the main process**, with **zero prefetching/overlap** against GPU compute. This is a strong, easy-to-check suspect. Also suspicious: frozen epoch 2 (which should be fast, since `CLIPImageCache` caches image embeddings after first use — see class docstring "Subsequent epochs are instant lookups") was **not** meaningfully faster than epoch 1 in the observed logs — if data-loading/deserialization dominates regardless of image caching, that would explain why the promised epoch-2+ speedup never showed up either. Worth checking:
   - Does raising `num_workers` from 0 to e.g. 4 in the frozen loaders change epoch time?
   - Is `CLIPImageCache` actually being reused across epochs (same instance, not accidentally recreated per epoch)? Check `run_frozen.py`'s `run()` function — is `image_cache` constructed once outside the epoch loop, or could it be getting rebuilt somewhere?

**2. Image decoding cost for non-frozen runs.** `VLMDataset.__getitem__` (`qnlp/domain/datasets/dataset.py`) calls `torchvision.io.read_image(...).float().div(255.0)` + a transform, per sample, every epoch, for all ~452K training rows (no caching — images are always re-decoded from disk). This *is* parallelized via `num_workers=4` (the `get_dataloaders` default), but real disk I/O + JPEG decode cost at this row count could still dominate, independent of anything on the text side. Worth profiling directly: how much wall-clock time per epoch is spent waiting on the DataLoader (i.e., time between `for batch in loader:` iterations where the model/optimizer isn't running) versus actual forward+backward+step time?

**3. Anything else that scales with row/batch count and wasn't touched by this optimization** — e.g. the `TTNImageModel` forward/backward (non-frozen only), the optimizer step itself, gradient clipping, MLflow logging overhead per batch, or Python-level overhead in `_run_epoch`'s manual metric accumulation (`totals[k] += float(v) * bs` — note the `float(v)` here forces a CPU sync on every single batch for frozen scripts, unlike the non-frozen `Trainer`'s `MetricsAccumulator`, which explicitly uses `torchmetrics.MeanMetric` to *avoid* per-batch `.item()`/CPU-sync calls, per its own docstring: "Avoids .item() / CPU sync in the hot loop — sync happens once at epoch end via compute()". This is a real, structural difference between the frozen and non-frozen loops worth investigating — per-batch CPU syncs can serialize the GPU queue and silently dominate wall-clock time, especially at ~1700+ batches/epoch.

### Recommended approach

Don't guess again — measure directly, since the float64 guess already turned out wrong once. Suggested method:

1. Add lightweight step-level timing to the training loop (both `Trainer._run_epoch` in `qnlp/core/training/trainer.py`, and the standalone `_run_epoch` functions in both `run_frozen.py` scripts) — wrap with `time.perf_counter()` around three phases: (a) waiting for the next batch from the DataLoader, (b) forward pass, (c) backward + optimizer step. Log the average per-epoch breakdown (or use `torch.profiler` for a more detailed trace if preferred).
2. Run one short training job (a handful of epochs is enough) with this instrumentation on both a frozen and non-frozen script, and inspect where the time actually goes.
3. Cross-check against `nvidia-smi`/GPU utilization during a run — if the GPU is mostly idle (low utilization) while wall-clock time passes, that confirms a CPU-bound/data-loading bottleneck rather than a compute-bound one.
4. Fix whatever the profiling reveals — don't assume it's `num_workers=0` or image decoding without confirming; those are informed suspects based on code inspection, not confirmed root causes.

---

## File index

| File | Role |
|---|---|
| `qnlp/discoviz/models/einsum_model.py` | Text encoder; batched fast path; the 3 already-fixed correctness bugs |
| `qnlp/domain/datasets/topology_bucket_sampler.py` | `TopologyBucketSampler`, `set_loader_epoch` helper |
| `qnlp/domain/datasets/dataloader.py` | `get_dataloaders` (non-frozen loader construction, incl. `topology_bucketing=` wiring) |
| `qnlp/domain/datasets/dataset.py` | `VLMDataset` — image decoding happens in `__getitem__` here |
| `qnlp/core/training/trainer.py` | Generic `Trainer` class used by non-frozen `run.py` scripts |
| `qnlp/scripts/coco_single_caption/run.py` | Non-frozen single-caption training entry point |
| `qnlp/scripts/coco_single_caption/run_frozen.py` | Frozen single-caption training entry point (own `_run_epoch`, `FrozenCOCODataset`, `CLIPImageCache`) |
| `qnlp/scripts/coco_multi_caption/run.py` | Non-frozen multi-caption training entry point |
| `qnlp/scripts/coco_multi_caption/run_frozen.py` | Frozen multi-caption training entry point |
| `scripts/benchmark_fp32_vs_fp64_contraction.py` | The benchmark that ruled out float64 as the bottleneck (already run, results above) |
| `scripts/submit_benchmark_fp32_vs_fp64.sh` | qsub wrapper for the above |
| `scripts/submit_coco_single_caption_tree_no_type_linear.sh` | Submit script, non-frozen single-caption (job name `coco_sc_tree_lin`) |
| `scripts/submit_coco_single_caption_tree_no_type_linear_frozen.sh` | Submit script, frozen single-caption (job name `coco_sc_frozen_tree_lin`) |
| `scripts/submit_coco_multi_caption_tree_no_type_linear.sh` | Submit script, non-frozen multi-caption (job name `coco_mc_tree_lin`) |
| `scripts/submit_coco_multi_caption_tree_no_type_linear_frozen.sh` | Submit script, frozen multi-caption (job name `coco_frozen_tree_lin`) |

## Cluster/environment conventions

- Always run Python via `conda run -n qnlp <command>` (or on the cluster, the env is at `/SAN/intelsys/discoviz/envs/qnlp311`).
- Jobs submitted via `qsub scripts/submit_*.sh`; output logs land in `job_outputs/<job_name>.o<job_id>` (locally referenced as `/SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/` on the cluster).
- `PARSER_VERSION=tree_no_type` env var selects the tree-reader parser variant (vs. the original Bobcat parser) — this is what all current training runs use; it doesn't affect the performance investigation, just note it if you see `tree_no_type` in dataset/job names.
- Security: never hardcode `HF_TOKEN` — pass via `qsub -v HF_TOKEN=...` only, per existing project convention.
- Do not push to any remote unless explicitly asked. Only commit when explicitly asked.

## What "done" looks like

A clear, evidence-based (not guessed) identification of where the 40–70 minutes per epoch is actually spent, and a fix that measurably reduces it — ideally validated by a short training run showing epoch time dropping substantially, consistent with the fact that the text contraction itself now costs only seconds per epoch.
