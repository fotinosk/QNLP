"""Rough comparison run: ARO contrastive training with a FROZEN image tower.

The image side is the pre-computed ViT-B/32 LookupEmbedding (frozen, never
trained) keyed by image basename — i.e. a fixed cache of image embeddings. Only
the text model (EinsumModel) is trained. Everything else (dataset, loss, config)
matches the trainable-image run so the text side is comparable.

This is a self-contained probe: it does not touch the dataset parquets or any
shared framework module, and it reuses the same aro_*.parquet datasets.

Note (intentional deviations for a rough check):
  * No alignment heads — text output is compared directly to the frozen image
    embedding (like the original frozen example), unlike the trainable-image run.
  * Honors cfg.use_non_linear_contractions so frozen linear vs non-linear text
    can both be probed.

Usage:
    ML_USE_NON_LINEAR_CONTRACTIONS=true python -m qnlp.scripts.aro_contrastive.run_frozen
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mlflow
import orjson
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.lookup_embeddings import LookupEmbedding
from qnlp.domain.datasets.dataset import _deserialize_symbols, collect_symbol_sizes
from qnlp.scripts.aro_contrastive.config import ExperimentConfig
from qnlp.scripts.aro_contrastive.evaluate import _load_task_map
from qnlp.utils.early_stopping import EarlyStopping, ModelTrainingStatus
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

EXPERIMENT_NAME = "aro_frozen"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
SPLIT_PARQUETS = {
    "train": DATASETS_PATH / "aro_train.parquet",
    "val": DATASETS_PATH / "aro_val.parquet",
    "test": DATASETS_PATH / "aro_test.parquet",
}
LOOKUP_PATH = "models/lookup_embedding_ViT-B_32.pt"
SYMBOL_COLS = ["true_symbols", "false_symbols"]


class FrozenARODataset(Dataset):
    """Yields captions + the image basename (no pixel decode — the image is a
    frozen lookup keyed by basename)."""

    def __init__(self, parquet_path: Path, use_non_linear_contractions: bool):
        self.df = pl.read_parquet(parquet_path)
        self.use_nlc = use_non_linear_contractions

    def __len__(self) -> int:
        return len(self.df)

    def _caption(self, row: dict, prefix: str) -> tuple:
        diagram = row[f"{prefix}_diagram"]
        symbols = _deserialize_symbols(row[f"{prefix}_symbols"])
        if self.use_nlc:
            raw_path = row[f"{prefix}_path"]
            path = [tuple(step) for step in orjson.loads(raw_path)] if raw_path else None
            return (diagram, symbols, path)
        return (diagram, symbols)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)
        return {
            "true_caption": self._caption(row, "true"),
            "false_caption": self._caption(row, "false"),
            "image_key": Path(row["local_image_path"]).name,
            "sample_id": row["sample_id"],
        }


def _collate(batch: list[dict]) -> dict:
    return {k: [item[k] for item in batch] for k in batch[0]}


def _build_loaders(cfg) -> dict[str, DataLoader]:
    loaders = {}
    for split, path in SPLIT_PARQUETS.items():
        ds = FrozenARODataset(path, cfg.use_non_linear_contractions)
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=_collate,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return loaders


def _embed(text_model, image_lookup, batch, device) -> dict[str, torch.Tensor]:
    true_emb = text_model(batch["true_caption"])
    false_emb = text_model(batch["false_caption"])
    image_emb = F.normalize(image_lookup(batch["image_key"]).to(device), dim=-1)
    return {
        "image_embeddings": image_emb,
        "true_caption_embeddings": true_emb,
        "false_caption_embeddings": false_emb,
    }


def _run_epoch(text_model, image_lookup, loader, loss_fn, optimizer, device, train: bool) -> dict[str, float]:
    text_model.train(train)
    totals: dict[str, float] = defaultdict(float)
    n = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if train:
                optimizer.zero_grad()
            outputs = _embed(text_model, image_lookup, batch, device)
            loss, metrics = loss_fn(outputs)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(text_model.parameters(), max_norm=1.0)
                optimizer.step()

            with torch.no_grad():
                pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
                neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
                bs = len(batch["sample_id"])
                totals["loss"] += float(metrics["loss"]) * bs
                totals["infonce_loss"] += float(metrics["infonce_loss"]) * bs
                totals["accuracy"] += float(metrics["accuracy"]) * bs
                totals["hard_neg_acc"] += float((pos > neg).float().mean()) * bs
                totals["true_cosine_mean"] += float(pos.mean()) * bs
                totals["false_cosine_mean"] += float(neg.mean()) * bs
                n += bs

    return {k: v / n for k, v in totals.items()}


def _evaluate_by_task(text_model, image_lookup, loader, device) -> dict[str, dict]:
    text_model.eval()
    task_map = _load_task_map(set(loader.dataset.df["sample_id"].to_list()))
    correct_by_task: dict[str, list[bool]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            outputs = _embed(text_model, image_lookup, batch, device)
            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            correct = (pos > neg).tolist()
            for sid, c in zip(batch["sample_id"], correct):
                correct_by_task[task_map[sid]].append(bool(c))

    results = {}
    all_c: list[bool] = []
    for task in sorted(correct_by_task):
        c = correct_by_task[task]
        results[task] = {"n": len(c), "hard_neg_acc": sum(c) / len(c)}
        all_c += c
    results["overall"] = {"n": len(all_c), "hard_neg_acc": sum(all_c) / len(all_c)}
    return results


def run() -> None:
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()
    nlc = cfg.use_non_linear_contractions

    loaders = _build_loaders(cfg)
    train_ds, val_ds, test_ds = (loaders[s].dataset for s in ("train", "val", "test"))
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | non_linear={nlc}")

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim},
    )

    text_model = EinsumModel(symbols, sizes, non_linear_contractions=nlc).to(device)

    image_lookup = LookupEmbedding.load_from_checkpoint(LOOKUP_PATH, map_location=device).to(device)
    image_lookup.eval()
    for p in image_lookup.parameters():
        p.requires_grad_(False)

    img_dim = image_lookup.embeddings.embedding_dim
    if img_dim != cfg.embedding_dim:
        raise ValueError(
            f"Frozen image embedding dim ({img_dim}) != cfg.embedding_dim ({cfg.embedding_dim}). "
            "Set ML_EMBEDDING_DIM to match the lookup (ViT-B/32 = 512)."
        )

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    # Only the text model is trainable; the image tower is frozen.
    optimizer = torch.optim.AdamW(text_model.parameters(), lr=cfg.text_lr, weight_decay=cfg.text_weight_decay)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_text_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, minimize=False)
    params = {
        **cfg.model_dump(),
        "frozen_image_tower": "ViT-B/32",
        "text_model_params": sum(p.numel() for p in text_model.parameters()),
    }

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as _run:
        for epoch in range(1, cfg.max_epochs + 1):
            train_metrics = _run_epoch(
                text_model, image_lookup, loaders["train"], loss_fn, optimizer, device, train=True
            )
            mlflow.log_metrics({f"train/epoch_{k}": v for k, v in train_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} train: {train_metrics}")

            val_metrics = _run_epoch(text_model, image_lookup, loaders["val"], loss_fn, optimizer, device, train=False)
            mlflow.log_metrics({f"val/epoch_{k}": v for k, v in val_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} val: {val_metrics}")

            status = early_stopping(val_metrics["hard_neg_acc"])
            if status == ModelTrainingStatus.improved:
                torch.save({"text_model_state_dict": text_model.state_dict(), "epoch": epoch}, checkpoint_path)
                logger.info(f"Epoch {epoch}: new best — checkpoint saved.")
            elif status == ModelTrainingStatus.stop:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        # Test with best checkpoint, broken down by task.
        best = torch.load(checkpoint_path, map_location=device)
        text_model.load_state_dict(best["text_model_state_dict"])

        task_results = _evaluate_by_task(text_model, image_lookup, loaders["test"], device)
        logger.info("=== Frozen-image test hard-negative accuracy by task ===")
        for task, r in task_results.items():
            logger.info(f"  {task:<14} n={r['n']:>6}  acc={r['hard_neg_acc']:.4f}")
            mlflow.log_metric(f"test/{task}_hard_neg_acc", r["hard_neg_acc"])


if __name__ == "__main__":
    run()
