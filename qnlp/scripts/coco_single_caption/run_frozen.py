"""COCO single-caption training with a FROZEN image tower.

Images are encoded on-the-fly with CLIP ViT-B/32 and cached in memory, so
only the first epoch pays the encoding cost. Subsequent epochs are instant
lookups. Only the text model (EinsumModel) and a linear text head are trained.

Usage:
    python -m qnlp.scripts.coco_single_caption.run_frozen
    ML_USE_NON_LINEAR_CONTRACTIONS=true python -m qnlp.scripts.coco_single_caption.run_frozen
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import clip
import mlflow
import orjson
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from qnlp.constants import constants
from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.single_caption import SingleCaptionLoss
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataset import _deserialize_symbols, collect_symbol_sizes
from qnlp.scripts.coco_single_caption.config import ExperimentConfig
from qnlp.utils.early_stopping import EarlyStopping, ModelTrainingStatus
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "coco_single_caption_frozen"
CLIP_MODEL = "ViT-B/32"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
SYMBOL_COLS = ["symbols"]


class CLIPImageCache:
    """Encodes images with frozen CLIP on first seen; returns cached embeddings thereafter."""

    def __init__(self, model_name: str, device):
        self.device = device
        self._model, self._preprocess = clip.load(model_name, device=device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, image_paths: list[str]) -> torch.Tensor:
        missing = [p for p in image_paths if p not in self._cache]
        if missing:
            images = torch.stack([self._preprocess(self._load(p)) for p in missing]).to(self.device)
            embeddings = self._model.encode_image(images).float()
            embeddings = F.normalize(embeddings, dim=-1)
            for path, emb in zip(missing, embeddings):
                self._cache[path] = emb.cpu()

        stacked = torch.stack([self._cache[p] for p in image_paths])
        return stacked.to(self.device)

    @staticmethod
    def _load(path: str):
        from PIL import Image

        return Image.open(path).convert("RGB")

    @property
    def embedding_dim(self) -> int:
        return self._model.visual.output_dim


class FrozenCOCODataset(Dataset):
    def __init__(self, parquet_path: Path, use_non_linear_contractions: bool):
        self.df = pl.read_parquet(parquet_path)
        self.use_nlc = use_non_linear_contractions

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.row(idx, named=True)
        symbols = _deserialize_symbols(row["symbols"])
        if self.use_nlc:
            raw_path = row.get("path")
            path = [tuple(step) for step in orjson.loads(raw_path)] if raw_path else None
            caption = (row["diagram"], symbols, path)
        else:
            caption = (row["diagram"], symbols)
        return {
            "caption": caption,
            "image_path": row["local_image_path"],
            "sample_id": row["sample_id"],
        }


def _collate(batch: list[dict]) -> dict:
    return {k: [item[k] for item in batch] for k in batch[0]}


def _collect_retrieval_metrics(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: "CLIPImageCache",
    loader: DataLoader,
) -> dict[str, float]:
    text_model.eval()
    text_head.eval()
    img_embs, txt_embs = [], []
    with torch.no_grad():
        for batch in loader:
            img_e = image_cache(batch["image_path"])
            txt_e = F.normalize(text_head(text_model(batch["caption"])), dim=-1)
            finite = torch.isfinite(img_e).all(-1) & torch.isfinite(txt_e).all(-1)
            img_embs.append(img_e[finite].cpu())
            txt_embs.append(txt_e[finite].cpu())
    return retrieval_metrics(torch.cat(img_embs), torch.cat(txt_embs))


def _build_loaders(cfg: ExperimentConfig) -> dict[str, DataLoader]:
    dataset_name = cfg.dataset_name or (
        "coco_single_caption_nlc" if cfg.use_non_linear_contractions else "coco_single_caption"
    )
    splits = {
        "train": DATASETS_PATH / f"{dataset_name}_train.parquet",
        "val": DATASETS_PATH / f"{dataset_name}_val.parquet",
        "test": DATASETS_PATH / f"{dataset_name}_test.parquet",
    }
    return {
        split: DataLoader(
            FrozenCOCODataset(path, cfg.use_non_linear_contractions),
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=_collate,
            num_workers=0,  # cache lives in main process
        )
        for split, path in splits.items()
    }


def _run_epoch(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: CLIPImageCache,
    loader: DataLoader,
    loss_fn: SingleCaptionLoss,
    optimizer: torch.optim.Optimizer,
    device,
    train: bool,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    text_model.train(train)
    text_head.train(train)
    totals: dict[str, float] = defaultdict(float)
    n = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if train:
                optimizer.zero_grad()

            image_emb = image_cache(batch["image_path"])
            caption_emb = F.normalize(text_head(text_model(batch["caption"])), dim=-1)

            loss_inputs = {"image_embeddings": image_emb, "caption_embeddings": caption_emb}
            loss_inputs, n_dropped = drop_nonfinite_rows(loss_inputs, list(loss_inputs))

            if loss_inputs["image_embeddings"].shape[0] == 0:
                continue

            loss, metrics = loss_fn(loss_inputs)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(text_model.parameters()) + list(text_head.parameters()),
                    max_norm=max_grad_norm,
                )
                optimizer.step()

            bs = loss_inputs["image_embeddings"].shape[0]
            for k, v in metrics.items():
                totals[k] += float(v) * bs
            totals["n_dropped"] += n_dropped
            n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def run() -> None:
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    loaders = _build_loaders(cfg)
    train_ds, val_ds, test_ds = (loaders[s].dataset for s in ("train", "val", "test"))
    logger.info(
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} |"
        " non_linear={cfg.use_non_linear_contractions}"
    )

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )

    image_cache = CLIPImageCache(CLIP_MODEL, device)
    if image_cache.embedding_dim != cfg.embedding_dim:
        raise ValueError(
            f"CLIP embedding dim ({image_cache.embedding_dim}) != cfg.embedding_dim ({cfg.embedding_dim}). "
            "Set ML_EMBEDDING_DIM to match (ViT-B/32 = 512)."
        )

    text_model = EinsumModel(symbols, sizes, non_linear_contractions=cfg.use_non_linear_contractions).to(device)
    text_head = nn.Linear(cfg.embedding_dim, cfg.embedding_dim).to(device)

    loss_fn = SingleCaptionLoss(temperature=cfg.temperature, alignment_weight=0.0).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
            {"params": text_head.parameters(), "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay},
        ]
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, minimize=True)

    params = {
        **cfg.model_dump(),
        "frozen_image_tower": CLIP_MODEL,
        "text_model_params": sum(p.numel() for p in text_model.parameters()),
    }

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as run:
        for epoch in range(1, cfg.max_epochs + 1):
            train_metrics = _run_epoch(
                text_model,
                text_head,
                image_cache,
                loaders["train"],
                loss_fn,
                optimizer,
                device,
                train=True,
                max_grad_norm=cfg.max_grad_norm,
            )
            if mlflow.active_run():
                mlflow.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} train: {train_metrics}")

            val_metrics = _run_epoch(
                text_model, text_head, image_cache, loaders["val"], loss_fn, optimizer, device, train=False
            )
            if mlflow.active_run():
                mlflow.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} val: {val_metrics}")

            status = early_stopping(val_metrics.get("loss", 0.0))
            if status == ModelTrainingStatus.improved:
                torch.save(
                    {
                        "text_model_state_dict": text_model.state_dict(),
                        "text_head_state_dict": text_head.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
                logger.info(f"Epoch {epoch}: new best — checkpoint saved.")
            elif status == ModelTrainingStatus.stop:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        best = torch.load(checkpoint_path, map_location=device)
        text_model.load_state_dict(best["text_model_state_dict"])
        text_head.load_state_dict(best["text_head_state_dict"])

        test_metrics = _run_epoch(
            text_model, text_head, image_cache, loaders["test"], loss_fn, optimizer, device, train=False
        )
        logger.info(f"Test: {test_metrics}")

        retrieval = _collect_retrieval_metrics(text_model, text_head, image_cache, loaders["test"])
        logger.info(f"Test retrieval: {retrieval}")

        if mlflow.active_run():
            mlflow.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})
            mlflow.log_metrics({f"test/{k}": v for k, v in retrieval.items()})
            mlflow.log_artifact(str(checkpoint_path))
        send_training_finished_notification({"experiment": EXPERIMENT_NAME, "run": run.info.run_name, **test_metrics})


if __name__ == "__main__":
    run()
