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
from torch.utils.data import DataLoader, Dataset, Subset

from qnlp.constants import constants
from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.single_caption import SingleCaptionLoss
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataloader import vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset, _deserialize_symbols, collect_symbol_sizes
from qnlp.domain.datasets.winoground_dataset import WinogroundDataset, winoground_eval_collate_fn
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
TEST_SIZE = 5000

_ARO_COMPILED = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]
_SC_COMPILED = _ARO_COMPILED


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


def _dedup_loader(ds: FrozenCOCODataset, batch_size: int, max_images: int | None = None) -> DataLoader:
    """One caption per unique sample_id — required for retrieval eval."""
    seen: set[str] = set()
    indices: list[int] = []
    for i, sid in enumerate(ds.df["sample_id"].to_list()):
        if sid not in seen:
            seen.add(sid)
            indices.append(i)
            if max_images is not None and len(indices) >= max_images:
                break
    return DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )


def _eval_hard_neg_frozen(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: "CLIPImageCache",
    parquet,
    compiled_columns: list[tuple],
    batch_size: int,
    device,
    nlc: bool,
) -> dict[str, float]:
    """Hard-negative accuracy on an ARO/SugarCREPE-style contrastive pair parquet."""
    if not parquet.exists():
        logger.warning(f"Skipping benchmark — {parquet} not found.")
        return {"hard_neg_acc": float("nan"), "n_evaluated": 0, "n_skipped": 0}

    ds = VLMDataset(
        parquet,
        compiled_columns=compiled_columns,
        use_non_linear_contractions=nlc,
        return_image_paths=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    correct: list[bool] = []
    pos_cos: list[float] = []
    neg_cos: list[float] = []
    n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            true_caps = batch["true_caption"]
            false_caps = batch["false_caption"]
            paths = batch["local_image_path"]

            valid = [
                i
                for i in range(len(paths))
                if all(s in known for s in true_caps[i][1]) and all(s in known for s in false_caps[i][1])
            ]
            n_skipped += len(paths) - len(valid)
            if not valid:
                continue

            img_emb = image_cache([paths[i] for i in valid])
            true_emb = F.normalize(text_head(text_model([true_caps[i] for i in valid])), dim=-1)
            false_emb = F.normalize(text_head(text_model([false_caps[i] for i in valid])), dim=-1)

            pos = F.cosine_similarity(true_emb, img_emb)
            neg = F.cosine_similarity(false_emb, img_emb)
            finite = (torch.isfinite(pos) & torch.isfinite(neg)).tolist()

            for j in range(len(valid)):
                if not finite[j]:
                    n_skipped += 1
                    continue
                correct.append(bool((pos > neg)[j].item()))
                pos_cos.append(float(pos[j]))
                neg_cos.append(float(neg[j]))

    n = len(correct)
    return {
        "hard_neg_acc": sum(correct) / n if n else float("nan"),
        "true_cos": sum(pos_cos) / n if n else float("nan"),
        "false_cos": sum(neg_cos) / n if n else float("nan"),
        "n_evaluated": n,
        "n_skipped": n_skipped,
    }


def _eval_winoground_frozen(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: "CLIPImageCache",
    batch_size: int,
    device,
    nlc: bool,
) -> dict[str, float]:
    parquet = constants.datasets_path / "winoground_eval.parquet"
    if not parquet.exists():
        logger.warning(f"Skipping Winoground — {parquet} not found.")
        return {
            "text_score": float("nan"),
            "image_score": float("nan"),
            "group_score": float("nan"),
            "n_pairs": 0,
            "n_skipped": 0,
        }

    ds = WinogroundDataset(parquet, mode="eval", use_non_linear_contractions=nlc, return_image_paths=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=winoground_eval_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    text_correct = image_correct = group_correct = n_total = n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            caps0, caps1 = batch["captions_0"], batch["captions_1"]
            paths0, paths1 = batch["images_0"], batch["images_1"]

            valid = [
                i
                for i in range(len(caps0))
                if all(s in known for s in caps0[i][1]) and all(s in known for s in caps1[i][1])
            ]
            n_skipped += len(caps0) - len(valid)
            if not valid:
                continue

            img0 = image_cache([paths0[i] for i in valid])
            img1 = image_cache([paths1[i] for i in valid])
            cap0 = F.normalize(text_head(text_model([caps0[i] for i in valid])), dim=-1)
            cap1 = F.normalize(text_head(text_model([caps1[i] for i in valid])), dim=-1)

            s00 = F.cosine_similarity(img0, cap0, dim=-1)
            s01 = F.cosine_similarity(img0, cap1, dim=-1)
            s10 = F.cosine_similarity(img1, cap0, dim=-1)
            s11 = F.cosine_similarity(img1, cap1, dim=-1)

            finite = torch.isfinite(s00) & torch.isfinite(s01) & torch.isfinite(s10) & torch.isfinite(s11)
            n_skipped += (~finite).sum().item()

            text_correct += ((s00 > s01) & (s11 > s10) & finite).sum().item()
            image_correct += ((s00 > s10) & (s11 > s01) & finite).sum().item()
            group_correct += ((s00 > s01) & (s11 > s10) & (s00 > s10) & (s11 > s01) & finite).sum().item()
            n_total += finite.sum().item()

    return {
        "text_score": text_correct / n_total if n_total else float("nan"),
        "image_score": image_correct / n_total if n_total else float("nan"),
        "group_score": group_correct / n_total if n_total else float("nan"),
        "n_pairs": n_total,
        "n_skipped": n_skipped,
    }


def _print_final_summary(retrieval: dict, wino: dict, aro: dict, sc: dict) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("FINAL RESULTS")
    logger.info(sep)
    if retrieval:
        logger.info(f"COCO Retrieval (test, {TEST_SIZE} images)")
        for k in sorted(retrieval):
            logger.info(f"  {k}: {retrieval[k]:.4f}")
    logger.info("Winoground")
    logger.info(f"  text:  {wino['text_score']:.4f}")
    logger.info(f"  image: {wino['image_score']:.4f}")
    logger.info(f"  group: {wino['group_score']:.4f}")
    logger.info(f"  pairs: {wino['n_pairs']}  skipped: {wino['n_skipped']}")
    logger.info("ARO")
    logger.info(f"  acc: {aro['hard_neg_acc']:.4f}  true_cos: {aro['true_cos']:.4f}  false_cos: {aro['false_cos']:.4f}")
    logger.info(f"  evaluated: {aro['n_evaluated']}  skipped: {aro['n_skipped']}")
    logger.info("SugarCREPE (swap_obj)")
    logger.info(f"  acc: {sc['hard_neg_acc']:.4f}  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
    logger.info(sep)


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


def _build_loaders(cfg: ExperimentConfig) -> tuple[dict[str, DataLoader], dict[str, FrozenCOCODataset]]:
    dataset_name = cfg.dataset_name or (
        "coco_single_caption_nlc" if cfg.use_non_linear_contractions else "coco_single_caption"
    )
    datasets = {
        split: FrozenCOCODataset(DATASETS_PATH / f"{dataset_name}_{split}.parquet", cfg.use_non_linear_contractions)
        for split in ("train", "val", "test")
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=_collate,
            num_workers=0,
        ),
        "val": _dedup_loader(datasets["val"], cfg.batch_size),
        "test": _dedup_loader(datasets["test"], cfg.batch_size, max_images=TEST_SIZE),
    }
    return loaders, datasets


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
            # Recover from NaN gate before each batch — can happen at large embedding_dim
            gate = getattr(text_model, "nonlinear_gate", None)
            if gate is not None and not gate.isfinite():
                logger.warning("NLC gate is NaN — resetting to 0.1 and continuing.")
                with torch.no_grad():
                    gate.fill_(0.1)

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

    loaders, datasets = _build_loaders(cfg)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]
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

            status = early_stopping(val_metrics.get("loss", float("inf")))
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

        logger.info("--- Winoground ---")
        wino = _eval_winoground_frozen(
            text_model, text_head, image_cache, cfg.batch_size, device, cfg.use_non_linear_contractions
        )

        logger.info("--- ARO ---")
        aro = _eval_hard_neg_frozen(
            text_model,
            text_head,
            image_cache,
            constants.datasets_path / "aro_eval.parquet",
            _ARO_COMPILED,
            cfg.batch_size,
            device,
            cfg.use_non_linear_contractions,
        )

        logger.info("--- SugarCREPE (swap_obj) ---")
        sc = _eval_hard_neg_frozen(
            text_model,
            text_head,
            image_cache,
            constants.datasets_path / "sugarcrepe_swap_obj_eval.parquet",
            _SC_COMPILED,
            cfg.batch_size,
            device,
            cfg.use_non_linear_contractions,
        )

        _print_final_summary(retrieval, wino, aro, sc)

        if mlflow.active_run():
            mlflow.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})
            mlflow.log_metrics({f"retrieval/{k}": v for k, v in retrieval.items()})
            mlflow.log_metrics(
                {"wino/text": wino["text_score"], "wino/image": wino["image_score"], "wino/group": wino["group_score"]}
            )
            mlflow.log_metrics({"aro/hard_neg_acc": aro["hard_neg_acc"], "sugarcrepe/swap_obj": sc["hard_neg_acc"]})
            mlflow.log_artifact(str(checkpoint_path))

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
                **{f"retrieval/{k}": v for k, v in retrieval.items()},
                "wino_group": wino["group_score"],
                "aro_overall": aro["hard_neg_acc"],
                "sugarcrepe": sc["hard_neg_acc"],
            }
        )


if __name__ == "__main__":
    run()
