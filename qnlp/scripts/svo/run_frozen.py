"""
SVO-Probes / SVO-Swap training — FROZEN image tower (CLIP ViT-B/32).

Diagnostic control run, not a proposal to put classical capacity in the
quantum pipeline. `image_ablation.py` showed the from-scratch TTNImageModel
contributes ~nothing to SVO-Probes accuracy (ARO: collapsed to a constant
vector; SVO: varies per image but the variation isn't caption-relevant).
This swaps in a frozen, ImageNet/WebImageText-pretrained CLIP visual encoder
in place of TTNImageModel — everything else (EinsumModel text tower,
ImageContrastiveLoss + hard-negative triplet, SVO's own data) stays the
same. If SVO-Probes jumps to something well above chance, the from-scratch
image tower (not data volume, not loss design, not text tower) is
conclusively the bottleneck. If it stays at chance too, the ceiling is
something else (e.g. genuine task/data difficulty).

Images are encoded on-the-fly with CLIP ViT-B/32 (512-dim, matching
SVOExperimentConfig's default embedding_dim) and cached in memory keyed by
path — reuses CLIPImageCache from qnlp.scripts.coco_multi_caption.run_frozen.
Only the text model (EinsumModel) and a linear text head are trained; the
legacy "no learnable head" config doesn't apply here since the text tower
now has to learn to land in CLIP's fixed embedding space, not co-adapt with
a from-scratch image tower.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from qnlp.constants import constants
from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.image_contrastive import ImageContrastiveLoss
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataloader import vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset, collect_symbol_sizes
from qnlp.scripts.coco_multi_caption.run_frozen import CLIP_MODEL, CLIPImageCache
from qnlp.scripts.svo.config import SVOExperimentConfig
from qnlp.utils.early_stopping import EarlyStopping, ModelTrainingStatus
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "svo_probes_frozen"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
COMPILED_COLUMNS = [("diagram", "symbols", "caption", "path")]
SVO_IMAGE_COLUMNS = ["true_local_image_path", "false_local_image_path"]
SYMBOL_COLS = ["symbols"]
_SVO_SUBSETS = ["subj_neg", "verb_neg", "obj_neg"]

_SWAP_COMPILED = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]


def _make_probes_ds(parquet: Path, nlc: bool) -> VLMDataset:
    return VLMDataset(
        parquet,
        image_columns=SVO_IMAGE_COLUMNS,
        compiled_columns=COMPILED_COLUMNS,
        return_image_paths=True,
        use_non_linear_contractions=nlc,
    )


def _run_epoch(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: CLIPImageCache,
    loader: DataLoader,
    loss_fn: ImageContrastiveLoss,
    optimizer: torch.optim.Optimizer,
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

            true_emb = image_cache(batch["true_local_image_path"])
            false_emb = image_cache(batch["false_local_image_path"])
            caption_emb = F.normalize(text_head(text_model(batch["caption"])), dim=-1)

            outputs = {
                "caption_embeddings": caption_emb,
                "true_image_embeddings": true_emb,
                "false_image_embeddings": false_emb,
            }
            outputs, n_dropped = drop_nonfinite_rows(outputs, list(outputs))

            if outputs["caption_embeddings"].shape[0] == 0:
                continue

            loss, metrics = loss_fn(outputs)

            if train:
                loss.backward()
                params = list(text_model.parameters()) + list(text_head.parameters())
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
                for p in params:
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                optimizer.step()

            with torch.no_grad():
                pos_sim = F.cosine_similarity(outputs["caption_embeddings"], outputs["true_image_embeddings"])
                neg_sim = F.cosine_similarity(outputs["caption_embeddings"], outputs["false_image_embeddings"])
                metrics["hard_neg_acc"] = (pos_sim > neg_sim).float().mean()

            bs = outputs["caption_embeddings"].shape[0]
            for k, v in metrics.items():
                totals[k] += float(v) * bs
            totals["n_dropped"] += n_dropped
            n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


def _eval_svo_probes_frozen(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: CLIPImageCache,
    parquet: Path,
    batch_size: int,
    nlc: bool,
) -> dict:
    """SVO-Probes accuracy stratified by subj/verb/obj subset — frozen-image analogue
    of evaluate_svo_probes (qnlp/scripts/coco_multi_caption/evaluate.py)."""
    ds = _make_probes_ds(parquet, nlc)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    correct_by_subset: dict[str, list[bool]] = defaultdict(list)
    n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            captions = batch["caption"]
            valid = [i for i in range(len(captions)) if all(s in known for s in captions[i][1])]
            n_skipped += len(captions) - len(valid)
            if not valid:
                continue

            true_paths = [batch["true_local_image_path"][i] for i in valid]
            false_paths = [batch["false_local_image_path"][i] for i in valid]
            caps = [captions[i] for i in valid]

            true_img = image_cache(true_paths)
            false_img = image_cache(false_paths)
            cap_emb = F.normalize(text_head(text_model(caps)), dim=-1)

            pos = F.cosine_similarity(cap_emb, true_img)
            neg = F.cosine_similarity(cap_emb, false_img)
            correct = (pos > neg).tolist()
            finite = (torch.isfinite(pos) & torch.isfinite(neg)).tolist()

            for j, i in enumerate(valid):
                if not finite[j]:
                    n_skipped += 1
                    continue
                for s in _SVO_SUBSETS:
                    if bool(batch[s][i]):
                        correct_by_subset[s].append(bool(correct[j]))
                correct_by_subset["overall"].append(bool(correct[j]))

    def _acc(xs: list[bool]) -> dict:
        return {"n": len(xs), "hard_neg_acc": sum(xs) / len(xs) if xs else float("nan")}

    result = {s: _acc(correct_by_subset.get(s, [])) for s in [*_SVO_SUBSETS, "overall"]}
    result["n_skipped"] = n_skipped
    return result


def _eval_svo_swap_frozen(
    text_model: nn.Module,
    text_head: nn.Module,
    image_cache: CLIPImageCache,
    parquet: Path,
    batch_size: int,
    nlc: bool,
) -> dict[str, float]:
    """SVO-Swap: fixed image, true/false caption — same shape as ARO/SugarCREPE."""
    if not parquet.exists():
        logger.warning(f"Skipping SVO-Swap — {parquet} not found.")
        return {"hard_neg_acc": float("nan"), "n_evaluated": 0, "n_skipped": 0}

    ds = VLMDataset(parquet, compiled_columns=_SWAP_COMPILED, use_non_linear_contractions=nlc, return_image_paths=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=0)
    known = set(text_model.sym2weight.keys())

    correct: list[bool] = []
    n_skipped = 0

    text_model.eval()
    text_head.eval()
    with torch.no_grad():
        for batch in loader:
            true_caps, false_caps = batch["true_caption"], batch["false_caption"]
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

    n = len(correct)
    return {"hard_neg_acc": sum(correct) / n if n else float("nan"), "n_evaluated": n, "n_skipped": n_skipped}


def run() -> None:
    cfg = SVOExperimentConfig()
    set_seed()
    device = get_device()

    logger.info("========================================")
    logger.info("Experiment config (frozen-image control):")
    for k, v in cfg.model_dump().items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  device: {device}")
    logger.info("========================================")

    suffix = cfg.dataset_suffix
    TRAIN_PARQUET = DATASETS_PATH / f"svo_train_probes{suffix}.parquet"
    VAL_PARQUET = DATASETS_PATH / f"svo_val_probes{suffix}.parquet"
    TEST_PARQUET = DATASETS_PATH / f"svo_test_probes{suffix}.parquet"
    SWAP_PARQUET = DATASETS_PATH / f"svo_swap_eval{suffix}.parquet"

    train_ds = _make_probes_ds(TRAIN_PARQUET, cfg.use_non_linear_contractions)
    val_ds = _make_probes_ds(VAL_PARQUET, cfg.use_non_linear_contractions)
    test_ds = _make_probes_ds(TEST_PARQUET, cfg.use_non_linear_contractions)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=vlm_collate_fn, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=vlm_collate_fn, num_workers=0)

    logger.info(f"Train: {len(train_ds)} rows | Val: {len(val_ds)} rows | Test: {len(test_ds)} rows")

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )
    logger.info(f"Collected {len(symbols)} unique symbols.")

    image_cache = CLIPImageCache(CLIP_MODEL, device)
    if image_cache.embedding_dim != cfg.embedding_dim:
        raise ValueError(
            f"CLIP embedding dim ({image_cache.embedding_dim}) != cfg.embedding_dim "
            f"({cfg.embedding_dim}). Set SVO_ML_EMBEDDING_DIM=512 to match ViT-B/32."
        )

    text_model = EinsumModel(
        symbols, sizes, non_linear_contractions=cfg.use_non_linear_contractions, use_weight_norm=cfg.use_weight_norm
    ).to(device)
    text_head = nn.Linear(cfg.embedding_dim, cfg.embedding_dim).to(device)

    loss_fn = ImageContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
            {"params": text_head.parameters(), "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay},
        ]
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, minimize=False)

    params = {**cfg.model_dump(), "frozen_image_tower": CLIP_MODEL}

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as mlrun:
        for epoch in range(1, cfg.max_epochs + 1):
            train_metrics = _run_epoch(
                text_model,
                text_head,
                image_cache,
                train_loader,
                loss_fn,
                optimizer,
                train=True,
                max_grad_norm=cfg.max_grad_norm,
            )
            if mlflow.active_run():
                mlflow.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} train: {train_metrics}")

            val_metrics = _run_epoch(
                text_model,
                text_head,
                image_cache,
                val_loader,
                loss_fn,
                optimizer,
                train=False,
            )
            if mlflow.active_run():
                mlflow.log_metrics({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)
            logger.info(f"Epoch {epoch} val: {val_metrics}")

            status = early_stopping(val_metrics.get("hard_neg_acc", 0.0))
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
        logger.info(f"Loaded checkpoint from epoch {best['epoch']}.")

        logger.info("--- SVO-Probes ---")
        probes = _eval_svo_probes_frozen(
            text_model, text_head, image_cache, TEST_PARQUET, cfg.batch_size, cfg.use_non_linear_contractions
        )
        for subset in [*_SVO_SUBSETS, "overall"]:
            r = probes[subset]
            logger.info(f"  {subset:<10} N={r['n']:>6} acc={r['hard_neg_acc']:.4f}")

        logger.info("--- SVO-Swap ---")
        swap = _eval_svo_swap_frozen(
            text_model, text_head, image_cache, SWAP_PARQUET, cfg.batch_size, cfg.use_non_linear_contractions
        )
        logger.info(f"  acc={swap['hard_neg_acc']:.4f} evaluated={swap['n_evaluated']} skipped={swap['n_skipped']}")

        if mlflow.active_run():
            mlflow.log_metrics({f"svo_probes/{s}": probes[s]["hard_neg_acc"] for s in [*_SVO_SUBSETS, "overall"]})
            mlflow.log_metrics({"svo_swap/acc": swap["hard_neg_acc"]})
            mlflow.log_artifact(str(checkpoint_path))

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": mlrun.info.run_name,
                "svo_probes_overall": probes["overall"]["hard_neg_acc"],
                "svo_swap": swap["hard_neg_acc"],
            }
        )


if __name__ == "__main__":
    run()
