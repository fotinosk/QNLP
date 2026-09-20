"""
SVO-Probes contrastive training.

Matches the legacy ARO training setup (qnlp/scripts/aro_contrastive/), not
the newer COCO-style one — COCO is ~70x larger than SVO's training set and
has no hard negatives at all, so its plain in-batch-InfoNCE config doesn't
transfer. ARO is a small, explicit-hard-negative benchmark much closer to
SVO-Probes in scale and shape. Same model architecture as COCO/ARO
(EinsumModel + TTNImageModel + ContrastiveVLM) — only the loss, training
step, and hyperparameters differ.

Trains directly on (caption, true_image, false_image) triplets — the same
probes-shape parquet the final SVO-Probes/SVO-Swap benchmark uses — via
InfoNCE (in-batch negatives) plus a heavily-weighted triplet margin loss on
the explicit hard negative (mirrors AROContrastiveStep, with image/caption
roles swapped since SVO's hard negative is on the image side).

Fully independent of the COCO pipeline: own dataset, own checkpoint dir, own
benchmark battery. Not evaluated against Winoground/ARO/SugarCREPE.
"""

from datetime import datetime

import mlflow
import torch
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.image_contrastive import ImageContrastiveLoss
from qnlp.core.training.losses.structured_contrastive import StructuredContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import build_image_model, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.domain.models.vlm.score_heads import (
    AggregationScoreHead,
    BornRuleScoreHead,
    GatedResidualTrilinearScoreHead,
    RoleGroundedScoreHead,
    ScoreHead,
    TrilinearScoreHead,
)
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_svo, log_banner, print_full_report
from qnlp.scripts.svo.config import SVOExperimentConfig
from qnlp.scripts.svo.step import SVOHardNegStep
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "svo_probes"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
IMAGE_COLUMNS = ["true_local_image_path", "false_local_image_path"]
COMPILED_COLUMNS = [("diagram", "symbols", "caption", "path")]
SYMBOL_COLS = ["symbols"]


def _warm_start_from_aro(text_model: EinsumModel, image_model: torch.nn.Module, checkpoint_path: str, device) -> None:
    """Load an aro_contrastive checkpoint's image tower in full, and transfer
    text-tower symbols (words) that exist in both vocabularies with matching
    shape. See SVOExperimentConfig.pretrained_checkpoint for the rationale."""
    logger.info(f"Warm-starting from ARO checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    image_sd = {k[len("image_model.") :]: v for k, v in state_dict.items() if k.startswith("image_model.")}
    image_model.load_state_dict(image_sd, strict=True)
    logger.info("Image tower: loaded in full from ARO checkpoint.")

    text_sd: dict = {}
    for k, v in state_dict.items():
        if k in ("symbols_list", "sizes_list", "non_linear_contractions"):
            text_sd[k] = v
        elif k.startswith("text_model."):
            text_sd[k[len("text_model.") :]] = v
    pretrained_symbols = list(text_sd["symbols_list"])
    pretrained_sizes = list(text_sd["sizes_list"])
    pretrained_text = EinsumModel(pretrained_symbols, pretrained_sizes)
    pretrained_text.load_state_dict(text_sd, strict=True)

    common_symbols = [
        sym
        for sym in text_model.symbols
        if sym in pretrained_text.sym2weight
        and pretrained_text.sym2weight[sym].shape == text_model.sym2weight[sym].shape
    ]
    common_tensors = [pretrained_text.sym2weight[sym].detach().clone() for sym in common_symbols]
    text_model.set_weights(common_symbols, common_tensors)
    logger.info(
        f"Text tower: transferred {len(common_symbols)}/{len(text_model.symbols)} symbols "
        f"(SVO vocab) found in ARO's {len(pretrained_symbols)}-symbol vocab with matching shape; "
        "remaining symbols kept at random init."
    )


def _load_pretrained_image_tower(image_model: torch.nn.Module, checkpoint_path: str, freeze: bool, device) -> None:
    """F1/F2 (TTN_CIFAR_EXPERIMENTS.md's "Route B variations" — Orthogonal:
    freeze a CIFAR-pretrained image tower). SVO has ~8,600 training rows
    against a ~2.3M-parameter image tower — most of B1's memorisation
    capacity is plausibly in the backbone, not the ~8K-parameter score
    head. Loads a real, image-grounded backbone (saved by
    ttn_supervised_probe.py) instead of training from random init.
    freeze=True (F1) removes that capacity source entirely, training only
    the score head and text tower; freeze=False (F2) warm-starts but keeps
    it trainable, separating "good init" from "reduced capacity"."""
    logger.info(f"Loading pretrained image tower from {checkpoint_path} (freeze={freeze})")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone_sd = checkpoint["backbone_state_dict"]
    # ttn_supervised_probe.py's classifier probe and SVO use different
    # `embedding_dim`s (128 vs 512), so `head.weight`/`head.bias` (the final
    # Linear mapping pooled features -> embedding_dim) never match shape --
    # a plain strict=False does NOT skip these (PyTorch still raises on a
    # shape mismatch for a key present in both dicts; strict only affects
    # missing/unexpected keys), so they're dropped from the checkpoint's
    # dict explicitly before loading. This is fine here because every
    # score_head that would use a pretrained image tower reads
    # TTNImageModel.forward_regions directly (raw quadtree layer outputs),
    # which never touches `head` or `final_norm` at all -- only the
    # quadtree layers + patch embedding (the actual CIFAR-pretrained
    # capacity) are shared and get loaded.
    dropped = [k for k in backbone_sd if k.startswith(("head.", "final_norm."))]
    backbone_sd = {k: v for k, v in backbone_sd.items() if k not in dropped}
    result = image_model.load_state_dict(backbone_sd, strict=False)
    if result.unexpected_keys or set(result.missing_keys) - set(dropped):
        raise RuntimeError(
            f"Unexpected key mismatch loading pretrained image tower: "
            f"missing={set(result.missing_keys) - set(dropped)}, unexpected={result.unexpected_keys}"
        )
    logger.info(
        f"Image tower loaded (probe test_acc={checkpoint.get('test_acc', '?')}, "
        f"val_acc={checkpoint.get('val_acc', '?')}); skipped head/final_norm (embedding_dim differs, "
        f"unused by region-based score heads): {dropped}"
    )
    if freeze:
        for p in image_model.parameters():
            p.requires_grad_(False)
        image_model.eval()
        logger.info("Image tower frozen (F1): 0 trainable image-tower parameters.")


def _build_score_head(cfg: SVOExperimentConfig, image_model: torch.nn.Module) -> ScoreHead | None:
    """Routes A/B (TTN_CIFAR_EXPERIMENTS.md). None reproduces every existing
    run bit-for-bit (cosine head, ImageContrastiveLoss)."""
    if cfg.score_head == "cosine":
        return None

    n_regions = 4**cfg.region_level
    # P1's dim_at_level: layer i (0-indexed) has out_dim = bond_dim * 2**(i+1);
    # forward_regions(level) reads layers[len(layers)-1-level].
    region_dim = image_model.bond_dim * (2 ** (len(image_model.layers) - cfg.region_level))

    if cfg.score_head == "trilinear":
        return TrilinearScoreHead(
            text_dim=cfg.embedding_dim,
            region_dim=region_dim,
            n_regions=n_regions,
            score_dim=cfg.score_dim,
            rank=cfg.rank,
            region_level=cfg.region_level,
            tie_uv=cfg.tie_uv,
            normalize_terms=cfg.normalize_terms,
        )
    if cfg.score_head == "trilinear_gated":
        return GatedResidualTrilinearScoreHead(
            text_dim=cfg.embedding_dim,
            region_dim=region_dim,
            n_regions=n_regions,
            score_dim=cfg.score_dim,
            rank=cfg.rank,
            region_level=cfg.region_level,
            tie_uv=cfg.tie_uv,
            normalize_terms=cfg.normalize_terms,
        )
    if cfg.score_head == "born":
        return BornRuleScoreHead(
            text_dim=cfg.embedding_dim,
            region_dim=region_dim,
            n_regions=n_regions,
            score_dim=cfg.score_dim,
            region_level=cfg.region_level,
        )
    if cfg.score_head == "aggregation":
        return AggregationScoreHead(
            text_dim=cfg.embedding_dim,
            region_dim=region_dim,
            n_regions=n_regions,
            score_dim=cfg.score_dim,
            rank=cfg.rank,
            region_level=cfg.region_level,
            agg=cfg.aggregation_fn,
        )
    if cfg.score_head == "role_grounded":
        return RoleGroundedScoreHead(
            noun_dim=cfg.embedding_dim,
            verb_leg_dim=cfg.embedding_dim,
            region_dim=region_dim,
            n_regions=n_regions,
            score_dim=cfg.score_dim,
            region_level=cfg.region_level,
        )
    raise ValueError(f"Unknown score_head: {cfg.score_head!r}")


def run():
    cfg = SVOExperimentConfig()
    set_seed()
    device = get_device()

    logger.info("========================================")
    logger.info("Experiment config:")
    for k, v in cfg.model_dump().items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  device: {device}")
    logger.info("========================================")

    suffix = cfg.dataset_suffix
    TRAIN_PARQUET = DATASETS_PATH / f"svo_train_probes{suffix}.parquet"
    VAL_PARQUET = DATASETS_PATH / f"svo_val_probes{suffix}.parquet"
    TEST_PARQUET = DATASETS_PATH / f"svo_test_probes{suffix}.parquet"

    size = image_model_hyperparams.image_size
    # Train/val previously used the IDENTICAL transform (no augmentation at all) — a
    # real gap, not a deliberate legacy-matching choice: ARO's own image tower never
    # needed to learn anything (see SVO_EXPERIMENTS.md's "image tower is the
    # bottleneck" finding), so its transform was never a validated reference either
    # way. RandomResizedCrop handles SVO's widely varying native image sizes
    # (173x160 to 1067x900+) directly, forcing the tower to learn features robust to
    # scale/crop instead of memorising exact pixel layouts.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    loaders, datasets = get_dataloaders(
        train_parquet=TRAIN_PARQUET,
        val_parquet=VAL_PARQUET,
        test_parquet=TEST_PARQUET,
        batch_size=cfg.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        image_columns=IMAGE_COLUMNS,
        compiled_columns=COMPILED_COLUMNS,
        use_non_linear_contractions=cfg.use_non_linear_contractions,
    )
    train_loader, val_loader, test_loader = loaders
    train_ds, val_ds, test_ds = datasets

    logger.info(f"Train: {len(train_ds)} rows | Val: {len(val_ds)} rows | Test: {len(test_ds)} rows")

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )
    logger.info(f"Collected {len(symbols)} unique symbols.")

    text_model = EinsumModel(symbols, sizes, non_linear_contractions=cfg.use_non_linear_contractions).to(device)
    image_model = build_image_model(cfg.embedding_dim).to(device)

    if cfg.pretrained_checkpoint:
        _warm_start_from_aro(text_model, image_model, cfg.pretrained_checkpoint, device)

    if cfg.pretrained_image_tower_checkpoint:
        _load_pretrained_image_tower(
            image_model, cfg.pretrained_image_tower_checkpoint, cfg.freeze_pretrained_image_tower, device
        )

    score_head = _build_score_head(cfg, image_model)

    model = ContrastiveVLM(
        text_model,
        image_model,
        embedding_dim=cfg.embedding_dim,
        use_mlp_head=cfg.use_mlp_head,
        use_projection_head=cfg.use_alignment_head,
        score_head=score_head,
    ).to(device)

    if score_head is not None:
        logger.info(f"Score head: {cfg.score_head} (region_level={cfg.region_level}, score_dim={cfg.score_dim})")
        loss_fn = StructuredContrastiveLoss(
            score_head=model.score_head,
            temperature=cfg.temperature,
            triplet_weight=cfg.triplet_weight,
            triplet_margin=cfg.triplet_margin,
        ).to(device)
    else:
        loss_fn = ImageContrastiveLoss(
            temperature=cfg.temperature,
            triplet_weight=cfg.triplet_weight,
            triplet_margin=cfg.triplet_margin,
            distance=cfg.distance,
        ).to(device)
    step = SVOHardNegStep(loss_fn=loss_fn, device=device)

    param_groups = [
        {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
    ]
    # F1 (frozen pretrained image tower): exclude it from the optimizer
    # entirely rather than relying on requires_grad=False alone, so AdamW
    # never allocates momentum state for parameters that will never update.
    trainable_image_params = [p for p in image_model.parameters() if p.requires_grad]
    if trainable_image_params:
        param_groups.append(
            {"params": trainable_image_params, "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay}
        )
    # NoOpHead (use_alignment_head=False) has no parameters — AdamW errors on
    # an empty param group, so only add it when there's something to train.
    head_params = list(model.image_head.parameters()) + list(model.text_head.parameters())
    if model.score_head is not None:
        head_params += list(model.score_head.parameters())
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay})
    optimizer = torch.optim.AdamW(param_groups)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        **cfg.model_dump(),
        **image_model_hyperparams.model_dump(),
        "text_model_params": sum(p.numel() for p in text_model.parameters()),
        "image_model_params": sum(p.numel() for p in image_model.parameters()),
    }

    run_info = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint_path": str(checkpoint_path),
        "non_linear_contractions": cfg.use_non_linear_contractions,
        "embedding_dim": cfg.embedding_dim,
        "bond_dim": cfg.bond_dim,
        "batch_size": cfg.batch_size,
        "n_symbols": len(symbols),
        "text_model_params": params["text_model_params"],
        "image_model_params": params["image_model_params"],
    }
    log_banner("TRAINING RUN — START", run_info)

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as run:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            step=step,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            monitor_metric="hard_neg_acc",
            minimize_metric=False,
            checkpoint_path=checkpoint_path,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )

        test_metrics = trainer.fit()

        logger.info("--- SVO-Probes / SVO-Swap ---")
        svo = evaluate_svo(
            model,
            device,
            cfg.batch_size,
            probes_parquet=TEST_PARQUET,
            swap_parquet=DATASETS_PATH / f"svo_swap_eval{suffix}.parquet",
        )

        report_info = {
            **run_info,
            "run_name": run.info.run_name,
            **{f"test/{k}": v for k, v in test_metrics.items()},
        }
        print_full_report(retrieval=None, benchmarks={}, info=report_info, svo=svo)

        if mlflow.active_run():
            probes = svo.get("svo_probes") or {}
            mlflow.log_metrics(
                {f"svo_probes/{subset}": res["hard_neg_acc"] for subset, res in probes.items() if isinstance(res, dict)}
            )
            swap = svo.get("svo_swap")
            if swap:
                mlflow.log_metrics({"svo_swap/acc": swap["hard_neg_acc"]})
            mlflow.log_artifact(str(checkpoint_path))

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
                "svo_probes_overall": (svo.get("svo_probes") or {})
                .get("overall", {})
                .get("hard_neg_acc", float("nan")),
                "svo_swap": (svo.get("svo_swap") or {}).get("hard_neg_acc", float("nan")),
            }
        )


if __name__ == "__main__":
    run()
