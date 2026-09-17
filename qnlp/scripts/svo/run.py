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
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
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

    TRAIN_PARQUET = DATASETS_PATH / "svo_train_probes.parquet"
    VAL_PARQUET = DATASETS_PATH / "svo_val_probes.parquet"
    TEST_PARQUET = DATASETS_PATH / "svo_test_probes.parquet"

    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
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
        train_transform=transform,
        val_transform=transform,
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
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(
        text_model,
        image_model,
        embedding_dim=cfg.embedding_dim,
        use_mlp_head=cfg.use_mlp_head,
        use_projection_head=cfg.use_alignment_head,
    ).to(device)

    loss_fn = ImageContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)
    step = SVOHardNegStep(loss_fn=loss_fn, device=device)

    param_groups = [
        {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
        {"params": image_model.parameters(), "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay},
    ]
    # NoOpHead (use_alignment_head=False) has no parameters — AdamW errors on
    # an empty param group, so only add it when there's something to train.
    head_params = list(model.image_head.parameters()) + list(model.text_head.parameters())
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
        svo = evaluate_svo(model, device, cfg.batch_size)

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
