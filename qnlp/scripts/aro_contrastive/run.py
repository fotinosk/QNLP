import os
from datetime import datetime

import mlflow
import torch
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.clip_text_model import build_text_model, caption_compiled_columns, text_model_hyperparams
from qnlp.discoviz.models.image_model import build_image_model, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.config import ExperimentConfig
from qnlp.scripts.aro_contrastive.step import AROContrastiveStep
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_aro, evaluate_sugarcrepe, evaluate_winoground
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "aro_contrastive"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path

SYMBOL_COLS = ["true_symbols", "false_symbols"]


def _compiled_columns() -> list[tuple]:
    """4th element is the pre-computed contraction path column (used only in
    non-linear mode). PAPER_EXPERIMENTS_PLAN.md's M3 (text_backbone=clip)
    gets the 2-tuple raw-passthrough form instead -- see caption_compiled_columns."""
    return [
        caption_compiled_columns("true_diagram", "true_symbols", "true_processed_text", "true_caption", "true_path"),
        caption_compiled_columns(
            "false_diagram", "false_symbols", "false_processed_text", "false_caption", "false_path"
        ),
    ]


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    suffix = cfg.dataset_suffix
    train_parquet = DATASETS_PATH / f"aro_train{suffix}.parquet"
    val_parquet = DATASETS_PATH / f"aro_val{suffix}.parquet"
    test_parquet = DATASETS_PATH / f"aro_test{suffix}.parquet"
    logger.info(f"Datasets (suffix='{suffix}'): {train_parquet.name}, {val_parquet.name}, {test_parquet.name}")

    # Checkpoint dir includes the suffix and PID so concurrent runs (e.g. _rtl and
    # _random launched in the same second) never collide on the same path.
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = f"{ts}{suffix or '_optimal'}_pid{os.getpid()}"
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / run_tag / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info(f"MODEL CHECKPOINT PATH: {checkpoint_path}")
    logger.info("=" * 70)

    size = image_model_hyperparams.image_size
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(size, padding=4, padding_mode="reflect"),
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

    nlc = cfg.use_non_linear_contractions

    loaders, datasets = get_dataloaders(
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_parquet,
        batch_size=cfg.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        compiled_columns=_compiled_columns(),
        use_non_linear_contractions=nlc,
    )
    train_loader, val_loader, test_loader = loaders
    train_ds, val_ds, test_ds = datasets

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )
    logger.info(f"Collected {len(symbols)} unique symbols.")
    logger.info(f"Text backbone: {text_model_hyperparams.text_backbone}")

    text_model = build_text_model(
        cfg.embedding_dim, symbols, sizes, non_linear_contractions=nlc, use_weight_norm=True
    ).to(device)
    image_model = build_image_model(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(
        text_model, image_model, embedding_dim=cfg.embedding_dim, use_projection_head=cfg.use_alignment_head
    ).to(device)

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = AROContrastiveStep(loss_fn=loss_fn, device=device)

    param_groups = [
        {
            "params": text_model.parameters(),
            "lr": cfg.text_lr,
            "weight_decay": cfg.text_weight_decay,
        },
        {
            "params": image_model.parameters(),
            "lr": cfg.image_lr,
            "weight_decay": cfg.image_weight_decay,
        },
    ]
    # NoOpHead (use_alignment_head=False) has no parameters - AdamW errors on
    # an empty param group, so only add it when there's something to train.
    head_params = list(model.image_head.parameters()) + list(model.text_head.parameters())
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay})
    optimizer = torch.optim.AdamW(param_groups)

    params = {
        **cfg.model_dump(),
        **image_model_hyperparams.model_dump(),
        "text_model_params": sum(p.numel() for p in text_model.parameters()),
        "image_model_params": sum(p.numel() for p in image_model.parameters()),
    }

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as run:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            step=step,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            monitor_metric="hard_neg_acc",
            checkpoint_path=checkpoint_path,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )

        test_metrics = trainer.fit()

        def _guard(name: str, fn):
            try:
                return fn()
            except Exception as e:
                logger.warning(f"{name}: eval skipped ({type(e).__name__}: {e})")
                return None

        # WinogroundDataset (unlike VLMDataset) has not been extended for
        # text_backbone=clip's raw-text passthrough -- guarded so an M3 run's
        # Winoground failure doesn't prevent the ARO numbers Step 1 actually
        # needs from being computed at all.
        logger.info("--- Winoground ---")
        wino = _guard("Winoground", lambda: evaluate_winoground(model, device, cfg.batch_size))

        # ARO on the HELD-OUT test split only (matching path suffix). The shared
        # aro_eval.parquet pools all splits, so it contains this run's training
        # rows — evaluating on it would be leakage. test_parquet is disjoint from
        # train/val and uses the same contraction paths the model trained on.
        logger.info("--- ARO (held-out test split) ---")
        aro = _guard("ARO", lambda: evaluate_aro(model, device, cfg.batch_size, parquet=test_parquet))

        logger.info("--- SugarCREPE (swap_obj) ---")
        sc = _guard("SugarCREPE", lambda: evaluate_sugarcrepe(model, device, cfg.batch_size))

        sep = "=" * 60
        logger.info(sep)
        logger.info("FINAL RESULTS")
        logger.info(sep)
        logger.info("Winoground")
        if wino:
            logger.info(f"  text:  {wino['text_score']:.4f}")
            logger.info(f"  image: {wino['image_score']:.4f}")
            logger.info(f"  group: {wino['group_score']:.4f}")
            logger.info(f"  pairs: {wino['n_pairs']}  skipped: {wino['n_skipped']}")
        else:
            logger.info("  (unavailable)")
        logger.info("ARO")
        if aro:
            logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
            for task in [*sorted(k for k in aro if k != "overall"), "overall"]:
                r = aro[task]
                logger.info(
                    f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}"
                )
        else:
            logger.info("  (unavailable)")
        logger.info("SugarCREPE (swap_obj)")
        if sc:
            logger.info(f"  acc: {sc['hard_neg_acc']:.4f}  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
        else:
            logger.info("  (unavailable)")
        logger.info(sep)

        if mlflow.active_run():
            if wino:
                mlflow.log_metrics(
                    {
                        "wino/text": wino["text_score"],
                        "wino/image": wino["image_score"],
                        "wino/group": wino["group_score"],
                    }
                )
            if aro:
                mlflow.log_metrics(
                    {f"aro/{task}/acc": res["hard_neg_acc"] for task, res in aro.items() if isinstance(res, dict)}
                )
            if sc:
                mlflow.log_metrics({"sugarcrepe/swap_obj": sc["hard_neg_acc"]})
            mlflow.log_artifact(checkpoint_path)

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
                "wino_group": wino["group_score"] if wino else float("nan"),
                "aro_overall": (aro or {}).get("overall", {}).get("hard_neg_acc", float("nan")),
                "sugarcrepe": sc["hard_neg_acc"] if sc else float("nan"),
            }
        )


if __name__ == "__main__":
    run()
