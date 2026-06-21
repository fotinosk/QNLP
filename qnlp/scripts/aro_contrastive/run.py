from datetime import datetime

import mlflow
import torch
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
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

TRAIN_PARQUET = DATASETS_PATH / "aro_train.parquet"
VAL_PARQUET = DATASETS_PATH / "aro_val.parquet"
TEST_PARQUET = DATASETS_PATH / "aro_test.parquet"

# 4th element is the pre-computed contraction path column (used only in non-linear mode).
COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]
SYMBOL_COLS = ["true_symbols", "false_symbols"]


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

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
        train_parquet=TRAIN_PARQUET,
        val_parquet=VAL_PARQUET,
        test_parquet=TEST_PARQUET,
        batch_size=cfg.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        compiled_columns=COMPILED_COLUMNS,
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

    text_model = EinsumModel(symbols, sizes, non_linear_contractions=nlc).to(device)
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = AROContrastiveStep(loss_fn=loss_fn, device=device)

    optimizer = torch.optim.AdamW(
        [
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
            {
                "params": list(model.image_head.parameters()) + list(model.text_head.parameters()),
                "lr": cfg.head_lr,
                "weight_decay": cfg.head_weight_decay,
            },
        ]
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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

        logger.info("--- Winoground ---")
        wino = evaluate_winoground(model, device, cfg.batch_size)

        logger.info("--- ARO ---")
        aro = evaluate_aro(model, device, cfg.batch_size)

        logger.info("--- SugarCREPE (swap_obj) ---")
        sc = evaluate_sugarcrepe(model, device, cfg.batch_size)

        sep = "=" * 60
        logger.info(sep)
        logger.info("FINAL RESULTS")
        logger.info(sep)
        logger.info("Winoground")
        logger.info(f"  text:  {wino['text_score']:.4f}")
        logger.info(f"  image: {wino['image_score']:.4f}")
        logger.info(f"  group: {wino['group_score']:.4f}")
        logger.info(f"  pairs: {wino['n_pairs']}  skipped: {wino['n_skipped']}")
        logger.info("ARO")
        logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
        for task in [*sorted(k for k in aro if k != "overall"), "overall"]:
            r = aro[task]
            logger.info(
                f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}"
            )
        logger.info("SugarCREPE (swap_obj)")
        logger.info(f"  acc: {sc['hard_neg_acc']:.4f}  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
        logger.info(sep)

        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "wino/text": wino["text_score"],
                    "wino/image": wino["image_score"],
                    "wino/group": wino["group_score"],
                }
            )
            mlflow.log_metrics(
                {f"aro/{task}/acc": res["hard_neg_acc"] for task, res in aro.items() if isinstance(res, dict)}
            )
            mlflow.log_metrics({"sugarcrepe/swap_obj": sc["hard_neg_acc"]})
            mlflow.log_artifact(checkpoint_path)

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
                "wino_group": wino["group_score"],
                "aro_overall": aro.get("overall", {}).get("hard_neg_acc", float("nan")),
                "sugarcrepe": sc["hard_neg_acc"],
            }
        )


if __name__ == "__main__":
    run()
