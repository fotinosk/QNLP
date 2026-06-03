from datetime import datetime

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.winoground_pair import WinogroundPairLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.datasets.winoground_dataset import (
    WinogroundDataset,
    winoground_eval_collate_fn,
)
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.winoground.config import ExperimentConfig
from qnlp.scripts.winoground.step import WinogroundPairStep
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "winoground"
logger = setup_logger(log_name=EXPERIMENT_NAME)

TRAIN_PARQUET = constants.datasets_path / "winoground_train.parquet"
VAL_PARQUET = constants.datasets_path / "winoground_val.parquet"
TEST_PARQUET = constants.datasets_path / "winoground_test.parquet"

SYMBOL_COLS = ["cap0_symbols", "cap1_symbols"]


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

    train_ds = WinogroundDataset(TRAIN_PARQUET, mode="eval", image_transform=train_transform)
    val_ds = WinogroundDataset(VAL_PARQUET, mode="eval", image_transform=val_transform)
    test_ds = WinogroundDataset(TEST_PARQUET, mode="eval", image_transform=val_transform)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    worker_kwargs = dict(num_workers=4, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=winoground_eval_collate_fn, **worker_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=winoground_eval_collate_fn, **worker_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=winoground_eval_collate_fn, **worker_kwargs
    )

    symbols, sizes = collect_symbol_sizes([train_ds, val_ds, test_ds], SYMBOL_COLS)
    logger.info(f"Collected {len(symbols)} unique symbols.")

    text_model = EinsumModel(symbols, sizes).to(device)
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    if cfg.pretrained_checkpoint:
        checkpoint = torch.load(cfg.pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded pretrained weights from {cfg.pretrained_checkpoint} (strict=False)")

    loss_fn = WinogroundPairLoss(
        margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = WinogroundPairStep(loss_fn=loss_fn, device=device)

    optimizer = torch.optim.AdamW(
        [
            {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
            {"params": image_model.parameters(), "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay},
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
            monitor_metric="group_acc",
            checkpoint_path=checkpoint_path,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )

        test_metrics = trainer.fit()

        mlflow.log_artifact(str(checkpoint_path))
        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
            }
        )


if __name__ == "__main__":
    run()
