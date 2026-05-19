"""Train EinsumModel against a frozen pretrained CLIP-ViT image encoder.

This is experiment (b) from the einsum/clip diagnosis: isolate EinsumModel's
compositional capacity by removing the image-side training signal entirely.

Architecture:
    text:   EinsumModel  (trainable)
    image:  FrozenClipVisionModel  (CLIP-ViT frozen + trainable Linear proj)
    heads:  AlignmentHead × 2  (trainable, identity-init)

Data:   coco_contrastive_{train,val,test}.parquet  (same as aro_contrastive)
Loss:   ContrastiveLoss with InfoNCE + triplet on synthetic word-order pairs.
"""
from datetime import datetime

import mlflow
import torch
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.step import AROContrastiveStep
from qnlp.scripts.einsum_frozen_clip.config import ExperimentConfig
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "einsum_frozen_clip"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path

TRAIN_PARQUET = DATASETS_PATH / "coco_contrastive_train.parquet"
VAL_PARQUET = DATASETS_PATH / "coco_contrastive_val.parquet"
TEST_PARQUET = DATASETS_PATH / "coco_contrastive_test.parquet"

COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption"),
    ("false_diagram", "false_symbols", "false_caption"),
]
SYMBOL_COLS = ["true_symbols", "false_symbols"]

# CLIP's expected preprocessing (official OpenAI weights).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    # Image transforms: resize/crop to CLIP's expected input, then CLIP-normalise.
    # Images arrive from VLMDataset already in float [0,1] (see dataset.py:70),
    # so Normalize works directly.
    size = cfg.clip_image_size
    train_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )

    loaders, datasets = get_dataloaders(
        train_parquet=TRAIN_PARQUET,
        val_parquet=VAL_PARQUET,
        test_parquet=TEST_PARQUET,
        batch_size=cfg.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        compiled_columns=COMPILED_COLUMNS,
    )
    train_loader, val_loader, test_loader = loaders
    train_ds, val_ds, test_ds = datasets

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    symbols, sizes = collect_symbol_sizes([train_ds, val_ds, test_ds], SYMBOL_COLS)
    logger.info(f"Collected {len(symbols)} unique symbols.")

    text_model = EinsumModel(symbols, sizes).to(device)
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    n_text = sum(p.numel() for p in text_model.parameters())
    n_image_total = sum(p.numel() for p in image_model.parameters())
    n_image_trainable = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    logger.info(
        f"Text params: {n_text:,} | "
        f"Image params (total/trainable): {n_image_total:,} / {n_image_trainable:,}"
    )

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = AROContrastiveStep(loss_fn=loss_fn, device=device)

    # Only the trainable parts go to the optimizer. CLIP backbone is frozen
    # (requires_grad=False), so we explicitly target image_model.proj.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": text_model.parameters(),
                "lr": cfg.text_lr,
                "weight_decay": cfg.text_weight_decay,
            },
            {
                "params": image_model.proj.parameters(),
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
        "text_model_params": n_text,
        "image_model_params_total": n_image_total,
        "image_model_params_trainable": n_image_trainable,
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

        mlflow.log_artifact(checkpoint_path)
        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
            }
        )


if __name__ == "__main__":
    run()
