"""Train EinsumModel + frozen CLIP-ViT with bond_dim=4.

Same setup as `einsum_frozen_clip` except that we remap the per-symbol tensor
sizes from the on-disk bond_dim (10) to a smaller bond_dim (default 4) before
allocating model parameters. The CCG diagrams + symbol identities don't depend
on bond_dim — only the contraction-axis sizes do — so this is a valid
bond_dim=4 experiment without re-running the BERT preprocessing.

Why: at bond_dim=10, each (10, 512, 10) typed tensor has 51 200 free params and
36 % of them are seen exactly once → pure memorisation. Dropping bond_dim to 4
cuts per-symbol params 6.25× and may reduce overfitting on the long tail.
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
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.scripts.einsum_frozen_clip_bond2.config import ExperimentConfig
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "einsum_frozen_clip_bond2"
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

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _remap_sizes(sizes, original_bond_dim: int, new_bond_dim: int):
    """Replace every occurrence of `original_bond_dim` in each size tuple with
    `new_bond_dim`. Embedding-dim axes (512) are left untouched."""
    return [
        tuple(new_bond_dim if d == original_bond_dim else d for d in size)
        for size in sizes
    ]


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

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

    symbols, sizes_raw = collect_symbol_sizes([train_ds, val_ds, test_ds], SYMBOL_COLS)
    sizes = _remap_sizes(sizes_raw, cfg.original_bond_dim, cfg.bond_dim)
    logger.info(
        f"Collected {len(symbols)} unique typed symbols. "
        f"Remapped sizes: bond_dim {cfg.original_bond_dim} → {cfg.bond_dim} "
        f"(e.g. {sizes_raw[0]} → {sizes[0]})"
    )

    text_model = EinsumModel(symbols, sizes).to(device)
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    n_text = sum(p.numel() for p in text_model.parameters())
    n_image_total = sum(p.numel() for p in image_model.parameters())
    n_image_trainable = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    logger.info(
        f"Text params (bond_dim={cfg.bond_dim}): {n_text:,} | "
        f"Image params (total/trainable): {n_image_total:,} / {n_image_trainable:,}"
    )

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
            {"experiment": EXPERIMENT_NAME, "run": run.info.run_name, **test_metrics}
        )


if __name__ == "__main__":
    run()
