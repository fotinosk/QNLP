"""
COCO ARO-style contrastive training.

Uses TF-IDF hard negatives (see create_dataset.py) with the same
ContrastiveLoss (InfoNCE + triplet, weight=40k) and AROContrastiveStep
that makes ARO training succeed. The negatives are baked into the dataset
so no online mining is needed.

Key differences from previous COCO runs:
  - Hard negatives: TF-IDF-similar captions from different images (not random)
  - Loss: InfoNCE + triplet (not InfoNCE alone)
  - Monitor: hard_neg_acc — direct binary ranking signal
  - Image augmentation: matches ARO (RandomCrop + ColorJitter + flip)
  - Dims: embedding_dim=512, bond_dim=10 (native to coco_single_caption_nlc)

After training, runs:
  - COCO val retrieval (R@1/R@5/R@10 image→text and text→image)
  - Full compositional benchmark suite via coco_multi_caption.evaluate
    (Winoground / ARO hard-neg-acc / SugarCREPE swap_obj)
"""

from datetime import datetime

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders, vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset, collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.step import AROContrastiveStep
from qnlp.scripts.coco_aro_style.config import ExperimentConfig
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "coco_aro_style"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path

# ARO-style pair schema matching create_dataset.py output
COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption", "true_path"),
    ("false_diagram", "false_symbols", "false_caption", "false_path"),
]
SYMBOL_COLS = ["true_symbols", "false_symbols"]

# Single-caption schema for COCO retrieval eval (uses coco_single_caption_val)
_SINGLE_COMPILED = [("diagram", "symbols", "caption", "path")]


def _coco_retrieval_eval(model: ContrastiveVLM, device: torch.device, batch_size: int, nlc: bool) -> dict[str, float]:
    """
    Evaluate COCO val retrieval using deduplicated coco_single_caption_val.parquet.
    One caption per unique sample_id; diagonal of similarity matrix = true pair.
    """
    val_path = DATASETS_PATH / "coco_single_caption_val.parquet"
    if not val_path.exists():
        logger.warning(f"Retrieval eval skipped — {val_path} not found.")
        return {}

    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = VLMDataset(
        val_path, compiled_columns=_SINGLE_COMPILED, image_transform=transform, use_non_linear_contractions=nlc
    )

    # Deduplicate to one caption per image for retrieval evaluation
    seen: set[str] = set()
    indices: list[int] = []
    for i, sid in enumerate(ds.df["sample_id"].to_list()):
        if sid not in seen:
            seen.add(sid)
            indices.append(i)

    loader = DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model.eval()
    img_embs, txt_embs = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["local_image_path"].to(device)
            outputs = model(images, batch["caption"])
            img_e = outputs["image_embeddings"]
            txt_e = outputs["true_caption_embeddings"]
            finite = torch.isfinite(img_e).all(-1) & torch.isfinite(txt_e).all(-1)
            img_embs.append(img_e[finite].cpu())
            txt_embs.append(txt_e[finite].cpu())

    if not img_embs:
        return {}

    metrics = retrieval_metrics(torch.cat(img_embs), torch.cat(txt_embs))
    logger.info(f"COCO val retrieval ({len(indices)} images): {metrics}")
    return metrics


def run() -> None:
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    logger.info("=" * 60)
    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    for k, v in cfg.model_dump().items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  device: {device}")
    logger.info("=" * 60)

    nlc = cfg.use_non_linear_contractions
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

    loaders, datasets = get_dataloaders(
        train_parquet=DATASETS_PATH / "coco_aro_style_train.parquet",
        val_parquet=DATASETS_PATH / "coco_aro_style_val.parquet",
        test_parquet=DATASETS_PATH / "coco_aro_style_test.parquet",
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

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as mlflow_run:
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

        # fit() restores best checkpoint in-memory before returning
        test_metrics = trainer.fit()

        retrieval = _coco_retrieval_eval(model, device, cfg.batch_size, nlc)

        if mlflow.active_run():
            mlflow.log_metrics({f"coco_retrieval/{k}": v for k, v in retrieval.items()})
            mlflow.log_artifact(str(checkpoint_path))

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": mlflow_run.info.run_name,
                **test_metrics,
                **{f"retrieval/{k}": v for k, v in retrieval.items()},
            }
        )

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(
        "To evaluate on Winoground/ARO/SugarCREPE run:\n"
        f"  python -m qnlp.scripts.coco_multi_caption.evaluate {checkpoint_path}"
    )


if __name__ == "__main__":
    run()
