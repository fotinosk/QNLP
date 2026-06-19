"""
Multi-caption COCO training with hard negative mining.

Two improvements over coco_single_caption:
  1. All 5 captions per image are used during training (each epoch each image
     is seen once per caption, in a different random order). No dataset change
     needed — SingleCaptionStrategy is a pass-through that keeps all atoms.
     Val/test loaders are deduplicated to one caption per image so retrieval
     evaluation uses the standard diagonal = true-positive assumption.

  2. Hard negative mining: after hard_neg_warmup_epochs, a full pass over the
     train set builds a pool of the most confusable text embeddings. Each batch's
     i2t logits are augmented with hard_neg_sample_k samples from this pool,
     replacing easy random negatives with globally hard ones.
"""

from datetime import datetime

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.single_caption import SingleCaptionLoss
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders, vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset, collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.coco_multi_caption.config import ExperimentConfig
from qnlp.scripts.coco_multi_caption.step import MultiCaptionHardNegStep
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "coco_multi_caption"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
COMPILED_COLUMNS = [("diagram", "symbols", "caption", "path")]
SYMBOL_COLS = ["symbols"]

TEST_SIZE = 5000


def _dedup_loader(ds: VLMDataset, batch_size: int, num_workers: int = 4) -> DataLoader:
    """DataLoader over the first caption per unique sample_id.

    COCO has 5 captions per image. Retrieval evaluation assumes row i of
    img_embs is paired with row i of txt_embs (diagonal = true positive), so
    we must use exactly one caption per image at eval time.
    """
    seen: set[str] = set()
    indices: list[int] = []
    for i, sid in enumerate(ds.df["sample_id"].to_list()):
        if sid not in seen:
            seen.add(sid)
            indices.append(i)
    subset = Subset(ds, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _collect_retrieval_metrics(model, loader, device) -> dict[str, float]:
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
    return retrieval_metrics(torch.cat(img_embs), torch.cat(txt_embs))


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    logger.info("========================================")
    logger.info("Experiment config:")
    for k, v in cfg.model_dump().items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  device: {device}")
    logger.info("========================================")

    # Reuse the existing single-caption dataset — it already contains all 5
    # captions per image as separate rows (SingleCaptionStrategy is pass-through).
    dataset = cfg.dataset_name or (
        "coco_single_caption_nlc" if cfg.use_non_linear_contractions else "coco_single_caption"
    )
    TRAIN_PARQUET = DATASETS_PATH / f"{dataset}_train.parquet"
    VAL_PARQUET = DATASETS_PATH / f"{dataset}_val.parquet"
    TEST_PARQUET = DATASETS_PATH / f"{dataset}_test.parquet"
    logger.info(f"Using dataset '{dataset}' (non_linear={cfg.use_non_linear_contractions})")

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
        compiled_columns=COMPILED_COLUMNS,
        use_non_linear_contractions=cfg.use_non_linear_contractions,
        test_size=TEST_SIZE,
    )
    train_loader, _, _ = loaders
    train_ds, val_ds, test_ds = datasets

    # Deduplicated loaders for eval — one caption per image
    val_loader_dedup = _dedup_loader(val_ds, cfg.batch_size)
    test_loader_dedup = _dedup_loader(test_ds, cfg.batch_size)

    logger.info(
        f"Train: {len(train_ds)} rows | "
        f"Val: {len(val_ds)} rows ({len(val_loader_dedup.dataset)} unique images) | "
        f"Test: {len(test_ds)} rows ({len(test_loader_dedup.dataset)} unique images)"
    )

    symbols, sizes = collect_symbol_sizes(
        [train_ds, val_ds, test_ds],
        SYMBOL_COLS,
        remap={constants.embedding_dim: cfg.embedding_dim, constants.bond_dim: cfg.bond_dim},
    )
    logger.info(f"Collected {len(symbols)} unique symbols.")

    text_model = EinsumModel(symbols, sizes, non_linear_contractions=cfg.use_non_linear_contractions).to(device)
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    loss_fn = SingleCaptionLoss(temperature=cfg.temperature, alignment_weight=0.0).to(device)

    step = MultiCaptionHardNegStep(
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device,
        hard_neg_warmup_epochs=cfg.hard_neg_warmup_epochs,
        hard_neg_refresh_epochs=cfg.hard_neg_refresh_epochs,
        hard_neg_pool_size=cfg.hard_neg_pool_size,
        hard_neg_sample_k=cfg.hard_neg_sample_k,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
            {"params": image_model.parameters(), "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay},
            {
                "params": list(model.image_head.parameters()) + list(model.text_head.parameters()),
                "lr": cfg.head_lr,
                "weight_decay": cfg.head_weight_decay,
            },
            {"params": loss_fn.parameters(), "lr": cfg.text_lr, "weight_decay": 0.0},
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
        # Wrap on_epoch_start to pass the model (needed for mining updates).
        # Trainer calls step.on_epoch_start(epoch) without the model; we
        # intercept by monkeypatching before handing step to Trainer.
        _original_on_epoch_start = step.on_epoch_start

        def _on_epoch_start_with_model(epoch: int) -> None:
            _original_on_epoch_start(epoch, model=model)

        step.on_epoch_start = _on_epoch_start_with_model

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            step=step,
            train_loader=train_loader,
            val_loader=val_loader_dedup,
            test_loader=test_loader_dedup,
            monitor_metric="hard_neg_accuracy",
            minimize_metric=False,
            checkpoint_path=checkpoint_path,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )

        test_metrics = trainer.fit()

        retrieval = _collect_retrieval_metrics(model, test_loader_dedup, device)
        if mlflow.active_run():
            mlflow.log_metrics({f"test/{k}": v for k, v in retrieval.items()})
            mlflow.log_artifact(str(checkpoint_path))
        logger.info(f"Test retrieval: {retrieval}")
        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
            }
        )


if __name__ == "__main__":
    run()
