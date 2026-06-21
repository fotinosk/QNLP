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
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_aro, evaluate_sugarcrepe, evaluate_winoground
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


RETRIEVAL_MAX_IMAGES = 5000


def _coco_retrieval_eval(model: ContrastiveVLM, device: torch.device, batch_size: int, nlc: bool) -> dict[str, float]:
    """
    COCO test retrieval using coco_single_caption_test.parquet, capped at
    RETRIEVAL_MAX_IMAGES unique images for a fair comparison across runs.
    """
    test_path = DATASETS_PATH / "coco_single_caption_test.parquet"
    if not test_path.exists():
        logger.warning(f"Retrieval eval skipped — {test_path} not found.")
        return {}

    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = VLMDataset(
        test_path, compiled_columns=_SINGLE_COMPILED, image_transform=transform, use_non_linear_contractions=nlc
    )

    # One caption per unique image, capped at RETRIEVAL_MAX_IMAGES
    seen: set[str] = set()
    indices: list[int] = []
    for i, sid in enumerate(ds.df["sample_id"].to_list()):
        if sid not in seen:
            seen.add(sid)
            indices.append(i)
            if len(indices) >= RETRIEVAL_MAX_IMAGES:
                break

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

    return retrieval_metrics(torch.cat(img_embs), torch.cat(txt_embs))


def _print_final_summary(
    retrieval: dict[str, float],
    wino: dict,
    aro: dict,
    sc: dict,
) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("FINAL RESULTS")
    logger.info(sep)

    if retrieval:
        logger.info(f"COCO Retrieval (test, {RETRIEVAL_MAX_IMAGES} images)")
        for k in sorted(retrieval):
            logger.info(f"  {k}: {retrieval[k]:.4f}")

    logger.info("Winoground")
    logger.info(f"  text:  {wino['text_score']:.4f}")
    logger.info(f"  image: {wino['image_score']:.4f}")
    logger.info(f"  group: {wino['group_score']:.4f}")
    logger.info(f"  pairs: {wino['n_pairs']}  skipped: {wino['n_skipped']}")

    logger.info("ARO")
    logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
    for task in [*sorted(k for k in aro if k != "overall"), "overall"]:
        r = aro[task]
        logger.info(f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}")

    logger.info("SugarCREPE (swap_obj)")
    logger.info(f"  acc: {sc['hard_neg_acc']:.4f}  evaluated: {sc['n_evaluated']}  skipped: {sc['n_skipped']}")
    logger.info(sep)


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

        logger.info("--- COCO Retrieval ---")
        retrieval = _coco_retrieval_eval(model, device, cfg.batch_size, nlc)

        logger.info("--- Winoground ---")
        wino = evaluate_winoground(model, device, cfg.batch_size)

        logger.info("--- ARO ---")
        aro = evaluate_aro(model, device, cfg.batch_size)

        logger.info("--- SugarCREPE (swap_obj) ---")
        sc = evaluate_sugarcrepe(model, device, cfg.batch_size)

        _print_final_summary(retrieval, wino, aro, sc)

        if mlflow.active_run():
            mlflow.log_metrics({f"retrieval/{k}": v for k, v in retrieval.items()})
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
            mlflow.log_artifact(str(checkpoint_path))

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": mlflow_run.info.run_name,
                **test_metrics,
                **{f"retrieval/{k}": v for k, v in retrieval.items()},
                "wino_group": wino["group_score"],
                "aro_overall": aro.get("overall", {}).get("hard_neg_acc", float("nan")),
                "sugarcrepe": sc["hard_neg_acc"],
            }
        )

    logger.info(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    run()
