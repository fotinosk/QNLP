from datetime import datetime

import mlflow
import torch
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.single_caption import SingleCaptionLoss
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.datasets.dataset import collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_all_benchmarks, log_banner, print_full_report
from qnlp.scripts.coco_single_caption.config import ExperimentConfig
from qnlp.scripts.coco_single_caption.step import COCOSingleCaptionStep
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "coco_single_caption"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path

# 4th element is the pre-computed contraction path column (used only in non-linear mode).
COMPILED_COLUMNS = [("diagram", "symbols", "caption", "path")]
SYMBOL_COLS = ["symbols"]


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

    # Linear and non-linear use different datasets: the linear one is built fast
    # without contraction paths; the non-linear one carries the `path` column.
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
        test_size=5000,
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

    text_model = EinsumModel(symbols, sizes, non_linear_contractions=cfg.use_non_linear_contractions).to(device)
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    loss_fn = SingleCaptionLoss(
        temperature=cfg.temperature,
        alignment_weight=cfg.alignment_weight,
    ).to(device)

    step = COCOSingleCaptionStep(
        loss_fn=loss_fn,
        device=device,
        warmup_epochs=cfg.alignment_warmup_epochs,
        warmup_alignment_weight=cfg.alignment_weight,
    )

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
            {
                # Learnable temperature — no weight decay on a scalar parameter
                "params": loss_fn.parameters(),
                "lr": cfg.text_lr,
                "weight_decay": 0.0,
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

    run_info = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint_path": str(checkpoint_path),
        "dataset": dataset,
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

        retrieval = _collect_retrieval_metrics(model, test_loader, device)

        # Full benchmark battery: Winoground (+ per-tag), ARO (attribution/relation),
        # SugarCREPE (swap_obj / full / ++). Each guarded against missing datasets.
        benchmarks = evaluate_all_benchmarks(model, device, cfg.batch_size)

        report_info = {
            **run_info,
            "run_name": run.info.run_name,
            **{f"test/{k}": v for k, v in test_metrics.items()},
        }
        print_full_report(retrieval, benchmarks, report_info)

        if mlflow.active_run():
            mlflow.log_metrics({f"retrieval/{k}": v for k, v in retrieval.items()})
            wino = benchmarks.get("winoground")
            if wino:
                mlflow.log_metrics(
                    {
                        "wino/text": wino["text_score"],
                        "wino/image": wino["image_score"],
                        "wino/group": wino["group_score"],
                    }
                )
            aro = benchmarks.get("aro") or {}
            mlflow.log_metrics(
                {f"aro/{task}/acc": res["hard_neg_acc"] for task, res in aro.items() if isinstance(res, dict)}
            )
            for name, key in [("full", "sugarcrepe_full"), ("pp", "sugarcrepepp")]:
                sc = benchmarks.get(key)
                if sc:
                    mlflow.log_metrics({f"sugarcrepe/{name}": sc["hard_neg_acc"]})
            mlflow.log_artifact(str(checkpoint_path))

        send_training_finished_notification(
            {
                "experiment": EXPERIMENT_NAME,
                "run": run.info.run_name,
                **test_metrics,
                **{f"retrieval/{k}": v for k, v in retrieval.items()},
                "wino_group": (benchmarks.get("winoground") or {}).get("group_score", float("nan")),
                "aro_overall": (benchmarks.get("aro") or {}).get("overall", {}).get("hard_neg_acc", float("nan")),
                "sugarcrepe_full": (benchmarks.get("sugarcrepe_full") or {}).get("hard_neg_acc", float("nan")),
                "sugarcrepepp": (benchmarks.get("sugarcrepepp") or {}).get("hard_neg_acc", float("nan")),
            }
        )


if __name__ == "__main__":
    run()
