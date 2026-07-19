"""
Multi-caption COCO contrastive training.

All 5 captions per image are used as separate training rows. Val/test loaders
are deduplicated to one caption per image for correct retrieval evaluation
(diagonal = true-positive assumption).

Temperature is fixed (not learnable) to avoid collapse. No hard negative
mining — plain symmetric InfoNCE with in-batch negatives only.
"""

from datetime import datetime

import mlflow
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.batch_utils import drop_nonfinite_rows
from qnlp.core.training.losses.single_caption import SingleCaptionLoss
from qnlp.core.training.retrieval_eval import retrieval_metrics
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import get_dataloaders, vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset, collect_symbol_sizes
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.coco_multi_caption.config import ExperimentConfig
from qnlp.scripts.coco_multi_caption.evaluate import evaluate_all_benchmarks, log_banner, print_full_report
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


class SimpleCaptionStep:
    def __init__(self, loss_fn: SingleCaptionLoss, device: torch.device):
        self.loss_fn = loss_fn
        self.device = device

    def __call__(self, model, batch: dict, train: bool) -> tuple[Tensor, dict]:
        images = batch["local_image_path"].to(self.device)
        outputs = model(images, batch["caption"])

        loss_inputs = {
            "image_embeddings": outputs["image_embeddings"],
            "caption_embeddings": outputs["true_caption_embeddings"],
        }
        loss_inputs, n_dropped = drop_nonfinite_rows(loss_inputs, list(loss_inputs))

        if loss_inputs["image_embeddings"].shape[0] == 0:
            return torch.zeros((), device=self.device, requires_grad=True), {}

        loss, metrics = self.loss_fn(loss_inputs)

        if n_dropped:
            metrics["n_skipped"] = images.new_tensor(float(n_dropped))

        gate = getattr(model.text_model, "nonlinear_gate", None)
        if gate is not None:
            metrics["nonlinear_gate"] = gate.detach()

        return loss, metrics


def _dedup_loader(ds: VLMDataset, batch_size: int, num_workers: int = 4, max_images: int | None = None) -> DataLoader:
    """One caption per unique sample_id — required for retrieval eval."""
    seen: set[str] = set()
    indices: list[int] = []
    for i, sid in enumerate(ds.df["sample_id"].to_list()):
        if sid not in seen:
            seen.add(sid)
            indices.append(i)
            if max_images is not None and len(indices) >= max_images:
                break
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
        # Batching by shared diagram topology only helps EinsumModel's batched
        # fast path in linear mode — NLC always uses the per-sample path.
        topology_bucketing=not cfg.use_non_linear_contractions,
    )
    train_loader, _, _ = loaders
    train_ds, val_ds, test_ds = datasets

    val_loader_dedup = _dedup_loader(val_ds, cfg.batch_size)
    test_loader_dedup = _dedup_loader(test_ds, cfg.batch_size, max_images=TEST_SIZE)

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
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim, use_mlp_head=cfg.use_mlp_head).to(
        device
    )

    # Fixed temperature — loss_fn is NOT in the optimizer so logit_scale won't move.
    loss_fn = SingleCaptionLoss(temperature=cfg.temperature, alignment_weight=0.0).to(device)
    step = SimpleCaptionStep(loss_fn=loss_fn, device=device)

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
            val_loader=val_loader_dedup,
            test_loader=test_loader_dedup,
            monitor_metric="accuracy",
            minimize_metric=False,
            checkpoint_path=checkpoint_path,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            max_grad_norm=cfg.max_grad_norm,
            device=device,
        )

        test_metrics = trainer.fit()

        logger.info("--- COCO Retrieval (test, 5000 images) ---")
        retrieval = _collect_retrieval_metrics(model, test_loader_dedup, device)

        # Full benchmark battery: Winoground (+ per-tag), ARO (attribution/relation),
        # SugarCREPE (swap_obj / full / ++). Each guarded against missing datasets.
        benchmarks = evaluate_all_benchmarks(model, device, cfg.batch_size)

        report_info = {
            **run_info,
            "run_name": run.info.run_name,
            **{f"test/{k}": v for k, v in test_metrics.items()},
        }
        print_full_report(retrieval, benchmarks, report_info)

        logger.info(
            f"Forward path statistics: {text_model.fast_path_batches} batches on batched fast path, "
            f"{text_model.fallback_batches} batches on sequential fallback path; "
            f"{text_model.fast_path_samples} samples contracted in same-topology groups, "
            f"{text_model.fallback_samples} samples contracted one-by-one."
        )

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
