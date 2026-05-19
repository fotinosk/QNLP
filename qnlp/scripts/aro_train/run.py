"""Train EinsumModel + TTNImageModel on the ARO train set (v1-style).

This is the closest reproduction we have of the historical v1 setup that
produced 0.78 hard_neg_acc on the ARO test set. Trains on ARO's real
word-order hard negatives (vs the synthetic random negatives used in
einsum_frozen_clip on COCO).

Architecture:
    text:   EinsumModel       (trainable, free per-typed-symbol tensors)
    image:  TTNImageModel     (trainable, same as v1)
    heads:  AlignmentHead × 2

Data: ProcessedARODataset over visual_genome_{relation,attribution}/{train,val,test}
      with CCG-compiled *_processed_512.jsonl sidecars.

The ARO train set has ~36 k pairs (vs 450 k in COCO) and real hard negatives —
the kind of training signal that exercises the EinsumModel's typed-CCG
compositional bias rather than letting it memorise random pairings.
"""
from datetime import datetime
from pathlib import Path

import json
import mlflow
import torch
from lambeq.backend.symbol import Symbol
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.dataset.aro_dataset import ProcessedARODataset
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_contrastive.step import AROContrastiveStep
from qnlp.scripts.aro_train.config import ExperimentConfig
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device
from qnlp.utils.training_notifications import send_training_finished_notification

EXPERIMENT_NAME = "aro_train"
logger = setup_logger(log_name=EXPERIMENT_NAME)

ARO_ROOT = constants.discoviz_root / "data" / "aro" / "processed" if hasattr(
    constants, "discoviz_root"
) else Path("data/aro/processed")


def aro_step_collate_fn(batch):
    """Collate ProcessedARODataset outputs into the keys AROContrastiveStep expects."""
    return {
        "local_image_path": torch.stack([b["image"] for b in batch]),
        "true_caption": [b["true_caption"] for b in batch],
        "false_caption": [b["false_caption"] for b in batch],
    }


def gather_symbols_from_jsonl(sidecar_paths):
    """Collect all unique (Symbol, size) pairs from CCG-compiled sidecar files."""
    seen: dict = {}
    for path in sidecar_paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for entry in rec.get("symbols", []):
                    if not (isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], dict)):
                        continue
                    d = entry[0]
                    sym = Symbol(
                        name=d["name"],
                        directed_dom=d["directed_dom"],
                        directed_cod=d["directed_cod"],
                    )
                    size = tuple(entry[1]) if isinstance(entry[1], list) else entry[1]
                    if sym in seen and seen[sym] != size:
                        raise ValueError(f"Symbol {sym} conflicting sizes: {seen[sym]} vs {size}")
                    seen[sym] = size
    return list(seen.keys()), list(seen.values())


def build_split_datasets(aro_root: Path, subset: str, train_tfm, eval_tfm):
    """Build per-split datasets for the chosen subset.

    'combined' uses the pre-merged data/aro/processed/combined/ split (single
    JSON per split) which is the closest match to v1's training setup.
    'relation' / 'attribution' use the per-benchmark subdirs.
    """
    if subset == "combined":
        # Pre-merged single JSON + sidecar per split.
        kind_dirs = [(aro_root / "combined", "")]
    elif subset == "relation":
        kind_dirs = [(aro_root / "visual_genome_relation", "")]
    elif subset == "attribution":
        kind_dirs = [(aro_root / "visual_genome_attribution", "")]
    else:
        raise ValueError(f"Unknown subset: {subset}")

    sidecars = []
    out = {}
    for split, tfm in (("train", train_tfm), ("val", eval_tfm), ("test", eval_tfm)):
        per_kind = []
        for dir_path, _ in kind_dirs:
            json_path = dir_path / f"{split}.json"
            sidecar = dir_path / f"{split}_processed_512.jsonl"
            sidecars.append(sidecar)
            if sidecar.exists():
                per_kind.append(ProcessedARODataset(
                    data_path=str(json_path),
                    return_images=True,
                    image_processing_fn=tfm,
                ))
        if not per_kind:
            raise FileNotFoundError(
                f"No processed sidecar found for split={split} in {[str(d) for d,_ in kind_dirs]}"
            )
        out[split] = ConcatDataset(per_kind) if len(per_kind) > 1 else per_kind[0]
    return out, sidecars


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    # TTN expects 64x64. Use PIL-input transforms because ProcessedARODataset
    # uses Image.open(...) and passes a PIL Image.
    size = image_model_hyperparams.image_size
    train_tfm = transforms.Compose([
        transforms.Resize(size + 8),
        transforms.RandomCrop(size),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    aro_root = Path("data/aro/processed").resolve()
    aro_images = Path("data/aro/images").resolve()
    # Resolve the checkpoint path NOW (before chdir below changes cwd).
    checkpoints_root = constants.checkpoints_path.resolve()

    datasets, sidecars = build_split_datasets(aro_root, cfg.aro_subset, train_tfm, eval_tfm)
    # ProcessedARODataset opens images by bare filename; chdir so they resolve.
    import os
    os.chdir(aro_images)
    logger.info(
        f"ARO subset='{cfg.aro_subset}' | "
        f"train: {len(datasets['train'])} | val: {len(datasets['val'])} | test: {len(datasets['test'])}"
    )

    symbols, sizes = gather_symbols_from_jsonl(sidecars)
    logger.info(f"Collected {len(symbols)} unique typed symbols from ARO sidecars.")

    text_model = EinsumModel(symbols, sizes).to(device)
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    n_text = sum(p.numel() for p in text_model.parameters())
    n_image = sum(p.numel() for p in image_model.parameters())
    logger.info(f"Text params: {n_text:,} | Image params: {n_image:,}")

    loaders = {}
    for split in ("train", "val", "test"):
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=aro_step_collate_fn,
            drop_last=(split == "train"),
        )

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = AROContrastiveStep(loss_fn=loss_fn, device=device)

    optimizer = torch.optim.AdamW([
        {"params": text_model.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
        {"params": image_model.parameters(), "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay},
        {
            "params": list(model.image_head.parameters()) + list(model.text_head.parameters()),
            "lr": cfg.head_lr,
            "weight_decay": cfg.head_weight_decay,
        },
    ])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = checkpoints_root / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        **cfg.model_dump(),
        **image_model_hyperparams.model_dump(),
        "text_model_params": n_text,
        "image_model_params": n_image,
        "n_train": len(datasets["train"]),
        "n_val": len(datasets["val"]),
        "n_test": len(datasets["test"]),
        "n_symbols": len(symbols),
    }

    with setup_mlflow_run(EXPERIMENT_NAME, params, 8080) as run:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            step=step,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            test_loader=loaders["test"],
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
