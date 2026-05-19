"""Train TTNImageModel against a frozen pretrained CLIP text encoder.

Inverse of einsum_frozen_clip:
    text:   FrozenClipTextModel  (CLIP text backbone frozen + tiny Linear proj)
    image:  TTNImageModel        (trainable, same as v1's image side)
    heads:  AlignmentHead × 2

Data: coco_single_caption_*.parquet (raw captions, like clip_baseline).
Synthetic negatives in-batch via random derangement (ClipBaselineStep).

Goal: test whether TTNImageModel can learn an alignment at full COCO scale
when given a perfect text signal. If yes, v1's bottleneck was the EinsumModel
text side, not the TTN image side.
"""
from datetime import datetime

import mlflow
import polars as pl
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from transformers import AutoTokenizer

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.models.vlm.contrastive_vlm import AlignmentHead
from qnlp.scripts.clip_baseline.step import ClipBaselineStep
from qnlp.scripts.clip_text_ttn.config import ExperimentConfig
from qnlp.scripts.clip_text_ttn.text_model import FrozenClipTextModel
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

EXPERIMENT_NAME = "clip_text_ttn"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
TRAIN_PARQUET = DATASETS_PATH / "coco_single_caption_train.parquet"
VAL_PARQUET = DATASETS_PATH / "coco_single_caption_val.parquet"
TEST_PARQUET = DATASETS_PATH / "coco_single_caption_test.parquet"


class ClipTextTtnVLM(nn.Module):
    """Same shape as ContrastiveVLM but with kwarg-splatting text forward."""

    def __init__(self, text_model: FrozenClipTextModel, image_model: TTNImageModel, embedding_dim: int):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.image_head = AlignmentHead(embedding_dim)
        self.text_head = AlignmentHead(embedding_dim)

    def forward(self, images, true_text: dict, false_text: dict | None = None) -> dict:
        image_emb = self.image_head(self.image_model(images))
        true_emb = self.text_head(self.text_model(**true_text))
        outputs = {
            "image_embeddings": image_emb,
            "true_caption_embeddings": true_emb,
        }
        if false_text is not None:
            outputs["false_caption_embeddings"] = self.text_head(self.text_model(**false_text))
        return outputs


class CaptionImageDataset(Dataset):
    def __init__(self, parquet_path, transform):
        df = pl.read_parquet(parquet_path).select(["local_image_path", "processed_text"])
        self.image_paths = df["local_image_path"].to_list()
        self.captions = df["processed_text"].to_list()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx], mode=ImageReadMode.RGB)
        img = self.transform(img)
        return img, self.captions[idx]


def make_collate(tokenizer, max_len: int):
    def _collate(batch):
        images = torch.stack([b[0] for b in batch])
        texts = [b[1] for b in batch]
        tok = tokenizer(
            texts,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "image": images,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }
    return _collate


def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    # CLIP's own tokenizer (BPE), distinct from BERT's WordPiece used by clip_baseline.
    tokenizer = AutoTokenizer.from_pretrained(cfg.clip_model_name)

    # TTNImageModel uses the existing image_model_hyperparams (64x64 patches).
    # Some COCO entries have been scraped at thumbnail sizes (e.g. 59×80), so we
    # must Resize first; aro_contrastive's plain RandomCrop with padding=4 fails
    # on those.
    image_size = image_model_hyperparams.image_size
    train_tfm = T.Compose([
        T.Resize(image_size + 8),         # min dim → 72
        T.RandomCrop(image_size),         # → 64
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CaptionImageDataset(TRAIN_PARQUET, train_tfm)
    val_ds = CaptionImageDataset(VAL_PARQUET, eval_tfm)
    test_ds = CaptionImageDataset(TEST_PARQUET, eval_tfm)

    collate = make_collate(tokenizer, cfg.text_max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True,
                             collate_fn=collate)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    text_model = FrozenClipTextModel(cfg.clip_model_name, cfg.embedding_dim).to(device)
    image_model = TTNImageModel(cfg.embedding_dim).to(device)
    model = ClipTextTtnVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    n_text_total = sum(p.numel() for p in text_model.parameters())
    n_text_trainable = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    n_image = sum(p.numel() for p in image_model.parameters())
    logger.info(f"Text params  (total/trainable): {n_text_total:,} / {n_text_trainable:,}  "
                f"(frozen CLIP {cfg.clip_model_name} + Linear adapter)")
    logger.info(f"Image params: {n_image:,}  (TTNImageModel, trainable)")

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = ClipBaselineStep(loss_fn=loss_fn, device=device)

    # Trainable params only: CLIP-text adapter Linear, TTN, two AlignmentHeads.
    optimizer = torch.optim.AdamW([
        {"params": text_model.proj.parameters(), "lr": cfg.text_lr, "weight_decay": cfg.text_weight_decay},
        {"params": image_model.parameters(), "lr": cfg.image_lr, "weight_decay": cfg.image_weight_decay},
        {
            "params": list(model.image_head.parameters()) + list(model.text_head.parameters()),
            "lr": cfg.head_lr,
            "weight_decay": cfg.head_weight_decay,
        },
    ])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        **cfg.model_dump(),
        **image_model_hyperparams.model_dump(),
        "text_model_params_total": n_text_total,
        "text_model_params_trainable": n_text_trainable,
        "image_model_params": n_image,
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
        logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    run()
