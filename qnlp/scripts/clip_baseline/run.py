"""CLIP-from-scratch baseline.

Same data, loss, optimizer, trainer, monitor metric, early stopping as the
aro_contrastive run — only the encoders differ:

  text_encoder  : small BERT-style transformer (4 layers, 256 hidden, 4 heads)
                   built via transformers.BertModel.from_config — NO pretrained
                   weights.
  image_encoder : torchvision ResNet-18 with weights=None (random init), final
                   fc replaced to project into the shared embedding_dim.

Both encoders run on GPU and process the batch in parallel — no cotengra
path-finding, no per-sentence Python loop. We use the existing
coco_single_caption_*.parquet (which already has `processed_text` per row) and
generate "false" captions per-batch via random derangement in the training step.

This is the apples-to-apples baseline for the EinsumModel run: if val
hard_neg_acc rises significantly above 0.5 here, the EinsumModel's vocabulary-
tail problem is the bottleneck rather than the contrastive task itself.
"""
from datetime import datetime

import mlflow
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models import resnet18
from transformers import AutoTokenizer, BertConfig, BertModel

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.domain.models.vlm.contrastive_vlm import AlignmentHead
from qnlp.scripts.clip_baseline.config import ExperimentConfig
from qnlp.scripts.clip_baseline.step import ClipBaselineStep
from qnlp.utils.logging import setup_logger
from qnlp.utils.mlflow_utils import setup_mlflow_run
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

EXPERIMENT_NAME = "clip_baseline"
logger = setup_logger(log_name=EXPERIMENT_NAME)

DATASETS_PATH = constants.datasets_path
TRAIN_PARQUET = DATASETS_PATH / "coco_single_caption_train.parquet"
VAL_PARQUET = DATASETS_PATH / "coco_single_caption_val.parquet"
TEST_PARQUET = DATASETS_PATH / "coco_single_caption_test.parquet"


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------

class MiniCLIPText(nn.Module):
    """Small BERT-style encoder, random init."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden: int,
        n_layers: int,
        n_heads: int,
        max_len: int,
    ):
        super().__init__()
        cfg = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=hidden * 4,
            max_position_embeddings=max_len,
            pad_token_id=0,
            type_vocab_size=2,
        )
        self.bert = BertModel(cfg)
        self.proj = nn.Linear(hidden, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [B, hidden]
        return self.proj(cls)               # [B, embedding_dim]


class MiniCLIPImage(nn.Module):
    """ResNet-18 with random init, final fc → embedding_dim."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.backbone = resnet18(weights=None)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ----------------------------------------------------------------------------
# Wrapper — same shape as ContrastiveVLM (image + text + alignment heads)
# ----------------------------------------------------------------------------

class ClipBaselineVLM(nn.Module):
    """Mirrors ContrastiveVLM but takes tokenised text instead of (diagram, syms)."""

    def __init__(self, text_model: MiniCLIPText, image_model: MiniCLIPImage, embedding_dim: int):
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


# ----------------------------------------------------------------------------
# Dataset & dataloaders
# ----------------------------------------------------------------------------

class CaptionImageDataset(Dataset):
    """Returns (image_tensor, caption_string). Tokenisation happens in collate."""

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


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

def run():
    cfg = ExperimentConfig()
    set_seed()
    device = get_device()

    # Tokenizer — bert-base-uncased's vocab.txt only (we don't use its weights).
    # AutoTokenizer downloads ~250 KB the first time; subsequent runs hit cache.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    image_size = 224
    train_tfm = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
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
    logger.info(f"Vocab size: {vocab_size}")

    text_model = MiniCLIPText(
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        hidden=cfg.text_hidden,
        n_layers=cfg.text_layers,
        n_heads=cfg.text_heads,
        max_len=cfg.text_max_len,
    ).to(device)
    image_model = MiniCLIPImage(cfg.embedding_dim).to(device)
    model = ClipBaselineVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)

    n_text = sum(p.numel() for p in text_model.parameters())
    n_image = sum(p.numel() for p in image_model.parameters())
    logger.info(f"Text params : {n_text:,}")
    logger.info(f"Image params: {n_image:,}")

    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature,
        triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin,
        distance=cfg.distance,
    ).to(device)

    step = ClipBaselineStep(loss_fn=loss_fn, device=device)

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
    checkpoint_path = constants.checkpoints_path / EXPERIMENT_NAME / ts / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        **cfg.model_dump(),
        "text_model_params": n_text,
        "image_model_params": n_image,
        "vocab_size": vocab_size,
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
