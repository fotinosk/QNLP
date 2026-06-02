"""Tree-shape ablation for TreeNeuralComposer on the aligned COCO subset.

Mirrors tree_neural_aro_run.py, with two changes so the ablation sits on the
SAME footing as the COCO model lineup:

  * Data    : coco_single_caption_{split}.parquet (aligned image-disjoint
              subset). Negatives are a seeded random derangement of the
              captions within each split — matching how COCO contrastive
              negatives are built (random, not hard).
  * Image   : FrozenClipVisionModel (CLIP-ViT-B/32, frozen + trainable proj),
              identical to einsum_frozen_clip — so the ablation's ccg_single
              number is directly comparable to the lineup's TreeReader entry.

SHAPE env picks the tree source + MLP mode:
  ccg_single | left | right | balanced | random   (all single shared MLP)
  ccg                                              (per-rule MLPs)
"""
import json, os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from qnlp.constants import constants
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.core.training.trainer import Trainer
from qnlp.discoviz.models.tree_neural_composer import TreeNeuralComposer, _rule_alias
from qnlp.domain.models.vlm.contrastive_vlm import AlignmentHead
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.utils.logging import setup_logger
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device

SHAPE = os.environ.get("SHAPE", "ccg_single")
EXPERIMENT_NAME = "tree_neural_coco"
logger = setup_logger(log_name=f"{EXPERIMENT_NAME}_{SHAPE}")

SUBSET = os.environ.get("SUBSET", "data/datasets_coco_cmp")
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def tree_subdir(shape: str) -> str:
    if shape in ("ccg", "ccg_single"):
        return "coco_cmp_trees"
    return f"coco_cmp_bintrees_{shape}"


def load_trees(shape: str, split: str) -> dict:
    path = f"data/aro/processed/{tree_subdir(shape)}/{split}_trees.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["caption"]] = r["tree"]
    return out


def collect_vocab_and_rules(trees: dict):
    lemmas, rules = set(), set()
    def walk(node):
        if "leaf" in node:
            lemmas.add((node.get("lemma") or node["leaf"]).lower())
        else:
            rules.add(_rule_alias(node.get("rule", "?")))
            for c in node.get("children", []):
                walk(c)
    for t in trees.values():
        if t is not None:
            walk(t)
    return sorted(lemmas), sorted(rules)


def derangement(n: int, seed: int) -> np.ndarray:
    """Random permutation with no fixed points (false != true)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    return perm


class TreeCocoDataset(Dataset):
    def __init__(self, split: str, trees: dict, tfm, seed: int):
        df = pl.read_parquet(f"{SUBSET}/coco_single_caption_{split}.parquet",
                             columns=["local_image_path", "processed_text"])
        df = df.filter(pl.col("processed_text").is_in(list(trees.keys())))
        self.paths = df["local_image_path"].to_list()
        self.caps = df["processed_text"].to_list()
        self.trees = trees
        self.tfm = tfm
        self.neg = derangement(len(self.caps), seed)

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, i):
        img = read_image(self.paths[i], mode=ImageReadMode.RGB).float().div(255.0)
        true_cap = self.caps[i]
        false_cap = self.caps[int(self.neg[i])]
        return {
            "image": self.tfm(img),
            "true_tree": self.trees[true_cap],
            "false_tree": self.trees[false_cap],
        }


def collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "true_trees": [b["true_tree"] for b in batch],
        "false_trees": [b["false_tree"] for b in batch],
    }


class TreeNeuralVLM(nn.Module):
    def __init__(self, text_model, image_model, embedding_dim):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.image_head = AlignmentHead(embedding_dim)
        self.text_head = AlignmentHead(embedding_dim)

    def forward(self, images, true_trees, false_trees=None):
        img = self.image_head(self.image_model(images))
        true_emb = self.text_head(self.text_model(true_trees))
        out = {"image_embeddings": img, "true_caption_embeddings": true_emb}
        if false_trees is not None:
            out["false_caption_embeddings"] = self.text_head(self.text_model(false_trees))
        return out


class TreeNeuralStep:
    def __init__(self, loss_fn, device):
        self.loss_fn = loss_fn
        self.device = device

    def __call__(self, model, batch, train):
        images = batch["images"].to(self.device)
        out = model(images, batch["true_trees"], batch["false_trees"])
        loss, metrics = self.loss_fn(out)
        with torch.no_grad():
            ps = F.cosine_similarity(out["true_caption_embeddings"], out["image_embeddings"])
            ns = F.cosine_similarity(out["false_caption_embeddings"], out["image_embeddings"])
            metrics["hard_neg_acc"] = (ps > ns).float().mean()
            metrics["true_cosine_mean"] = ps.mean()
            metrics["false_cosine_mean"] = ns.mean()
        return loss, metrics


def run():
    set_seed(int(os.environ.get("ART_SEED", "42")))
    device = get_device()
    EMB = 512
    HIDDEN = int(os.environ.get("ART_HIDDEN", "512"))
    BATCH = int(os.environ.get("ART_BATCH", "256"))
    MAX_EPOCHS = int(os.environ.get("ART_MAX_EPOCHS", "40"))
    PATIENCE = int(os.environ.get("ART_PATIENCE", "8"))
    SINGLE_MLP = (SHAPE != "ccg")
    logger.info(f"SHAPE={SHAPE}  SINGLE_MLP={SINGLE_MLP}  subdir={tree_subdir(SHAPE)}")

    trees_train = load_trees(SHAPE, "train")
    trees_val = load_trees(SHAPE, "val")
    trees_test = load_trees(SHAPE, "test")
    all_trees = {**trees_train, **trees_val, **trees_test}
    vocab, rules = collect_vocab_and_rules(all_trees)
    logger.info(f"vocab={len(vocab)}  rule_names={rules}")

    train_tfm = transforms.Compose([
        transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CLIP_SIZE),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CLIP_SIZE),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    train_ds = TreeCocoDataset("train", trees_train, train_tfm, seed=20260528)
    val_ds = TreeCocoDataset("val", trees_val, eval_tfm, seed=20260529)
    test_ds = TreeCocoDataset("test", trees_test, eval_tfm, seed=20260530)
    logger.info(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if SINGLE_MLP:
        rules = ["COMP"]
    text_model = TreeNeuralComposer(vocab=vocab, rule_names=rules, d=EMB, hidden=HIDDEN).to(device)
    image_model = FrozenClipVisionModel(CLIP_MODEL, EMB).to(device)
    model = TreeNeuralVLM(text_model, image_model, EMB).to(device)
    n_text = sum(p.numel() for p in text_model.parameters())
    n_img_tr = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    logger.info(f"text params: {n_text:,} | image trainable: {n_img_tr:,}")

    mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh, num_workers=4,
                                   pin_memory=True, persistent_workers=True, collate_fn=collate)
    train_loader, val_loader, test_loader = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    loss_fn = ContrastiveLoss(
        temperature=0.07,
        triplet_weight=float(os.environ.get("ML_TRIPLET_WEIGHT", "40000")),
        triplet_margin=0.2, distance="cosine",
    ).to(device)

    optim = torch.optim.AdamW([
        {"params": text_model.parameters(), "lr": float(os.environ.get("ART_TEXT_LR", "0.0003"))},
        {"params": [p for p in image_model.parameters() if p.requires_grad], "lr": 0.0001},
        {"params": model.image_head.parameters(), "lr": 0.001},
        {"params": model.text_head.parameters(), "lr": 0.001},
    ])

    step = TreeNeuralStep(loss_fn, device)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = constants.checkpoints_path / EXPERIMENT_NAME / SHAPE / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model, optimizer=optim, step=step,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        monitor_metric="hard_neg_acc",
        checkpoint_path=str(ckpt_dir / "best_model.pt"),
        max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=0.0001,
        max_grad_norm=1.0, device=device,
    )
    test_metrics = trainer.fit()
    logger.info(f"[{SHAPE}] test_metrics: {test_metrics}")


if __name__ == "__main__":
    run()
