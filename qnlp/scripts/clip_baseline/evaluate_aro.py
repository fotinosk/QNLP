"""Evaluate the clip_baseline checkpoint on the ARO Visual-Genome benchmarks.

Same `hard_neg_acc` metric we used at training time, but against the real
ARO hard negatives (word-order swaps that flip relational meaning or swap
attributes) rather than random in-batch negatives.

Usage:
    python -m qnlp.scripts.clip_baseline.evaluate_aro \
        --checkpoint runs/checkpoints/clip_baseline/<ts>/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image

from qnlp.scripts.clip_baseline.config import ExperimentConfig
from qnlp.scripts.clip_baseline.run import ClipBaselineVLM, MiniCLIPImage, MiniCLIPText
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device
from transformers import AutoTokenizer

logger = setup_logger(log_name="evaluate_aro_clip")

# Resolved at import time; both layouts work since data/aro lives next to project root.
ARO_ROOT = Path("data/aro")


class AROPairDataset(Dataset):
    """Yields (image_tensor, true_caption, false_caption) per ARO test entry.

    ARO test JSON entries look like:
        {"true_caption": "...", "false_caption": "...", "image_path": "1234.jpg", ...}
    """

    def __init__(self, json_path: Path, image_dir: Path, transform):
        with open(json_path) as f:
            self.entries = json.load(f)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img_path = self.image_dir / e["image_path"]
        img = read_image(str(img_path), mode=ImageReadMode.RGB)
        img = self.transform(img)
        return img, e["true_caption"], e["false_caption"]


def make_collate(tokenizer, max_len: int):
    def _collate(batch):
        images = torch.stack([b[0] for b in batch])
        true_texts = [b[1] for b in batch]
        false_texts = [b[2] for b in batch]
        true_tok = tokenizer(true_texts, padding="max_length", max_length=max_len,
                             truncation=True, return_tensors="pt")
        false_tok = tokenizer(false_texts, padding="max_length", max_length=max_len,
                              truncation=True, return_tensors="pt")
        return {
            "image": images,
            "true_ids": true_tok["input_ids"],
            "true_mask": true_tok["attention_mask"],
            "false_ids": false_tok["input_ids"],
            "false_mask": false_tok["attention_mask"],
        }
    return _collate


@torch.no_grad()
def evaluate_split(model: ClipBaselineVLM, loader: DataLoader, device) -> dict:
    model.eval()
    n_total = 0
    n_correct = 0
    sum_pos = 0.0
    sum_neg = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        true_text = {"input_ids": batch["true_ids"].to(device, non_blocking=True),
                     "attention_mask": batch["true_mask"].to(device, non_blocking=True)}
        false_text = {"input_ids": batch["false_ids"].to(device, non_blocking=True),
                      "attention_mask": batch["false_mask"].to(device, non_blocking=True)}
        outputs = model(images, true_text, false_text)
        pos_sim = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
        neg_sim = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
        n_correct += (pos_sim > neg_sim).sum().item()
        n_total += images.size(0)
        sum_pos += pos_sim.sum().item()
        sum_neg += neg_sim.sum().item()
    return {
        "n": n_total,
        "hard_neg_acc": n_correct / n_total,
        "true_cosine_mean": sum_pos / n_total,
        "false_cosine_mean": sum_neg / n_total,
    }


def build_model(cfg: ExperimentConfig, vocab_size: int, device) -> ClipBaselineVLM:
    text_model = MiniCLIPText(
        vocab_size=vocab_size,
        embedding_dim=cfg.embedding_dim,
        hidden=cfg.text_hidden,
        n_layers=cfg.text_layers,
        n_heads=cfg.text_heads,
        max_len=cfg.text_max_len,
    ).to(device)
    image_model = MiniCLIPImage(cfg.embedding_dim).to(device)
    return ClipBaselineVLM(text_model, image_model, embedding_dim=cfg.embedding_dim).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--aro-root", type=Path, default=ARO_ROOT)
    args = parser.parse_args()

    device = get_device()
    cfg = ExperimentConfig()
    logger.info(f"Loading tokenizer (bert-base-uncased vocab only)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    logger.info(f"Building model...")
    model = build_model(cfg, vocab_size=tokenizer.vocab_size, device=device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded ({n_params:,} params); best epoch in ckpt: {ckpt.get('epoch')}")

    eval_tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_dir = args.aro_root / "images"
    collate = make_collate(tokenizer, cfg.text_max_len)

    results = {}
    for kind in ("visual_genome_relation", "visual_genome_attribution"):
        json_path = args.aro_root / "processed" / kind / "test.json"
        if not json_path.exists():
            logger.warning(f"missing {json_path}, skipping")
            continue
        ds = AROPairDataset(json_path, image_dir, eval_tfm)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4,
                            pin_memory=True, persistent_workers=False, collate_fn=collate)
        logger.info(f"\nEvaluating {kind} ({len(ds):,} pairs)...")
        t0 = time.time()
        metrics = evaluate_split(model, loader, device)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        results[kind] = metrics
        logger.info(f"  {kind}:")
        for k, v in metrics.items():
            logger.info(f"    {k:20s} : {v}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL: clip_baseline on ARO")
    print("=" * 70)
    for kind, m in results.items():
        print(f"  {kind:30s}  hard_neg_acc = {m['hard_neg_acc']:.4f}  (n={m['n']:,})")
    print(f"\nFor reference, v1 ARO baseline (EinsumModel + TTN) : 0.78")
    print(f"Random baseline                                     : 0.50")


if __name__ == "__main__":
    main()
