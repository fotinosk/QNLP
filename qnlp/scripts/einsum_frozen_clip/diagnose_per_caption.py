"""For each val caption, compute:
  - the minimum train-frequency of any typed symbol it contains
  - the average train-frequency
  - whether the model gets it right (cos(image, true) > cos(image, false))

Then group val captions by "weakest symbol" frequency bucket and report
per-bucket accuracy. If overfitting is rare-symbol-driven, accuracy should
rise sharply with the min-freq threshold.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import polars as pl
import torch
import torch.nn.functional as F
from lambeq.backend.symbol import Symbol
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.datasets.dataloader import get_dataloaders
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.einsum_frozen_clip.config import ExperimentConfig
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.utils.torch_utils import get_device


COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption"),
    ("false_diagram", "false_symbols", "false_caption"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-parquet", type=Path, default=Path("data/datasets/coco_contrastive_train.parquet"))
    parser.add_argument("--val-parquet", type=Path, default=Path("data/datasets/coco_contrastive_val.parquet"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("/tmp/diagnose_per_caption.json"))
    args = parser.parse_args()

    cfg = ExperimentConfig()
    device = get_device()

    # ── 1. Count train frequencies ───────────────────────────────────────────
    print(f"counting train symbol frequencies from {args.train_parquet}...")
    freq: Counter = Counter()
    df_train = pl.read_parquet(args.train_parquet)
    for col in ("true_symbols", "false_symbols"):
        for raw in df_train[col].to_list():
            if raw is None:
                continue
            entries = orjson.loads(raw) if isinstance(raw, str) else raw
            for e in entries:
                if isinstance(e, (list, tuple)) and e and isinstance(e[0], dict):
                    d = e[0]
                    freq[(d["name"], d["directed_dom"], d["directed_cod"])] += 1
    print(f"  {len(freq):,} unique symbols, {sum(freq.values()):,} occurrences")

    # ── 2. Per val-caption: min/avg/max symbol frequency in train ────────────
    print(f"\nanalyzing val captions...")
    df_val = pl.read_parquet(args.val_parquet)
    caption_stats = []
    for i in range(len(df_val)):
        row = df_val.row(i, named=True)
        raw = row["true_symbols"]
        if raw is None:
            continue
        entries = orjson.loads(raw) if isinstance(raw, str) else raw
        freqs = []
        for e in entries:
            if isinstance(e, (list, tuple)) and e and isinstance(e[0], dict):
                d = e[0]
                freqs.append(freq.get((d["name"], d["directed_dom"], d["directed_cod"]), 0))
        if not freqs:
            continue
        caption_stats.append({
            "row": i,
            "min_freq": min(freqs),
            "max_freq": max(freqs),
            "avg_freq": sum(freqs) / len(freqs),
            "n_symbols": len(freqs),
            "n_unseen": sum(1 for f in freqs if f == 0),
            "n_rare": sum(1 for f in freqs if f <= 2),
        })

    n = len(caption_stats)
    print(f"  {n:,} val captions analyzed")

    # ── 3. Distribution of caption "vocab quality" ───────────────────────────
    n_with_unseen = sum(1 for c in caption_stats if c["n_unseen"] > 0)
    n_with_rare = sum(1 for c in caption_stats if c["n_rare"] > 0)
    print(f"\nval captions containing at least one NEVER-trained symbol: "
          f"{n_with_unseen:,} ({100*n_with_unseen/n:.1f}%)")
    print(f"val captions containing at least one RARE (freq≤2) symbol: "
          f"{n_with_rare:,} ({100*n_with_rare/n:.1f}%)")

    # ── 4. Run model on val, record per-row prediction correctness ───────────
    print(f"\nloading model + running val inference...")
    text_model = EinsumModel()
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    size = cfg.clip_image_size
    val_tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # Build a val-only loader. get_dataloaders takes train+val+test; pass val for all three.
    loaders, _ = get_dataloaders(
        train_parquet=args.val_parquet,
        val_parquet=args.val_parquet,
        test_parquet=args.val_parquet,
        batch_size=args.batch_size,
        train_transform=val_tfm,
        val_transform=val_tfm,
        compiled_columns=COMPILED_COLUMNS,
    )
    val_loader = loaders[1]

    correct_per_row: dict[int, bool] = {}
    row_idx = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["local_image_path"].to(device)
            outputs = model(images, batch["true_caption"], batch["false_caption"])
            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            correct = (pos > neg).cpu().tolist()
            for c in correct:
                correct_per_row[row_idx] = bool(c)
                row_idx += 1
    print(f"  {row_idx:,} val rows scored")

    # Merge with caption_stats
    rows_in_order = [s for s in caption_stats if s["row"] in correct_per_row]
    for s in rows_in_order:
        s["correct"] = correct_per_row[s["row"]]

    # ── 5. Bucket by min_freq, report accuracy per bucket ────────────────────
    print("\n" + "=" * 78)
    print("VAL ACCURACY BY 'WEAKEST SYMBOL' TRAIN FREQUENCY")
    print("=" * 78)
    buckets = [
        ("min_freq == 0 (any never-trained)", lambda s: s["min_freq"] == 0),
        ("min_freq == 1            ", lambda s: s["min_freq"] == 1),
        ("min_freq == 2            ", lambda s: s["min_freq"] == 2),
        ("min_freq ∈ [3, 9]        ", lambda s: 3 <= s["min_freq"] <= 9),
        ("min_freq ∈ [10, 99]      ", lambda s: 10 <= s["min_freq"] <= 99),
        ("min_freq ≥ 100           ", lambda s: s["min_freq"] >= 100),
    ]
    print(f"{'bucket':<40} {'n':>8} {'accuracy':>10}")
    summary = []
    for label, fn in buckets:
        sub = [s for s in rows_in_order if fn(s)]
        if not sub: continue
        acc = sum(1 for s in sub if s["correct"]) / len(sub)
        print(f"{label:<40} {len(sub):>8,} {acc:>10.4f}")
        summary.append({"bucket": label, "n": len(sub), "accuracy": acc})

    overall_acc = sum(1 for s in rows_in_order if s["correct"]) / len(rows_in_order)
    print(f"\nOverall val accuracy: {overall_acc:.4f}")

    with open(args.out, "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "n_val": len(rows_in_order),
            "overall_accuracy": overall_acc,
            "buckets": summary,
            "n_with_unseen_pct": 100*n_with_unseen/n,
            "n_with_rare_pct": 100*n_with_rare/n,
        }, f, indent=2)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
