"""Evaluate an aro_train checkpoint on each ARO subset × split separately.

Reports hard_neg_acc for:
  visual_genome_relation/val, visual_genome_relation/test,
  visual_genome_attribution/val, visual_genome_attribution/test.

The model was trained on the combined train set (relation + attribution merged)
but the eval splits each benchmark separately so we can compare to v1's
per-benchmark numbers.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.discoviz.dataset.aro_dataset import ProcessedARODataset
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.aro_train.config import ExperimentConfig
from qnlp.scripts.aro_train.run import aro_step_collate_fn
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="evaluate_aro_per_subset")


@torch.no_grad()
def evaluate_split(model, loader, device) -> dict:
    model.eval()
    n_total = 0
    n_correct = 0
    sum_pos = 0.0
    sum_neg = 0.0
    for batch in loader:
        images = batch["local_image_path"].to(device, non_blocking=True)
        true_captions = batch["true_caption"]
        false_captions = batch["false_caption"]
        outputs = model(images, true_captions, false_captions)
        pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
        neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
        n_correct += (pos > neg).sum().item()
        n_total += images.size(0)
        sum_pos += pos.sum().item()
        sum_neg += neg.sum().item()
    return {
        "n": n_total,
        "hard_neg_acc": n_correct / n_total,
        "true_cosine_mean": sum_pos / n_total,
        "false_cosine_mean": sum_neg / n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--aro-root", type=Path, default=Path("data/aro/processed"),
    )
    parser.add_argument(
        "--aro-images", type=Path, default=Path("data/aro/images"),
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    device = get_device()

    # Build on CPU, load checkpoint, then move to device.
    text_model = EinsumModel()
    image_model = TTNImageModel(cfg.embedding_dim)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    size = image_model_hyperparams.image_size
    eval_tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    aro_root = args.aro_root.resolve()
    aro_images = args.aro_images.resolve()
    os.chdir(aro_images)

    results: dict = {}
    for kind in ("visual_genome_relation", "visual_genome_attribution"):
        for split in ("val", "test"):
            json_path = aro_root / kind / f"{split}.json"
            if not json_path.exists():
                logger.warning(f"missing {json_path}")
                continue
            ds = ProcessedARODataset(
                data_path=str(json_path),
                return_images=True,
                image_processing_fn=eval_tfm,
            )
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                pin_memory=True, collate_fn=aro_step_collate_fn,
            )
            t0 = time.time()
            m = evaluate_split(model, loader, device)
            m["elapsed_s"] = round(time.time() - t0, 1)
            results[(kind, split)] = m
            logger.info(f"  {kind}/{split}: hard_neg_acc={m['hard_neg_acc']:.4f}  (n={m['n']:,})")

    # Final summary
    print("\n" + "=" * 78)
    print("FINAL: aro_train checkpoint on per-subset val + test splits")
    print("=" * 78)
    for kind in ("visual_genome_relation", "visual_genome_attribution"):
        for split in ("val", "test"):
            if (kind, split) in results:
                m = results[(kind, split)]
                print(f"  {kind:30s} {split:5s}  hard_neg_acc = {m['hard_neg_acc']:.4f}  "
                      f"(n={m['n']:>5,}, cos_gap={m['true_cosine_mean']-m['false_cosine_mean']:+.3f})")
    print(f"\nFor reference:")
    print(f"  v1 (EinsumModel+TTN, trained on ARO+Wino): ARO ≈ 0.78")
    print(f"  Random baseline:                            ARO   = 0.50")


if __name__ == "__main__":
    main()
