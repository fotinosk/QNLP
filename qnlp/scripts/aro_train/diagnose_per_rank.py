"""Per-rank diagnostic on an aro_train checkpoint.

Measures whether rank-1 noun tensors learned less than rank-3 verb tensors
during training, by comparing checkpoint weights to a fresh re-initialisation
with the same random seed.

Reports per-rank:
  - mean ‖θ_final − θ_init‖₂ / ‖θ_init‖₂ (relative movement)
  - mean ‖θ_final‖₂
  - mean ‖θ_init‖₂
  - count of symbols of that rank
  - per-rank gradient norm on a small batch (forward + backward)
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.discoviz.dataset.aro_dataset import ProcessedARODataset
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.core.training.losses.contrastive import ContrastiveLoss
from qnlp.scripts.aro_train.config import ExperimentConfig
from qnlp.scripts.aro_train.run import aro_step_collate_fn
from qnlp.utils.seeding import set_seed
from qnlp.utils.torch_utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--aro-root", type=Path, default=Path("data/aro/processed"))
    parser.add_argument("--aro-images", type=Path, default=Path("data/aro/images"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    device = get_device()

    # Build CPU model + load trained state
    text_model = EinsumModel()
    image_model = TTNImageModel(cfg.embedding_dim)
    trained = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    trained.load_state_dict(sd, strict=True)

    # Rebuild with same seed for a fair init reference
    set_seed()
    text_model_init = EinsumModel(list(text_model.symbols), list(text_model.sizes))

    # ── 1. Per-rank parameter movement ────────────────────────────────────
    rank_stats: dict[int, dict] = defaultdict(lambda: {
        "count": 0, "init_norm_sum": 0.0, "final_norm_sum": 0.0,
        "delta_norm_sum": 0.0, "rel_movement_sum": 0.0,
        "init_norms": [], "final_norms": [], "rel_movements": [],
    })
    for sym, p_final, p_init in zip(text_model.symbols, text_model.weights, text_model_init.weights):
        r = len(p_final.shape)  # rank (number of axes)
        i_norm = p_init.detach().norm().item()
        f_norm = p_final.detach().norm().item()
        d_norm = (p_final.detach() - p_init.detach()).norm().item()
        rel = d_norm / (i_norm + 1e-12)
        s = rank_stats[r]
        s["count"] += 1
        s["init_norm_sum"] += i_norm
        s["final_norm_sum"] += f_norm
        s["delta_norm_sum"] += d_norm
        s["rel_movement_sum"] += rel
        s["init_norms"].append(i_norm)
        s["final_norms"].append(f_norm)
        s["rel_movements"].append(rel)

    print("=" * 78)
    print(f"PER-RANK PARAMETER MOVEMENT: {args.checkpoint}")
    print("=" * 78)
    print(f"{'Rank':<5} {'Count':>7} {'avg ‖θ_init‖':>14} {'avg ‖θ_final‖':>15} "
          f"{'avg ‖Δ‖':>10} {'avg ‖Δ‖/‖θ_init‖':>20}")
    for r in sorted(rank_stats):
        s = rank_stats[r]
        n = s["count"]
        print(f"{r:<5} {n:>7} {s['init_norm_sum']/n:>14.4f} {s['final_norm_sum']/n:>15.4f} "
              f"{s['delta_norm_sum']/n:>10.4f} {s['rel_movement_sum']/n:>20.4f}")

    # ── 2. Per-rank gradient norms on a small batch ───────────────────────
    print()
    print("=" * 78)
    print("PER-RANK GRADIENT NORMS (single batch through trained model)")
    print("=" * 78)
    trained.to(device).train()
    aro_root = args.aro_root.resolve()
    aro_images = args.aro_images.resolve()
    eval_tfm = transforms.Compose([
        transforms.Resize((image_model_hyperparams.image_size, image_model_hyperparams.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    os.chdir(aro_images)
    ds = ProcessedARODataset(
        data_path=str(aro_root / "combined" / "train.json"),
        return_images=True,
        image_processing_fn=eval_tfm,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                        pin_memory=True, collate_fn=aro_step_collate_fn)
    loss_fn = ContrastiveLoss(
        temperature=cfg.temperature, triplet_weight=cfg.triplet_weight,
        triplet_margin=cfg.triplet_margin, distance=cfg.distance,
    ).to(device)
    batch = next(iter(loader))
    images = batch["local_image_path"].to(device)
    outputs = trained(images, batch["true_caption"], batch["false_caption"])
    loss, _ = loss_fn(outputs)
    trained.zero_grad()
    loss.backward()

    rank_grad: dict[int, list[float]] = defaultdict(list)
    for p in trained.text_model.weights:
        if p.grad is None:
            continue
        r = len(p.shape)
        rank_grad[r].append(p.grad.norm().item())

    print(f"{'Rank':<5} {'count':>7} {'mean ‖∇θ‖':>14} {'max ‖∇θ‖':>14} {'min ‖∇θ‖':>14}")
    for r in sorted(rank_grad):
        gs = rank_grad[r]
        n = len(gs)
        print(f"{r:<5} {n:>7} {sum(gs)/n:>14.6f} {max(gs):>14.6f} {min(gs):>14.6f}")
    print(f"\nLoss on this batch: {loss.item():.4f}")


if __name__ == "__main__":
    main()
