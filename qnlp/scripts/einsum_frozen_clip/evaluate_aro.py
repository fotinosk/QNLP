"""Evaluate an einsum_frozen_clip checkpoint on real ARO test sets.

Uses the same `hard_neg_acc` metric as training, but against the real ARO
word-order hard negatives (VG-relation, VG-attribution) rather than the
synthetic in-distribution negatives we trained with.

Assumes ARO captions have already been CCG-compiled into the sidecar
`{stem}_processed_512.jsonl` files that `ProcessedARODataset` expects —
these are produced by the existing ARO preprocessing pipeline used by v1.

Usage:
    python -m qnlp.scripts.einsum_frozen_clip.evaluate_aro \
        --checkpoint runs/checkpoints/einsum_frozen_clip/<ts>/best_model.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.discoviz.dataset.aro_dataset import ProcessedARODataset, aro_tn_collate_fn
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.scripts.einsum_frozen_clip.config import ExperimentConfig
from qnlp.scripts.einsum_frozen_clip.image_model import FrozenClipVisionModel
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="evaluate_aro_einsum_frozen_clip")

ARO_ROOT = Path("data/aro")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _gather_aro_symbols(aro_root: Path):
    """Collect all unique typed Symbols (and their sizes) from ARO test sidecars.

    The CCG-compiled sidecar files (`{stem}_processed_512.jsonl`) hold the
    compiled diagrams + symbols for ARO. We collect across both relation and
    attribution splits so the EinsumModel can be augmented in one shot.
    """
    import json as _json
    from lambeq.backend.symbol import Symbol as _Symbol
    syms_seen: dict = {}
    for kind in ("visual_genome_relation", "visual_genome_attribution"):
        sidecar = aro_root / "processed" / kind / "test_processed_512.jsonl"
        if not sidecar.exists():
            continue
        with open(sidecar) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = _json.loads(line)
                for entry in rec.get("symbols", []):
                    if not (isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], dict)):
                        continue
                    d = entry[0]
                    sym = _Symbol(
                        name=d["name"],
                        directed_dom=d["directed_dom"],
                        directed_cod=d["directed_cod"],
                    )
                    size = tuple(entry[1]) if isinstance(entry[1], list) else entry[1]
                    syms_seen[sym] = size
    return list(syms_seen.keys()), list(syms_seen.values())


def build_image_preprocess(size: int):
    """PIL Image → CLIP-normalised tensor (matches training's val_transform)."""
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


@torch.no_grad()
def evaluate_split(model: ContrastiveVLM, loader: DataLoader, device) -> dict:
    model.eval()
    n_total = 0
    n_correct = 0
    sum_pos = 0.0
    sum_neg = 0.0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        true_captions = batch["true_captions"]
        false_captions = batch["false_captions"]
        outputs = model(images, true_captions, false_captions)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--aro-root", type=Path, default=ARO_ROOT)
    args = parser.parse_args()

    device = get_device()
    cfg = ExperimentConfig()

    # Build model on CPU first to avoid GPU mem doubling during load_state_dict.
    text_model = EinsumModel()
    image_model = FrozenClipVisionModel(cfg.clip_model_name, cfg.embedding_dim)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=cfg.embedding_dim)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)

    # Augment the EinsumModel with any symbols that appear in ARO test sets
    # but were never seen during training. Without this, the forward pass
    # KeyErrors on the first ARO-only symbol. New symbols get random init,
    # which is the standard "ARO eval on a COCO-trained model" behaviour.
    aro_root_abs = args.aro_root.resolve()
    extra_syms, extra_sizes = _gather_aro_symbols(aro_root_abs)
    known = set(text_model.symbols)
    new_syms = [s for s in extra_syms if s not in known]
    new_sizes = [extra_sizes[i] for i, s in enumerate(extra_syms) if s not in known]
    if new_syms:
        logger.info(
            f"Adding {len(new_syms)} ARO-only symbols (random-init) to the EinsumModel"
        )
        text_model.add_symbols(new_syms, new_sizes)

    # Move the fully-loaded + augmented model to GPU in one shot.
    model.to(device)
    model.eval()

    preprocess = build_image_preprocess(cfg.clip_image_size)

    # Resolve absolute paths now, then chdir to the images dir so that
    # ProcessedARODataset's bare-filename Image.open(...) calls resolve.
    aro_root_abs = args.aro_root.resolve()
    json_paths = {
        kind: (aro_root_abs / "processed" / kind / "test.json")
        for kind in ("visual_genome_relation", "visual_genome_attribution")
    }
    import os
    os.chdir(aro_root_abs / "images")

    results = {}
    for kind, json_path in json_paths.items():
        if not json_path.exists():
            logger.warning(f"missing {json_path}, skipping")
            continue
        ds = ProcessedARODataset(
            data_path=str(json_path),
            return_images=True,
            image_processing_fn=preprocess,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=aro_tn_collate_fn,
        )
        logger.info(f"\nEvaluating {kind} ({len(ds):,} pairs)...")
        t0 = time.time()
        metrics = evaluate_split(model, loader, device)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        results[kind] = metrics
        logger.info(f"  {kind}:")
        for k, v in metrics.items():
            logger.info(f"    {k:20s} : {v}")

    print("\n" + "=" * 70)
    print("FINAL: einsum_frozen_clip on ARO")
    print("=" * 70)
    for kind, m in results.items():
        print(f"  {kind:30s}  hard_neg_acc = {m['hard_neg_acc']:.4f}  (n={m['n']:,})")
    print(f"\nFor reference:")
    print(f"  v1 (EinsumModel+TTN, historical):   ARO ≈ 0.78")
    print(f"  clip_baseline (ResNet18+BERT):      ARO ≈ 0.50")
    print(f"  Random baseline:                    ARO   = 0.50")


if __name__ == "__main__":
    main()
