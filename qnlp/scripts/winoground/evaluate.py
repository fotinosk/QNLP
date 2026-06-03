"""
Evaluate a trained model on the Winoground benchmark.

Computes the three canonical Winoground scores for each pair:
    text  — sim(image_0, cap_0) > sim(image_0, cap_1)
            AND sim(image_1, cap_1) > sim(image_1, cap_0)
    image — sim(image_0, cap_0) > sim(image_1, cap_0)
            AND sim(image_1, cap_1) > sim(image_0, cap_1)
    group — text AND image both correct

Usage:
    python -m qnlp.scripts.winoground.evaluate <checkpoint_path>
    python -m qnlp.scripts.winoground.evaluate <checkpoint_path> --split val
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.constants import constants
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.winoground_dataset import WinogroundDataset, winoground_eval_collate_fn
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="evaluate_winoground")

EMBEDDING_DIM = 512
BATCH_SIZE = 32

SPLIT_PARQUETS = {
    "train": constants.datasets_path / "winoground_train.parquet",
    "val": constants.datasets_path / "winoground_val.parquet",
    "test": constants.datasets_path / "winoground_test.parquet",
}

DEFAULT_CHECKPOINT = "runs/checkpoints/winoground/2026-05-25_14-18-09/best_model.pt"


def _load_model(checkpoint_path: Path, device: torch.device) -> ContrastiveVLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    text_model = EinsumModel()
    image_model = TTNImageModel(EMBEDDING_DIM)
    model = ContrastiveVLM(text_model, image_model, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    saved_epoch = checkpoint.get("epoch", "unknown")
    logger.info(f"Loaded checkpoint (epoch {saved_epoch}) from {checkpoint_path}")
    return model


def evaluate(
    checkpoint_path: Path,
    parquet: Path,
    batch_size: int = BATCH_SIZE,
) -> dict[str, float]:
    device = get_device()

    size = image_model_hyperparams.image_size
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = WinogroundDataset(parquet, mode="eval", image_transform=transform)
    logger.info(f"Evaluating on {parquet.name} — {len(ds)} pairs")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=winoground_eval_collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = _load_model(checkpoint_path, device)
    known_symbols = set(model.text_model.sym2weight.keys())

    def _all_known(caption) -> bool:
        _, symbols = caption
        return all(s in known_symbols for s in symbols)

    text_correct = 0
    image_correct = 0
    group_correct = 0
    n_total = 0
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            captions_0 = batch["captions_0"]
            captions_1 = batch["captions_1"]

            valid = [i for i in range(len(captions_0)) if _all_known(captions_0[i]) and _all_known(captions_1[i])]
            n_skipped += len(captions_0) - len(valid)

            if not valid:
                continue

            images_0 = torch.stack([batch["images_0"][i] for i in valid]).to(device)
            images_1 = torch.stack([batch["images_1"][i] for i in valid]).to(device)
            captions_0 = [captions_0[i] for i in valid]
            captions_1 = [captions_1[i] for i in valid]

            # Two forward passes to get all four embeddings
            out0 = model(images_0, captions_0, captions_1)
            out1 = model(images_1, captions_1, captions_0)

            img0_emb = out0["image_embeddings"]
            img1_emb = out1["image_embeddings"]
            cap0_emb = out0["true_caption_embeddings"]
            cap1_emb = out0["false_caption_embeddings"]

            s00 = F.cosine_similarity(img0_emb, cap0_emb, dim=-1)
            s01 = F.cosine_similarity(img0_emb, cap1_emb, dim=-1)
            s10 = F.cosine_similarity(img1_emb, cap0_emb, dim=-1)
            s11 = F.cosine_similarity(img1_emb, cap1_emb, dim=-1)

            text_score = (s00 > s01) & (s11 > s10)
            image_score = (s00 > s10) & (s11 > s01)
            group_score = text_score & image_score

            text_correct += text_score.sum().item()
            image_correct += image_score.sum().item()
            group_correct += group_score.sum().item()
            n_total += images_0.shape[0]

    if n_skipped:
        logger.warning(
            f"Skipped {n_skipped} pairs with unknown symbols ({n_skipped / (n_total + n_skipped):.1%} of total)"
        )

    metrics = {
        "text_score": text_correct / n_total if n_total else 0.0,
        "image_score": image_correct / n_total if n_total else 0.0,
        "group_score": group_correct / n_total if n_total else 0.0,
        "n_pairs": n_total,
        "n_skipped": n_skipped,
    }

    logger.info("=== Winoground Evaluation Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    args = sys.argv[1:]
    checkpoint = Path(args[0]) if args else Path(DEFAULT_CHECKPOINT)

    split = "test"
    if "--split" in args:
        split = args[args.index("--split") + 1]

    evaluate(checkpoint, parquet=SPLIT_PARQUETS[split], batch_size=BATCH_SIZE)
