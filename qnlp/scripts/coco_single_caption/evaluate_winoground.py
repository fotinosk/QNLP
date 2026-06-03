"""
Evaluate a trained COCO single-caption model on the Winoground benchmark.

For each sample the model sees one image, a true caption and a foil caption.
Hard-negative accuracy measures how often cosine_similarity(image, true) beats
cosine_similarity(image, false).

Usage:
    python -m qnlp.scripts.coco_single_caption.evaluate_winoground
    python -m qnlp.scripts.coco_single_caption.evaluate_winoground <checkpoint_path>
    python -m qnlp.scripts.coco_single_caption.evaluate_winoground <checkpoint_path> --split val
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qnlp.discoviz.image_transforms.aro import create_aro_image_transforms
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.aro.aro_dataset import ProcessedARODataset, aro_tn_collate_fn
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="evaluate_winoground")

EMBEDDING_DIM = 512
BATCH_SIZE = 128

SPLIT_PATHS = {
    "train": Path("data/winoground/processed/train.json"),
    "val": Path("data/winoground/processed/val.json"),
    "test": Path("data/winoground/processed/test.json"),
}

DEFAULT_CHECKPOINT = "runs/checkpoints/coco_single_caption/2026-05-02_11-27-49/best_model.pt"


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
    data_path: Path,
    batch_size: int = BATCH_SIZE,
) -> dict[str, float]:
    device = get_device()

    _, val_preprocess = create_aro_image_transforms(image_model_hyperparams.image_size)

    ds = ProcessedARODataset(data_path=str(data_path), return_images=True, image_processing_fn=val_preprocess)
    logger.info(f"Evaluating on {data_path.name} — {len(ds)} samples")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=aro_tn_collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = _load_model(checkpoint_path, device)

    known_symbols = set(model.text_model.sym2weight.keys())

    def _all_symbols_known(caption) -> bool:
        _, symbols = caption
        return all(s in known_symbols for s in symbols)

    n_correct = 0
    n_total = 0
    n_skipped = 0
    sum_true_sim = 0.0
    sum_false_sim = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"]
            true_captions = batch["true_captions"]
            false_captions = batch["false_captions"]

            valid = [
                i
                for i in range(len(true_captions))
                if _all_symbols_known(true_captions[i]) and _all_symbols_known(false_captions[i])
            ]
            n_skipped += len(true_captions) - len(valid)

            if not valid:
                continue

            images = torch.stack([images[i] for i in valid]).to(device)
            true_captions = [true_captions[i] for i in valid]
            false_captions = [false_captions[i] for i in valid]

            outputs = model(images, true_captions, false_captions)
            img_emb = outputs["image_embeddings"]
            true_emb = outputs["true_caption_embeddings"]
            false_emb = outputs["false_caption_embeddings"]

            true_sim = F.cosine_similarity(img_emb, true_emb, dim=-1)
            false_sim = F.cosine_similarity(img_emb, false_emb, dim=-1)

            n_correct += (true_sim > false_sim).sum().item()
            n_total += images.shape[0]
            sum_true_sim += true_sim.sum().item()
            sum_false_sim += false_sim.sum().item()

    if n_skipped:
        logger.warning(
            f"Skipped {n_skipped} samples with unknown symbols ({n_skipped / (n_total + n_skipped):.1%} of total)"
        )

    metrics = {
        "hard_neg_accuracy": n_correct / n_total if n_total > 0 else 0.0,
        "true_cosine_similarity": sum_true_sim / n_total if n_total > 0 else 0.0,
        "false_cosine_similarity": sum_false_sim / n_total if n_total > 0 else 0.0,
        "margin": (sum_true_sim - sum_false_sim) / n_total if n_total > 0 else 0.0,
        "n_samples": n_total,
        "n_skipped": n_skipped,
    }

    logger.info("=== Winoground Evaluation Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    args = sys.argv[1:]
    checkpoint = Path(args[0]) if args else Path(DEFAULT_CHECKPOINT)

    split = "train"
    if "--split" in args:
        split = args[args.index("--split") + 1]

    evaluate(checkpoint, data_path=SPLIT_PATHS[split], batch_size=BATCH_SIZE)
