"""
Evaluate a trained COCO single-caption model on ARO-style binary choice.

For each sample the model sees one image, a true caption and a foil caption.
Hard-negative accuracy measures how often cosine_similarity(image, true) beats
cosine_similarity(image, false) — directly comparable to the ARO benchmark and
to the 78% target from train_aro_clean.py.

Usage:
    python -m qnlp.scripts.coco_single_caption.evaluate_aro <checkpoint_path>
    python -m qnlp.scripts.coco_single_caption.evaluate_aro <checkpoint_path> --split val
    python -m qnlp.scripts.coco_single_caption.evaluate_aro <checkpoint_path> --parquet /path/to/custom.parquet
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from qnlp.constants import constants
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.image_model import TTNImageModel, image_model_hyperparams
from qnlp.domain.datasets.dataloader import vlm_collate_fn
from qnlp.domain.datasets.dataset import VLMDataset
from qnlp.domain.models.vlm.contrastive_vlm import ContrastiveVLM
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="evaluate_aro")

EMBEDDING_DIM = 512
BATCH_SIZE = 128

SPLIT_PARQUETS = {
    "train": constants.datasets_path / "coco_contrastive_train.parquet",
    "val": constants.datasets_path / "coco_contrastive_val.parquet",
    "test": constants.datasets_path / "coco_contrastive_test.parquet",
}

COMPILED_COLUMNS = [
    ("true_diagram", "true_symbols", "true_caption"),
    ("false_diagram", "false_symbols", "false_caption"),
]


def _load_model(checkpoint_path: Path, device: torch.device) -> ContrastiveVLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # EinsumModel reconstructs its symbols/sizes from the state dict
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

    ds = VLMDataset(parquet, compiled_columns=COMPILED_COLUMNS, image_transform=transform)
    logger.info(f"Evaluating on {parquet.name} — {len(ds)} samples")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
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
            images = batch["local_image_path"]
            true_captions = batch["true_caption"]
            false_captions = batch["false_caption"]

            # Filter samples whose true or false caption contains unknown symbols
            valid = [
                i
                for i in range(len(images))
                if _all_symbols_known(true_captions[i]) and _all_symbols_known(false_captions[i])
            ]
            n_skipped += len(images) - len(valid)

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

            B = images.shape[0]
            n_correct += (true_sim > false_sim).sum().item()
            n_total += B
            sum_true_sim += true_sim.sum().item()
            sum_false_sim += false_sim.sum().item()

    if n_skipped:
        logger.warning(
            f"Skipped {n_skipped} samples with unknown symbols ({n_skipped / (n_total + n_skipped):.1%} of total)"
        )

    metrics = {
        "hard_neg_accuracy": n_correct / n_total,
        "true_cosine_similarity": sum_true_sim / n_total,
        "false_cosine_similarity": sum_false_sim / n_total,
        "margin": (sum_true_sim - sum_false_sim) / n_total,
        "n_samples": n_total,
        "n_skipped": n_skipped,
    }

    logger.info("=== ARO Evaluation Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    evaluate(
        "runs/checkpoints/coco_single_caption/2026-05-01_22-36-22/best_model.pt",
        parquet=SPLIT_PARQUETS["test"],
        batch_size=512,
    )
