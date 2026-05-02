"""
Full-corpus retrieval evaluation for COCO single-caption models.

Encodes every test image and caption once, builds a [N × N] similarity
matrix, and measures how well the model ranks each ground-truth pair.

Metrics (image-to-text and text-to-image, then averaged):
  R@1, R@5, R@10   — recall: fraction where ground truth is in top-K
  Median rank       — median position of ground truth (lower = better)
  Mean rank         — mean position of ground truth
  MRR               — mean reciprocal rank (higher = better)

Batch accuracy is also reported: for a sliding window of `batch_size`
samples, how often does the model rank the correct caption first
(directly comparable to the InfoNCE training accuracy metric).


"""

import re
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

logger = setup_logger(log_name="evaluate_retrieval")

EMBEDDING_DIM = 512
BATCH_SIZE = 128

SPLIT_PARQUETS = {
    "train": constants.datasets_path / "coco_single_caption_train.parquet",
    "val": constants.datasets_path / "coco_single_caption_val.parquet",
    "test": constants.datasets_path / "coco_single_caption_test.parquet",
}

COMPILED_COLUMNS = [("diagram", "symbols", "caption")]


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


def _retrieval_metrics(sim_matrix: torch.Tensor) -> dict[str, float]:
    """
    Compute retrieval metrics from a [N, N] similarity matrix where
    ground truth for row i is column i.

    Returns R@1/5/10, median rank, mean rank, MRR.
    """
    N = sim_matrix.shape[0]

    # rank of the diagonal (ground truth), 1-indexed
    # argsort descending: position of each index in the ranking
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)  # [N, N]
    ranks = (sorted_indices == torch.arange(N, device=sim_matrix.device).unsqueeze(1)).nonzero(as_tuple=False)[
        :, 1
    ] + 1  # 1-indexed, shape [N]

    r1 = (ranks <= 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r10 = (ranks <= 10).float().mean().item()
    median_rank = ranks.float().median().item()
    mean_rank = ranks.float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()

    return {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "median_rank": median_rank,
        "mean_rank": mean_rank,
        "MRR": mrr,
    }


def evaluate(
    checkpoint_path: Path,
    parquet: Path,
    batch_size: int = BATCH_SIZE,
    batch_accuracy_size: int = 512,
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

    def _is_1d_diagram(diagram: str) -> bool:
        if "->" not in diagram:
            return True
        output_part = diagram.split("->")[1].strip()
        return len(re.findall(r"[a-zA-Z]", output_part)) == 1

    def _caption_valid(caption) -> bool:
        diagram, symbols = caption
        return _is_1d_diagram(diagram) and all(s in known_symbols for s in symbols)

    all_image_embs = []
    all_text_embs = []
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["local_image_path"]
            captions = batch["caption"]

            valid = [i for i in range(len(images)) if _caption_valid(captions[i])]
            n_skipped += len(images) - len(valid)

            if not valid:
                continue

            images = torch.stack([images[i] for i in valid]).to(device)
            captions = [captions[i] for i in valid]

            outputs = model(images, captions)
            all_image_embs.append(outputs["image_embeddings"].cpu())
            all_text_embs.append(outputs["true_caption_embeddings"].cpu())

    if not all_image_embs:
        raise RuntimeError("No valid samples found — check EinsumModel vocabulary.")

    image_embs = torch.cat(all_image_embs, dim=0)  # [N, D]
    text_embs = torch.cat(all_text_embs, dim=0)  # [N, D]
    N = image_embs.shape[0]

    if n_skipped:
        logger.warning(f"Skipped {n_skipped} samples with unknown symbols ({n_skipped / (N + n_skipped):.1%} of total)")
    logger.info(f"Encoding complete: {N} valid samples")

    # Both embeddings are already L2-normalised by the alignment heads.
    # Normalise again to be safe (idempotent for unit vectors).
    image_embs = F.normalize(image_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    # Full corpus similarity matrix [N, N]
    sim = image_embs @ text_embs.t()

    i2t = _retrieval_metrics(sim)
    t2i = _retrieval_metrics(sim.t())

    # Average i2t and t2i
    averaged = {k: (i2t[k] + t2i[k]) / 2 for k in i2t}

    # Batch-level accuracy: slide a window of `batch_accuracy_size` samples,
    # ask whether the correct caption is ranked #1 within that window.
    window = min(batch_accuracy_size, N)
    batch_correct = 0
    batch_total = 0
    for start in range(0, N - window + 1, window):
        end = start + window
        sub_sim = sim[start:end, start:end]  # [W, W]
        batch_correct += (sub_sim.argmax(dim=1) == torch.arange(window)).sum().item()
        batch_total += window
    batch_acc = batch_correct / batch_total if batch_total else 0.0

    metrics = {
        # Full-corpus retrieval (image-to-text)
        "i2t_R@1": i2t["R@1"],
        "i2t_R@5": i2t["R@5"],
        "i2t_R@10": i2t["R@10"],
        "i2t_median_rank": i2t["median_rank"],
        "i2t_mean_rank": i2t["mean_rank"],
        "i2t_MRR": i2t["MRR"],
        # Full-corpus retrieval (text-to-image)
        "t2i_R@1": t2i["R@1"],
        "t2i_R@5": t2i["R@5"],
        "t2i_R@10": t2i["R@10"],
        "t2i_median_rank": t2i["median_rank"],
        "t2i_mean_rank": t2i["mean_rank"],
        "t2i_MRR": t2i["MRR"],
        # Averaged
        "avg_R@1": averaged["R@1"],
        "avg_R@5": averaged["R@5"],
        "avg_R@10": averaged["R@10"],
        "avg_median_rank": averaged["median_rank"],
        "avg_MRR": averaged["MRR"],
        # Batch-level accuracy (comparable to training InfoNCE accuracy)
        f"batch_{window}_accuracy": batch_acc,
        "n_samples": N,
        "n_skipped": n_skipped,
    }

    logger.info("=== Retrieval Evaluation Results ===")
    logger.info(f"  Corpus size: {N}  (skipped: {n_skipped})")
    logger.info(f"  {'Metric':<25} {'i2t':>8}  {'t2i':>8}  {'avg':>8}")
    logger.info(f"  {'-' * 51}")
    for k in ("R@1", "R@5", "R@10", "MRR"):
        logger.info(f"  {k:<25} {i2t[k]:>8.4f}  {t2i[k]:>8.4f}  {averaged[k]:>8.4f}")
    logger.info(f"  {'Median rank':<25} {i2t['median_rank']:>8.1f}  {t2i['median_rank']:>8.1f}")
    logger.info(f"  {'Mean rank':<25} {i2t['mean_rank']:>8.1f}  {t2i['mean_rank']:>8.1f}")
    logger.info(f"  Batch-{window} accuracy: {batch_acc:.4f}")

    return metrics


if __name__ == "__main__":
    evaluate(
        checkpoint_path=Path("runs/checkpoints/coco_single_caption/2026-05-02_00-59-08/best_model.pt"),
        parquet=SPLIT_PARQUETS["test"],
        batch_size=512,
    )
