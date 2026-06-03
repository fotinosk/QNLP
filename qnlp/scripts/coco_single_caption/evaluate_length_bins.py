"""
Retrieval evaluation stratified by caption length.

Encodes all test samples once, builds the full [N x N] similarity matrix,
then reports R@1, R@5, R@10, MRR and median rank for each word-count bin.
Metrics are computed against the full corpus (not within-bin only), so the
difficulty is identical to the global retrieval task — only the subset of
queries differs between bins.

Usage:
    python -m qnlp.scripts.coco_single_caption.evaluate_length_bins <checkpoint>
    python -m qnlp.scripts.coco_single_caption.evaluate_length_bins <checkpoint> --split val
    python -m qnlp.scripts.coco_single_caption.evaluate_length_bins <checkpoint> --parquet /path/to.parquet
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

logger = setup_logger(log_name="evaluate_length_bins")

EMBEDDING_DIM = 512
BATCH_SIZE = 128

SPLIT_PARQUETS = {
    "train": constants.datasets_path / "coco_single_caption_train.parquet",
    "val": constants.datasets_path / "coco_single_caption_val.parquet",
    "test": constants.datasets_path / "coco_single_caption_test.parquet",
}

COMPILED_COLUMNS = [("diagram", "symbols", "caption")]

# Word-count bin edges (right-inclusive).  The last bucket is open-ended.
BIN_EDGES = [0, 7, 10, 13, 999]
BIN_LABELS = ["≤7", "8–10", "11–13", "≥14"]


def _word_count(text: str) -> int:
    return len(text.split())


def _bin_index(n: int) -> int:
    for i, edge in enumerate(BIN_EDGES[1:]):
        if n <= edge:
            return i
    return len(BIN_LABELS) - 1


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


def _retrieval_metrics(sim_rows: torch.Tensor, ground_truth_indices: torch.Tensor) -> dict[str, float]:
    """
    Compute retrieval metrics for a subset of queries.

    sim_rows:            [M, N] similarity scores for M query rows against all N corpus items
    ground_truth_indices: [M]   the column index that is the correct match for each row
    """
    M = sim_rows.shape[0]
    sorted_indices = sim_rows.argsort(dim=1, descending=True)  # [M, N]
    ranks = (
        (sorted_indices == ground_truth_indices.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1  # 1-indexed
    )

    return {
        "R@1": (ranks <= 1).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MRR": (1.0 / ranks.float()).mean().item(),
        "median_rank": ranks.float().median().item(),
        "mean_rank": ranks.float().mean().item(),
        "n": M,
    }


def evaluate(
    checkpoint_path: Path,
    parquet: Path,
    batch_size: int = BATCH_SIZE,
) -> dict:
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
        return len(re.findall(r"[a-zA-Z]", diagram.split("->")[1].strip())) == 1

    def _caption_valid(caption) -> bool:
        diagram, symbols = caption
        return _is_1d_diagram(diagram) and all(s in known_symbols for s in symbols)

    all_image_embs = []
    all_text_embs = []
    all_word_counts = []
    n_skipped = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["local_image_path"]
            captions = batch["caption"]
            texts = batch["processed_text"]

            valid = [i for i in range(len(images)) if _caption_valid(captions[i])]
            n_skipped += len(images) - len(valid)
            if not valid:
                continue

            images_t = torch.stack([images[i] for i in valid]).to(device)
            captions_v = [captions[i] for i in valid]

            outputs = model(images_t, captions_v)
            all_image_embs.append(outputs["image_embeddings"].cpu())
            all_text_embs.append(outputs["true_caption_embeddings"].cpu())
            all_word_counts.extend(_word_count(texts[i]) for i in valid)

    if not all_image_embs:
        raise RuntimeError("No valid samples found — check EinsumModel vocabulary.")

    image_embs = F.normalize(torch.cat(all_image_embs, dim=0), dim=-1)  # [N, D]
    text_embs = F.normalize(torch.cat(all_text_embs, dim=0), dim=-1)  # [N, D]
    word_counts = torch.tensor(all_word_counts, dtype=torch.long)
    N = image_embs.shape[0]

    if n_skipped:
        logger.warning(f"Skipped {n_skipped} samples ({n_skipped / (N + n_skipped):.1%} of total)")
    logger.info(f"Encoding complete: {N} valid samples")

    sim = image_embs @ text_embs.t()  # [N, N]
    gt = torch.arange(N)  # ground truth: row i matches column i

    # Global metrics
    global_i2t = _retrieval_metrics(sim, gt)
    global_t2i = _retrieval_metrics(sim.t(), gt)

    # Per-bin metrics
    bin_results = []
    for b, label in enumerate(BIN_LABELS):
        mask = torch.tensor([_bin_index(wc.item()) == b for wc in word_counts])
        idx = mask.nonzero(as_tuple=False).squeeze(1)

        if idx.numel() == 0:
            bin_results.append(None)
            continue

        i2t = _retrieval_metrics(sim[idx], idx)
        t2i = _retrieval_metrics(sim.t()[idx], idx)
        avg = {k: (i2t[k] + t2i[k]) / 2 for k in ("R@1", "R@5", "R@10", "MRR", "median_rank")}
        bin_results.append({"label": label, "n": idx.numel(), "i2t": i2t, "t2i": t2i, "avg": avg})

    # Logging
    logger.info("=== Length-Bin Retrieval Results ===")
    logger.info(f"  Corpus size: {N}  (skipped: {n_skipped})")
    logger.info(f"  {'Bin':<8} {'N':>6}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'MRR':>7}  {'Med.Rank':>9}")
    logger.info(f"  {'-' * 60}")

    def _log_row(label, n, avg):
        logger.info(
            f"  {label:<8} {n:>6}  "
            f"{avg['R@1']:>7.4f}  {avg['R@5']:>7.4f}  {avg['R@10']:>7.4f}  "
            f"{avg['MRR']:>7.4f}  {avg['median_rank']:>9.1f}"
        )

    global_avg = {k: (global_i2t[k] + global_t2i[k]) / 2 for k in ("R@1", "R@5", "R@10", "MRR", "median_rank")}
    _log_row("ALL", N, global_avg)
    logger.info(f"  {'-' * 60}")
    for res in bin_results:
        if res is not None:
            _log_row(res["label"], res["n"], res["avg"])

    return {
        "global": {"i2t": global_i2t, "t2i": global_t2i, "avg": global_avg},
        "bins": {res["label"]: res for res in bin_results if res is not None},
        "n_samples": N,
        "n_skipped": n_skipped,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    parquet = args.parquet or SPLIT_PARQUETS[args.split]
    evaluate(args.checkpoint, parquet=parquet, batch_size=args.batch_size)
