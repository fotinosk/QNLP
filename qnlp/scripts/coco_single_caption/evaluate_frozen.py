"""
Standalone evaluation of a frozen checkpoint against all benchmarks.

Reconstructs the model entirely from the checkpoint (symbols, sizes, and
non_linear_contractions are stored inside the EinsumModel state_dict), so no
dataset rebuild is needed. Prints the same contiguous FINAL TRAINING REPORT as
run_frozen.py: COCO retrieval, Winoground (overall + per-tag), ARO
(attribution/relation/overall), SugarCREPE full + ++.

Usage:
    python -m qnlp.scripts.coco_single_caption.evaluate_frozen \\
        --checkpoint /path/to/best_model.pt

    python -m qnlp.scripts.coco_single_caption.evaluate_frozen \\
        --checkpoint /path/to/best_model.pt --batch_size 128
"""

import argparse

import torch
from torch import nn

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.scripts.coco_multi_caption.evaluate import (
    evaluate_all_benchmarks_frozen,
    log_banner,
    print_full_report,
)
from qnlp.scripts.coco_single_caption.run_frozen import (
    CLIP_MODEL,
    DATASETS_PATH,
    TEST_SIZE,
    CLIPImageCache,
    FrozenCOCODataset,
    _collect_retrieval_metrics,
    _dedup_loader,
)
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="evaluate_frozen")


def evaluate(checkpoint_path: str, batch_size: int = 256) -> None:
    device = get_device()

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    # The frozen checkpoint holds only the text model + head (no optimizer state),
    # so it is small; load to CPU first anyway, then move the reconstructed modules.
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    sd = ckpt["text_model_state_dict"]
    nlc = bool(sd.get("non_linear_contractions", False))

    # load_state_dict recreates self.weights as new CPU tensors, so .to(device)
    # must come AFTER load_state_dict to move everything consistently.
    text_model = EinsumModel(non_linear_contractions=nlc)
    text_model.load_state_dict(sd)
    text_model = text_model.to(device)
    text_model.eval()
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Epoch: {epoch}  non_linear_contractions: {nlc}  symbols: {len(text_model.symbols)}")

    head_sd = ckpt["text_head_state_dict"]
    embedding_dim = head_sd["weight"].shape[0]
    text_head = nn.Linear(embedding_dim, embedding_dim)
    text_head.load_state_dict(head_sd)
    text_head = text_head.to(device)
    text_head.eval()
    logger.info(f"embedding_dim: {embedding_dim}")

    image_cache = CLIPImageCache(CLIP_MODEL, device)
    if image_cache.embedding_dim != embedding_dim:
        raise ValueError(
            f"CLIP embedding dim ({image_cache.embedding_dim}) != checkpoint embedding_dim ({embedding_dim})."
        )

    dataset_name = "coco_single_caption_nlc" if nlc else "coco_single_caption"
    test_ds = FrozenCOCODataset(DATASETS_PATH / f"{dataset_name}_test.parquet", use_non_linear_contractions=nlc)
    test_loader = _dedup_loader(test_ds, batch_size, max_images=TEST_SIZE, topology_bucketing=not nlc)
    logger.info(f"Test set: {len(test_ds)} rows → {len(test_loader.dataset)} unique images (capped at {TEST_SIZE})")

    log_banner(
        "FROZEN EVALUATION — START",
        {
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "non_linear_contractions": nlc,
            "embedding_dim": embedding_dim,
            "n_symbols": len(text_model.symbols),
            "frozen_image_tower": CLIP_MODEL,
            "batch_size": batch_size,
        },
    )

    logger.info("--- COCO Retrieval ---")
    retrieval = _collect_retrieval_metrics(text_model, text_head, image_cache, test_loader)

    benchmarks = evaluate_all_benchmarks_frozen(text_model, text_head, image_cache, batch_size, device, nlc)

    report_info = {
        "checkpoint_path": checkpoint_path,
        "epoch": epoch,
        "non_linear_contractions": nlc,
        "embedding_dim": embedding_dim,
        "n_symbols": len(text_model.symbols),
        "frozen_image_tower": CLIP_MODEL,
    }
    print_full_report(retrieval, benchmarks, report_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint against all benchmarks.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.batch_size)
