"""
Standalone evaluation of a coco_single_caption_frozen checkpoint.

Reconstructs the model entirely from the checkpoint (symbols, sizes, and
non_linear_contractions are stored inside the EinsumModel state_dict), so no
dataset rebuild is needed. Runs COCO retrieval, Winoground, ARO, and
SugarCREPE using the same functions as the training script.

Usage:
    python -m qnlp.scripts.coco_single_caption.evaluate_frozen \\
        --checkpoint /path/to/best_model.pt

    python -m qnlp.scripts.coco_single_caption.evaluate_frozen \\
        --checkpoint /path/to/best_model.pt --batch_size 128
"""

import argparse

import torch
from torch import nn

from qnlp.constants import constants
from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.scripts.coco_single_caption.run_frozen import (
    _ARO_COMPILED,
    _SC_COMPILED,
    CLIP_MODEL,
    DATASETS_PATH,
    TEST_SIZE,
    CLIPImageCache,
    FrozenCOCODataset,
    _collect_retrieval_metrics,
    _dedup_loader,
    _eval_hard_neg_frozen,
    _eval_winoground_frozen,
    _print_final_summary,
)
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="coco_single_caption_eval_frozen")


def evaluate(checkpoint_path: str, batch_size: int = 256) -> None:
    device = get_device()

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Peek at the nlc flag before constructing so the gate parameter exists
    # before load_state_dict tries to populate it.
    sd = ckpt["text_model_state_dict"]
    nlc = bool(sd.get("non_linear_contractions", False))

    text_model = EinsumModel(non_linear_contractions=nlc).to(device)
    text_model.load_state_dict(sd)
    text_model.eval()
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Epoch: {epoch}  non_linear_contractions: {nlc}  symbols: {len(text_model.symbols)}")

    # Infer embedding_dim from the linear head weight shape
    head_sd = ckpt["text_head_state_dict"]
    embedding_dim = head_sd["weight"].shape[0]
    text_head = nn.Linear(embedding_dim, embedding_dim).to(device)
    text_head.load_state_dict(head_sd)
    text_head.eval()

    logger.info(f"embedding_dim: {embedding_dim}")

    image_cache = CLIPImageCache(CLIP_MODEL, device)
    if image_cache.embedding_dim != embedding_dim:
        raise ValueError(
            f"CLIP embedding dim ({image_cache.embedding_dim}) != checkpoint embedding_dim ({embedding_dim})."
        )

    dataset_name = "coco_single_caption_nlc" if nlc else "coco_single_caption"
    test_ds = FrozenCOCODataset(DATASETS_PATH / f"{dataset_name}_test.parquet", use_non_linear_contractions=nlc)
    test_loader = _dedup_loader(test_ds, batch_size, max_images=TEST_SIZE)
    logger.info(f"Test set: {len(test_ds)} rows → {len(test_loader.dataset)} unique images (capped at {TEST_SIZE})")

    logger.info("--- COCO Retrieval ---")
    retrieval = _collect_retrieval_metrics(text_model, text_head, image_cache, test_loader)

    logger.info("--- Winoground ---")
    wino = _eval_winoground_frozen(text_model, text_head, image_cache, batch_size, device, nlc)

    logger.info("--- ARO ---")
    aro = _eval_hard_neg_frozen(
        text_model,
        text_head,
        image_cache,
        constants.datasets_path / "aro_eval.parquet",
        _ARO_COMPILED,
        batch_size,
        device,
        nlc,
    )

    logger.info("--- SugarCREPE (swap_obj) ---")
    sc = _eval_hard_neg_frozen(
        text_model,
        text_head,
        image_cache,
        constants.datasets_path / "sugarcrepe_swap_obj_eval.parquet",
        _SC_COMPILED,
        batch_size,
        device,
        nlc,
    )

    _print_final_summary(retrieval, wino, aro, sc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a coco_single_caption_frozen checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.batch_size)
