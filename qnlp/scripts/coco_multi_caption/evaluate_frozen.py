"""
Evaluate a trained COCO checkpoint (frozen CLIP image tower) on retrieval +
compositional benchmarks, without retraining.

For run_frozen.py runs where training completed and the checkpoint was saved,
but the post-training in-process eval crashed (e.g. the checkpoint-reload CUDA
OOM inside EinsumModel.load_state_dict — see HARD_NEG_PI_SWEEP_PLAN.md /
RESULTS.md) — this reloads the saved checkpoint fresh, on a clean process/GPU
context, and reruns just the eval instead of re-training from scratch.

Architecture (embedding_dim, non-linear mode) is inferred from the checkpoint
so no config flags are needed, mirroring evaluate.py's non-frozen counterpart.

Usage:
    python -m qnlp.scripts.coco_multi_caption.evaluate_frozen <checkpoint_path>
    python -m qnlp.scripts.coco_multi_caption.evaluate_frozen <checkpoint_path> \
        --dataset coco_single_caption_nlc --batch_size 64
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.scripts.coco_multi_caption.evaluate import (
    evaluate_all_benchmarks_frozen,
    print_full_report,
)
from qnlp.scripts.coco_multi_caption.run_frozen import (
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

logger = setup_logger(log_name="coco_evaluate_frozen")


def _infer_non_linear(text_model_state_dict: dict) -> bool:
    return any("nonlinear_gate" in k for k in text_model_state_dict)


def _infer_embedding_dim(text_head_state_dict: dict) -> int:
    return text_head_state_dict["weight"].shape[0]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[EinsumModel, nn.Linear, int]:
    # Load to CPU first, build on CPU, then move — same reasoning as evaluate.py's
    # load_model and the trainer.py / run_frozen.py checkpoint-reload OOM fix:
    # EinsumModel.load_state_dict rebuilds its weight ParameterList on whichever
    # device the model is already on, so building fresh (empty, CPU) modules and
    # loading before any .to(device) avoids ever holding two copies on the GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    text_state = checkpoint["text_model_state_dict"]
    head_state = checkpoint["text_head_state_dict"]
    non_linear = _infer_non_linear(text_state)
    embedding_dim = _infer_embedding_dim(head_state)
    logger.info(
        f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}) | "
        f"embedding_dim={embedding_dim} | non_linear={non_linear}"
    )

    text_model = EinsumModel(non_linear_contractions=non_linear)
    text_model.load_state_dict(text_state)
    text_head = nn.Linear(embedding_dim, embedding_dim)
    text_head.load_state_dict(head_state)
    del checkpoint, text_state, head_state

    text_model.to(device).eval()
    text_head.to(device).eval()
    return text_model, text_head, embedding_dim


def evaluate_all(
    checkpoint_path: Path,
    batch_size: int = 128,
    dataset_name: str | None = None,
) -> dict[str, dict | None]:
    device = get_device()
    text_model, text_head, embedding_dim = load_model(checkpoint_path, device)
    non_linear = text_model.non_linear_contractions

    dataset = dataset_name or ("coco_single_caption_nlc" if non_linear else "coco_single_caption")
    test_ds = FrozenCOCODataset(DATASETS_PATH / f"{dataset}_test.parquet", non_linear)
    # Topology bucketing only helps EinsumModel's batched fast path in linear mode.
    test_loader = _dedup_loader(test_ds, batch_size, max_images=TEST_SIZE, topology_bucketing=not non_linear)

    image_cache = CLIPImageCache(CLIP_MODEL, device)
    if image_cache.embedding_dim != embedding_dim:
        raise ValueError(
            f"CLIP embedding dim ({image_cache.embedding_dim}) != checkpoint embedding_dim ({embedding_dim})."
        )

    try:
        retrieval = _collect_retrieval_metrics(text_model, text_head, image_cache, test_loader)
    except Exception as e:
        logger.warning(f"Retrieval: eval skipped ({type(e).__name__}: {e})")
        retrieval = None

    benchmarks = evaluate_all_benchmarks_frozen(text_model, text_head, image_cache, batch_size, device, non_linear)
    info = {
        "checkpoint": str(checkpoint_path),
        "embedding_dim": embedding_dim,
        "non_linear": non_linear,
        "dataset": dataset,
    }
    print_full_report(retrieval=retrieval, benchmarks=benchmarks, info=info)
    return {"retrieval": retrieval, **benchmarks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen-image-tower COCO checkpoint on retrieval + compositional benchmarks."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to best_model.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the retrieval test-set dataset name (e.g. coco_single_caption_nlc). "
        "Required if training used ML_DATASET_NAME to override the per-mode default — the "
        "checkpoint doesn't record which dataset it was trained on, so the built-in "
        "linear/non-linear heuristic can silently grab the wrong test set.",
    )
    args = parser.parse_args()

    evaluate_all(args.checkpoint, batch_size=args.batch_size, dataset_name=args.dataset)
