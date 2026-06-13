"""ARO hard-negative evaluation for FROZEN-image-tower checkpoints, by task.

Mirrors `evaluate.py`, but the image side is the frozen ViT-B/32 LookupEmbedding
(keyed by image basename) and the checkpoint holds only the text model
(`text_model_state_dict`, saved by run_frozen.py) — there is no ContrastiveVLM
or image head.

Leakage safety (identical to evaluate.py):
  * Model inputs come ONLY from the selected split parquet (default: test).
  * `task` labels come from the manifest, restricted to the split's sample_ids,
    reading only [sample_id, task]; no other-split data is loaded.
  * Non-linear mode is inferred from the checkpoint.

Usage:
    python -m qnlp.scripts.aro_contrastive.evaluate_frozen <checkpoint>
    python -m qnlp.scripts.aro_contrastive.evaluate_frozen <checkpoint> --split val
"""

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qnlp.discoviz.models.einsum_model import EinsumModel
from qnlp.discoviz.models.lookup_embeddings import LookupEmbedding
from qnlp.scripts.aro_contrastive.evaluate import _load_task_map
from qnlp.scripts.aro_contrastive.run_frozen import (
    LOOKUP_PATH,
    SPLIT_PARQUETS,
    FrozenARODataset,
    _collate,
    _embed,
)
from qnlp.utils.logging import setup_logger
from qnlp.utils.torch_utils import get_device

logger = setup_logger(log_name="aro_evaluate_frozen")


def _infer_non_linear(state_dict: dict) -> bool:
    return any("nonlinear_gate" in k for k in state_dict)


def evaluate(checkpoint_path: Path, split: str = "test", batch_size: int = 128) -> dict:
    device = get_device()
    parquet = SPLIT_PARQUETS[split]

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["text_model_state_dict"]
    non_linear = _infer_non_linear(state_dict)
    logger.info(
        f"Checkpoint epoch {checkpoint.get('epoch', '?')} | non_linear_contractions={non_linear} | split={split}"
    )

    # Text model: load_state_dict rebuilds the symbol table on CPU, so move to
    # device AFTER loading (this is the bug that crashed run_frozen's inline eval).
    text_model = EinsumModel(non_linear_contractions=non_linear)
    text_model.load_state_dict(state_dict)
    text_model.to(device).eval()

    image_lookup = LookupEmbedding.load_from_checkpoint(LOOKUP_PATH, map_location=device).to(device)
    image_lookup.eval()

    ds = FrozenARODataset(parquet, use_non_linear_contractions=non_linear)
    logger.info(f"Evaluating on {parquet.name} — {len(ds)} pairs")
    task_map = _load_task_map(set(ds.df["sample_id"].to_list()))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )

    correct_by_task: dict[str, list[bool]] = defaultdict(list)
    pos_by_task: dict[str, list[float]] = defaultdict(list)
    neg_by_task: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            outputs = _embed(text_model, image_lookup, batch, device)
            pos = F.cosine_similarity(outputs["true_caption_embeddings"], outputs["image_embeddings"])
            neg = F.cosine_similarity(outputs["false_caption_embeddings"], outputs["image_embeddings"])
            correct = (pos > neg).tolist()
            for sid, c, p, n in zip(batch["sample_id"], correct, pos.tolist(), neg.tolist()):
                task = task_map[sid]
                correct_by_task[task].append(bool(c))
                pos_by_task[task].append(float(p))
                neg_by_task[task].append(float(n))

    def _acc(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    results: dict[str, dict] = {}
    all_c: list[bool] = []
    all_p: list[float] = []
    all_n: list[float] = []
    for task in sorted(correct_by_task):
        c, p, n = correct_by_task[task], pos_by_task[task], neg_by_task[task]
        results[task] = {"n": len(c), "hard_neg_acc": _acc(c), "true_cos": _mean(p), "false_cos": _mean(n)}
        all_c += c
        all_p += p
        all_n += n
    results["overall"] = {
        "n": len(all_c),
        "hard_neg_acc": _acc(all_c),
        "true_cos": _mean(all_p),
        "false_cos": _mean(all_n),
    }

    logger.info("=== Frozen-image ARO hard-negative accuracy by task ===")
    logger.info(f"  {'task':<14}{'N':>7}{'acc':>9}{'true_cos':>10}{'false_cos':>11}")
    logger.info(f"  {'-' * 50}")
    for task in [*sorted(correct_by_task), "overall"]:
        r = results[task]
        logger.info(f"  {task:<14}{r['n']:>7}{r['hard_neg_acc']:>9.4f}{r['true_cos']:>10.4f}{r['false_cos']:>11.4f}")

    return results


if __name__ == "__main__":
    import argparse

    checkpoint = "runs/checkpoints/aro_frozen/2026-06-12_22-32-36/best_text_model.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    evaluate(checkpoint, split=args.split, batch_size=args.batch_size)
