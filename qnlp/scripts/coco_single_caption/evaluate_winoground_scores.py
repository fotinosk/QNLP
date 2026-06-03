"""
Evaluate a COCO-trained model on the Winoground benchmark using the canonical
text / image / group scores.

Usage:
    python -m qnlp.scripts.coco_single_caption.evaluate_winoground_scores
    python -m qnlp.scripts.coco_single_caption.evaluate_winoground_scores <checkpoint_path>
    python -m qnlp.scripts.coco_single_caption.evaluate_winoground_scores <checkpoint_path> --split val
"""

import sys
from pathlib import Path

from qnlp.constants import constants
from qnlp.scripts.winoground.evaluate import evaluate

EMBEDDING_DIM = 256
BATCH_SIZE = 32

SPLIT_PARQUETS = {
    "train": constants.datasets_path / "winoground_train.parquet",
    "val": constants.datasets_path / "winoground_val.parquet",
    "test": constants.datasets_path / "winoground_test.parquet",
}

DEFAULT_CHECKPOINT = "runs/checkpoints/coco_single_caption/2026-05-22_20-01-07/best_model.pt"


if __name__ == "__main__":
    args = sys.argv[1:]
    checkpoint = Path(args[0]) if args else Path(DEFAULT_CHECKPOINT)

    split = "test"
    if "--split" in args:
        split = args[args.index("--split") + 1]

    evaluate(checkpoint, parquet=SPLIT_PARQUETS[split], batch_size=BATCH_SIZE)
