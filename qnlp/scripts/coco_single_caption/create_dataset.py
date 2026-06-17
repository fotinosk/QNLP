"""Create COCO single-caption train/val/test datasets.

Two variants (same seed -> same split):
    (default)  linear:     no contraction paths, fast    -> coco_single_caption_*
    --paths    non-linear: contraction paths, slow       -> coco_single_caption_nlc_*
"""

import argparse

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_train_val_test_datasets
from qnlp.core.data_engine.dataset_creator.strategies.single_caption import SingleCaptionStrategy

COCO_DERIVED_DIR = constants.atlases_path / "coco" / "derived_test"


def run(paths: bool) -> None:
    output_name = "coco_single_caption_nlc" if paths else "coco_single_caption"
    create_train_val_test_datasets(
        derived_dirs=[COCO_DERIVED_DIR],
        strategy=SingleCaptionStrategy(),
        output_name=output_name,
        compute_contraction_paths=paths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", action="store_true", help="compute non-linear contraction paths (slow)")
    args = parser.parse_args()
    run(args.paths)
