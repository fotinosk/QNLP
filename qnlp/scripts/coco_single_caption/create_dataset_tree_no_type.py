"""Create COCO single-caption train/val/test datasets from the tree-no-type atlas.

Set PARSER_VERSION=tree_no_type before importing constants so the derived dir
and dataset output names are automatically versioned. Used by both single- and
multi-caption COCO training (select via ML_DATASET_NAME=coco_single_caption_nlc_tree_no_type).

Always computed with contraction paths (non-linear). The path column is simply
ignored in linear training, so a single non-linear dataset serves both modes and
there is no need for a separate linear build.

Output: coco_single_caption_nlc_tree_no_type_{train,val,test}.parquet
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_train_val_test_datasets
from qnlp.core.data_engine.dataset_creator.strategies.single_caption import SingleCaptionStrategy

COCO_DERIVED_DIR = constants.atlases_path / "coco" / constants.derived_name


def run() -> None:
    create_train_val_test_datasets(
        derived_dirs=[COCO_DERIVED_DIR],
        strategy=SingleCaptionStrategy(),
        output_name="coco_single_caption_nlc",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
