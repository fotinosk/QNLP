"""Create the Winoground evaluation dataset from the tree-no-type atlas.

Set PARSER_VERSION=tree_no_type before importing constants so the derived dir
and dataset output name are automatically versioned (-> winoground_eval_tree_no_type).
Eval-only: a single parquet, no splits.
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.winoground_pair import WinogroundPairStrategy

WINO_DERIVED_DIR = constants.atlases_path / "winoground" / constants.derived_name


def run() -> None:
    create_dataset(
        derived_dirs=[WINO_DERIVED_DIR],
        strategy=WinogroundPairStrategy(),
        output_name="winoground_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
