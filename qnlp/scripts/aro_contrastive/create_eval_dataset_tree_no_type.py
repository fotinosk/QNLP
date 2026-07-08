"""Create the ARO evaluation dataset from the tree-no-type atlas.

Set PARSER_VERSION=tree_no_type before importing constants so the derived dir
and dataset output name are automatically versioned (-> aro_eval_tree_no_type).
Uses all ARO data regardless of split; composes contrastive (true/false) pairs.
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

ARO_DERIVED_DIR = constants.atlases_path / "aro" / constants.derived_name


def run() -> None:
    create_dataset(
        derived_dirs=[ARO_DERIVED_DIR],
        strategy=ContrastivePairStrategy(),
        output_name="aro_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
