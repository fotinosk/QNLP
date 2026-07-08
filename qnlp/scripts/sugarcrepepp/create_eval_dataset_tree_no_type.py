"""Create the SugarCREPE++ evaluation dataset from the tree-no-type atlas.

Set PARSER_VERSION=tree_no_type before importing constants so the derived dir
and dataset output name are auto-versioned (-> sugarcrepepp_eval_tree_no_type).
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

ATLAS_NAME = "sugarcrepepp"
ATLAS_DERIVED_DIR = constants.atlases_path / ATLAS_NAME / constants.derived_name


def run() -> None:
    create_dataset(
        derived_dirs=[ATLAS_DERIVED_DIR],
        strategy=ContrastivePairStrategy(),
        output_name="sugarcrepepp_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
