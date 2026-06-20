"""Create the SugarCREPE swap_att evaluation dataset (eval-only — a single parquet, no splits).

Enriches atoms with diagrams/symbols, pre-computes non-linear contraction
paths, and composes them into contrastive (true/false) pairs for COCO model
evaluation.
"""

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

SUBSET = "swap_obj"
ATLAS_DERIVED_DIR = constants.atlases_path / f"sugarcrepe_{SUBSET}" / "derived_v1"


def run() -> None:
    create_dataset(
        derived_dirs=[ATLAS_DERIVED_DIR],
        strategy=ContrastivePairStrategy(),
        output_name=f"sugarcrepe_{SUBSET}_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
