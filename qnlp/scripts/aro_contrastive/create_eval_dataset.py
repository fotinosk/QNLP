"""Create the ARO evaluation dataset (eval-only — a single parquet, no splits).

Uses all ARO data regardless of the split column. Enriches atoms with
diagrams/symbols, pre-computes non-linear contraction paths, and composes
them into contrastive (true/false) pairs for COCO model evaluation.
"""

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

ARO_DERIVED_DIR = constants.atlases_path / "aro" / "derived_v1"


def run() -> None:
    create_dataset(
        derived_dirs=[ARO_DERIVED_DIR],
        strategy=ContrastivePairStrategy(),
        output_name="aro_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
