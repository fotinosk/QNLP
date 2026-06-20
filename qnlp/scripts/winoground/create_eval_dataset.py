"""Create the Winoground evaluation dataset (eval-only — a single parquet, no splits).

Enriches atoms with diagrams/symbols, pre-computes non-linear contraction
paths, and composes them into paired (caption_0/caption_1) rows via
WinogroundPairStrategy for COCO model evaluation.
"""

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.winoground_pair import WinogroundPairStrategy

WINO_DERIVED_DIR = constants.atlases_path / "winoground" / "derived_v1"


def run() -> None:
    create_dataset(
        derived_dirs=[WINO_DERIVED_DIR],
        strategy=WinogroundPairStrategy(),
        output_name="winoground_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
