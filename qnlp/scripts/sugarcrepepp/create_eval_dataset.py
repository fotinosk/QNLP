from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_dataset
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

ATLAS_NAME = "sugarcrepepp"
ATLAS_DERIVED_DIR = constants.atlases_path / ATLAS_NAME / "derived_v1"


def run() -> None:
    create_dataset(
        derived_dirs=[ATLAS_DERIVED_DIR],
        strategy=ContrastivePairStrategy(),
        output_name="sugarcrepepp_eval",
        compute_contraction_paths=True,
    )


if __name__ == "__main__":
    run()
