from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_train_val_test_datasets
from qnlp.core.data_engine.dataset_creator.strategies.single_caption import SingleCaptionStrategy

COCO_DERIVED_DIR = constants.atlases_path / "coco" / "derived_test"

if __name__ == "__main__":
    create_train_val_test_datasets(
        derived_dirs=[COCO_DERIVED_DIR],
        strategy=SingleCaptionStrategy(),
        output_name="coco_single_caption",
    )
