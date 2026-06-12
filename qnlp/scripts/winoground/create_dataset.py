import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_train_val_test_datasets
from qnlp.core.data_engine.dataset_creator.strategies.winoground_pair import WinogroundPairStrategy

WINO_DERIVED_DIR = constants.atlases_path / "winoground" / "derived_v1"


if __name__ == "__main__":
    create_train_val_test_datasets(
        derived_dirs=[WINO_DERIVED_DIR],
        strategy=WinogroundPairStrategy(),
        output_name="winoground",
        pre_split_hook=lambda atoms: atoms.with_columns(
            pl.col("sample_id").str.split("__").list.first().alias("pair_id")
        ),
        group_column="pair_id",
        compute_contraction_paths=True,
    )
