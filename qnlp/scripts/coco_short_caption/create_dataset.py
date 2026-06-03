"""
Create train/val/test datasets restricted to short captions (< MAX_WORDS words).

Uses the same derived atlas and pipeline as coco_single_caption but applies a
word-count filter before the train/val/test split, so all three splits are drawn
exclusively from short captions and the split boundaries are determined by the
filtered pool (no leakage from longer captions influencing which sample_ids land
in which split).

Output:
    data/datasets/coco_short_caption_train.parquet
    data/datasets/coco_short_caption_val.parquet
    data/datasets/coco_short_caption_test.parquet
"""

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import create_train_val_test_datasets
from qnlp.core.data_engine.dataset_creator.strategies.single_caption import SingleCaptionStrategy

COCO_DERIVED_DIR = constants.atlases_path / "coco" / "derived_test"

MAX_WORDS = 10


def _filter_short(atoms: pl.DataFrame) -> pl.DataFrame:
    word_counts = atoms["processed_text"].map_elements(lambda t: len(t.split()), return_dtype=pl.Int32)
    filtered = atoms.filter(word_counts < MAX_WORDS)
    original = len(atoms)
    kept = len(filtered)
    print(f"Length filter (<{MAX_WORDS} words): kept {kept} / {original} atoms ({100 * kept / original:.1f}%)")
    return filtered


if __name__ == "__main__":
    create_train_val_test_datasets(
        derived_dirs=[COCO_DERIVED_DIR],
        strategy=SingleCaptionStrategy(),
        output_name="coco_short_caption",
        pre_split_hook=_filter_short,
    )
