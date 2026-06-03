"""
Create an ARO-style test set from the existing coco_short_caption test split.

Applies ContrastivePairStrategy to the held-out test parquet to synthesise
one negative caption per sample via random derangement. Using the test split
as the source (rather than coco_contrastive_test.parquet) avoids data leakage:
the short-caption model was trained on coco_short_caption_train.parquet, whose
sample_ids are disjoint from the test split by construction.

Output:
    data/datasets/coco_short_caption_aro_test.parquet
"""

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy

SRC = constants.datasets_path / "coco_short_caption_test.parquet"
OUT = constants.datasets_path / "coco_short_caption_aro_test.parquet"


if __name__ == "__main__":
    atoms = pl.read_parquet(SRC)
    composed = ContrastivePairStrategy().compose(atoms)
    composed.write_parquet(OUT)
    print(f"Written {len(composed)} rows to {OUT}")
