"""Create ARO contrastive train/val/test datasets from the tree-no-type atlas.

Set PARSER_VERSION=tree_no_type before importing constants so the derived dir
and dataset output names are automatically versioned. Honors ARO's pre-defined
splits (the `split` column) rather than re-splitting randomly.

Always computed with contraction paths (non-linear); the path column is ignored
in linear training, so a single non-linear build serves both modes.

Output: aro_{train,val,test}_tree_no_type.parquet
    (selected at train time via ML_DATASET_SUFFIX=_tree_no_type)
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import enrich_atoms
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="aro_create_dataset_tree")

ARO_DERIVED_DIR = constants.atlases_path / "aro" / constants.derived_name
SPLITS = ["train", "val", "test"]


def run() -> None:
    atoms = enrich_atoms(
        [ARO_DERIVED_DIR],
        compute_contraction_paths=True,
    )
    logger.info(f"Enriched {len(atoms)} atoms across splits: {atoms['split'].value_counts().to_dict()}")

    strategy = ContrastivePairStrategy()

    for split in SPLITS:
        split_atoms = atoms.filter(pl.col("split") == split)
        composed = strategy.compose(split_atoms)

        out_path = constants.datasets_path / f"aro_{split}{constants.artifact_suffix}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.write_parquet(out_path)
        logger.info(f"{split}: {len(composed)} contrastive pairs -> {out_path}")


if __name__ == "__main__":
    run()
