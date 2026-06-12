"""Create ARO contrastive train/val/test datasets from the compiled atlas.

Honors ARO's pre-defined splits (the `split` column carried through the pipeline)
rather than re-splitting randomly. Contraction paths are pre-computed once over
all atoms (deduped by topology) and atoms with no feasible path are excluded.
"""

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import enrich_atoms
from qnlp.core.data_engine.dataset_creator.strategies.contrastive_pair import ContrastivePairStrategy
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="aro_create_dataset")

ARO_DERIVED_DIR = constants.atlases_path / "aro" / "derived_v1"
SPLITS = ["train", "val", "test"]


def run() -> None:
    # Enrich once: join diagram/symbols from LMDB, compute contraction paths,
    # and exclude atoms whose path is infeasible (timeout / too-large intermediate).
    atoms = enrich_atoms(
        [ARO_DERIVED_DIR],
        compute_contraction_paths=True,
    )
    logger.info(f"Enriched {len(atoms)} atoms across splits: {atoms['split'].value_counts().to_dict()}")

    strategy = ContrastivePairStrategy()

    for split in SPLITS:
        split_atoms = atoms.filter(pl.col("split") == split)
        composed = strategy.compose(split_atoms)

        out_path = constants.datasets_path / f"aro_{split}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.write_parquet(out_path)
        logger.info(f"{split}: {len(composed)} contrastive pairs -> {out_path}")


if __name__ == "__main__":
    run()
