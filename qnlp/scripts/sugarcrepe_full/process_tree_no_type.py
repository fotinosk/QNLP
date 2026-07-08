"""Compile SugarCrepe-full atlas into CCG tensor diagrams using tree-reader (NO_TYPE mode).

Set PARSER_VERSION=tree_no_type before importing constants so all paths are
automatically versioned. Bobcat CCG trees are reused from the existing diskcache.
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import FlattenContrastivePairStep, RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="sugarcrepe_full_process_tree")

ATLAS_NAME = "sugarcrepe_full"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME
MAX_WORKERS = 4
CHUNK_SIZE = 2000


def run() -> None:
    compiler = CCGCompilerStep(
        lmdb_path=constants.lmdb_path,
        text_column="processed_text",
        embedding_dim=constants.embedding_dim,
        bond_dim=constants.bond_dim,
        max_workers=MAX_WORKERS,
        cache_path=str(constants.bobcat_cache_path),
        tree_no_type=True,
    )
    steps = [FlattenContrastivePairStep(), RemoveTrailingDotsStep(), compiler, UnifyEinsumRankStep()]
    pipeline = Pipeline(
        atlas_dir=ATLAS_DIR,
        steps=steps,
        lmdb_path=constants.lmdb_path,
        derived_name=constants.derived_name,
        keep_columns=["label"],
    )
    try:
        pipeline.run(chunk_size=CHUNK_SIZE)
    finally:
        compiler.teardown()

    logger.info("SugarCrepe-full tree-no-type compilation complete.")


if __name__ == "__main__":
    run()
