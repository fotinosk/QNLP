"""Winoground CCG processing pipeline using TreeReader in NO_TYPE mode.

Set PARSER_VERSION=tree_no_type before importing constants so all paths
(LMDB, derived dir, bobcat diskcache) are automatically versioned.
Bobcat CCG trees are reused from the existing diskcache; only the diagram
conversion differs (TreeReader instead of grammatical cups).
"""

import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.preprocessing_pipelines.winoground.pipeline import WinogroundPipeline
from qnlp.preprocessing_pipelines.winoground.steps import WinogroundFlattenStep


def build_pipeline() -> WinogroundPipeline:
    flatten_step = WinogroundFlattenStep()
    remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
    lemma_step = LemmatizeStep(text_column="processed_text")
    ccg_step = CCGCompilerStep(
        lmdb_path=constants.lmdb_path,
        bond_dim=constants.bond_dim,
        embedding_dim=constants.embedding_dim,
        cache_path=str(constants.bobcat_cache_path),
        max_workers=4,
        worker_batch_size=250,
        tree_no_type=True,
    )
    unify_step = UnifyEinsumRankStep()

    return WinogroundPipeline(
        atlas_dir=constants.atlases_path / "winoground",
        lmdb_path=constants.lmdb_path,
        steps=[flatten_step, remove_dots_step, lemma_step, ccg_step, unify_step],
        derived_name=constants.derived_name,
    )


if __name__ == "__main__":
    build_pipeline().run(chunk_size=100)
