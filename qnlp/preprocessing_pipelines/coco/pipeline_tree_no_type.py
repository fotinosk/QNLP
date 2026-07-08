"""COCO CCG processing pipeline using TreeReader in NO_TYPE mode.

Set PARSER_VERSION=tree_no_type before importing constants so all paths
(LMDB, derived dir, bobcat diskcache) are automatically versioned.
Bobcat CCG trees are reused from the existing diskcache; only the diagram
conversion differs (TreeReader instead of grammatical cups).
"""

import argparse
import os

os.environ["PARSER_VERSION"] = "tree_no_type"

from qnlp.constants import constants
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.preprocessing_pipelines.coco.steps import COCOFlattenStep, RemoveTrailingDotsStep, SchemaMappingStep


def build_pipeline(max_workers: int, worker_batch_size: int) -> Pipeline:
    flatten_step = COCOFlattenStep("sentences_raw")
    schema_step = SchemaMappingStep(column_mapping={"sentences_raw": "processed_text"})
    remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
    lemma_step = LemmatizeStep(text_column="processed_text")
    ccg_parsing_step = CCGCompilerStep(
        lmdb_path=constants.lmdb_path,
        bond_dim=constants.bond_dim,
        embedding_dim=constants.embedding_dim,
        cache_path=str(constants.bobcat_cache_path),
        max_workers=max_workers,
        worker_batch_size=worker_batch_size,
        tree_no_type=True,
    )
    unification_step = UnifyEinsumRankStep()

    return Pipeline(
        atlas_dir=constants.atlases_path / "coco",
        lmdb_path=constants.lmdb_path,
        steps=[flatten_step, schema_step, remove_dots_step, lemma_step, ccg_parsing_step, unification_step],
        derived_name=constants.derived_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the COCO tree-no-type preprocessing pipeline.")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--worker-batch-size", type=int, default=1000)
    args = parser.parse_args()

    pipeline = build_pipeline(max_workers=args.max_workers, worker_batch_size=args.worker_batch_size)
    pipeline.run(chunk_size=args.chunk_size)
