"""COCO preprocessing pipeline using lambeq's TreeReader (CCG tree shape,
words as vectors, rule names as boxes) + CustomMPSAnsatz.

Writes to a separate LMDB and derived-dir from the default bobcat pipeline
so the two can coexist on disk.

Run with:
    python -m qnlp.preprocessing_pipelines.coco.pipeline_tree
"""
import os
from pathlib import Path

from qnlp.constants import constants
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.preprocessing_pipelines.coco.steps import COCOFlattenStep, RemoveTrailingDotsStep, SchemaMappingStep

# Scratch-aware paths: on hosts with tight NFS quotas (e.g. vanilla, 20 GB
# home), put LMDB + derived chunks on local scratch instead of $HOME.
# Override via PIPELINE_TREE_SCRATCH=/var/tmp/...
_SCRATCH = os.environ.get("PIPELINE_TREE_SCRATCH")
if _SCRATCH:
    _SCRATCH_ROOT = Path(_SCRATCH)
    LMDB_PATH = _SCRATCH_ROOT / "sentence_mapping_tree"
    ATLAS_DIR = _SCRATCH_ROOT / "atlases" / "coco"  # must contain data_manifest.parquet
else:
    LMDB_PATH = constants.lmdb_path.parent / "sentence_mapping_tree"
    ATLAS_DIR = constants.atlases_path / "coco"
DERIVED_NAME = "derived_tree"
DEVICE = os.environ.get("PIPELINE_TREE_DEVICE", "cpu")

flatten_step = COCOFlattenStep("sentences_raw")
schema_step = SchemaMappingStep(column_mapping={"sentences_raw": "processed_text"})
remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
lemma_step = LemmatizeStep(text_column="processed_text")
ccg_parsing_step = CCGCompilerStep(
    lmdb_path=LMDB_PATH,
    bond_dim=constants.bond_dim,
    embedding_dim=constants.embedding_dim,
    device=DEVICE,
    parser_batch_size=64,
    max_workers=8,
    reader="tree",
    tree_reader_mode="RULE_TYPE",
)
unification_step = UnifyEinsumRankStep()

coco_pipeline_tree = Pipeline(
    atlas_dir=ATLAS_DIR,
    lmdb_path=LMDB_PATH,
    steps=[flatten_step, schema_step, remove_dots_step, lemma_step, ccg_parsing_step, unification_step],
    derived_name=DERIVED_NAME,
)


if __name__ == "__main__":
    coco_pipeline_tree.run(chunk_size=100)
