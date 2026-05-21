"""COCO preprocessing pipeline using lambeq's cups_reader (linear, all-rank-2
word boxes, cups contracting adjacent wires) + CustomMPSAnsatz.

The user explicitly wanted to check whether MPS does anything here — every
word box is at most rank-2, so it's worth confirming via the
[ansatz-instrumentation] logs whether any decomposition fires.

Writes to a separate LMDB and derived-dir from the default bobcat pipeline.

Run with:
    python -m qnlp.preprocessing_pipelines.coco.pipeline_cups
"""
from qnlp.constants import constants
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.preprocessing_pipelines.coco.steps import COCOFlattenStep, RemoveTrailingDotsStep, SchemaMappingStep

LMDB_PATH = constants.lmdb_path.parent / "sentence_mapping_cups"
DERIVED_NAME = "derived_cups"

flatten_step = COCOFlattenStep("sentences_raw")
schema_step = SchemaMappingStep(column_mapping={"sentences_raw": "processed_text"})
remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
lemma_step = LemmatizeStep(text_column="processed_text")
ccg_parsing_step = CCGCompilerStep(
    lmdb_path=LMDB_PATH,
    bond_dim=constants.bond_dim,
    embedding_dim=constants.embedding_dim,
    # cups_reader needs no parser, so device is moot. Keep workers high
    # because parsing is now CPU-only and very cheap per sentence.
    device="cpu",
    parser_batch_size=256,
    max_workers=8,
    reader="cups",
)
unification_step = UnifyEinsumRankStep()

coco_atlas = constants.atlases_path / "coco"

coco_pipeline_cups = Pipeline(
    atlas_dir=coco_atlas,
    lmdb_path=LMDB_PATH,
    steps=[flatten_step, schema_step, remove_dots_step, lemma_step, ccg_parsing_step, unification_step],
    derived_name=DERIVED_NAME,
)


if __name__ == "__main__":
    coco_pipeline_cups.run(chunk_size=100)
