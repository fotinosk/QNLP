from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep, SchemaMappingStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.pipeline import Pipeline

# SVO has one sentence per row — no flatten step needed.
# `sentence` is the positive caption; negatives are image-level, not text-level.
schema_step = SchemaMappingStep(column_mapping={"sentence": "processed_text"})
remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
lemma_step = LemmatizeStep(text_column="processed_text")
ccg_parsing_step = CCGCompilerStep(
    lmdb_path=constants.lmdb_path,
    bond_dim=constants.bond_dim,
    embedding_dim=constants.embedding_dim,
)
unification_step = UnifyEinsumRankStep()

svo_atlas = constants.atlases_path / "svo"

svo_pipeline = Pipeline(
    atlas_dir=svo_atlas,
    lmdb_path=constants.lmdb_path,
    steps=[schema_step, remove_dots_step, lemma_step, ccg_parsing_step, unification_step],
    derived_name="derived_v1",
)


if __name__ == "__main__":
    svo_pipeline.run(chunk_size=100)
