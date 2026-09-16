from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep, SchemaMappingStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.pipeline import Pipeline

# SVO has one sentence per row — no flatten step needed.
# `corrected_sentence` (Llama-3.2-3B grammar/spelling fix) is the positive caption;
# negatives are image-level, not text-level.
schema_step = SchemaMappingStep(column_mapping={"corrected_sentence": "processed_text"})
remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
lemma_step = LemmatizeStep(text_column="processed_text")
# max_workers matches submit_svo_pipeline.sh's `#$ -pe smp 2`. Kept deliberately
# small: scripts/submit_svo_compile_array.sh should run FIRST, sharding the
# unique captions across an SGE job array (many small nodes in parallel, which
# schedules far faster than one big multi-core request) and pre-populating the
# LMDB cache — at which point this step is just a cache-hit pass over rows that
# are already compiled, so it doesn't need many workers of its own. A bigger
# `-pe smp` request here mainly means a longer wait in the scheduler queue for
# no benefit once the cache is warm.
ccg_parsing_step = CCGCompilerStep(
    lmdb_path=constants.lmdb_path,
    bond_dim=constants.bond_dim,
    embedding_dim=constants.embedding_dim,
    max_workers=2,
    worker_batch_size=200,
)
unification_step = UnifyEinsumRankStep()

svo_atlas = constants.atlases_path / "svo"

# SVO's positive/negative pairing is image-level (one caption, two candidate
# images), not text-level like ARO/Winoground — so there's no "local_image_path"
# column to standardise on. Both image paths plus the subj/verb/obj metadata
# needed to build the SVO-Probes eval split and the SVO-Swap set are carried
# through via keep_columns instead.
svo_pipeline = Pipeline(
    atlas_dir=svo_atlas,
    lmdb_path=constants.lmdb_path,
    steps=[schema_step, remove_dots_step, lemma_step, ccg_parsing_step, unification_step],
    derived_name="derived_v1",
    keep_columns=[
        "pos_local_image_path",
        "neg_local_image_path",
        "pos_image_id",
        "neg_image_id",
        "subj",
        "verb",
        "obj",
        "subj_neg",
        "verb_neg",
        "obj_neg",
    ],
)


if __name__ == "__main__":
    # Larger chunk_size than the COCO default (100) — fewer outer-loop/LMDB
    # checkpoint round trips for SVO's much smaller (~26k row) manifest.
    svo_pipeline.run(chunk_size=2000)
