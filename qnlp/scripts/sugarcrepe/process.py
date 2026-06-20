"""Compile a SugarCrepe atlas into CCG tensor diagrams.

Flattens each (true_caption, false_caption) record into two atoms sharing a
sample_id, compiles every caption into an einsum diagram via Bobcat CCG, and
writes diagrams/symbols to the LMDB cache. Eval-only, so there is no split column.

Run after `load_sugarcrepe_to_atlas`.
"""

from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import FlattenContrastivePairStep, RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="sugarcrepe_process")

SUBSET = "swap_obj"
ATLAS_DIR = constants.atlases_path / f"sugarcrepe_{SUBSET}"

DEVICE = "cpu"  # bobcat parser device; "mps" is faster but riskier in subprocesses
MAX_WORKERS = 4
CHUNK_SIZE = 2000


def run() -> None:
    compiler = CCGCompilerStep(
        lmdb_path=constants.lmdb_path,
        text_column="processed_text",
        embedding_dim=constants.embedding_dim,
        bond_dim=constants.bond_dim,
        device=DEVICE,
        max_workers=MAX_WORKERS,
    )

    steps = [
        FlattenContrastivePairStep(),  # (true_caption, false_caption) -> two labelled atoms
        RemoveTrailingDotsStep(),
        compiler,  # processed_text -> diagram/symbols in LMDB
        UnifyEinsumRankStep(),  # force rank-1 (vector) outputs
    ]

    pipeline = Pipeline(
        atlas_dir=ATLAS_DIR,
        steps=steps,
        lmdb_path=constants.lmdb_path,
        keep_columns=["label"],  # carried into the derived parquets for the strategy
    )

    try:
        pipeline.run(chunk_size=CHUNK_SIZE)
    finally:
        compiler.teardown()

    logger.info(f"SugarCrepe '{SUBSET}' compilation complete.")


if __name__ == "__main__":
    run()
