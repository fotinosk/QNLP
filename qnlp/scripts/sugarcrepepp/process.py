from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import FlattenContrastivePairStep, RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="sugarcrepepp_process")

ATLAS_NAME = "sugarcrepepp"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME
DEVICE = "cpu"
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
        FlattenContrastivePairStep(true_column="pos", false_column="neg"),
        RemoveTrailingDotsStep(),
        compiler,
        UnifyEinsumRankStep(),
    ]
    pipeline = Pipeline(
        atlas_dir=ATLAS_DIR,
        steps=steps,
        lmdb_path=constants.lmdb_path,
        keep_columns=["label"],
    )
    try:
        pipeline.run(chunk_size=CHUNK_SIZE)
    finally:
        compiler.teardown()


if __name__ == "__main__":
    run()
