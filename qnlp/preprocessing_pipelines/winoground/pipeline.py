import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.pipeline import Pipeline
from qnlp.preprocessing_pipelines.winoground.steps import WinogroundFlattenStep


class WinogroundPipeline(Pipeline):
    """
    Extends Pipeline for Winoground's paired structure.

    Each manifest row (one Winoground item) produces two derived rows after
    flattening (one per caption/image pair). Delta detection is overridden to
    strip the __0/__1 suffix from derived sample_ids before joining against
    the manifest, so already-processed pairs are correctly skipped.
    """

    def _get_delta_sample_ids(self) -> list[str]:
        if not self.data_manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.data_manifest_path}")

        raw_lf = pl.scan_parquet(self.data_manifest_path)
        derived_files = list(self.derived_dir.glob("*.parquet")) if self.derived_dir.exists() else []

        if derived_files:
            derived_lf = pl.scan_parquet(derived_files)
            # Strip the __0 / __1 suffix to recover the original manifest sample_id
            processed_pair_ids = derived_lf.select(
                pl.col("sample_id").str.split("__").list.first().alias("pair_id")
            ).unique()
            delta_lf = raw_lf.join(processed_pair_ids, left_on="sample_id", right_on="pair_id", how="anti")
        else:
            delta_lf = raw_lf

        return delta_lf.select("sample_id").collect().get_column("sample_id").to_list()


flatten_step = WinogroundFlattenStep()
remove_dots_step = RemoveTrailingDotsStep(text_column="processed_text")
lemma_step = LemmatizeStep(text_column="processed_text")
ccg_step = CCGCompilerStep(
    lmdb_path=constants.lmdb_path,
    bond_dim=constants.bond_dim,
    embedding_dim=constants.embedding_dim,
)
unify_step = UnifyEinsumRankStep()

winoground_pipeline = WinogroundPipeline(
    atlas_dir=constants.atlases_path / "winoground",
    lmdb_path=constants.lmdb_path,
    steps=[flatten_step, remove_dots_step, lemma_step, ccg_step, unify_step],
    derived_name="derived_v1",
)


if __name__ == "__main__":
    winoground_pipeline.run(chunk_size=100)
