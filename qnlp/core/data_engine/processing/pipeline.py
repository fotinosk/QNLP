from pathlib import Path
from typing import List

import polars as pl

from qnlp.core.data_engine.processing.steps import PipelineStep


class Pipeline:
    def __init__(self, atlas_dir: Path | str, derived_name: str = "derived_v1.parquet"):
        self.atlas_dir = Path(atlas_dir)
        self.data_manifest_path = self.atlas_dir / "data_manifest.parquet"
        self.derived_path = self.atlas_dir / derived_name
        self.steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep):
        self.steps.append(step)

    def _append_to_parquet(self, chunk_df: pl.DataFrame):
        """Append a chunk to the derived parquet file while keeping memory usage bounded."""
        if not self.derived_path.exists():
            chunk_df.write_parquet(self.derived_path)
            return

        temp_chunk_path = self.derived_path.with_suffix(".tmp_chunk.parquet")
        chunk_df.write_parquet(temp_chunk_path)

        new_derived_path = self.derived_path.with_suffix(".new.parquet")

        lazy_combined = pl.concat([pl.scan_parquet(self.derived_path), pl.scan_parquet(temp_chunk_path)])
        lazy_combined.sink_parquet(new_derived_path)

        new_derived_path.replace(self.derived_path)
        temp_chunk_path.unlink()

    def run(self, chunk_size: int = 1000):
        if not self.data_manifest_path.exists():
            print("No data manifest found.")
            return

        raw_lf = pl.scan_parquet(self.data_manifest_path)

        if self.derived_path.exists():
            derived_lf = pl.scan_parquet(self.derived_path)
            delta_lf = raw_lf.join(derived_lf.select("sample_id"), on="sample_id", how="anti")
        else:
            delta_lf = raw_lf

        delta_sample_ids_df = delta_lf.select("sample_id").collect()
        if delta_sample_ids_df.is_empty():
            print("No new data to process.")
            return

        delta_sample_ids = delta_sample_ids_df.get_column("sample_id").to_list()

        for i in range(0, len(delta_sample_ids), chunk_size):
            chunk_ids = delta_sample_ids[i : i + chunk_size]

            chunk_df = raw_lf.filter(pl.col("sample_id").is_in(chunk_ids)).collect()

            for step in self.steps:
                chunk_df = step.process(chunk_df)

            # Filter to strict contract
            if "local_image_path" in chunk_df.columns:

                def resolve_path(p: str) -> str:
                    return str(Path(p).resolve())

                chunk_df = chunk_df.with_columns(
                    pl.col("local_image_path").map_elements(resolve_path, return_dtype=pl.String)
                )

            required_cols = ["sample_id", "local_image_path", "processed_text", "text_hash"]
            available_cols = [c for c in required_cols if c in chunk_df.columns]
            chunk_df = chunk_df.select(available_cols)

            self._append_to_parquet(chunk_df)

        print(
            f"Processed {len(delta_sample_ids)} records in "
            "{(len(delta_sample_ids) + chunk_size - 1) // chunk_size} chunks."
        )
