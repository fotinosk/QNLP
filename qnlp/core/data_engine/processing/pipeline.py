from pathlib import Path
from typing import Protocol

import polars as pl

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="processing_pipeline")


class PipelineStep(Protocol):
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process a dataframe chunk and return the modified dataframe."""
        ...


class Pipeline:
    def __init__(
        self,
        atlas_dir: Path | str,
        steps: list[PipelineStep],
        lmdb_path: Path | str | None = None,
        derived_name: str = "derived_v1",
        lmdb_map_size: int = 10 * 1024 * 1024 * 1024,
    ):
        self.atlas_dir = Path(atlas_dir)
        self.data_manifest_path = self.atlas_dir / "data_manifest.parquet"
        self.derived_dir = self.atlas_dir / derived_name
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        self.lmdb_map_size = lmdb_map_size
        self.steps = steps

    def _get_delta_sample_ids(self) -> list[str]:
        """Identify sample_ids that haven't been processed yet."""
        if not self.data_manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.data_manifest_path}")

        raw_lf = pl.scan_parquet(self.data_manifest_path)
        derived_files = list(self.derived_dir.glob("*.parquet")) if self.derived_dir.exists() else []

        if derived_files:
            derived_lf = pl.scan_parquet(derived_files)
            delta_lf = raw_lf.join(derived_lf.select("sample_id"), on="sample_id", how="anti")
        else:
            delta_lf = raw_lf

        delta_ids_df = delta_lf.select("sample_id").collect()
        return delta_ids_df.get_column("sample_id").to_list()

    def _process_chunk(self, chunk_df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, bytes]]:
        """Apply pipeline steps, validate, resolve paths, and extract LMDB payloads."""
        logger.info(f"Processing chunk with {len(self.steps)} step(s)...")
        for i, step in enumerate(self.steps):
            logger.debug(f"Running step {i + 1}/{len(self.steps)}: {step.__class__.__name__}")
            chunk_df = step.process(chunk_df)

        if "sample_id" not in chunk_df.columns:
            raise ValueError("Pipeline step dropped required column 'sample_id'")

        if "local_image_path" in chunk_df.columns:
            chunk_df = chunk_df.with_columns(
                pl.col("local_image_path")
                .map_elements(lambda p: str(Path(p).resolve()), return_dtype=pl.String, skip_nulls=True)
                .alias("local_image_path")
            )

        lmdb_entries: dict[str, bytes] = {}
        if {"compiled_bytes", "text_hash"}.issubset(chunk_df.columns):
            valid = chunk_df.filter(pl.col("compiled_bytes").is_not_null())
            for row in valid.iter_rows(named=True):
                h = row["text_hash"]
                if h is not None:
                    lmdb_entries[str(h)] = row["compiled_bytes"]
            chunk_df = chunk_df.drop("compiled_bytes")

        required_cols = ["sample_id", "local_image_path", "processed_text", "text_hash"]
        available_cols = [c for c in required_cols if c in chunk_df.columns]
        chunk_df = chunk_df.select(available_cols)

        return chunk_df, lmdb_entries

    def _write_lmdb_chunk(self, env, entries: dict[str, bytes], chunk_idx: int) -> None:
        """Atomically write compiled bytes to LMDB using batched cursor."""
        if not entries or not env:
            return

        with env.begin(write=True) as txn:
            kv_pairs = ((k.encode("utf-8"), v) for k, v in entries.items())
            curs = txn.cursor()
            written = curs.putmulti(kv_pairs, overwrite=True)

            if written != len(entries):
                logger.warning(
                    "Chunk %d: LMDB putmulti wrote %d/%d entries. Duplicates or errors may have occurred.",
                    chunk_idx,
                    written,
                    len(entries),
                )

    def _write_parquet_chunk(self, df: pl.DataFrame, chunk_offset: int) -> None:
        """Write processed metadata to Parquet."""
        chunk_path = self.derived_dir / f"chunk_{chunk_offset:06d}.parquet"
        df.write_parquet(chunk_path)

    def run(self, chunk_size: int = 1000, dry_run: bool = False) -> None:
        """Execute the pipeline across all unprocessed samples."""
        logger.info("Starting pipeline execution")
        delta_sample_ids = self._get_delta_sample_ids()
        if not delta_sample_ids:
            logger.info("No new data to process.")
            return

        logger.info(f"Detected {len(delta_sample_ids)} new row(s) to process.")

        total_chunks = (len(delta_sample_ids) + chunk_size - 1) // chunk_size

        if not dry_run:
            self.derived_dir.mkdir(parents=True, exist_ok=True)
            if self.lmdb_path:
                self.lmdb_path.parent.mkdir(parents=True, exist_ok=True)

        lmdb_env = None
        if not dry_run and self.lmdb_path:
            import lmdb

            lmdb_env = lmdb.open(
                str(self.lmdb_path),
                max_readers=128,
                map_size=self.lmdb_map_size,
                create=True,
                writemap=True,
            )

        try:
            raw_lf = pl.scan_parquet(self.data_manifest_path)
            for i in range(0, len(delta_sample_ids), chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_ids = delta_sample_ids[i : i + chunk_size]
                chunk_df = raw_lf.filter(pl.col("sample_id").is_in(chunk_ids)).collect()

                processed_df, lmdb_entries = self._process_chunk(chunk_df)

                if dry_run:
                    logger.info(
                        "[DRY RUN] Chunk %d/%d | Schema: %s | Shape: %s",
                        chunk_idx,
                        total_chunks,
                        processed_df.columns,
                        processed_df.shape,
                    )
                    if i == 0:
                        logger.info("Sample output:\n%s", processed_df.head(3))
                else:
                    self._write_lmdb_chunk(lmdb_env, lmdb_entries, chunk_idx)
                    self._write_parquet_chunk(processed_df, i)

        finally:
            if lmdb_env:
                lmdb_env.close()

        mode_str = "[DRY RUN] " if dry_run else ""
        logger.info("%sProcessed and committed %d records in %d chunks.", mode_str, len(delta_sample_ids), total_chunks)
