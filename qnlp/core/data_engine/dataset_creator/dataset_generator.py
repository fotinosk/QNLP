from pathlib import Path
from typing import Optional

import lmdb
import numpy as np
import orjson
import polars as pl

from qnlp.constants import constants
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="dataset_generator")


def _fetch_lmdb_fields(dataset_df: pl.DataFrame, lmdb_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetches 'diagram' and 'symbols' from LMDB using deduplicated hashes."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    hash_to_diagram, hash_to_symbols = {}, {}

    try:
        unique_hashes = dataset_df["text_hash"].drop_nulls().unique().to_list()
        with env.begin() as txn:
            for h in unique_hashes:
                val = txn.get(h.encode("utf-8"))
                if val:
                    data = orjson.loads(val)
                    hash_to_diagram[h] = data.get("diagram")
                    hash_to_symbols[h] = data.get("symbols")
    finally:
        env.close()

    diag_df = pl.DataFrame(
        {"text_hash": list(hash_to_diagram.keys()), "diagram": list(hash_to_diagram.values())},
        schema={"text_hash": pl.String, "diagram": pl.String},
    )
    sym_df = pl.DataFrame(
        {"text_hash": list(hash_to_symbols.keys()), "symbols": list(hash_to_symbols.values())},
        schema={"text_hash": pl.String, "symbols": pl.Object},
    )
    return diag_df, sym_df


def _enforce_schema(df: pl.DataFrame, output_schema: pl.Schema) -> pl.DataFrame:
    """Ensures DataFrame matches the expected output schema exactly."""
    exprs = [
        pl.col(c)
        if dtype == pl.Object
        else pl.col(c).cast(dtype, strict=False)
        if c in df.columns
        else pl.lit(None, dtype=dtype).alias(c)
        for c, dtype in output_schema.items()
    ]
    return df.select(exprs)


def _enrich_and_format_dataset(df: pl.DataFrame, output_schema: pl.Schema, lmdb_path: str) -> pl.DataFrame:
    """Reusable post-collection pipeline: LMDB enrichment + schema enforcement."""
    diag_df, sym_df = _fetch_lmdb_fields(df, lmdb_path)
    df = df.join(diag_df, on="text_hash", how="left").join(sym_df, on="text_hash", how="left")

    if "compiled_data" in df.columns and "compiled_data" not in output_schema:
        df = df.drop("compiled_data")

    if "symbols" in df.columns:
        df = df.with_columns(
            pl.col("symbols").map_elements(
                lambda x: orjson.dumps(x).decode() if x is not None else None,
                return_dtype=pl.String,
            )
        )

    return _enforce_schema(df, output_schema)


def _sample_manifest_lazy(
    manifest_path: Path, ratio: float, num_rows_total: int, exclude_sample_ids: Optional[list[str]] = None
) -> pl.LazyFrame:
    """Lazily samples rows from a parquet manifest using index-based selection."""
    df = pl.scan_parquet(manifest_path, row_index_name="index")
    if exclude_sample_ids:
        df = df.filter(~pl.col("sample_id").is_in(exclude_sample_ids))

    num_rows_from_df = int(ratio * num_rows_total)
    max_idx = df.select(pl.col("index")).max().collect()[0, "index"]

    subsample_indices = np.random.choice(max_idx + 1, num_rows_from_df, replace=False)
    return df.filter(pl.col("index").is_in(subsample_indices))


def _get_unique_sample_ids(manifests: list[Path]) -> list[str]:
    """Lazily extracts deduplicated sample_ids across all manifests."""
    lazy_ids = [pl.scan_parquet(m).select("sample_id") for m in manifests]
    return pl.concat(lazy_ids).unique().collect()["sample_id"].to_list()


def _split_ids(ids: list[str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Shuffles and splits IDs deterministically."""
    assert abs(sum(ratios.values()) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    shuffled = np.array(ids)
    rng.shuffle(shuffled)

    splits, cumsum = {}, 0
    for name, ratio in ratios.items():
        end = int(cumsum + ratio * len(shuffled))
        splits[name] = shuffled[cumsum:end].tolist()
        cumsum = end
    return splits


def _build_split_dataset(
    manifests: list[Path], include_ids: list[str], output_schema: pl.Schema, lmdb_path: str
) -> pl.DataFrame:
    """Filters manifests by allowed IDs, collects, enriches, and formats."""
    id_series = pl.Series("sample_id", include_ids)
    lazy_frames = [pl.scan_parquet(m).filter(pl.col("sample_id").is_in(id_series)) for m in manifests]
    combined = pl.concat(lazy_frames, how="vertical_relaxed").collect()
    return _enrich_and_format_dataset(combined, output_schema, lmdb_path)


def create_dataset(
    dataset_name: str,
    manifests: list[Path],
    num_rows: int,
    output_schema: pl.Schema,
    dataset_composition: list[float] | None = None,
    exclude_sample_ids: list[str] | None = None,
) -> None:
    """Creates a single dataset, enriches it, enforces schema, and writes to Parquet."""
    if dataset_composition:
        assert len(manifests) == len(dataset_composition)
        assert abs(sum(dataset_composition) - 1.0) < 1e-9
        composition = dataset_composition
    else:
        composition = [1 / len(manifests)] * len(manifests)

    lazy_datasets = [_sample_manifest_lazy(m, r, num_rows, exclude_sample_ids) for m, r in zip(manifests, composition)]

    dataset_df = pl.concat(lazy_datasets, how="vertical_relaxed").collect()
    final_df = _enrich_and_format_dataset(dataset_df, output_schema, str(constants.lmdb_path))

    out_path = Path(constants.datasets_path) / f"{dataset_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(out_path)
    logger.info(f"Dataset written to {out_path}")


def create_train_test_val_datasets(
    dataset_name: str,
    manifests: list[Path],
    output_schema: pl.Schema,
    split_ratios: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    seed: int = 42,
) -> None:
    """
    Creates mutually exclusive train/val/test datasets.
    Group-aware: all rows sharing a sample_id stay in the same split.
    Writes each split to Parquet with a suffix.
    """
    unique_ids = _get_unique_sample_ids(manifests)
    split_ids = _split_ids(unique_ids, split_ratios, seed)

    for split, ids in split_ids.items():
        df = _build_split_dataset(manifests, ids, output_schema, str(constants.lmdb_path))
        out_path = Path(constants.datasets_path) / f"{dataset_name}_{split}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path)
        logger.info(f"{split} split written to {out_path}")
