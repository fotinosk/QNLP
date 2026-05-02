import re
from pathlib import Path

import lmdb
import numpy as np
import orjson
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.composition_strategy import CompositionStrategy
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="dataset_generator")


def _is_1d_diagram(diagram: str) -> bool:
    if "->" not in diagram:
        return True
    output_part = diagram.split("->")[1].strip()
    return len(re.findall(r"[a-zA-Z]", output_part)) == 1


def _fetch_lmdb_fields(atoms: pl.DataFrame, lmdb_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch diagram and symbols from LMDB for all unique text_hashes in atoms."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    hash_to_diagram: dict[str, str] = {}
    hash_to_symbols: dict[str, str] = {}

    try:
        unique_hashes = atoms["text_hash"].drop_nulls().unique().to_list()
        with env.begin() as txn:
            for h in unique_hashes:
                val = txn.get(h.encode("utf-8"))
                if val:
                    data = orjson.loads(val)
                    hash_to_diagram[h] = data.get("diagram")
                    # Serialize symbols to JSON string for parquet storage
                    raw_symbols = data.get("symbols")
                    hash_to_symbols[h] = orjson.dumps(raw_symbols).decode() if raw_symbols is not None else None
    finally:
        env.close()

    diag_df = pl.DataFrame(
        {"text_hash": list(hash_to_diagram.keys()), "diagram": list(hash_to_diagram.values())},
        schema={"text_hash": pl.String, "diagram": pl.String},
    )
    sym_df = pl.DataFrame(
        {"text_hash": list(hash_to_symbols.keys()), "symbols": list(hash_to_symbols.values())},
        schema={"text_hash": pl.String, "symbols": pl.String},
    )
    return diag_df, sym_df


def _split_ids(
    ids: list[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle sample_ids deterministically and split into train/val/test."""
    assert abs(sum(ratios) - 1.0) < 1e-9, f"Ratios must sum to 1.0, got {sum(ratios)}"
    rng = np.random.default_rng(seed)
    shuffled = np.array(ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(ratios[0] * n)
    val_end = train_end + int(ratios[1] * n)

    return (
        shuffled[:train_end].tolist(),
        shuffled[train_end:val_end].tolist(),
        shuffled[val_end:].tolist(),
    )


def enrich_atoms(
    derived_dirs: list[Path],
    lmdb_path: Path = constants.lmdb_path,
    filter_2d_outputs: bool = True,
) -> pl.DataFrame:
    """
    Read all chunk_*.parquet files from the given derived dirs, concatenate,
    and enrich each atom with diagram and symbols from LMDB.

    Preserves all columns present in the derived parquets (e.g. label).
    """
    chunk_files = []
    for d in derived_dirs:
        chunk_files.extend(sorted(Path(d).glob("chunk_*.parquet")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.parquet files found in: {derived_dirs}")

    atoms = pl.concat([pl.scan_parquet(f) for f in chunk_files], how="vertical_relaxed").collect()

    diag_df, sym_df = _fetch_lmdb_fields(atoms, lmdb_path)
    atoms = atoms.join(diag_df, on="text_hash", how="left").join(sym_df, on="text_hash", how="left")

    before = len(atoms)
    atoms = atoms.filter(pl.col("diagram").is_not_null() & pl.col("symbols").is_not_null())
    dropped = before - len(atoms)
    if dropped:
        logger.warning(f"Dropped {dropped} atoms with null diagram/symbols (CCG compilation failures).")

    if filter_2d_outputs:
        before = len(atoms)
        atoms = atoms.filter(pl.col("diagram").map_elements(_is_1d_diagram, return_dtype=pl.Boolean))
        dropped_2d = before - len(atoms)
        if dropped_2d:
            logger.warning(f"Dropped {dropped_2d} atoms with 2D diagram outputs.")

    logger.info(f"Enriched {len(atoms)} atoms from {len(chunk_files)} chunk(s).")
    return atoms


def split_by_groups(
    atoms: pl.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split atoms into train/val/test ensuring all atoms sharing a sample_id
    land in the same split. Guarantees no overlap between splits.
    """
    unique_ids = atoms["sample_id"].unique().to_list()
    train_ids, val_ids, test_ids = _split_ids(unique_ids, ratios, seed)

    return (
        atoms.filter(pl.col("sample_id").is_in(train_ids)),
        atoms.filter(pl.col("sample_id").is_in(val_ids)),
        atoms.filter(pl.col("sample_id").is_in(test_ids)),
    )


def create_dataset(
    derived_dirs: list[Path],
    strategy: CompositionStrategy,
    output_name: str,
    lmdb_path: Path = constants.lmdb_path,
    excluded_sample_ids: set[str] | None = None,
    filter_2d_outputs: bool = True,
) -> Path:
    """
    Create a single dataset parquet by enriching atoms and applying a composition strategy.

    Args:
        derived_dirs: Directories containing chunk_*.parquet files from one or more atlases.
        strategy: Defines how atoms are composed into task-specific samples.
        output_name: Written to data/datasets/<output_name>.parquet.
        excluded_sample_ids: sample_ids to exclude before composition (e.g. a held-out test set).
    """
    atoms = enrich_atoms(derived_dirs, lmdb_path, filter_2d_outputs=filter_2d_outputs)

    if excluded_sample_ids:
        before = len(atoms)
        atoms = atoms.filter(~pl.col("sample_id").is_in(excluded_sample_ids))
        logger.info(f"Excluded {before - len(atoms)} atoms via excluded_sample_ids.")

    composed = strategy.compose(atoms)

    out_path = Path(constants.datasets_path) / f"{output_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.write_parquet(out_path)
    logger.info(f"Dataset '{output_name}' written to {out_path} ({len(composed)} rows).")
    return out_path


def create_train_val_test_datasets(
    derived_dirs: list[Path],
    strategy: CompositionStrategy,
    output_name: str,
    lmdb_path: Path = constants.lmdb_path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    filter_2d_outputs: bool = True,
) -> tuple[Path, Path, Path]:
    """
    Create non-overlapping train/val/test datasets.

    Groups are split on sample_id before composition, so the strategy
    never sees atoms from different splits — preventing data leakage
    in synthesis scenarios (e.g. random negative sampling).
    """
    atoms = enrich_atoms(derived_dirs, lmdb_path, filter_2d_outputs=filter_2d_outputs)
    train_atoms, val_atoms, test_atoms = split_by_groups(atoms, ratios, seed)

    paths = []
    for split_name, split_atoms in [("train", train_atoms), ("val", val_atoms), ("test", test_atoms)]:
        composed = strategy.compose(split_atoms)
        out_path = Path(constants.datasets_path) / f"{output_name}_{split_name}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.write_parquet(out_path)
        logger.info(f"{split_name} split written to {out_path} ({len(composed)} rows).")
        paths.append(out_path)

    return tuple(paths)
