import re
import signal
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import lmdb
import numpy as np
import orjson
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.composition_strategy import CompositionStrategy
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="dataset_generator")


@contextmanager
def _time_limit(seconds: float):
    """Raise TimeoutError if the wrapped block runs longer than `seconds` (SIGALRM, main thread)."""

    def _handler(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def add_contraction_paths(atoms: pl.DataFrame, timeout_seconds: float = 10.0) -> pl.DataFrame:
    """Compute the non-linear contraction path for each atom's diagram and add it
    as a JSON-serialised `path` column.

    Derived purely from the already-fetched `diagram` and `symbols` (no LMDB cache
    changes). Atoms whose path times out or would produce an intermediate larger
    than MAX_INTERMEDIATE_ELEMENTS get `path = None` (excluded later via require_path).
    """
    from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import (
        MAX_INTERMEDIATE_ELEMENTS,
        get_contraction_path_and_cost,
    )

    diagrams = atoms["diagram"].to_list()
    symbols_list = atoms["symbols"].to_list()
    paths: list[str | None] = []
    n_ok = 0
    n_failed = 0

    # Memoise on (diagram, shapes): templated captions produce the same topology
    # thousands of times, and path planning is the expensive part.
    cache: dict[tuple, str | None] = {}

    for diagram, sym_json in zip(diagrams, symbols_list):
        if diagram is None or sym_json is None:
            paths.append(None)
            continue

        shapes = tuple(tuple(entry[1]) for entry in orjson.loads(sym_json))
        key = (diagram, shapes)
        if key in cache:
            result = cache[key]
            paths.append(result)
            n_ok += result is not None
            n_failed += result is None
            continue

        try:
            with _time_limit(timeout_seconds):
                path, largest_intermediate = get_contraction_path_and_cost(diagram, shapes)
            if largest_intermediate > MAX_INTERMEDIATE_ELEMENTS:
                raise ValueError(f"largest intermediate {largest_intermediate:,} > {MAX_INTERMEDIATE_ELEMENTS:,}")
            result = orjson.dumps(path).decode()
            n_ok += 1
        except Exception as e:
            result = None
            n_failed += 1
            logger.warning(f"Contraction path failed for diagram '{diagram[:60]}...': {e}")

        cache[key] = result
        paths.append(result)

    logger.info(f"Contraction paths computed: {n_ok} ok, {n_failed} failed ({len(cache)} unique topologies).")
    return atoms.with_columns(pl.Series("path", paths, dtype=pl.String))


_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def _is_1d_diagram(diagram: str) -> bool:
    if "->" not in diagram:
        return True
    output_part = diagram.rsplit("->", 1)[-1].strip()
    return len(_LETTER_RE.findall(output_part)) == 1


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
    compute_contraction_paths: bool = False,
    path_timeout_seconds: float = 10.0,
) -> pl.DataFrame:
    """
    Read all chunk_*.parquet files from the given derived dirs, concatenate,
    and enrich each atom with diagram and symbols from LMDB.

    Preserves all columns present in the derived parquets (e.g. label).

    If compute_contraction_paths is set, a `path` column is added and atoms whose
    non-linear contraction path could not be computed (timeout / too-large
    intermediate) are excluded.
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

    if compute_contraction_paths:
        atoms = add_contraction_paths(atoms, path_timeout_seconds)
        before = len(atoms)
        atoms = atoms.filter(pl.col("path").is_not_null())
        dropped_path = before - len(atoms)
        if dropped_path:
            logger.warning(f"Dropped {dropped_path} atoms with no computable contraction path.")

    logger.info(f"Enriched {len(atoms)} atoms from {len(chunk_files)} chunk(s).")
    return atoms


def split_by_groups(
    atoms: pl.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_column: str = "sample_id",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split atoms into train/val/test ensuring all atoms sharing a group_column
    value land in the same split. Guarantees no overlap between splits.

    group_column defaults to "sample_id". Pass a different column (e.g.
    "pair_id") for datasets where the natural grouping differs from sample_id.
    """
    unique_ids = atoms[group_column].unique().to_list()
    train_ids, val_ids, test_ids = _split_ids(unique_ids, ratios, seed)

    return (
        atoms.filter(pl.col(group_column).is_in(train_ids)),
        atoms.filter(pl.col(group_column).is_in(val_ids)),
        atoms.filter(pl.col(group_column).is_in(test_ids)),
    )


def create_dataset(
    derived_dirs: list[Path],
    strategy: CompositionStrategy,
    output_name: str,
    lmdb_path: Path = constants.lmdb_path,
    excluded_sample_ids: set[str] | None = None,
    filter_2d_outputs: bool = True,
    compute_contraction_paths: bool = False,
) -> Path:
    """
    Create a single dataset parquet by enriching atoms and applying a composition strategy.

    Args:
        derived_dirs: Directories containing chunk_*.parquet files from one or more atlases.
        strategy: Defines how atoms are composed into task-specific samples.
        output_name: Written to data/datasets/<output_name>.parquet.
        excluded_sample_ids: sample_ids to exclude before composition (e.g. a held-out test set).
        compute_contraction_paths: Pre-compute non-linear contraction paths and exclude
            atoms whose path is infeasible.
    """
    atoms = enrich_atoms(
        derived_dirs,
        lmdb_path,
        filter_2d_outputs=filter_2d_outputs,
        compute_contraction_paths=compute_contraction_paths,
    )

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
    group_column: str = "sample_id",
    pre_split_hook: "Callable[[pl.DataFrame], pl.DataFrame] | None" = None,
    compute_contraction_paths: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Create non-overlapping train/val/test datasets.

    Groups are split on group_column before composition, so the strategy
    never sees atoms from different splits — preventing data leakage
    in synthesis scenarios (e.g. random negative sampling).

    pre_split_hook: optional transform applied to atoms before splitting,
        useful for deriving a group_column that doesn't exist in the raw
        atoms (e.g. extracting pair_id from sample_id for Winoground).
    """
    atoms = enrich_atoms(
        derived_dirs,
        lmdb_path,
        filter_2d_outputs=filter_2d_outputs,
        compute_contraction_paths=compute_contraction_paths,
    )
    if pre_split_hook is not None:
        atoms = pre_split_hook(atoms)
    train_atoms, val_atoms, test_atoms = split_by_groups(atoms, ratios, seed, group_column)

    paths = []
    for split_name, split_atoms in [("train", train_atoms), ("val", val_atoms), ("test", test_atoms)]:
        composed = strategy.compose(split_atoms)
        out_path = Path(constants.datasets_path) / f"{output_name}_{split_name}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.write_parquet(out_path)
        logger.info(f"{split_name} split written to {out_path} ({len(composed)} rows).")
        paths.append(out_path)

    return tuple(paths)
