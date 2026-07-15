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


def add_contraction_paths(
    atoms: pl.DataFrame,
    timeout_seconds: float = 10.0,
    max_symbols: int = 200,
    strategy: str = "optimal",
) -> pl.DataFrame:
    """Compute the non-linear contraction path for each atom's diagram and add it
    as a JSON-serialised `path` column.

    strategy:
      "optimal"      — branch-2 opt_einsum (default)
      "right_to_left" — fold right; t0⊗(t1⊗(t2⊗t3))
      "random"       — uniformly random pairwise order (memory-checked, retried up to 10x)

    Atoms get `path = None` when they exceed max_symbols, time out (optimal only),
    or produce an intermediate larger than MAX_INTERMEDIATE_ELEMENTS.
    """
    from tqdm import tqdm

    from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import (
        MAX_INTERMEDIATE_ELEMENTS,
        PATH_OPTIMIZER,
        get_contraction_path_and_cost,
        get_random_path,
        get_right_to_left_path,
    )

    opt_name = PATH_OPTIMIZER if strategy == "optimal" else strategy
    logger.info(
        f"Planning contraction paths: strategy={strategy} "
        f"(opt_einsum optimizer='{opt_name}', max_intermediate={MAX_INTERMEDIATE_ELEMENTS:,} elems)."
    )

    from collections import Counter

    diagrams = atoms["diagram"].to_list()
    symbols_list = atoms["symbols"].to_list()
    paths: list[str | None] = []
    n_ok = 0
    n_failed = 0

    # Diagnostics: why each unique topology failed, an example per reason, and the
    # sizes of over-limit intermediates — logged as a summary so the job output
    # shows what actually fails without flooding one warning per atom.
    fail_reasons: Counter = Counter()
    fail_examples: dict[str, tuple[str, str]] = {}
    too_big_sizes: list[int] = []

    # Memoise on (diagram, shapes): templated captions produce the same topology
    # thousands of times, and path planning is the expensive part. A cache miss on a
    # long diagram can take up to `timeout_seconds`, so the bar may crawl on novel ones.
    cache: dict[tuple, str | None] = {}

    bar = tqdm(zip(diagrams, symbols_list), total=len(diagrams), desc="Computing contraction paths")
    for diagram, sym_json in bar:
        bar.set_postfix(ok=n_ok, failed=n_failed, topologies=len(cache))
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

        if len(shapes) > max_symbols:
            result = None
            n_failed += 1
            fail_reasons["too_many_symbols"] += 1
            fail_examples.setdefault(
                "too_many_symbols", (diagram[:80], f"n_tensors={len(shapes)} > max_symbols={max_symbols}")
            )
        elif strategy == "right_to_left":
            try:
                import opt_einsum

                path = get_right_to_left_path(len(shapes))
                _, info = opt_einsum.contract_path(diagram, *shapes, shapes=True, optimize=path)
                if info.largest_intermediate > MAX_INTERMEDIATE_ELEMENTS:
                    raise ValueError(
                        f"largest intermediate {info.largest_intermediate:,} > {MAX_INTERMEDIATE_ELEMENTS:,}"
                    )
                result = orjson.dumps(path).decode()
                n_ok += 1
            except Exception as e:
                result = None
                n_failed += 1
                logger.warning(f"RTL path failed for diagram '{diagram[:60]}...': {e}")
        elif strategy == "random":
            result = None
            for attempt in range(10):
                try:
                    import opt_einsum

                    path = get_random_path(diagram, seed=attempt)
                    _, info = opt_einsum.contract_path(diagram, *shapes, shapes=True, optimize=path)
                    if info.largest_intermediate <= MAX_INTERMEDIATE_ELEMENTS:
                        result = orjson.dumps(path).decode()
                        n_ok += 1
                        break
                except Exception:
                    pass
            else:
                n_failed += 1
                logger.warning(f"Random path failed (all 10 attempts) for diagram '{diagram[:60]}...'")
        else:
            try:
                with _time_limit(timeout_seconds):
                    path, largest_intermediate = get_contraction_path_and_cost(diagram, shapes)
                if largest_intermediate > MAX_INTERMEDIATE_ELEMENTS:
                    result = None
                    n_failed += 1
                    fail_reasons["intermediate_too_big"] += 1
                    too_big_sizes.append(largest_intermediate)
                    fail_examples.setdefault(
                        "intermediate_too_big",
                        (diagram[:80], f"{largest_intermediate:,} elems (n_tensors={len(shapes)})"),
                    )
                else:
                    result = orjson.dumps(path).decode()
                    n_ok += 1
            except Exception as e:
                result = None
                n_failed += 1
                reason = type(e).__name__
                fail_reasons[reason] += 1
                fail_examples.setdefault(reason, (diagram[:80], str(e)[:100]))

        cache[key] = result
        paths.append(result)

    logger.info(f"Contraction paths computed: {n_ok} ok, {n_failed} failed ({len(cache)} unique topologies).")
    if fail_reasons:
        logger.info(f"Path failure breakdown (by unique topology): {dict(fail_reasons)}")
        for reason, (diag, detail) in fail_examples.items():
            logger.info(f"  example [{reason}]: {detail}  ::  {diag}")
        if too_big_sizes:
            s = sorted(too_big_sizes)
            logger.info(
                f"  over-limit intermediate sizes: min={s[0]:,} median={s[len(s) // 2]:,} "
                f"max={s[-1]:,}  (limit={MAX_INTERMEDIATE_ELEMENTS:,})"
            )
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
        from tqdm import tqdm

        unique_hashes = atoms["text_hash"].drop_nulls().unique().to_list()
        with env.begin() as txn:
            for h in tqdm(unique_hashes, desc="Fetching diagrams from LMDB"):
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
    lmdb_path: Path | None = None,
    filter_2d_outputs: bool = True,
    compute_contraction_paths: bool = False,
    path_timeout_seconds: float = 10.0,
    path_strategy: str = "optimal",
) -> pl.DataFrame:
    """
    Read all chunk_*.parquet files from the given derived dirs, concatenate,
    and enrich each atom with diagram and symbols from LMDB.

    Preserves all columns present in the derived parquets (e.g. label).

    If compute_contraction_paths is set, a `path` column is added and atoms whose
    non-linear contraction path could not be computed (timeout / too-large
    intermediate) are excluded.
    """
    lmdb_path = lmdb_path or constants.lmdb_path
    chunk_files = []
    for d in derived_dirs:
        chunk_files.extend(sorted(Path(d).glob("chunk_*.parquet")))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.parquet files found in: {derived_dirs}")

    logger.info(f"Reading {len(chunk_files)} derived chunk(s)...")
    atoms = pl.concat([pl.scan_parquet(f) for f in chunk_files], how="vertical_relaxed").collect()
    logger.info(f"Read {len(atoms)} atoms. Fetching diagrams/symbols from LMDB...")

    diag_df, sym_df = _fetch_lmdb_fields(atoms, lmdb_path)
    atoms = atoms.join(diag_df, on="text_hash", how="left").join(sym_df, on="text_hash", how="left")

    before = len(atoms)
    atoms = atoms.filter(pl.col("diagram").is_not_null() & pl.col("symbols").is_not_null())
    dropped = before - len(atoms)
    if dropped:
        logger.warning(f"Dropped {dropped} atoms with null diagram/symbols (CCG compilation failures).")

    if filter_2d_outputs:
        logger.info("Filtering 2D-output diagrams...")
        before = len(atoms)
        atoms = atoms.filter(pl.col("diagram").map_elements(_is_1d_diagram, return_dtype=pl.Boolean))
        dropped_2d = before - len(atoms)
        if dropped_2d:
            logger.warning(f"Dropped {dropped_2d} atoms with 2D diagram outputs.")

    if compute_contraction_paths:
        logger.info(f"Computing contraction paths for {len(atoms)} atoms (strategy={path_strategy})...")
        # 200, not 20: tree-reader (NO_TYPE) diagrams have ~3x the tensors of the
        # grammatical-cups diagrams (more boxes, no rewriter compaction) — n_tensors
        # of 37-65+ is normal — so a cap of 20 rejected every tree diagram outright.
        # dp path planning stays fast on tree topologies, and path_timeout_seconds
        # guards any pathologically large one.
        atoms = add_contraction_paths(atoms, path_timeout_seconds, max_symbols=200, strategy=path_strategy)
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
    lmdb_path: Path | None = None,
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

    # artifact_suffix keeps new-parser datasets parallel to the bobcat ones.
    out_name = f"{output_name}{constants.artifact_suffix}"
    out_path = Path(constants.datasets_path) / f"{out_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.write_parquet(out_path)
    logger.info(f"Dataset '{out_name}' written to {out_path} ({len(composed)} rows).")
    return out_path


def create_train_val_test_datasets(
    derived_dirs: list[Path],
    strategy: CompositionStrategy,
    output_name: str,
    lmdb_path: Path | None = None,
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
        out_path = Path(constants.datasets_path) / f"{output_name}{constants.artifact_suffix}_{split_name}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.write_parquet(out_path)
        logger.info(f"{split_name} split written to {out_path} ({len(composed)} rows).")
        paths.append(out_path)

    return tuple(paths)
