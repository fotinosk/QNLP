"""Create a small COCO dataset for local development and fast feedback.

Samples N unique image IDs from the existing derived chunks and writes
train/val/test splits. No CCG parsing — reuses whatever is in LMDB locally.

Usage:
    python -m qnlp.scripts.coco_single_caption.create_tiny_dataset
    python -m qnlp.scripts.coco_single_caption.create_tiny_dataset --n 500 --paths
"""

import argparse

import numpy as np
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import (
    _is_1d_diagram,
    add_contraction_paths,
    split_by_groups,
)
from qnlp.core.data_engine.dataset_creator.strategies.single_caption import SingleCaptionStrategy
from qnlp.utils.logging import setup_logger

COCO_DERIVED_DIR = constants.atlases_path / "coco" / "derived_test"
logger = setup_logger(log_name="create_tiny_dataset")


def run(n: int, paths: bool, seed: int) -> None:
    output_name = f"coco_single_caption{'_nlc' if paths else ''}_tiny"

    chunk_files = sorted(COCO_DERIVED_DIR.glob("chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.parquet files found in {COCO_DERIVED_DIR}")

    atoms = pl.concat([pl.scan_parquet(f) for f in chunk_files], how="vertical_relaxed").collect()
    logger.info(f"Loaded {len(atoms)} total atoms across {len(chunk_files)} chunks.")

    unique_ids = atoms["sample_id"].unique().to_list()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique_ids, size=min(n, len(unique_ids)), replace=False).tolist()
    atoms = atoms.filter(pl.col("sample_id").is_in(chosen))
    logger.info(f"Sampled {len(atoms)} atoms from {len(chosen)} image IDs.")

    atoms = _enrich(atoms, paths)

    train_atoms, val_atoms, test_atoms = split_by_groups(atoms, ratios=(0.8, 0.1, 0.1), seed=seed)
    strategy = SingleCaptionStrategy()

    for split_name, split_atoms in [("train", train_atoms), ("val", val_atoms), ("test", test_atoms)]:
        composed = strategy.compose(split_atoms)
        out_path = constants.datasets_path / f"{output_name}_{split_name}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed.write_parquet(out_path)
        logger.info(f"{split_name}: {len(composed)} rows → {out_path}")


def _enrich(atoms: pl.DataFrame, compute_contraction_paths: bool) -> pl.DataFrame:
    import lmdb
    import orjson
    from tqdm import tqdm

    env = lmdb.open(str(constants.lmdb_path), readonly=True, lock=False, readahead=False)
    hash_to_diagram: dict[str, str] = {}
    hash_to_symbols: dict[str, str] = {}

    unique_hashes = atoms["text_hash"].drop_nulls().unique().to_list()
    with env.begin() as txn:
        for h in tqdm(unique_hashes, desc="Fetching from LMDB"):
            val = txn.get(h.encode("utf-8"))
            if val:
                data = orjson.loads(val)
                hash_to_diagram[h] = data.get("diagram")
                raw_symbols = data.get("symbols")
                hash_to_symbols[h] = orjson.dumps(raw_symbols).decode() if raw_symbols is not None else None
    env.close()

    diag_df = pl.DataFrame({"text_hash": list(hash_to_diagram), "diagram": list(hash_to_diagram.values())})
    sym_df = pl.DataFrame({"text_hash": list(hash_to_symbols), "symbols": list(hash_to_symbols.values())})

    atoms = atoms.join(diag_df, on="text_hash", how="left").join(sym_df, on="text_hash", how="left")
    atoms = atoms.filter(pl.col("diagram").is_not_null() & pl.col("symbols").is_not_null())
    atoms = atoms.filter(pl.col("diagram").map_elements(_is_1d_diagram, return_dtype=pl.Boolean))

    if compute_contraction_paths:
        atoms = add_contraction_paths(atoms, max_symbols=20)
        atoms = atoms.filter(pl.col("path").is_not_null())

    logger.info(f"Enriched {len(atoms)} atoms.")
    return atoms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a tiny COCO dataset for local dev.")
    parser.add_argument("--n", type=int, default=300, help="Number of unique image IDs to sample (default: 300)")
    parser.add_argument("--paths", action="store_true", help="Compute non-linear contraction paths (slow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n=args.n, paths=args.paths, seed=args.seed)
