"""
Compile one shard of the unique SVO captions into its own LMDB store, for use
as an SGE job-array task (scripts/submit_svo_compile_array.sh).

Rationale: this cluster's nodes are mostly 4-8 cores (few have 16+), so one
large `-pe smp N` job queues behind whichever handful of big nodes are free.
An array of many small tasks schedules on almost any node and runs in
parallel across them instead. Each task compiles a disjoint slice of the
~8.4k unique captions into its own LMDB (avoiding concurrent-write risk on
a single shared store over NFS), then merge_shard_lmdbs.py consolidates all
shards into the real LMDB store in one single-writer pass.

Sharding is over unique `corrected_sentence` -> processed_text values (not
raw manifest rows) so no compute is wasted recompiling the ~3.1x duplicate
captions that share the same underlying sentence.

Usage (shard index/count from SGE_TASK_ID/SGE_TASK_LAST, or explicit args):
    python -m qnlp.scripts.svo.compile_shard --shard-index 0 --num-shards 8
"""

import argparse
import os
from pathlib import Path

import lmdb
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.preprocessing_pipelines.svo.pipeline import lemma_step, remove_dots_step
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="svo_compile_shard")

SHARD_LMDB_ROOT = constants.lmdb_path.parent / f"{constants.lmdb_path.name}_svo_shards"
# Mirrors the LMDB-output sharding above: `CCGCompilerStep`'s default
# `cache_path` (~/.cache/lambeq/bobcat/diskcache) is a single shared sqlite
# file. Concurrent SGE array tasks all writing to it caused silent, large-
# scale corruption ("database disk image is malformed") that discarded
# ~31% of the SVO corpus in one past run — every task needs its own cache
# dir, exactly like the LMDB shards, not just a differently-located shared
# one (constants.bobcat_cache_path is still a single shared path).
SHARD_BOBCAT_CACHE_ROOT = constants.bobcat_cache_path.parent / "svo_shards"


def _resolve_shard_args() -> tuple[int, int, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=None, help="0-indexed shard (default: from SGE_TASK_ID)")
    parser.add_argument("--num-shards", type=int, default=None, help="total shards (default: from SGE_TASK_LAST)")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    shard_index = args.shard_index
    if shard_index is None:
        # SGE_TASK_ID is 1-indexed.
        shard_index = int(os.environ["SGE_TASK_ID"]) - 1
    num_shards = args.num_shards or int(os.environ["SGE_TASK_LAST"])
    return shard_index, num_shards, args.max_workers


def _unique_processed_texts() -> list[str]:
    """Recreate the exact processed_text values svo_pipeline would produce,
    without running the full Pipeline (we only need distinct texts to compile,
    keyed the same way schema_step + remove_dots_step + lemma_step would)."""
    manifest = pl.read_parquet(constants.atlases_path / "svo" / "data_manifest.parquet")
    unique_sentences = sorted(manifest["corrected_sentence"].unique().drop_nulls().to_list())

    df = pl.DataFrame({"processed_text": unique_sentences})
    df = remove_dots_step.process(df)
    df = lemma_step.process(df)
    return sorted(set(df["processed_text"].drop_nulls().to_list()))


def _write_lmdb(lmdb_path: Path, entries: dict[str, bytes]) -> None:
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), max_readers=128, map_size=2 * 1024 * 1024 * 1024, create=True, writemap=True)
    try:
        with env.begin(write=True) as txn:
            kv_pairs = ((k.encode("utf-8"), v) for k, v in entries.items())
            txn.cursor().putmulti(kv_pairs, overwrite=True)
    finally:
        env.close()


def run() -> None:
    shard_index, num_shards, max_workers = _resolve_shard_args()
    logger.info(f"Shard {shard_index}/{num_shards} (max_workers={max_workers})")

    all_texts = _unique_processed_texts()
    shard_texts = all_texts[shard_index::num_shards]
    logger.info(f"{len(all_texts)} unique captions total; this shard: {len(shard_texts)}")

    if not shard_texts:
        logger.info("Empty shard, nothing to do.")
        return

    shard_cache_path = SHARD_BOBCAT_CACHE_ROOT / f"shard_{shard_index}"
    shard_cache_path.mkdir(parents=True, exist_ok=True)
    compiler = CCGCompilerStep(
        lmdb_path=constants.lmdb_path,  # only used to skip already-cached texts
        bond_dim=constants.bond_dim,
        embedding_dim=constants.embedding_dim,
        max_workers=max_workers,
        worker_batch_size=max(10, len(shard_texts) // (max_workers * 4) or 1),
        cache_path=str(shard_cache_path),
    )
    df = pl.DataFrame({"processed_text": shard_texts})
    df = compiler.process(df)

    compiled = df.filter(pl.col("compiled_bytes").is_not_null())
    entries = dict(zip(compiled["text_hash"].to_list(), compiled["compiled_bytes"].to_list()))
    n_failed = len(df) - len(compiled)
    if n_failed:
        logger.warning(f"{n_failed}/{len(df)} captions in this shard failed to compile.")

    shard_lmdb_path = SHARD_LMDB_ROOT / f"shard_{shard_index}"
    _write_lmdb(shard_lmdb_path, entries)
    logger.info(f"Wrote {len(entries)} entries to {shard_lmdb_path}")

    compiler.teardown()


if __name__ == "__main__":
    run()
