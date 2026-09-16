"""
Merge all per-shard LMDB stores produced by compile_shard.py into the real
LMDB store (constants.lmdb_path), in a single writer pass — avoids any
concurrent-write risk from having array tasks write directly to a shared
store on (likely NFS-backed) cluster storage.

Run once after every task in the compile array job has finished, before
scripts/submit_svo_pipeline.sh.

Usage:
    python -m qnlp.scripts.svo.merge_shard_lmdbs
"""

import lmdb

from qnlp.constants import constants
from qnlp.scripts.svo.compile_shard import SHARD_LMDB_ROOT
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="svo_merge_shard_lmdbs")


def run() -> None:
    shard_dirs = sorted(p for p in SHARD_LMDB_ROOT.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard LMDBs found under {SHARD_LMDB_ROOT}")

    dest = constants.lmdb_path
    dest.mkdir(parents=True, exist_ok=True)
    dest_env = lmdb.open(str(dest), max_readers=128, map_size=10 * 1024 * 1024 * 1024, create=True, writemap=True)

    total = 0
    try:
        for shard_dir in shard_dirs:
            src_env = lmdb.open(str(shard_dir), readonly=True, lock=False, readahead=False)
            try:
                with src_env.begin() as src_txn, dest_env.begin(write=True) as dst_txn:
                    cursor = src_txn.cursor()
                    n = 0
                    for key, value in cursor:
                        dst_txn.put(key, value, overwrite=True)
                        n += 1
                logger.info(f"Merged {n} entries from {shard_dir}")
                total += n
            finally:
                src_env.close()
    finally:
        dest_env.close()

    logger.info(f"Merge complete: {total} entries merged from {len(shard_dirs)} shards into {dest}")


if __name__ == "__main__":
    run()
