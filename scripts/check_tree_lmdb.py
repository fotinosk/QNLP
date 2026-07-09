"""Per-dataset health check for the shared tree_no_type LMDB diagram store.

The tree LMDB (data/sentence_mapping_tree_no_type/) is keyed by text_hash with no
dataset label. This script maps each dataset's text_hashes (from its
derived_tree_no_type/*.parquet chunks) back to the LMDB and tallies how many
diagrams are ok / errored / null / missing per dataset, so you can see whether the
diskcache corruption hit only COCO or every dataset.

Run with:
    /SAN/intelsys/discoviz/envs/qnlp311/bin/python scripts/check_tree_lmdb.py
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import lmdb
import polars as pl


def classify(txn: lmdb.Transaction, h: str) -> tuple[str, str]:
    v = txn.get(h.encode("utf-8"))
    if v is None:
        return "missing", ""
    d = json.loads(v)
    if d.get("error"):
        return "error", d["error"][:60]
    if d.get("diagram") is None:
        return "none", ""
    return "ok", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/sentence_mapping_tree_no_type")
    ap.add_argument("--atlases", default="data/atlases")
    ap.add_argument("--derived", default="derived_tree_no_type")
    args = ap.parse_args()

    derived_dirs = sorted(Path(args.atlases).glob(f"*/{args.derived}"))
    if not derived_dirs:
        print(f"No */{args.derived} dirs found under {args.atlases}. Nothing has been tree-processed yet.")
        return

    env = lmdb.open(args.lmdb, readonly=True, lock=False, max_readers=256)

    print(f"LMDB: {args.lmdb}")
    print(f"{'dataset':<20}{'unique_hashes':>14}{'ok':>9}{'error':>9}{'none':>7}{'missing':>9}{'ok%':>7}")
    print("-" * 75)

    grand_err_kinds: Counter[str] = Counter()

    with env.begin() as txn:
        for d in derived_dirs:
            dataset = d.parent.name
            chunks = list(d.glob("*.parquet"))
            if not chunks:
                continue

            lf = pl.scan_parquet(chunks)
            if "text_hash" not in lf.collect_schema().names():
                print(f"  ! {dataset}: no text_hash column (has {lf.collect_schema().names()})")
                continue
            hashes = lf.select("text_hash").unique().collect().get_column("text_hash").drop_nulls().to_list()

            status: Counter[str] = Counter()
            for h in hashes:
                s, msg = classify(txn, h)
                status[s] += 1
                if s == "error":
                    grand_err_kinds[msg] += 1

            n = len(hashes) or 1
            print(
                f"{dataset:<20}{len(hashes):>14}{status['ok']:>9}{status['error']:>9}"
                f"{status['none']:>7}{status['missing']:>9}{100 * status['ok'] / n:>6.1f}%"
            )

    if grand_err_kinds:
        print("\nError breakdown (all datasets):")
        for msg, c in grand_err_kinds.most_common(10):
            print(f"  {c:>8}  {msg}")


if __name__ == "__main__":
    main()
