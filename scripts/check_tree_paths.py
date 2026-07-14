"""Test whether the contraction-path failures for tree_no_type diagrams are a
heuristic problem (branch-2) or fundamental (every path exceeds the limit).

Reads real stored diagrams from the tree LMDB and, for a sample of unique
topologies, computes the largest contraction intermediate under a few opt_einsum
optimizers. If e.g. `dp` yields feasible paths where `branch-2` does not, the fix
is just to change the optimizer in get_contraction_path_and_cost. If ALL are
too_big, the tree diagrams are genuinely too expensive at embedding_dim=512.

Prints one line per topology so it is never silent. `auto-hq` (cotengra) is
omitted on purpose — its path search alone can take minutes per diagram.

Run: /SAN/intelsys/discoviz/envs/qnlp311/bin/python scripts/check_tree_paths.py --n 50
"""

import argparse
import signal

import lmdb
import opt_einsum
import orjson

MAX_INTERMEDIATE_ELEMENTS = 50_000_000
OPTIMIZERS = ["branch-2", "greedy", "dp"]


class _TO(Exception):
    pass


def _run(diagram, shapes, optimizer, timeout):
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        _, info = opt_einsum.contract_path(diagram, *shapes, shapes=True, optimize=optimizer)
        return int(info.largest_intermediate)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/sentence_mapping_tree_no_type")
    ap.add_argument("--n", type=int, default=50, help="unique topologies to sample")
    ap.add_argument("--timeout", type=float, default=8.0, help="per-optimizer time budget (s)")
    args = ap.parse_args()

    counts = {o: {"ok": 0, "too_big": 0, "timeout": 0, "err": 0} for o in OPTIMIZERS}

    def fmt(x):
        if isinstance(x, int):
            return f"{x:,}" + ("" if x <= MAX_INTERMEDIATE_ELEMENTS else "!")
        return x

    env = lmdb.open(args.lmdb, readonly=True, lock=False, max_readers=64)
    seen: set[str] = set()
    print(f"sampling up to {args.n} unique topologies (timeout {args.timeout}s each)\n", flush=True)
    print(f"{'#':>4}  {'n_tensors':>9}  " + "  ".join(f"{o:>14}" for o in OPTIMIZERS), flush=True)

    with env.begin() as txn:
        for _, v in txn.cursor():
            d = orjson.loads(v)
            diagram = d.get("diagram")
            if d.get("error") or diagram is None or d.get("symbols") is None:
                continue
            if diagram in seen:
                continue
            seen.add(diagram)
            shapes = [tuple(entry[1]) for entry in d["symbols"]]
            cells = []
            for o in OPTIMIZERS:
                try:
                    li = _run(diagram, shapes, o, args.timeout)
                    counts[o]["ok" if li <= MAX_INTERMEDIATE_ELEMENTS else "too_big"] += 1
                    cells.append(fmt(li))
                except _TO:
                    counts[o]["timeout"] += 1
                    cells.append("timeout")
                except Exception as e:
                    counts[o]["err"] += 1
                    cells.append(type(e).__name__)
            print(f"{len(seen):>4}  {len(shapes):>9}  " + "  ".join(f"{c:>14}" for c in cells), flush=True)
            if len(seen) >= args.n:
                break

    print(f"\n=== totals over {len(seen)} topologies (feasible <= {MAX_INTERMEDIATE_ELEMENTS:,}) ===", flush=True)
    for o in OPTIMIZERS:
        print(f"  {o:<10} {counts[o]}", flush=True)


if __name__ == "__main__":
    main()
