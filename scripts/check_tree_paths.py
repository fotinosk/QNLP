"""Test whether the contraction-path failures for tree_no_type diagrams are a
heuristic problem (branch-2) or fundamental (every path exceeds the limit).

Reads real stored diagrams from the tree LMDB and, for a sample of unique
topologies, computes the largest contraction intermediate under several
opt_einsum optimizers. If e.g. `dp`/`optimal` yields many feasible paths where
`branch-2` yields none, the fix is just to change the optimizer in
get_contraction_path_and_cost. If ALL optimizers exceed the limit, the tree
diagrams are genuinely too expensive at embedding_dim=512 and need a different
approach (smaller dim / different ansatz).

Run: /SAN/intelsys/discoviz/envs/qnlp311/bin/python scripts/check_tree_paths.py
"""

import argparse
import signal

import lmdb
import opt_einsum
import orjson

MAX_INTERMEDIATE_ELEMENTS = 50_000_000


class _TO(Exception):
    pass


def _limit(seconds):
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(_TO()))
    signal.setitimer(signal.ITIMER_REAL, seconds)


def _clear():
    signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/sentence_mapping_tree_no_type")
    ap.add_argument("--n", type=int, default=200, help="unique topologies to sample")
    ap.add_argument("--timeout", type=float, default=20.0, help="per-optimizer time budget (s)")
    args = ap.parse_args()

    optimizers = ["branch-2", "greedy", "dp", "auto-hq"]
    counts = {o: {"ok": 0, "too_big": 0, "timeout": 0, "err": 0} for o in optimizers}
    medians = {o: [] for o in optimizers}

    env = lmdb.open(args.lmdb, readonly=True, lock=False, max_readers=64)
    seen: set[str] = set()
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
            for o in optimizers:
                try:
                    _limit(args.timeout)
                    _, info = opt_einsum.contract_path(diagram, *shapes, shapes=True, optimize=o)
                    _clear()
                    li = int(info.largest_intermediate)
                    medians[o].append(li)
                    if li <= MAX_INTERMEDIATE_ELEMENTS:
                        counts[o]["ok"] += 1
                    else:
                        counts[o]["too_big"] += 1
                except _TO:
                    counts[o]["timeout"] += 1
                except Exception:
                    _clear()
                    counts[o]["err"] += 1
            if len(seen) >= args.n:
                break

    print(f"Sampled {len(seen)} unique topologies from {args.lmdb}\n")
    print(f"{'optimizer':<10}{'ok':>7}{'too_big':>9}{'timeout':>9}{'err':>6}{'median_intermediate':>22}")
    print("-" * 63)
    for o in optimizers:
        s = sorted(medians[o])
        med = s[len(s) // 2] if s else 0
        c = counts[o]
        print(f"{o:<10}{c['ok']:>7}{c['too_big']:>9}{c['timeout']:>9}{c['err']:>6}{med:>22,}")
    print(f"\n(feasible = largest_intermediate <= {MAX_INTERMEDIATE_ELEMENTS:,})")


if __name__ == "__main__":
    main()
