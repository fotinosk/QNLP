"""Validate the Phase A0 rule port against the colleague's released negatives.

For captions of ours that EXACT-match his (same cocoid + lowercased caption
string — the ~66% subset unchanged by our LemmatizeStep), compare the swap
candidates as (type, {word1, word2}) sets — form-insensitive, so his raw-form
words and our lemma-normalized words still align through norm().

Expect high overlap; systematic differences indicate a rule-port bug (or CCG
parse differences between his trees and ours — spot-check a few by hand before
concluding either way).

Run on the cluster from PROJECT_DIR:
  python -m qnlp.scripts.coco_multi_caption.validate_hard_negatives
"""

import argparse
import json
import re
import string

import polars as pl

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="validate_hard_negatives")

NEGS_JSONL = "/SAN/intelsys/discoviz/systematic/data_release_v3/negs_karpathy_dedup.jsonl"
SPECS = "data/datasets/coco_hard_negs_train.parquet"
TRAIN = "data/datasets/coco_single_caption_nlc_train.parquet"
_ID_RX = re.compile(r"_(\d+)\.jpg$")


def _norm(w: str) -> str:
    return (w or "").lower().strip(string.punctuation)


def run(specs_path: str, train_path: str, negs_path: str, sample: int) -> None:
    train = pl.read_parquet(train_path, columns=["text_hash", "processed_text", "local_image_path"]).unique(
        subset=["text_hash"]
    )
    specs = pl.read_parquet(specs_path, columns=["text_hash", "t", "w1", "w2"])
    ours_by_hash: dict[str, set] = {}
    for th, t, w1, w2 in specs.iter_rows():
        ours_by_hash.setdefault(th, set()).add((t, frozenset((w1, w2))))

    his: dict[tuple[int, str], list[dict]] = {}
    with open(negs_path) as f:
        for line in f:
            d = json.loads(line)
            his[(d["cocoid"], d["caption"])] = d["negs"]

    def his_pairset(negs: list[dict], caption: str) -> set:
        """Recover (t, {w1,w2}) from his materialized strings by diffing caption vs neg."""
        cap_toks = caption.split()
        out = set()
        for nrec in negs:
            neg_toks = nrec["neg"].split()
            if len(neg_toks) != len(cap_toks):
                continue
            diff = [k for k, (a, b) in enumerate(zip(cap_toks, neg_toks)) if a != b]
            if len(diff) == 2:
                out.add((nrec["t"], frozenset((_norm(cap_toks[diff[0]]), _norm(cap_toks[diff[1]])))))
        return out

    n_matched = 0
    jaccards, exact, ours_only_ex, his_only_ex = [], 0, [], []
    for th, text, img in train.iter_rows():
        m = _ID_RX.search(img)
        if not m:
            continue
        cocoid = int(m.group(1))
        key = (cocoid, text.lower())
        if key not in his:
            continue
        n_matched += 1
        h_set = his_pairset(his[key], text.lower())
        o_set = ours_by_hash.get(th, set())
        union = h_set | o_set
        jac = len(h_set & o_set) / len(union) if union else 1.0
        jaccards.append(jac)
        if h_set == o_set:
            exact += 1
        elif len(ours_only_ex) < sample and o_set - h_set:
            ours_only_ex.append((text, sorted(o_set - h_set)))
        elif len(his_only_ex) < sample and h_set - o_set:
            his_only_ex.append((text, sorted(h_set - o_set)))

    n = max(1, n_matched)
    logger.info(
        f"Compared {n_matched} exact-match captions: "
        f"identical candidate sets {exact / n:.1%}, mean Jaccard {sum(jaccards) / n:.3f}"
    )
    for label, examples in (("OURS-ONLY", ours_only_ex), ("HIS-ONLY", his_only_ex)):
        for text, diff in examples:
            logger.info(f"  [{label}] {text!r}: {diff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=SPECS)
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--negs", default=NEGS_JSONL)
    ap.add_argument("--sample", type=int, default=10, help="Example mismatches to print per side.")
    args = ap.parse_args()
    run(args.specs, args.train, args.negs, args.sample)
