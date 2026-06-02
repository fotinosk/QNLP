"""Binary trees in 4 fixed shapes for the aligned COCO comparison subset.

Mirrors build_binary_trees.py, but the caption source is the
coco_single_caption_{split}.parquet `processed_text` column.

Shapes (SHAPE env): left | right | balanced | random
All use a single rule name 'COMP' (one shared composition MLP).

Output: data/aro/processed/coco_cmp_bintrees_{shape}/{split}_trees.jsonl
"""
import json, os, random, re

import polars as pl

_PUNCT = re.compile(r"[^\w\s']")
SUBSET = "data/datasets_coco_cmp"


def tokenize(caption: str) -> list[str]:
    t = _PUNCT.sub(" ", caption.lower())
    return [w for w in t.split() if w]


def leaf(token: str) -> dict:
    return {"leaf": token, "lemma": token, "type": "w"}


def left_tree(toks):
    if not toks: return leaf("")
    out = leaf(toks[0])
    for t in toks[1:]:
        out = {"rule": "COMP", "children": [out, leaf(t)]}
    return out


def right_tree(toks):
    if not toks: return leaf("")
    out = leaf(toks[-1])
    for t in reversed(toks[:-1]):
        out = {"rule": "COMP", "children": [leaf(t), out]}
    return out


def balanced_tree(toks):
    if len(toks) == 1: return leaf(toks[0])
    if not toks: return leaf("")
    mid = len(toks) // 2
    return {"rule": "COMP", "children": [balanced_tree(toks[:mid]), balanced_tree(toks[mid:])]}


def random_tree(toks, rng):
    if len(toks) == 1: return leaf(toks[0])
    if not toks: return leaf("")
    split = rng.randint(1, len(toks) - 1)
    return {"rule": "COMP", "children": [random_tree(toks[:split], rng), random_tree(toks[split:], rng)]}


SHAPES = {"left": left_tree, "right": right_tree, "balanced": balanced_tree}


def captions_for(split):
    df = pl.read_parquet(f"{SUBSET}/coco_single_caption_{split}.parquet",
                         columns=["processed_text"])
    return sorted(set(df["processed_text"].to_list()))


def main():
    shape = os.environ.get("SHAPE", "left")
    out_dir = f"data/aro/processed/coco_cmp_bintrees_{shape}"
    os.makedirs(out_dir, exist_ok=True)
    seed_base = int(os.environ.get("SEED", "42"))
    for split in ["val", "test", "train"]:
        dst = f"{out_dir}/{split}_trees.jsonl"
        caps = captions_for(split)
        print(f"[{split}] {len(caps)} captions -> {dst}", flush=True)
        with open(dst, "w") as f:
            for c in caps:
                toks = tokenize(c)
                if shape == "random":
                    rng = random.Random(seed_base + hash(c) % 1000000)
                    t = random_tree(toks, rng)
                else:
                    t = SHAPES[shape](toks)
                f.write(json.dumps({"caption": c, "tree": t}) + "\n")
        print("  done", flush=True)


if __name__ == "__main__":
    main()
