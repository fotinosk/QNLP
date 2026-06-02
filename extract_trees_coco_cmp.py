"""Extract CCG trees (bobcat) for the aligned COCO comparison subset.

Mirrors extract_trees_all3.py, but the caption source is the
coco_single_caption_{split}.parquet `processed_text` column (the aligned
image-disjoint subset built by build_coco_cmp_subset.py).

Output: data/aro/processed/coco_cmp_trees/{split}_trees.jsonl
  one line per caption: {"caption": <text>, "tree": <recursive dict>}
Resume-safe: skips captions already present in the output file.
"""
import json, os, signal, time

import polars as pl

class _Timeout(Exception): pass
def _ah(s, f): raise _Timeout()
signal.signal(signal.SIGALRM, _ah)
TIMEOUT = 20

SUBSET = "data/datasets_coco_cmp"
OUT_DIR = "data/aro/processed/coco_cmp_trees"

print("loading bobcat...", flush=True)
from qnlp.discoviz.parser.cached_bobcat import CachedBobcatParser
parser = CachedBobcatParser(device=os.environ.get("CCG_DEVICE", "cuda"))


def tree_to_dict(t):
    if t is None:
        return None
    if t.is_leaf:
        return {"leaf": (t.text or "").lower(),
                "lemma": (getattr(t, "lemma", None) or t.text or "").lower(),
                "type": str(t.biclosed_type)}
    rule = getattr(t, "rule", None)
    rule_name = rule.name if rule is not None else "?"
    return {"rule": rule_name,
            "type": str(t.biclosed_type),
            "children": [tree_to_dict(c) for c in t.children]}


def captions_for(split):
    df = pl.read_parquet(f"{SUBSET}/coco_single_caption_{split}.parquet",
                         columns=["processed_text"])
    return sorted(set(df["processed_text"].to_list()))


def process_split(split):
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = f"{OUT_DIR}/{split}_trees.jsonl"
    caps = captions_for(split)
    done = set()
    if os.path.exists(dst):
        with open(dst) as f:
            for line in f:
                try: done.add(json.loads(line)["caption"])
                except: pass
    todo = [c for c in caps if c not in done]
    print(f"[{split}] total={len(caps)} done={len(done)} todo={len(todo)}", flush=True)
    fout = open(dst, "a")
    t0 = time.time()
    BATCH = 64
    n_ok = n_fail = 0
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        try:
            trees = parser.sentences2trees(batch, suppress_exceptions=True)
        except Exception:
            trees = [None] * len(batch)
        for c, tr in zip(batch, trees):
            if tr is None:
                n_fail += 1; continue
            try:
                signal.alarm(TIMEOUT)
                d = tree_to_dict(tr)
                signal.alarm(0)
                fout.write(json.dumps({"caption": c, "tree": d}) + "\n")
                n_ok += 1
            except Exception:
                signal.alarm(0); n_fail += 1
        fout.flush()
        if i % (BATCH * 8) == 0:
            rate = (n_ok + n_fail) / max(time.time() - t0, 1e-6)
            print(f"  [{split}] {n_ok+n_fail}/{len(todo)} ok={n_ok} fail={n_fail} ({rate:.1f}/s)", flush=True)
    fout.close()
    print(f"[{split}] DONE: {n_ok}/{n_ok+n_fail}", flush=True)


for s in ["val", "test", "train"]:
    process_split(s)
print("all done", flush=True)
