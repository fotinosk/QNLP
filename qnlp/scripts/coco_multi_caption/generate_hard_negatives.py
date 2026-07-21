"""Generate hard-negative swap candidates for OUR COCO captions (Phase A0).

Port of the colleague's caption-level generator (discoviz-repo/llm/systematic/
gen_hard_negs_v2.py + score_negs_hardness.py) onto our artifacts.

IMPORTANT (found via smoke test, job 7081219, 2026-07-20): the CachedBobcatParser
diskcache is NOT a complete mirror of the training set. CCGCompilerStep checks the
LMDB diagram store (keyed by text_hash) FIRST and only invokes the parser — hence
only populates the tree diskcache — for captions that were not already compiled
there at some point in the past. So most captions have NO cached tree and must be
PARSED here. This is a real, CPU-bound bobcat parsing job (hours), not a fast
lookup pass.

Two stages, run as separate cluster jobs (see submit scripts):
  enumerate — CPU only. For each caption: get its CCGTree (diskcache hit, or a real
    bobcat parse via a worker pool with recycling — the same memory-leak mitigation
    used by CCGCompilerStep, since lambeq CCG objects leak cumulatively), enumerate
    obj/attr swaps, write resumable per-chunk part files (no hardness yet).
  score — GPU. Loads all parts, CLIP-embeds the unique swapped words once, joins on
    h, writes the final output.

Swap rules (verbatim from gen_hard_negs_v2, CCG grammar):
  OBJECT   : two distinct head nouns (type "n") with a predicate leaf (type
             containing "s\\" or the core-preposition type "(np\\np)/np")
             strictly between them in surface order. Coordinate pairs (a conj
             between and no predicate) are skipped; no predicate at all -> skip.
  ATTRIBUTE: two adjectives (leaf type "n/n" as left child of a forward
             application) modifying DIFFERENT head nouns; identical adjectives
             skipped; one adjective kept per distinct head noun.
Both capped at MAXK=6 candidates per type per caption. Word identity uses the
normalized form (lemma if available, lowercased, punctuation-stripped).

Hardness h = cosine similarity of the two swapped words' CLIP text embeddings
(openai/clip-vit-base-patch32 via transformers, template "a photo of a {}"),
higher = harder. Matches score_negs_hardness.py exactly.

Output: data/datasets/coco_hard_neg_specs.parquet with columns
  text_hash, processed_text, neg_text, t ("obj"|"attr"), h, w1, w2
keyed by the POSITIVE row's text_hash (joins 1:1 onto both train parquets).

Run on the cluster from PROJECT_DIR:
  python -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate
  python -m qnlp.scripts.coco_multi_caption.generate_hard_negatives score
"""

import argparse
import gc
import multiprocessing as mp
import string
import time
from pathlib import Path

import polars as pl

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="generate_hard_negatives")

MAXK = 6
CLIP_NAME = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {}"
CHUNK = 50_000  # captions per resumable part-file

BOBCAT_CACHE = "/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat/diskcache"

DEFAULT_PARQUETS = [
    "data/datasets/coco_single_caption_nlc_train.parquet",
    "data/datasets/coco_single_caption_nlc_tree_no_type_train.parquet",
]
OUTPUT = "data/datasets/coco_hard_neg_specs.parquet"


# ---------------------------------------------------------------- tree access


def _leaves(tree) -> list:
    """lambeq CCGTree leaves in surface order."""
    if tree.is_leaf:
        return [tree]
    out = []
    for c in tree.children:
        out.extend(_leaves(c))
    return out


def _norm(word: str) -> str:
    return (word or "").lower().strip(string.punctuation)


class LeafRecord:
    __slots__ = ("text", "lemma", "type")

    def __init__(self, text: str, lemma: str | None, type_: str):
        self.text = text
        self.lemma = lemma
        self.type = type_

    @property
    def norm(self) -> str:
        return _norm(self.lemma or self.text)


def _is_pred_type(ty: str) -> bool:
    """Verb (contains s\\) or core preposition (np\\np)/np — gen_hard_negs_v2.is_pred_type."""
    ty = ty or ""
    return ("s\\" in ty) or (ty == "(np\\np)/np")


def _is_conj(leaf: LeafRecord) -> bool:
    return leaf.type == "conj" or leaf.text.lower() in ("and", "or")


# ---------------------------------------------------------------- swap rules


def object_swaps(order: list[LeafRecord]) -> list[tuple[int, int]]:
    nouns = [i for i, l in enumerate(order) if l.type == "n"]
    out = []
    for a in range(len(nouns)):
        for b in range(a + 1, len(nouns)):
            i, j = nouns[a], nouns[b]
            wi, wj = order[i].norm, order[j].norm
            if not wi or wi == wj:
                continue
            btw = order[i + 1 : j]
            has_conj = any(_is_conj(l) for l in btw)
            has_pred = any(_is_pred_type(l.type) for l in btw)
            if has_conj and not has_pred:  # coordinate — symmetric, skip
                continue
            if not has_pred:  # no clear relation — conservative skip
                continue
            out.append((i, j))
            if len(out) >= MAXK:
                return out
    return out


def attribute_swaps(tree, order_index: dict[int, int], order: list[LeafRecord]) -> list[tuple[int, int]]:
    """Adjective (n/n leaf as left child of a forward application) + the LAST head
    noun in its sibling subtree. One adjective per distinct head noun; all
    cross-noun pairs. Mirrors gen_hard_negs_v2.attribute_swaps."""
    pairs: list[tuple[int, str]] = []

    def walk(node):
        if node.is_leaf:
            return
        ch = node.children
        rule = getattr(getattr(node, "rule", None), "name", "")
        if (
            rule in ("FA", "FORWARD_APPLICATION")
            and len(ch) == 2
            and ch[0].is_leaf
            and str(ch[0].biclosed_type) == "n/n"
        ):
            head_nouns = [l for l in _leaves(ch[1]) if str(l.biclosed_type) == "n"]
            if head_nouns:
                adj_pos = order_index[id(ch[0])]
                hn_leaf = head_nouns[-1]
                hn_word = order[order_index[id(hn_leaf)]].norm
                pairs.append((adj_pos, hn_word))
        for c in ch:
            walk(c)

    walk(tree)
    by_head: dict[str, int] = {}
    for pos, head in pairs:
        by_head.setdefault(head, pos)
    positions = list(by_head.values())
    out = []
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            i, j = sorted((positions[a], positions[b]))
            if order[i].norm and order[i].norm != order[j].norm:
                out.append((i, j))
                if len(out) >= MAXK:
                    return out
    return out


def materialize(order: list[LeafRecord], i: int, j: int) -> str:
    tokens = [l.text for l in order]
    tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def _enumerate_from_tree(tree, tokens: list[str], tokenizer) -> list[tuple[str, str, str]]:
    """(t, w1, w2, neg_text) candidates for one already-parsed tree."""
    leaves = _leaves(tree)
    lemmas = tokenizer.lemmatize(tokens) if len(tokens) == len(leaves) else [None] * len(leaves)
    order = [LeafRecord(l.text, lemma, str(l.biclosed_type)) for l, lemma in zip(leaves, lemmas)]
    order_index = {id(l): k for k, l in enumerate(leaves)}
    obj = object_swaps(order)
    attr = attribute_swaps(tree, order_index, order)
    out = []
    for t, swaps in (("obj", obj), ("attr", attr)):
        for i, j in swaps:
            w1, w2 = sorted((order[i].norm, order[j].norm))
            out.append((t, w1, w2, materialize(order, i, j)))
    return out


# ---------------------------------------------------------------- worker pool
# Mirrors compiler_step.py's pattern: heavy CachedBobcatParser loaded once per
# worker, maxtasksperchild forces periodic restart to bound the lambeq CCG
# compile memory leak (see memory: project_lambeq_tree_memory_leak).

_worker_parser = None
_worker_tokenizer = None


def _worker_init(cache_path: str):
    global _worker_parser, _worker_tokenizer
    from qnlp.discoviz.models.bobcat_text_processor import Tokenizer
    from qnlp.discoviz.parser.cached_bobcat import CachedBobcatParser

    _worker_tokenizer = Tokenizer()
    _worker_parser = CachedBobcatParser(device="cpu", cache_path=cache_path, load_parser=True)


def _worker_process_batch(items: list[tuple[str, str]]) -> list[tuple[str, str, list]]:
    """items: (text_hash, processed_text). Returns (text_hash, processed_text, candidates)
    where candidates = [(t, w1, w2, neg_text), ...]."""
    global _worker_parser, _worker_tokenizer
    texts = [t for _, t in items]
    tokens_list = [_worker_tokenizer.tokenize(t) for t in texts]
    trees = _worker_parser.sentences2trees(tokens_list, tokenised=True, suppress_exceptions=True, verbose="suppress")
    results = []
    for (text_hash, text), tokens, tree in zip(items, tokens_list, trees):
        cands = _enumerate_from_tree(tree, tokens, _worker_tokenizer) if tree is not None else []
        results.append((text_hash, text, cands))
    gc.collect()
    return results


# ---------------------------------------------------------------- stage: enumerate


def load_unique_captions(parquets: list[str]) -> pl.DataFrame:
    frames = []
    for p in parquets:
        if not Path(p).exists():
            logger.warning(f"Parquet not found, skipping: {p}")
            continue
        frames.append(pl.read_parquet(p, columns=["text_hash", "processed_text"]))
    df = pl.concat(frames).unique(subset=["text_hash"], keep="first")
    logger.info(f"{df.height} unique captions across {len(frames)} parquets.")
    return df


def enumerate_stage(
    parquets: list[str],
    output: str,
    limit: int | None,
    max_workers: int,
    worker_batch_size: int,
    max_tasks_per_child: int,
) -> None:
    df = load_unique_captions(parquets).sort("text_hash")
    if limit:
        df = df.head(limit)

    parts_dir = Path(str(output).replace(".parquet", "_parts"))
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_parts = (df.height + CHUNK - 1) // CHUNK
    t0 = time.time()

    pool = mp.get_context("spawn").Pool(
        processes=max_workers, initializer=_worker_init, initargs=(BOBCAT_CACHE,), maxtasksperchild=max_tasks_per_child
    )
    try:
        for k in range(n_parts):
            part_path = parts_dir / f"part_{k:05d}.parquet"
            if part_path.exists():
                logger.info(f"Part {k + 1}/{n_parts} exists — skipping (resume).")
                continue
            chunk_df = df.slice(k * CHUNK, CHUNK)
            items = list(chunk_df.iter_rows())
            batches = [items[i : i + worker_batch_size] for i in range(0, len(items), worker_batch_size)]

            rows = []
            n_with_any = 0
            for batch_results in pool.imap_unordered(_worker_process_batch, batches, chunksize=1):
                for text_hash, text, cands in batch_results:
                    if cands:
                        n_with_any += 1
                    for t, w1, w2, neg_text in cands:
                        rows.append((text_hash, text, neg_text, t, w1, w2))

            part = pl.DataFrame(rows, schema=["text_hash", "processed_text", "neg_text", "t", "w1", "w2"], orient="row")
            tmp = part_path.with_suffix(".tmp.parquet")
            part.write_parquet(tmp)
            tmp.rename(part_path)
            logger.info(
                f"Part {k + 1}/{n_parts}: {len(items)} captions -> {part.height} candidates, "
                f"{n_with_any} captions with >=1 candidate ({time.time() - t0:.0f}s elapsed)"
            )
    finally:
        pool.close()
        pool.join()

    all_parts = pl.concat([pl.read_parquet(p) for p in sorted(parts_dir.glob("part_*.parquet"))])
    n_covered = all_parts["text_hash"].n_unique()
    n_obj = (all_parts["t"] == "obj").sum()
    n_attr = (all_parts["t"] == "attr").sum()
    logger.info(
        f"Enumeration done: {df.height} captions, {n_covered} with >=1 candidate "
        f"({100 * n_covered / max(1, df.height):.1f}%), "
        f"obj/cap={n_obj / max(1, df.height):.2f} attr/cap={n_attr / max(1, df.height):.2f} "
        f"({n_obj} obj, {n_attr} attr) in {time.time() - t0:.0f}s. "
        f"Reference (colleague's release): 98.8% coverage, 3.62 cand/caption."
    )


# ---------------------------------------------------------------- stage: score


def clip_word_similarities(pairs: set[tuple[str, str]], device: str) -> dict[tuple[str, str], float]:
    """h per word pair = cos of CLIP text embeddings, template 'a photo of a {}'."""
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    words = sorted({w for p in pairs for w in p})
    logger.info(f"Encoding {len(words)} unique swap words with {CLIP_NAME} on {device}...")
    model = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)
    emb: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        B = 512
        for k in range(0, len(words), B):
            chunk = words[k : k + B]
            tok = proc(text=[TEMPLATE.format(w) for w in chunk], return_tensors="pt", padding=True, truncation=True).to(
                device
            )
            feats = F.normalize(model.get_text_features(**tok), dim=-1).cpu()
            for w, v in zip(chunk, feats):
                emb[w] = v
    return {(a, b): round(float(emb[a] @ emb[b]), 4) for a, b in pairs}


def score_stage(output: str) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parts_dir = Path(str(output).replace(".parquet", "_parts"))
    part_files = sorted(parts_dir.glob("part_*.parquet"))
    if not part_files:
        raise RuntimeError(f"No parts found in {parts_dir} — run the 'enumerate' stage first.")
    all_parts = pl.concat([pl.read_parquet(p) for p in part_files])

    word_pairs = {(w1, w2) for w1, w2 in all_parts.select("w1", "w2").unique().iter_rows()}
    h_by_pair = clip_word_similarities(word_pairs, device)

    out = all_parts.with_columns(
        pl.struct(["w1", "w2"])
        .map_elements(lambda s: h_by_pair[(s["w1"], s["w2"])], return_dtype=pl.Float64)
        .alias("h")
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    final_tmp = Path(str(output) + ".tmp")
    out.write_parquet(final_tmp)
    final_tmp.rename(output)

    h = out["h"]
    logger.info(
        f"Wrote {out.height} candidates for {out['text_hash'].n_unique()} captions -> {output}\n"
        f"  mean cand/caption (over covered): {out.height / max(1, out['text_hash'].n_unique()):.2f}\n"
        f"  h: mean={h.mean():.3f} p5={h.quantile(0.05):.3f} p95={h.quantile(0.95):.3f} "
        f"frac(h>0.95)={(h > 0.95).mean():.3%}"
    )


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate hard-negative swap candidates (Phase A0).")
    ap.add_argument("stage", choices=["enumerate", "score"])
    ap.add_argument("--parquets", nargs="+", default=DEFAULT_PARQUETS)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--limit", type=int, default=None, help="Cap #captions (smoke test, enumerate stage only).")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--worker-batch-size", type=int, default=200)
    ap.add_argument("--max-tasks-per-child", type=int, default=5)
    args = ap.parse_args()

    if args.stage == "enumerate":
        enumerate_stage(
            args.parquets, args.output, args.limit, args.max_workers, args.worker_batch_size, args.max_tasks_per_child
        )
    else:
        score_stage(args.output)
