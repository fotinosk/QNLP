"""Generate hard-negative swap candidates for OUR COCO captions (Phase A0).

Port of the colleague's caption-level generator (discoviz-repo/llm/systematic/
gen_hard_negs_v2.py + score_negs_hardness.py) onto our artifacts: instead of his
trees_train jsonl we read lambeq CCGTree objects straight from the
CachedBobcatParser diskcache, keyed by the exact post-lemmatize training strings
in our train parquets. No parsing happens here — cache hits only.

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

Run on the cluster from PROJECT_DIR (caches + parquets are there):
  python -m qnlp.scripts.coco_multi_caption.generate_hard_negatives
"""

import argparse
import string
import time
from pathlib import Path

import diskcache
import polars as pl
import torch
import torch.nn.functional as F

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="generate_hard_negatives")

MAXK = 6
CLIP_NAME = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {}"

BOBCAT_CACHE = "/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat/diskcache"
TREE_CACHE = "/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat_tree_no_type/diskcache"

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


# ---------------------------------------------------------------- hardness


def clip_word_similarities(pairs: set[tuple[str, str]], device: str) -> dict[tuple[str, str], float]:
    """h per word pair = cos of CLIP text embeddings, template 'a photo of a {}'."""
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


# ---------------------------------------------------------------- main


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


def fetch_tree(caches: list[diskcache.Cache], tokenizer, text: str):
    """Replicates CachedBobcatParser's key: str((tokens, tokenised=True, suppress_exceptions=False))."""
    tokens = tokenizer.tokenize(text)
    key = str((tokens, True, False))
    for cache in caches:
        if key in cache:
            return tokens, cache[key]
    return tokens, None


CHUNK = 50_000  # captions per resumable part-file


def _enumerate_chunk(chunk_df: pl.DataFrame, caches, tokenizer) -> tuple[list[tuple], int]:
    """Swap-enumerate one chunk of (text_hash, processed_text) rows.
    Returns (rows, cache_misses); rows = (text_hash, processed_text, neg_text, t, w1, w2)."""
    rows: list[tuple] = []
    misses = 0
    for text_hash, text in chunk_df.iter_rows():
        tokens, tree = fetch_tree(caches, tokenizer, text)
        if tree is None:
            misses += 1
            continue
        leaves = _leaves(tree)
        lemmas = tokenizer.lemmatize(tokens) if len(tokens) == len(leaves) else [None] * len(leaves)
        order = [LeafRecord(l.text, lemma, str(l.biclosed_type)) for l, lemma in zip(leaves, lemmas)]
        order_index = {id(l): k for k, l in enumerate(leaves)}

        obj = object_swaps(order)
        attr = attribute_swaps(tree, order_index, order)
        for t, swaps in (("obj", obj), ("attr", attr)):
            for i, j in swaps:
                w1, w2 = sorted((order[i].norm, order[j].norm))
                rows.append((text_hash, text, materialize(order, i, j), t, w1, w2))
    return rows, misses


def run(parquets: list[str], output: str, limit: int | None = None) -> None:
    from qnlp.discoviz.models.bobcat_text_processor import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer()
    caches = []
    for path in (BOBCAT_CACHE, TREE_CACHE):
        if Path(path).exists():
            caches.append(diskcache.Cache(path))
        else:
            logger.warning(f"diskcache not found: {path}")
    if not caches:
        raise RuntimeError("No bobcat diskcache available — run on the cluster from PROJECT_DIR.")

    # Deterministic order so chunk boundaries are identical across restarts.
    df = load_unique_captions(parquets).sort("text_hash")
    if limit:
        df = df.head(limit)

    # Resumable enumeration: one part-file per CHUNK captions; existing parts are
    # skipped, so a killed job picks up at the first missing part on rerun.
    # Parts are written atomically (tmp + rename) so a mid-write kill can't leave
    # a truncated part behind. h is NOT in the parts — it's scored over all parts
    # at the end (cheap), keeping parts independent of the global word vocabulary.
    parts_dir = Path(str(output).replace(".parquet", "_parts"))
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_parts = (df.height + CHUNK - 1) // CHUNK
    t0 = time.time()
    total_misses = 0

    for k in range(n_parts):
        part_path = parts_dir / f"part_{k:05d}.parquet"
        if part_path.exists():
            logger.info(f"Part {k + 1}/{n_parts} exists — skipping (resume).")
            continue
        chunk_df = df.slice(k * CHUNK, CHUNK)
        rows, misses = _enumerate_chunk(chunk_df, caches, tokenizer)
        total_misses += misses
        part = pl.DataFrame(rows, schema=["text_hash", "processed_text", "neg_text", "t", "w1", "w2"], orient="row")
        tmp = part_path.with_suffix(".tmp.parquet")
        part.write_parquet(tmp)
        tmp.rename(part_path)
        logger.info(
            f"Part {k + 1}/{n_parts}: {chunk_df.height} captions -> {part.height} candidates, "
            f"{misses} cache misses ({time.time() - t0:.0f}s elapsed)"
        )

    all_parts = pl.concat([pl.read_parquet(p) for p in sorted(parts_dir.glob("part_*.parquet"))])
    n_covered = all_parts["text_hash"].n_unique()
    n_obj = (all_parts["t"] == "obj").sum()
    n_attr = (all_parts["t"] == "attr").sum()
    logger.info(
        f"Enumeration done: {df.height} captions, {total_misses} cache misses THIS run "
        f"(misses in resumed parts not re-counted), {n_covered} with >=1 candidate "
        f"({100 * n_covered / max(1, df.height):.1f}%), "
        f"obj/cap={n_obj / max(1, df.height):.2f} attr/cap={n_attr / max(1, df.height):.2f} "
        f"({n_obj} obj, {n_attr} attr)"
    )

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
        f"frac(h>0.95)={(h > 0.95).mean():.3%}\n"
        f"  (colleague's reference: 98.8% coverage, mean 3.62 cand/caption)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate hard-negative swap candidates (Phase A0).")
    ap.add_argument("--parquets", nargs="+", default=DEFAULT_PARQUETS)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--limit", type=int, default=None, help="Cap #captions (smoke test).")
    args = ap.parse_args()
    run(args.parquets, args.output, args.limit)
