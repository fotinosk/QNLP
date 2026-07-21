"""Generate hard-negative training pairs for OUR COCO captions (combined design,
2026-07-21 architecture revision — supersedes the old enumerate-only Phase A0 +
separate Phase A recompile).

A swap only exchanges two leaves that already share a CCG type (two `n` nouns, or
two `n/n` adjectives), so it never changes the derivation tree's structure — only
two leaves' text. This means a negative's CCGTree is built by deep-copying the
POSITIVE's already-parsed (and lemmatized) tree and mutating two leaves, with no
fresh CCG search and no separate recompile pass:

  parse once -> lemmatize the tree once -> enumerate swaps on the lemmatized
  leaves -> per swap, deep-copy + mutate two leaves -> compile that tree straight
  to BOTH diagram types (bobcat grammatical-cups, tree_no_type) -> write two
  companion parquets, one per parser's training data.

Trained tensor symbol names are lemma-based (`{lemma}_{index}__{CCG_type}` —
see BobcatTextProcessor.lemmatize_tree), so lemmatizing BEFORE enumerating and
swapping lemma text directly (not raw surface tokens) is required for the
compiled negative symbols to land in the same vocabulary as the positives.

Each caption's two parquets are only in a parser's training data if `text_hash`
is in that parser's own `*_train.parquet` (bobcat and tree_no_type currently have
different train sets — ~59% overlap — so this conditional compile avoids wasted
work and keeps each output file precisely scoped to that parser's actual training
rows; see HARD_NEG_PI_SWEEP_PLAN.md "CORRECTION 2026-07-21").

Two stages, run as separate cluster jobs (see submit scripts):
  enumerate — CPU only. Parse + lemmatize + enumerate + dual-compile per caption.
    Resumable at BATCH granularity (worker_batch_size captions, not a large
    chunk): each batch's two part files are written the instant that batch's
    result comes back from the worker pool, never accumulated across batches in
    the main process. Safe to kill and restart at any point with the same
    command — only whichever batch(es) were in flight get redone (no hardness
    yet).
  score — GPU, short. CLIP-embeds the unique swapped words once, joins h onto
    both parsers' outputs by (w1, w2), writes the two final parquets.

Swap rules (verbatim from gen_hard_negs_v2, CCG grammar):
  OBJECT   : two distinct head nouns (type "n") with a predicate leaf (type
             containing "s\\" or the core-preposition type "(np\\np)/np")
             strictly between them in surface order. Coordinate pairs (a conj
             between and no predicate) are skipped; no predicate at all -> skip.
  ATTRIBUTE: two adjectives (leaf type "n/n" as left child of a forward
             application) modifying DIFFERENT head nouns; identical adjectives
             skipped; one adjective kept per distinct head noun.
Both capped at MAXK=6 candidates per type per caption. Word identity uses the
lemma-normalized form (lowercased, punctuation-stripped).

Hardness h = cosine similarity of the two swapped words' CLIP text embeddings
(openai/clip-vit-base-patch32 via transformers, template "a photo of a {}"),
higher = harder. Matches score_negs_hardness.py exactly.

Output (per parser, keyed by the POSITIVE row's text_hash):
  data/datasets/coco_hard_negs_train.parquet
  data/datasets/coco_hard_negs_tree_no_type_train.parquet
  columns: text_hash, processed_text, neg_text, t ("obj"|"attr"), w1, w2, h,
           diagram, symbols

Run on the cluster from PROJECT_DIR:
  python -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate
  python -m qnlp.scripts.coco_multi_caption.generate_hard_negatives score
"""

import argparse
import gc
import multiprocessing as mp
import re
import string
import time
from dataclasses import asdict
from pathlib import Path

import orjson
import polars as pl

from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="generate_hard_negatives")

MAXK = 6
CLIP_NAME = "openai/clip-vit-base-patch32"
TEMPLATE = "a photo of a {}"
EMBEDDING_DIM = 512  # matches every linear submit script's ML_EMBEDDING_DIM in this sweep
BOND_DIM = 10  # matches every linear submit script's ML_BOND_DIM in this sweep
# Random startup delay applied on every worker (re)start (see _worker_init) so
# maxtasksperchild-triggered recycles across workers can't converge into a
# synchronized reload burst (all workers reloading parser+tagger+ansatz at once
# was the real driver of the ~90G memory peaks — see HARD_NEG_PI_SWEEP_PLAN.md).
WORKER_INIT_JITTER_SECONDS = 60
# Exact rule list from CCGCompilerStep.__init__ — must match what positives were
# compiled with, or negatives' rewritten diagrams diverge from the trained topology.
REWRITE_RULES = [
    "auxiliary",
    "connector",
    "determiner",
    "postadverb",
    "preadverb",
    "prepositional_phrase",
    "coordination",
    "object_rel_pronoun",
    "subject_rel_pronoun",
]

# NOTE: resume only checks whether a part file exists, not whether it matches the
# current code. If you change enumerate_stage's logic, delete the parts_dir before
# rerunning — do not resume into parts built by older code.

BOBCAT_CACHE = "/SAN/intelsys/discoviz/fotinos/QNLP/.cache/lambeq/bobcat/diskcache"

BOBCAT_TRAIN = "data/datasets/coco_single_caption_nlc_train.parquet"
TREE_TRAIN = "data/datasets/coco_single_caption_nlc_tree_no_type_train.parquet"
OUTPUT_BOBCAT = "data/datasets/coco_hard_negs_train.parquet"
OUTPUT_TREE = "data/datasets/coco_hard_negs_tree_no_type_train.parquet"
PARTS_DIR = "data/datasets/coco_hard_negs_compiled_parts"

_ROW_SCHEMA = ["text_hash", "processed_text", "neg_text", "t", "w1", "w2", "diagram", "symbols"]
_LETTER_RE = re.compile(r"[^\W\d_]")


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
    __slots__ = ("text", "type")

    def __init__(self, text: str, type_: str):
        self.text = text
        self.type = type_

    @property
    def norm(self) -> str:
        return _norm(self.text)


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


# ---------------------------------------------------------------- compile


def _unify_rank(einsum_str: str) -> str:
    """Port of UnifyEinsumRankStep: if the diagram resolves to >1 open output wire,
    keep only the first output index (traces out the rest via implicit summation).
    Must match what positives went through so negatives share the same rank-1
    output convention the model expects."""
    if "->" not in einsum_str:
        return einsum_str
    lhs, rhs = einsum_str.rsplit("->", 1)
    return f"{lhs}->{rhs[:1]}" if len(rhs) > 1 else einsum_str


def _is_1d(einsum_str: str) -> bool:
    if "->" not in einsum_str:
        return True
    return len(_LETTER_RE.findall(einsum_str.rsplit("->", 1)[-1])) == 1


def _compile_diagram(diagram, ansatz) -> tuple[str, str] | None:
    """diagram -> (einsum_str, symbols_json) or None on rank/compile failure."""
    from qnlp.discoviz.models.bobcat_text_processor import tn_to_einsum

    circuit = ansatz(diagram)
    einsum_str, tensors = tn_to_einsum(circuit)
    einsum_str = _unify_rank(einsum_str)
    if not _is_1d(einsum_str):
        return None
    symbols = [[asdict(x[0]), x[1]] for x in tensors]
    return einsum_str, orjson.dumps(symbols).decode()


def _compile_bobcat(tree, ansatz, rewriter) -> tuple[str, str] | None:
    diagram = rewriter(tree.to_diagram()).remove_snakes()
    return _compile_diagram(diagram, ansatz)


def _compile_tree_no_type(tree, ansatz) -> tuple[str, str] | None:
    from lambeq import TreeReader, TreeReaderMode

    diagram = TreeReader.tree2diagram(tree, mode=TreeReaderMode.NO_TYPE)
    return _compile_diagram(diagram, ansatz)


def _lemmatize_tree(tree, lemmas: list[str]):
    """Port of BobcatTextProcessor.lemmatize_tree — deep-copies the tree and
    overwrites every leaf's text with its lemma, in surface order."""
    import copy

    tree = copy.deepcopy(tree)

    def traverse(node, lemma_iter):
        if node.is_leaf:
            node._text = next(lemma_iter)
        else:
            for child in node.children:
                traverse(child, lemma_iter)

    traverse(tree, iter(lemmas))
    return tree


def _swap_tree(lemma_tree, i: int, j: int):
    """Deep-copy the lemmatized tree and swap two leaves' text by surface position."""
    import copy

    swapped = copy.deepcopy(lemma_tree)
    leaves = _leaves(swapped)
    leaves[i]._text, leaves[j]._text = leaves[j]._text, leaves[i]._text
    return swapped


# ---------------------------------------------------------------- worker pool
# Mirrors compiler_step.py's pattern: heavy parser/ansatz loaded once per worker,
# maxtasksperchild forces periodic restart to bound the lambeq CCG compile memory
# leak (see memory: project_lambeq_tree_memory_leak).

_worker_parser = None
_worker_tokenizer = None
_worker_ansatz = None
_worker_rewriter = None
_worker_bobcat_hashes = None
_worker_tree_hashes = None


def _worker_init(cache_path: str, bobcat_train_path: str, tree_train_path: str):
    global _worker_parser, _worker_tokenizer, _worker_ansatz, _worker_rewriter
    global _worker_bobcat_hashes, _worker_tree_hashes
    import random
    import time as _time

    from lambeq import AtomicType, Rewriter
    from lambeq.backend.tensor import Dim

    # Runs on EVERY worker (re)start, including maxtasksperchild-triggered
    # respawns, not just the first launch — see WORKER_INIT_JITTER_SECONDS.
    _time.sleep(random.uniform(0, WORKER_INIT_JITTER_SECONDS))

    from qnlp.discoviz.models.bobcat_text_processor import Tokenizer
    from qnlp.discoviz.parser.asnsatz import CustomMPSAnsatz
    from qnlp.discoviz.parser.cached_bobcat import CachedBobcatParser

    _worker_tokenizer = Tokenizer()
    _worker_parser = CachedBobcatParser(device="cpu", cache_path=cache_path, load_parser=True)
    _worker_ansatz = CustomMPSAnsatz(
        {
            AtomicType.SENTENCE: Dim(EMBEDDING_DIM),
            AtomicType.NOUN: Dim(EMBEDDING_DIM),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(EMBEDDING_DIM),
        },
        bond_dim=BOND_DIM,
    )
    _worker_rewriter = Rewriter(REWRITE_RULES)
    _worker_bobcat_hashes = set(pl.read_parquet(bobcat_train_path, columns=["text_hash"])["text_hash"].to_list())
    _worker_tree_hashes = set(pl.read_parquet(tree_train_path, columns=["text_hash"])["text_hash"].to_list())


def _worker_process_batch(
    payload: tuple[int, list[tuple[str, str]]],
) -> tuple[int, list[tuple], list[tuple], int, int, int]:
    """payload: (batch_index, [(text_hash, processed_text), ...]). Returns
    (batch_index, bobcat_rows, tree_rows, n_with_any_candidate, n_compile_fail,
    n_items) where each row is (text_hash, processed_text, neg_text, t, w1, w2,
    diagram, symbols_json). The batch_index is carried through so the caller can
    match a result back to its batch even though imap_unordered returns results
    out of order."""
    global _worker_parser, _worker_tokenizer, _worker_ansatz, _worker_rewriter
    global _worker_bobcat_hashes, _worker_tree_hashes

    bi, items = payload
    texts = [t for _, t in items]
    tokens_list = [_worker_tokenizer.tokenize(t) for t in texts]
    trees = _worker_parser.sentences2trees(tokens_list, tokenised=True, suppress_exceptions=True, verbose="suppress")

    bobcat_rows, tree_rows = [], []
    n_with_any = 0
    n_fail = 0
    for (text_hash, text), tokens, tree in zip(items, tokens_list, trees):
        if tree is None:
            continue
        want_bobcat = text_hash in _worker_bobcat_hashes
        want_tree = text_hash in _worker_tree_hashes
        if not want_bobcat and not want_tree:
            continue

        lemmas = _worker_tokenizer.lemmatize(tokens)
        leaves = _leaves(tree)
        if len(lemmas) != len(leaves):
            continue
        lemma_tree = _lemmatize_tree(tree, lemmas)
        lemma_leaves = _leaves(lemma_tree)
        order = [LeafRecord(l._text, str(l.biclosed_type)) for l in lemma_leaves]
        order_index = {id(l): k for k, l in enumerate(lemma_leaves)}

        obj = object_swaps(order)
        attr = attribute_swaps(lemma_tree, order_index, order)
        candidates = [(t, i, j) for t, swaps in (("obj", obj), ("attr", attr)) for i, j in swaps]
        if candidates:
            n_with_any += 1

        for t, i, j in candidates:
            w1, w2 = sorted((order[i].norm, order[j].norm))
            swapped = _swap_tree(lemma_tree, i, j)
            neg_text = " ".join(l._text for l in _leaves(swapped))

            if want_bobcat:
                try:
                    res = _compile_bobcat(swapped, _worker_ansatz, _worker_rewriter)
                except Exception:
                    res = None
                if res is None:
                    n_fail += 1
                else:
                    diagram_str, symbols_json = res
                    bobcat_rows.append((text_hash, text, neg_text, t, w1, w2, diagram_str, symbols_json))

            if want_tree:
                try:
                    res = _compile_tree_no_type(swapped, _worker_ansatz)
                except Exception:
                    res = None
                if res is None:
                    n_fail += 1
                else:
                    diagram_str, symbols_json = res
                    tree_rows.append((text_hash, text, neg_text, t, w1, w2, diagram_str, symbols_json))

    gc.collect()
    return bi, bobcat_rows, tree_rows, n_with_any, n_fail, len(items)


# ---------------------------------------------------------------- stage: enumerate


def load_unique_captions(bobcat_train: str, tree_train: str) -> pl.DataFrame:
    frames = []
    for p in (bobcat_train, tree_train):
        if not Path(p).exists():
            logger.warning(f"Parquet not found, skipping: {p}")
            continue
        frames.append(pl.read_parquet(p, columns=["text_hash", "processed_text"]))
    df = pl.concat(frames).unique(subset=["text_hash"], keep="first")
    logger.info(f"{df.height} unique captions across the union of both parsers' train sets.")
    return df


def enumerate_stage(
    bobcat_train: str,
    tree_train: str,
    parts_dir: str,
    limit: int | None,
    max_workers: int,
    worker_batch_size: int,
    max_tasks_per_child: int,
    cache_path: str = BOBCAT_CACHE,
    num_shards: int = 1,
    shard_index: int = 0,
) -> None:
    """Resumable at BATCH granularity (not chunk/part): each worker_batch_size-sized
    batch gets its own two part files, written the moment that batch's result comes
    back — never accumulated across batches in the main process. If killed, restart
    with the same command; only the batch(es) in flight at kill time are redone.

    num_shards/shard_index: batch indices are GLOBAL (computed over the full sorted
    caption list regardless of sharding), so multiple independent processes/jobs can
    safely run concurrently against the SAME parts_dir — each only ever touches
    batches where `bi % num_shards == shard_index`, so there is no possibility of two
    shards writing the same file. Re-running a shard (or all shards) after a failure
    is just resubmitting the same command again — already-written batches (from this
    shard or any other) are skipped exactly as in the non-sharded case."""
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index ({shard_index}) must be in [0, num_shards={num_shards})")

    df = load_unique_captions(bobcat_train, tree_train).sort("text_hash")
    if limit:
        df = df.head(limit)

    parts_path = Path(parts_dir)
    parts_path.mkdir(parents=True, exist_ok=True)

    items = list(df.iter_rows())
    all_batches = [items[i : i + worker_batch_size] for i in range(0, len(items), worker_batch_size)]
    n_batches = len(all_batches)

    def _paths(bi: int) -> tuple[Path, Path]:
        return parts_path / f"part_bobcat_{bi:06d}.parquet", parts_path / f"part_tree_{bi:06d}.parquet"

    shard_batches = [(bi, batch) for bi, batch in enumerate(all_batches) if bi % num_shards == shard_index]
    pending = [(bi, batch) for bi, batch in shard_batches if not all(p.exists() for p in _paths(bi))]
    n_shard_total = len(shard_batches)
    n_skipped = n_shard_total - len(pending)
    logger.info(
        f"{len(items)} captions -> {n_batches} batches of ~{worker_batch_size} total "
        f"(shard {shard_index}/{num_shards}: {n_shard_total} batches assigned to this shard). "
        f"{n_skipped} of this shard's batches already done (resume), {len(pending)} remaining."
    )

    t0 = time.time()
    n_with_any_total = 0
    n_fail_total = 0
    n_done = 0

    pool = mp.get_context("spawn").Pool(
        processes=max_workers,
        initializer=_worker_init,
        initargs=(cache_path, bobcat_train, tree_train),
        maxtasksperchild=max_tasks_per_child,
    )
    try:
        for bi, bobcat_rows, tree_rows, with_any, fail, n_items in pool.imap_unordered(
            _worker_process_batch, pending, chunksize=1
        ):
            bobcat_path, tree_path = _paths(bi)
            bobcat_part = pl.DataFrame(bobcat_rows, schema=_ROW_SCHEMA, orient="row")
            tree_part = pl.DataFrame(tree_rows, schema=_ROW_SCHEMA, orient="row")
            for part, path in ((bobcat_part, bobcat_path), (tree_part, tree_path)):
                tmp = path.with_suffix(".tmp.parquet")
                part.write_parquet(tmp)
                tmp.rename(path)

            n_with_any_total += with_any
            n_fail_total += fail
            n_done += n_items
            logger.info(
                f"batch {bi} done ({n_items} captions, {len(bobcat_rows)} bobcat + {len(tree_rows)} tree rows): "
                f"{n_done}/{len(pending) * worker_batch_size} so far (approx), "
                f"{time.time() - t0:.0f}s elapsed ({(time.time() - t0) / max(1, n_done):.2f}s/caption avg), "
                f"{n_with_any_total} with >=1 candidate, {n_fail_total} compile failures so far"
            )
    finally:
        pool.close()
        pool.join()

    for label, prefix in (("bobcat", "part_bobcat_"), ("tree_no_type", "part_tree_")):
        parts = sorted(parts_path.glob(f"{prefix}*.parquet"))
        if not parts:
            continue
        all_parts = pl.concat([pl.read_parquet(p) for p in parts])
        n_covered = all_parts["text_hash"].n_unique()
        n_obj = (all_parts["t"] == "obj").sum()
        n_attr = (all_parts["t"] == "attr").sum()
        logger.info(
            f"[{label}] Enumeration+compile done: {n_covered} captions with >=1 candidate, "
            f"obj={n_obj} attr={n_attr} ({time.time() - t0:.0f}s elapsed). "
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


def _score_one(parts_glob: str, output: str, device: str, h_by_pair: dict[tuple[str, str], float] | None) -> dict:
    parts_path = Path(PARTS_DIR)
    part_files = sorted(parts_path.glob(parts_glob))
    if not part_files:
        raise RuntimeError(f"No parts found matching {parts_path / parts_glob} — run the 'enumerate' stage first.")
    all_parts = pl.concat([pl.read_parquet(p) for p in part_files])

    if h_by_pair is None:
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
    return h_by_pair


def score_stage(output_bobcat: str, output_tree: str) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Score once over the union of both parsers' (w1, w2) pairs — hardness is a
    # pure function of the two words, independent of t or which parser the
    # candidate ended up scoped to (see plan doc "join key" correction).
    parts_path = Path(PARTS_DIR)
    all_files = sorted(parts_path.glob("part_bobcat_*.parquet")) + sorted(parts_path.glob("part_tree_*.parquet"))
    if not all_files:
        raise RuntimeError(f"No parts found in {parts_path} — run the 'enumerate' stage first.")
    union_pairs = {
        (w1, w2) for p in all_files for w1, w2 in pl.read_parquet(p, columns=["w1", "w2"]).unique().iter_rows()
    }
    h_by_pair = clip_word_similarities(union_pairs, device)

    _score_one("part_bobcat_*.parquet", output_bobcat, device, h_by_pair)
    _score_one("part_tree_*.parquet", output_tree, device, h_by_pair)


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate hard-negative training pairs (combined design).")
    ap.add_argument("stage", choices=["enumerate", "score"])
    ap.add_argument("--bobcat-train", default=BOBCAT_TRAIN)
    ap.add_argument("--tree-train", default=TREE_TRAIN)
    ap.add_argument("--parts-dir", default=PARTS_DIR)
    ap.add_argument("--output-bobcat", default=OUTPUT_BOBCAT)
    ap.add_argument("--output-tree", default=OUTPUT_TREE)
    ap.add_argument("--limit", type=int, default=None, help="Cap #captions (smoke test, enumerate stage only).")
    ap.add_argument("--max-workers", type=int, default=4)
    # Each worker restarts every max_tasks_per_child*worker_batch_size captions
    # (lambeq CCG compile memory leak mitigation, see project_lambeq_tree_memory_leak
    # memory). The original pipeline's tuning (1000 captions/worker-lifetime) was for
    # 1 diagram/caption; this script compiles up to ~7 (up to 6 obj + 6 attr swap
    # candidates x 2 diagram types), so the equivalent-safe budget rescales to
    # ~1000/7 ~= 700-1000 captions/worker-lifetime. 100*10=1000 sits at the top of
    # that range — loosened from 500 once job 7085501 confirmed maxvmem~38G against
    # a 120G budget (tmem=24G x 5), well under budget, so less margin here is fine.
    # worker_batch_size ALSO sets write/resume granularity (a kill loses at most
    # ~worker_batch_size*max_workers captions of in-flight work) AND, critically, the
    # batch INDEX/filename scheme (part_{bi}.parquet where bi is positional over
    # worker_batch_size-sized spans) — changing it makes existing part files from a
    # different worker_batch_size silently mismatched with the new batch boundaries
    # (same filename, different caption span), NOT just "less resumable". Deliberately
    # kept at 100 (reverted from a 400 experiment) so job 7085501's already-written
    # part files stay resumable rather than forcing a restart. Bump max_tasks_per_child
    # instead if only reload FREQUENCY needs adjusting without this hazard.
    ap.add_argument("--worker-batch-size", type=int, default=100)
    ap.add_argument("--max-tasks-per-child", type=int, default=10)
    ap.add_argument(
        "--cache-path", default=BOBCAT_CACHE, help="Bobcat parse diskcache dir (override for local testing)."
    )
    # Sharding: run N independent processes (e.g. separate SGE array-job tasks),
    # each handling batches where bi % num_shards == shard_index. Safe to run
    # concurrently against the same --parts-dir (global batch indices, no file
    # collisions possible) — but give each shard its OWN --cache-path when running
    # concurrently, since the diskcache write-race seen earlier was plausibly caused
    # by multiple processes sharing one cache dir concurrently.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    if args.stage == "enumerate":
        enumerate_stage(
            args.bobcat_train,
            args.tree_train,
            args.parts_dir,
            args.limit,
            args.max_workers,
            args.worker_batch_size,
            args.max_tasks_per_child,
            args.cache_path,
            args.num_shards,
            args.shard_index,
        )
    else:
        PARTS_DIR = args.parts_dir
        score_stage(args.output_bobcat, args.output_tree)
