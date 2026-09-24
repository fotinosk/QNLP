"""
PAPER_EXPERIMENTS_PLAN.md's Step 2, phase 1: measure how much of Winoground
is reachable with ARO's trained vocabulary, and classify every pair.

A word is "covered" only if BOTH its base word AND its exact CCG-type
usage already exist in ARO's vocabulary -- reusing a trained parameter
requires the same (word, piece-index, type) symbol, not just the same
word (see CustomMPSAnsatz._split_ar: a word's MPS chain pieces are named
f"{word}_{i}__{type_suffix}", and EinsumModel.sym2weight looks up by the
full Symbol, not by word alone). Three outcomes per blocking word:

- covered: (word, type) already in ARO's vocabulary.
- type_mismatch: the WORD exists in ARO's vocabulary under some type(s),
  but never under this exact type -- e.g. a noun usage of a word ARO only
  ever saw as a verb. Per PAPER_EXPERIMENTS_PLAN.md's D3, these are
  DROPPED, not repaired: initialising the new-type symbol from the word's
  other-type tensor asserts a word means the same thing regardless of
  grammatical role, which a DisCoCat model denies by construction.
- new_word: the word never appears in ARO's vocabulary under any type --
  a genuine gap, and the only kind of blocker substitution can address
  (Rule 2: only if a real in-vocabulary synonym exists, never forced).

A PAIR (not a single caption) is classified by the union of blockers
across both its captions, since Winoground's design requires both
captions to parse under the substitution together (Rule 1: substitute
per pair, never per caption -- else the shared bag-of-words the
benchmark is built on breaks).

    untouched:     no blockers at all in either caption.
    dropped_type:  at least one blocker is a type_mismatch (undroppable).
    substitutable: every blocker is a new_word (no type_mismatch present).

Usage:
    python -m qnlp.scripts.winoground.coverage_analysis
"""

import re
from collections import defaultdict

import orjson
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import enrich_atoms
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="winoground_coverage")

_NAME_RE = re.compile(r"^(?P<word>.+?)_(?P<idx>\d+)(?:__(?P<type>.+))?$")


def _parse_symbol_name(name: str) -> tuple[str, str] | None:
    """Returns (word, type_suffix). type_suffix is "" for an untyped
    single-piece symbol (no __ suffix at all)."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    return m.group("word").lower(), (m.group("type") or "")


def _symbols_to_word_types(raw_symbols: str | bytes | None) -> set[tuple[str, str]]:
    if raw_symbols is None:
        return set()
    payload = orjson.loads(raw_symbols)
    out = set()
    for entry in payload:
        name = entry[0]["name"]
        parsed = _parse_symbol_name(name)
        if parsed:
            out.add(parsed)
    return out


def _build_aro_vocab() -> dict[str, set[str]]:
    """word -> set of type_suffixes ARO's trained vocabulary covers that
    word under, pooled across train/val/test (whatever a run's
    collect_symbol_sizes would see)."""
    vocab: dict[str, set[str]] = defaultdict(set)
    for split in ["train", "val", "test"]:
        path = constants.datasets_path / f"aro_{split}.parquet"
        df = pl.read_parquet(path)
        for col in ["true_symbols", "false_symbols"]:
            for raw in df[col].to_list():
                for word, type_suffix in _symbols_to_word_types(raw):
                    vocab[word].add(type_suffix)
    logger.info(f"ARO vocabulary: {len(vocab)} distinct words")
    return dict(vocab)


def _load_winoground_pair_symbols() -> dict[str, list[set[tuple[str, str]]]]:
    """pair sample_id ('winoground_<id>') -> [caption_0 word/types, caption_1 word/types].
    Derived rows are flattened per-caption with a __0/__1 suffix on sample_id
    (see WinogroundPipeline)."""
    derived_dir = constants.atlases_path / "winoground" / "derived_v1"
    # filter_2d_outputs=True (the default) matches what real training/eval
    # actually uses -- a pair whose diagram never reduces to rank-1 isn't
    # usable regardless of vocabulary coverage, so it shouldn't inflate or
    # deflate this measurement either way.
    atoms = enrich_atoms([derived_dir], compute_contraction_paths=False)

    pairs: dict[str, dict[int, set[tuple[str, str]]]] = defaultdict(dict)
    for row in atoms.iter_rows(named=True):
        sample_id = row["sample_id"]
        if "__" not in sample_id:
            logger.warning(f"Unexpected sample_id format (no __N suffix): {sample_id}")
            continue
        pair_id, caption_idx = sample_id.rsplit("__", 1)
        pairs[pair_id][int(caption_idx)] = _symbols_to_word_types(row["symbols"])

    out: dict[str, list[set[tuple[str, str]]]] = {}
    n_incomplete = 0
    for pair_id, captions in pairs.items():
        if 0 not in captions or 1 not in captions:
            n_incomplete += 1
            continue
        out[pair_id] = [captions[0], captions[1]]
    if n_incomplete:
        logger.warning(f"{n_incomplete} pairs have fewer than 2 successfully compiled captions -- excluded entirely.")
    return out


def run() -> None:
    aro_vocab = _build_aro_vocab()
    wino_pairs = _load_winoground_pair_symbols()
    logger.info(f"Winoground pairs with both captions compiled: {len(wino_pairs)}")

    untouched: list[str] = []
    dropped_type: list[str] = []
    substitutable: list[str] = []
    substitutable_new_words: dict[str, set[str]] = {}
    dropped_type_words: dict[str, set[str]] = {}

    for pair_id, (caps0, caps1) in wino_pairs.items():
        all_word_types = caps0 | caps1
        blockers_new_word: set[str] = set()
        blockers_type_mismatch: set[str] = set()

        for word, type_suffix in all_word_types:
            known_types = aro_vocab.get(word)
            if known_types is None:
                blockers_new_word.add(word)
            elif type_suffix not in known_types:
                blockers_type_mismatch.add(word)
            # else: covered, no action

        if not blockers_new_word and not blockers_type_mismatch:
            untouched.append(pair_id)
        elif blockers_type_mismatch:
            dropped_type.append(pair_id)
            dropped_type_words[pair_id] = blockers_type_mismatch
        else:
            substitutable.append(pair_id)
            substitutable_new_words[pair_id] = blockers_new_word

    total = len(wino_pairs)
    logger.info("=" * 72)
    logger.info("Winoground / ARO vocabulary coverage")
    logger.info("=" * 72)
    logger.info(f"  total pairs (both captions compiled): {total}")
    logger.info(f"  untouched     : {len(untouched):4d} ({100 * len(untouched) / total:.1f}%)")
    logger.info(f"  dropped_type  : {len(dropped_type):4d} ({100 * len(dropped_type) / total:.1f}%)")
    logger.info(f"  substitutable : {len(substitutable):4d} ({100 * len(substitutable) / total:.1f}%)")

    all_new_word_stems: set[str] = set()
    for words in substitutable_new_words.values():
        all_new_word_stems |= words
    logger.info(f"  distinct new-word stems across substitutable pairs: {len(all_new_word_stems)}")

    all_type_mismatch_stems: set[str] = set()
    for words in dropped_type_words.values():
        all_type_mismatch_stems |= words
    top_blockers = sorted(
        ((w, sum(w in ws for ws in dropped_type_words.values())) for w in all_type_mismatch_stems),
        key=lambda x: -x[1],
    )[:15]
    logger.info("  top dropped-type blocking stems:")
    for w, n in top_blockers:
        logger.info(f"    {w:<20} blocks {n} pairs")

    out_path = constants.datasets_path / "winoground_coverage_analysis.parquet"
    out_df = pl.DataFrame(
        [
            {
                "pair_id": pid,
                "classification": "untouched",
                "blocking_words": "",
            }
            for pid in untouched
        ]
        + [
            {
                "pair_id": pid,
                "classification": "dropped_type",
                "blocking_words": ",".join(sorted(dropped_type_words[pid])),
            }
            for pid in dropped_type
        ]
        + [
            {
                "pair_id": pid,
                "classification": "substitutable",
                "blocking_words": ",".join(sorted(substitutable_new_words[pid])),
            }
            for pid in substitutable
        ]
    )
    out_df.write_parquet(out_path)
    logger.info(f"Wrote per-pair classification to {out_path}")


if __name__ == "__main__":
    run()
