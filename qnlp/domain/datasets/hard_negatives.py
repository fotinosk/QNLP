import math
import random
from collections import defaultdict

import polars as pl

from qnlp.domain.datasets.dataset import _deserialize_symbols


class HardNegativeBank:
    """Loads a companion hard-negatives parquet (produced by
    qnlp/scripts/coco_multi_caption/generate_hard_negatives.py) and samples a
    training-time negative caption per positive row, keyed by the positive's
    text_hash.

    ONLY consult this from the TRAIN forward/loss step. text_hash is a hash of
    the caption STRING (not of (image, caption)), so a generic caption that
    happens to also appear verbatim in val/test would share a text_hash with a
    train row — harmless ONLY because this lookup is never called from eval
    code paths (see HARD_NEG_PI_SWEEP_PLAN.md "join key" note).
    """

    def __init__(self, parquet_path: str, h_max: float = 0.95, softmax_temp: float = 0.5):
        self.df = pl.read_parquet(parquet_path)
        if h_max is not None:
            self.df = self.df.filter(pl.col("h") <= h_max)
        self.softmax_temp = softmax_temp

        self._by_hash: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for text_hash, diagram, symbols, h in self.df.select("text_hash", "diagram", "symbols", "h").iter_rows():
            self._by_hash[text_hash].append((diagram, symbols, h))

    def sample(self, text_hash: str, pi: float, rng: random.Random) -> tuple[str, list] | None:
        """With probability pi, return one (diagram, [Symbol,...]) negative for
        this text_hash, sampled with probability proportional to
        exp((h - mean_h) / softmax_temp) over that caption's own candidates
        (favors locally-harder candidates for THIS caption, not a global rank).
        Returns None if pi doesn't fire, or this text_hash has no candidates
        (e.g. it was excluded by the per-parser conditional-compile scoping, or
        by h_max)."""
        if pi <= 0.0 or rng.random() >= pi:
            return None
        candidates = self._by_hash.get(text_hash)
        if not candidates:
            return None
        hs = [c[2] for c in candidates]
        mean_h = sum(hs) / len(hs)
        weights = [math.exp((h - mean_h) / self.softmax_temp) for h in hs]
        total = sum(weights)
        idx = rng.choices(range(len(candidates)), weights=[w / total for w in weights], k=1)[0]
        diagram, symbols_raw, _h = candidates[idx]
        return diagram, _deserialize_symbols(symbols_raw)


class _SymbolSource:
    """Minimal duck-typed wrapper so collect_symbol_sizes (which only reads
    `ds.df[col]`) can register symbols from an arbitrary parquet's DataFrame,
    e.g. a HardNegativeBank's, without going through VLMDataset's image/
    compiled-column machinery (irrelevant here — we never call __getitem__)."""

    def __init__(self, df: pl.DataFrame):
        self.df = df
