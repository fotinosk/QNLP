"""
Post-parse vocabulary consolidation: relabels each compiled symbol's word
stem to its WordNet base form (e.g. "holds" -> "hold", "children" -> "child"),
WITHOUT touching the CCG parse or diagram structure at all.

DISCOCLIP_REPRODUCTION_PLAN.md's structural comparison found that the
reference lemmatizes purely for symbol/vocabulary identity -- it parses the
raw, unlemmatised sentence, then relabels the resulting tree's leaves with
lemmas afterward. This is NOT the same operation as this project's
`LemmatizeStep` (which grammatically finitises verb forms *before* parsing,
specifically to guarantee the CCG parser resolves to a rank-1 "S" type --
removing it would risk breaking that guarantee, which `UnifyEinsumRankStep`
exists to catch as a safety net, not to rely on as the primary mechanism).

This step is the correct analogue of the reference's post-parse lemma
substitution: it runs AFTER CCGCompilerStep, in addition to (not instead
of) LemmatizeStep, and only affects vocabulary size/symbol identity.

Safe because symbol NAMES are looked up by string identity for parameter
sharing across rows, but the einsum expression only ever references
symbols by their position in the row's own symbols list -- renaming a
symbol's word stem never requires touching the diagram string.
"""

import orjson
import polars as pl
from nltk.stem import WordNetLemmatizer

from qnlp.core.data_engine.processing.pipeline import PipelineStep
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="symbol_lemmatize_step")

_lemmatizer = WordNetLemmatizer()


def _lemmatize_word(word: str) -> str:
    """Try verb-form reduction first (catches "holds"/"held"/"holding" ->
    "hold"), then noun-form reduction (catches "dogs" -> "dog") -- a simple,
    POS-agnostic heuristic: WordNetLemmatizer leaves a word unchanged if it
    doesn't recognise it under the given POS, so trying both in sequence
    and keeping whichever one actually changed something is safe."""
    w = word.lower()
    v = _lemmatizer.lemmatize(w, pos="v")
    if v != w:
        return v
    return _lemmatizer.lemmatize(w, pos="n")


def _relabel_symbols(raw: bytes | None) -> bytes | None:
    if raw is None:
        return None
    payload = orjson.loads(raw)
    symbols = payload.get("symbols")
    if payload.get("error") or symbols is None:
        return raw
    for entry in symbols:
        sym_dict = entry[0]
        name = sym_dict.get("name", "")
        if "_" not in name:
            continue
        base, rest = name.split("_", 1)
        lemma = _lemmatize_word(base)
        if lemma != base:
            sym_dict["name"] = f"{lemma}_{rest}"
    return orjson.dumps(payload)


class SymbolLemmatizeStep(PipelineStep):
    """Relabels compiled symbols' word stems to their WordNet base form.
    Must run after CCGCompilerStep (needs `compiled_bytes` to already
    exist)."""

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info(f"Starting SymbolLemmatizeStep for chunk of size {len(df)}")
        if "compiled_bytes" not in df.columns:
            return df
        return df.with_columns(
            pl.col("compiled_bytes").map_elements(_relabel_symbols, return_dtype=pl.Binary).alias("compiled_bytes")
        )
