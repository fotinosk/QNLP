"""
Build the SVO-Swap evaluation set: pairs where the subject and object of an
SVO-Probes caption are swapped, restricted to cases where both refer to a
human or animal (so the swap stays semantically well-formed).

Subject/object identification uses the raw Bobcat CCG derivation tree (not
the corrected-CSV `subj`/`obj` columns, which are single-word extractions
that may not align with the actual constituent span) — for a simple
transitive clause the tree always decomposes as:

    S            (root, via {FORWARD,BACKWARD}_APPLICATION)
    +-- NP           <- subject span
    +-- S\\NP
         +-- (S\\NP)/NP   (verb)
         +-- NP           <- object span

The CSV's `subj`/`obj` single-word columns are still used for the human/animal
WordNet check (they're already the lemmatized head noun).

Output: data/datasets/svo_swap_eval.parquet in the SugarCREPE-compatible
schema (sample_id, local_image_path, true_diagram, true_symbols, true_path,
false_diagram, false_symbols, false_path) — reusable by evaluate_sugarcrepe.

Usage:
    python -m qnlp.scripts.svo.build_svo_swap
"""

import re

import orjson
import polars as pl
from lambeq import AtomicType, BobcatParser, Rewriter
from lambeq.backend.tensor import Dim
from nltk.corpus import wordnet as wn

from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.non_linear_contraction.determine_optimal_contraction_path import get_contraction_path_and_cost
from qnlp.discoviz.models.bobcat_text_processor import BobcatTextProcessor
from qnlp.discoviz.parser.asnsatz import CustomMPSAnsatz
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="build_svo_swap")

CCG_RULES = [
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
_TRAILING_DOTS_RE = re.compile(r"\.+$")


# ---------------------------------------------------------------------------
# Subject/object span identification via the raw CCG tree
# ---------------------------------------------------------------------------


def _find_subj_np_and_vp(node):
    """Descend through unary/leaf wrapping to find an S node that splits into
    an NP (subject) and an S\\NP (verb phrase). Returns (subj_node, vp_node)
    or None."""
    if node.is_leaf:
        return None
    if node.right is None:
        return _find_subj_np_and_vp(node.left)
    if (
        str(node.biclosed_type) == "s"
        and str(node.left.biclosed_type) == "np"
        and str(node.right.biclosed_type) == r"s\np"
    ):
        return node.left, node.right
    found = _find_subj_np_and_vp(node.left)
    if found:
        return found
    return _find_subj_np_and_vp(node.right)


def _find_object_np(vp_node):
    """Given an S\\NP node, find its NP (object) child. Returns node or None."""
    if vp_node.is_leaf or vp_node.right is None:
        return None
    if str(vp_node.right.biclosed_type) == "np":
        return vp_node.right
    return None


def find_subject_object_spans(parser: BobcatParser, text: str) -> tuple[str, str] | None:
    """Parse `text` and return (subject_span, object_span) surface text, or
    None if it isn't a simple transitive clause."""
    stripped = _TRAILING_DOTS_RE.sub("", text)
    try:
        tree = parser.sentence2tree(stripped)
    except Exception as e:
        logger.debug(f"Parse failed for '{text}': {e}")
        return None
    if tree is None:
        return None
    split = _find_subj_np_and_vp(tree)
    if split is None:
        return None
    subj_node, vp_node = split
    obj_node = _find_object_np(vp_node)
    if obj_node is None:
        return None
    return subj_node.text, obj_node.text


def swap_spans(text: str, span_a: str, span_b: str) -> str | None:
    """Swap the first occurrence of span_a with the first occurrence of span_b."""
    if span_a not in text or span_b not in text or span_a == span_b:
        return None
    placeholder = "\x00SWAP\x00"
    new_text = text.replace(span_a, placeholder, 1)
    new_text = new_text.replace(span_b, span_a, 1)
    new_text = new_text.replace(placeholder, span_b, 1)
    return new_text


# ---------------------------------------------------------------------------
# Human/animal check
# ---------------------------------------------------------------------------

_PERSON_SYNSET = wn.synset("person.n.01")
_ANIMAL_SYNSET = wn.synset("animal.n.01")


def _is_human_or_animal(word: str) -> bool:
    """Check only the first (most frequent) WordNet sense — checking all senses
    lets rare slang meanings through (e.g. "grass" as informer -> person.n.01,
    when captions almost always mean the plant)."""
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return False
    syn = synsets[0]
    hypernyms = {h for path in syn.hypernym_paths() for h in path}
    hypernyms.add(syn)
    return _PERSON_SYNSET in hypernyms or _ANIMAL_SYNSET in hypernyms


# ---------------------------------------------------------------------------
# Diagram compilation (mirrors CCGCompilerStep's worker, single-process)
# ---------------------------------------------------------------------------


def _build_text_processor() -> BobcatTextProcessor:
    ansatz = CustomMPSAnsatz(
        {
            AtomicType.SENTENCE: Dim(constants.embedding_dim),
            AtomicType.NOUN: Dim(constants.embedding_dim),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(constants.embedding_dim),
        },
        bond_dim=constants.bond_dim,
    )
    parser = BobcatParser(verbose="suppress")
    return BobcatTextProcessor(ccg_parser=parser, ansatz=ansatz, rewriter=Rewriter(CCG_RULES))


def _preprocess_text(text: str, remove_dots: RemoveTrailingDotsStep, lemmatize: LemmatizeStep) -> str | None:
    df = pl.DataFrame({"processed_text": [text]})
    df = remove_dots.process(df)
    df = lemmatize.process(df)
    out = df["processed_text"][0]
    return out if out else None


def _compile_diagram(text_processor: BobcatTextProcessor, text: str) -> tuple[str, str, str] | None:
    try:
        out = text_processor([text])
        diagram, symbol_size_pairs = out["einsum_inputs"][0]
    except Exception as e:
        logger.debug(f"Compile failed for '{text}': {e}")
        return None

    symbols = [[_asdict_symbol(sym), size] for sym, size in symbol_size_pairs]
    shapes = tuple(size for _, size in symbols)
    try:
        path, _largest = get_contraction_path_and_cost(diagram, shapes)
    except Exception as e:
        logger.debug(f"Contraction path failed for '{text}': {e}")
        return None

    return diagram, orjson.dumps(symbols).decode(), orjson.dumps(path).decode()


def _asdict_symbol(sym) -> dict:
    from dataclasses import asdict

    return asdict(sym)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    manifest = pl.read_parquet(constants.atlases_path / "svo" / "data_manifest.parquet")
    logger.info(f"Loaded {len(manifest)} SVO manifest rows.")

    parser = BobcatParser(verbose="suppress")
    text_processor = _build_text_processor()
    remove_dots = RemoveTrailingDotsStep(text_column="processed_text")
    lemmatize = LemmatizeStep(text_column="processed_text")

    rows = []
    n_candidates = 0
    n_swapped = 0
    seen_sentences: dict[str, str | None] = {}  # corrected_sentence -> swapped sentence (memoise parse+swap)

    for row in manifest.iter_rows(named=True):
        sentence = row["corrected_sentence"]
        subj_word = (row.get("subj") or "").lower()
        obj_word = (row.get("obj") or "").lower()
        if not subj_word or not obj_word:
            continue
        if not (_is_human_or_animal(subj_word) and _is_human_or_animal(obj_word)):
            continue
        n_candidates += 1

        if sentence in seen_sentences:
            swapped = seen_sentences[sentence]
        else:
            spans = find_subject_object_spans(parser, sentence)
            swapped = None
            if spans is not None:
                subj_span, obj_span = spans
                swapped = swap_spans(sentence, subj_span, obj_span)
            seen_sentences[sentence] = swapped

        if swapped is None:
            continue

        true_text = _preprocess_text(sentence, remove_dots, lemmatize)
        false_text = _preprocess_text(swapped, remove_dots, lemmatize)
        if not true_text or not false_text:
            continue

        true_compiled = _compile_diagram(text_processor, true_text)
        false_compiled = _compile_diagram(text_processor, false_text)
        if true_compiled is None or false_compiled is None:
            continue

        true_diagram, true_symbols, true_path = true_compiled
        false_diagram, false_symbols, false_path = false_compiled

        rows.append(
            {
                "sample_id": row["sample_id"],
                "local_image_path": row["pos_local_image_path"],
                "true_diagram": true_diagram,
                "true_symbols": true_symbols,
                "true_path": true_path,
                "false_diagram": false_diagram,
                "false_symbols": false_symbols,
                "false_path": false_path,
            }
        )
        n_swapped += 1

    logger.info(f"Human/animal subj+obj candidates: {n_candidates}")
    logger.info(f"Successfully swapped + compiled: {n_swapped}")

    out = pl.DataFrame(rows)
    out_path = constants.datasets_path / "svo_swap_eval.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    logger.info(f"svo_swap_eval.parquet written to {out_path} ({len(out)} rows).")


if __name__ == "__main__":
    run()
