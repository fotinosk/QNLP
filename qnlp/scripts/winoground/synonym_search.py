"""
PAPER_EXPERIMENTS_PLAN.md's Step 2, phase 2 (Rule 2): for every new-word
stem blocking a "substitutable" Winoground pair, propose ranked
in-vocabulary synonym candidates for manual review.

This is a CANDIDATE proposal step, not a final decision -- semantic
nearness (CLIP text embeddings) is a good filter for what to look at
first, but it does not guarantee the candidate will actually parse into
a CCG-compatible type once substituted (Rule 3: substitute -> recompile
-> re-check symbol-level coverage is a separate, later loop). Per Rule 2,
no candidate is ever auto-applied: every row here needs a human to
confirm the semantic substitution is actually valid before it's used,
and a blocking word with no acceptable candidate is left unsubstituted
rather than forced.

Candidates are drawn from ARO's actual trained vocabulary (the same word
list built by coverage_analysis.py's _build_aro_vocab), ranked by cosine
similarity between frozen CLIP text embeddings of the two words in
isolation (e.g. "kiss" vs "hug", not embedded in a sentence context).

Usage:
    python -m qnlp.scripts.winoground.synonym_search
"""

import polars as pl
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from qnlp.constants import constants
from qnlp.scripts.winoground.coverage_analysis import _build_aro_vocab
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="winoground_synonym_search")

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_K = 8


def _embed_words(words: list[str], model: CLIPModel, tokenizer: CLIPTokenizer, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(words, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_text_features(**tokens)
    return F.normalize(feats, p=2, dim=-1)


def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)

    aro_vocab = _build_aro_vocab()
    aro_words = sorted(aro_vocab.keys())
    logger.info(f"ARO vocabulary: {len(aro_words)} candidate words")

    coverage = pl.read_parquet(constants.datasets_path / "winoground_coverage_analysis.parquet")
    substitutable = coverage.filter(pl.col("classification") == "substitutable")

    # Count how many pairs each blocking word affects, to prioritise review.
    pair_counts: dict[str, int] = {}
    for row in substitutable.iter_rows(named=True):
        for word in row["blocking_words"].split(","):
            pair_counts[word] = pair_counts.get(word, 0) + 1
    blocking_words = sorted(pair_counts, key=lambda w: -pair_counts[w])
    logger.info(f"Distinct blocking stems: {len(blocking_words)}")

    aro_embeddings = _embed_words(aro_words, model, tokenizer, device)
    blocking_embeddings = _embed_words(blocking_words, model, tokenizer, device)

    sims = blocking_embeddings @ aro_embeddings.T  # [n_blocking, n_aro]
    topk_sims, topk_idx = sims.topk(k=min(TOP_K, len(aro_words)), dim=-1)

    rows = []
    for i, word in enumerate(blocking_words):
        for rank in range(topk_sims.shape[1]):
            candidate = aro_words[topk_idx[i, rank].item()]
            rows.append(
                {
                    "blocking_word": word,
                    "n_pairs_blocked": pair_counts[word],
                    "rank": rank + 1,
                    "candidate": candidate,
                    "cosine_sim": topk_sims[i, rank].item(),
                }
            )

    out = pl.DataFrame(rows)
    out_path = constants.datasets_path / "winoground_synonym_candidates.parquet"
    out.write_parquet(out_path)
    logger.info(f"Wrote {len(out)} candidate rows ({len(blocking_words)} words x top-{TOP_K}) to {out_path}")

    logger.info("=" * 72)
    logger.info("Top candidates for the most-blocking stems (for manual review)")
    logger.info("=" * 72)
    for word in blocking_words[:20]:
        cands = out.filter(pl.col("blocking_word") == word).sort("rank")
        cand_str = ", ".join(f"{r['candidate']}({r['cosine_sim']:.2f})" for r in cands.iter_rows(named=True))
        logger.info(f"  {word:<15} [{pair_counts[word]:>2} pairs]  {cand_str}")


if __name__ == "__main__":
    run()
