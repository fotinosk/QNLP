"""
Create coco_aro_style {train,val,test} datasets with TF-IDF hard negatives.

For each caption, the top_k most lexically similar captions from DIFFERENT
images are found via TF-IDF cosine similarity. One is sampled uniformly from
this pool as the hard negative. The output follows the ARO contrastive-pair
schema and plugs directly into the AROContrastiveStep / ContrastiveLoss
training loop with no further changes.

This directly addresses the root failure of previous COCO training runs:
random-derangement negatives are too easy (overlapping descriptions from
similar scenes give contradictory gradients). TF-IDF negatives share
vocabulary with the true caption, forcing the model to learn fine-grained
semantic distinctions.

Three strategies are available via --strategy:
  bm25_hard   (default) sample from top-10 most similar
  bm25_medium sample from top-50 most similar (easier)
  random      pure random derangement (baseline for ablation)

Output:
    data/datasets/coco_aro_style_train.parquet
    data/datasets/coco_aro_style_val.parquet
    data/datasets/coco_aro_style_test.parquet

Schema (ARO-compatible):
    sample_id, local_image_path,
    true_diagram, true_symbols, true_path,
    false_diagram, false_symbols, false_path

Usage:
    python -m qnlp.scripts.coco_aro_style.create_dataset
    python -m qnlp.scripts.coco_aro_style.create_dataset --top_k 5 --strategy bm25_hard
"""

import argparse
from collections import defaultdict

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from qnlp.constants import constants
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="coco_aro_style_create_dataset")

DATASETS_PATH = constants.datasets_path
SOURCE_PREFIX = "coco_single_caption"
OUTPUT_PREFIX = "coco_aro_style"

TFIDF_MAX_FEATURES = 30_000
BATCH_SIZE = 5_000


def _random_derangement(sample_ids: list[str], rng: np.random.Generator) -> list[int]:
    """Assign each row a random index from a DIFFERENT sample_id (derangement over groups)."""
    unique_sids = list(dict.fromkeys(sample_ids))
    neg_sids = unique_sids.copy()
    while any(a == b for a, b in zip(unique_sids, neg_sids)):
        rng.shuffle(neg_sids)
    sid_to_neg = dict(zip(unique_sids, neg_sids))

    # For each row, pick a random index from the negative group
    sid_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(sample_ids):
        sid_to_indices[sid].append(i)

    chosen = []
    for sid in sample_ids:
        neg_sid = sid_to_neg[sid]
        chosen.append(int(rng.choice(sid_to_indices[neg_sid])))
    return chosen


def _tfidf_hard_negatives(
    df: pl.DataFrame,
    vectorizer: TfidfVectorizer,
    top_k: int,
    seed: int,
) -> list[int]:
    """
    For each row in df, find the top_k most TF-IDF-similar captions from
    DIFFERENT sample_ids and sample one uniformly as the hard negative.

    Uses NearestNeighbors(n_jobs=-1) to parallelise the brute-force search
    and avoid materialising the full N×N similarity matrix.

    Returns a list of row indices (one per row in df) pointing to the selected
    hard negative.
    """
    rng = np.random.default_rng(seed)
    texts = df["processed_text"].to_list()
    sample_ids = df["sample_id"].to_list()
    N = len(texts)

    # Map each sample_id → all row indices sharing that image
    sid_to_indices: dict[str, set[int]] = defaultdict(set)
    for i, sid in enumerate(sample_ids):
        sid_to_indices[sid].add(i)

    X = vectorizer.transform(texts)

    # Request enough neighbours that after filtering same-image rows there
    # are still at least top_k candidates. COCO has ~5 captions per image on
    # average, so top_k * 10 + 50 gives a comfortable margin.
    n_neighbors = min(top_k * 10 + 50, N - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)

    neg_indices: list[int] = []

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Hard negatives"):
        end = min(start + BATCH_SIZE, N)
        knn = nn.kneighbors(X[start:end], return_distance=False)  # [batch, n_neighbors]

        for local_i, global_i in enumerate(range(start, end)):
            sid = sample_ids[global_i]
            exclude = sid_to_indices[sid]

            candidates = [int(j) for j in knn[local_i] if j not in exclude][:top_k]

            if not candidates:
                candidates = [j for j in range(N) if j not in exclude]

            neg_indices.append(int(rng.choice(candidates)))

    return neg_indices


def _build_pairs(
    df: pl.DataFrame,
    vectorizer: TfidfVectorizer,
    strategy: str,
    top_k: int,
    seed: int,
) -> pl.DataFrame:
    """Pair each row with a hard (or random) negative and return an ARO-schema DataFrame."""
    if strategy == "random":
        rng = np.random.default_rng(seed)
        neg_indices = _random_derangement(df["sample_id"].to_list(), rng)
    else:
        neg_indices = _tfidf_hard_negatives(df, vectorizer, top_k=top_k, seed=seed)

    false_rows = df[neg_indices]
    return pl.DataFrame(
        {
            "sample_id": df["sample_id"].to_list(),
            "local_image_path": df["local_image_path"].to_list(),
            "true_diagram": df["diagram"].to_list(),
            "true_symbols": df["symbols"].to_list(),
            "true_path": df["path"].to_list(),
            "false_diagram": false_rows["diagram"].to_list(),
            "false_symbols": false_rows["symbols"].to_list(),
            "false_path": false_rows["path"].to_list(),
        }
    )


def run(top_k: int = 10, strategy: str = "bm25_hard", seed: int = 42) -> None:
    train_df = pl.read_parquet(DATASETS_PATH / f"{SOURCE_PREFIX}_train.parquet")
    val_df = pl.read_parquet(DATASETS_PATH / f"{SOURCE_PREFIX}_val.parquet")
    test_df = pl.read_parquet(DATASETS_PATH / f"{SOURCE_PREFIX}_test.parquet")

    logger.info(f"Strategy: {strategy}  top_k: {top_k}  seed: {seed}")
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Fit TF-IDF on train split only — shared tokenisation across splits
    logger.info(f"Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES}, ngram_range=(1,2))")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    vectorizer.fit(train_df["processed_text"].to_list())

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"--- {split} ({len(df)} rows) ---")
        pairs = _build_pairs(df, vectorizer, strategy=strategy, top_k=top_k, seed=seed)
        out_path = DATASETS_PATH / f"{OUTPUT_PREFIX}_{split}.parquet"
        pairs.write_parquet(out_path)
        logger.info(f"Written {len(pairs)} pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build COCO ARO-style dataset with TF-IDF hard negatives.")
    parser.add_argument(
        "--strategy",
        choices=["bm25_hard", "bm25_medium", "random"],
        default="bm25_hard",
        help="bm25_hard: top-10; bm25_medium: top-50; random: random derangement (baseline)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override top-K pool size (default: 10 for bm25_hard, 50 for bm25_medium)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    top_k = args.top_k
    if top_k is None:
        top_k = {"bm25_hard": 10, "bm25_medium": 50, "random": 1}[args.strategy]

    run(top_k=top_k, strategy=args.strategy, seed=args.seed)
