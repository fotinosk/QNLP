"""
Build the final SVO-Probes train/val/test datasets from compiled atoms.

Drop rows containing a word with corpus frequency below WORD_FREQ_THRESHOLD
(originally 50, matching the paper; lowered to recover more training rows —
see the constant's comment), then split 60/20/20 with no POSITIVE-image overlap between
splits — an image that is some row's pos_image never appears as another
row's pos_image in a different split. Negative images ARE allowed to repeat
across splits.

This is deliberately looser than full pos+neg connectivity. SVO-Probes
negatives are frequently *borrowed* from elsewhere in the dataset (a
subject/object-swap negative is typically some other row's positive image),
so grouping by full connectivity chains almost the entire corpus into one
giant component via these borrowed images — verified empirically: 65% of
all post-filter rows collapsed into a single component, and because
subject/object-swap negatives account for a disproportionate share of that
chaining, val/test were left with only 1-5 subj_neg/obj_neg examples each,
too few to report a meaningful per-subset breakdown. Splitting on
pos_image_id alone keeps each split's subj/verb/obj proportions close to
the full corpus's, at the cost of a mild, common-in-practice leakage: a
model may see an image as a training positive and later see the same image
again as an eval negative (it can only help correctly reject that negative,
since embeddings for it were already learned well).

Outputs (data/datasets/):
    svo_train.parquet       — sample_id, local_image_path, processed_text,
                               text_hash, diagram, symbols, path (positive
                               pairs only; SingleCaptionStrategy shape)
    svo_val_probes.parquet  — sample_id, true_local_image_path,
    svo_test_probes.parquet   false_local_image_path, diagram, symbols, path,
                               subj_neg, verb_neg, obj_neg (SVO-Probes eval shape)

Usage:
    python -m qnlp.scripts.svo.prepare_datasets
"""

import re
from collections import Counter

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.dataset_creator.dataset_generator import enrich_atoms, split_by_groups
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="svo_prepare_datasets")

WORD_FREQ_THRESHOLD = 10
# Was 50, matching the original paper. Every clean training run has landed
# at chance regardless of architecture/loss/hyperparameters (see
# SVO_EXPERIMENTS.md), and SVO's training set is ~7x smaller than ARO's
# (the only config with a validated result). Lowering to 10 recovers 14,284
# of 17,782 enriched rows (vs 9,107 at 50) — a real ~57% increase in
# available training data, while each surviving word still occurs at least
# 10 times (SVO's vocabulary is far smaller than COCO's, so this is a much
# milder relaxation than the number alone suggests).
SPLIT_RATIOS = (0.6, 0.2, 0.2)
SPLIT_SEED = 42
_WORD_RE = re.compile(r"[a-z']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _filter_by_word_frequency(atoms: pl.DataFrame, threshold: int) -> pl.DataFrame:
    counts: Counter[str] = Counter()
    for text in atoms["processed_text"].to_list():
        counts.update(_words(text))

    def _all_words_frequent(text: str) -> bool:
        return all(counts[w] >= threshold for w in _words(text))

    before = len(atoms)
    atoms = atoms.filter(pl.col("processed_text").map_elements(_all_words_frequent, return_dtype=pl.Boolean))
    logger.info(f"Word-frequency filter (<{threshold} occurrences): {before} -> {len(atoms)} rows")
    return atoms


def _assign_image_groups(atoms: pl.DataFrame) -> pl.DataFrame:
    """Group atoms by pos_image_id only — see module docstring for why this is
    looser than full pos+neg connectivity."""
    return atoms.with_columns(pl.col("pos_image_id").cast(pl.String).alias("image_group"))


def _build_train_split(atoms: pl.DataFrame) -> pl.DataFrame:
    has_path = "path" in atoms.columns
    cols = ["sample_id", "pos_local_image_path", "processed_text", "text_hash", "diagram", "symbols"] + (
        ["path"] if has_path else []
    )
    return atoms.select(cols).rename({"pos_local_image_path": "local_image_path"})


def _build_probes_split(atoms: pl.DataFrame) -> pl.DataFrame:
    has_path = "path" in atoms.columns
    cols = (
        ["sample_id", "pos_local_image_path", "neg_local_image_path", "diagram", "symbols"]
        + (["path"] if has_path else [])
        + ["subj_neg", "verb_neg", "obj_neg"]
    )
    return atoms.select(cols).rename(
        {"pos_local_image_path": "true_local_image_path", "neg_local_image_path": "false_local_image_path"}
    )


def run() -> None:
    derived_dir = constants.atlases_path / "svo" / constants.derived_name
    atoms = enrich_atoms([derived_dir], lmdb_path=constants.lmdb_path, compute_contraction_paths=True)

    atoms = _filter_by_word_frequency(atoms, WORD_FREQ_THRESHOLD)
    atoms = _assign_image_groups(atoms)

    train_atoms, val_atoms, test_atoms = split_by_groups(
        atoms, ratios=SPLIT_RATIOS, seed=SPLIT_SEED, group_column="image_group"
    )

    # Assert no POSITIVE image leaks across splits (negative images may repeat).
    def _pos_image_ids(split: pl.DataFrame) -> set[str]:
        return set(split["pos_image_id"].cast(pl.String).to_list())

    train_imgs, val_imgs, test_imgs = (
        _pos_image_ids(train_atoms),
        _pos_image_ids(val_atoms),
        _pos_image_ids(test_atoms),
    )
    assert not (train_imgs & val_imgs), "train/val positive-image overlap"
    assert not (train_imgs & test_imgs), "train/test positive-image overlap"
    assert not (val_imgs & test_imgs), "val/test positive-image overlap"

    datasets_path = constants.datasets_path
    datasets_path.mkdir(parents=True, exist_ok=True)

    # Positive-pairs / train-shape parquets: svo_train.parquet is what the model
    # trains on; svo_val.parquet / svo_test.parquet are the same shape, needed
    # because Trainer requires a val_loader (epoch monitoring/early stopping)
    # and a test_loader in the exact batch structure SimpleCaptionStep expects
    # (local_image_path + caption) — the real SVO-Probes accuracy is reported
    # separately via evaluate_svo on the *_probes.parquet files below.
    for split_name, split_atoms in [("train", train_atoms), ("val", val_atoms), ("test", test_atoms)]:
        out = _build_train_split(split_atoms)
        out_path = datasets_path / f"svo_{split_name}.parquet"
        out.write_parquet(out_path)
        logger.info(f"svo_{split_name}.parquet: {len(out)} rows")

    # Eval-shape parquets (true/false image pairs): the actual SVO-Probes
    # benchmark accuracy, computed for val (mid-training sanity check) and test
    # (final reported number). Also built for train — the ARO-style hard-negative
    # training step (SVOHardNegStep) trains directly on this shape, matching the
    # legacy ARO pipeline's explicit-triplet training rather than in-batch-only.
    for split_name, split_atoms, split_imgs in [
        ("train", train_atoms, train_imgs),
        ("val", val_atoms, val_imgs),
        ("test", test_atoms, test_imgs),
    ]:
        out = _build_probes_split(split_atoms)
        out_path = datasets_path / f"svo_{split_name}_probes.parquet"
        out.write_parquet(out_path)
        logger.info(f"svo_{split_name}_probes.parquet: {len(out)} rows ({len(split_imgs)} unique positive images)")


if __name__ == "__main__":
    run()
