"""
Build the final SVO-Probes train/val/test datasets from compiled atoms.

Mirrors the paper's preprocessing: drop rows containing a word with corpus
frequency <50, then split 60/20/20 with NO image overlap between splits.
"Image overlap" is enforced across pos_image_id AND neg_image_id together —
an atom's two images are unioned into the same connected component via
union-find, so an image can never appear as a positive in one split and a
negative in another.

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

WORD_FREQ_THRESHOLD = 50
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


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def _assign_image_groups(atoms: pl.DataFrame) -> pl.DataFrame:
    """Group atoms by connected component over (pos_image_id, neg_image_id) pairs
    so no image can straddle a split boundary."""
    uf = _UnionFind()
    pos_ids = atoms["pos_image_id"].cast(pl.String).to_list()
    neg_ids = atoms["neg_image_id"].cast(pl.String).to_list()
    for p, n in zip(pos_ids, neg_ids):
        uf.union(p, n)

    groups = [uf.find(p) for p in pos_ids]
    return atoms.with_columns(pl.Series("image_group", groups))


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

    # Assert no image leaks across splits.
    def _image_ids(split: pl.DataFrame) -> set[str]:
        return set(split["pos_image_id"].cast(pl.String).to_list()) | set(
            split["neg_image_id"].cast(pl.String).to_list()
        )

    train_imgs, val_imgs, test_imgs = _image_ids(train_atoms), _image_ids(val_atoms), _image_ids(test_atoms)
    assert not (train_imgs & val_imgs), "train/val image overlap"
    assert not (train_imgs & test_imgs), "train/test image overlap"
    assert not (val_imgs & test_imgs), "val/test image overlap"

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
    # (final reported number).
    for split_name, split_atoms, split_imgs in [("val", val_atoms, val_imgs), ("test", test_atoms, test_imgs)]:
        out = _build_probes_split(split_atoms)
        out_path = datasets_path / f"svo_{split_name}_probes.parquet"
        out.write_parquet(out_path)
        logger.info(f"svo_{split_name}_probes.parquet: {len(out)} rows ({len(split_imgs)} unique images)")


if __name__ == "__main__":
    run()
