"""Phase 0: unify bobcat/tree_no_type train/val/test splits (see
HARD_NEG_PI_SWEEP_PLAN.md "NEW PHASE 0"). Bobcat and tree_no_type currently
have DIFFERENT splits (~59% train-set overlap) because the atlas was ingested
incrementally between the two builds — same split function, same seed, but
over different-sized universes at build time. This isn't a bug, just a
consequence, but it's a methodological confound for the whole pi-sweep:
bobcat-vs-tree cells train on different image pools, not just different
parsers.

Verified (read-only, see plan doc): bobcat's full pool (463,075 captions) is a
STRICT SUBSET of tree's (542,040) — every bobcat caption already exists
somewhere in tree's data too. So a matched split needs zero re-parsing or
re-compiling — purely a reshuffle+reslice of existing rows:

  1. Concat bobcat's own train+val+test -> bobcat's full pool (the common pool
     by construction, since it's the smaller/limiting side).
  2. Run split_by_groups ONCE on that pool (group_column="sample_id", the
     existing (0.8, 0.1, 0.1)/seed=42 convention, CALLING the existing function
     directly rather than reimplementing its shuffle logic, to avoid any
     reproducibility risk from a subtly different reimplementation) -> this
     directly IS bobcat's matched output (no further reslicing needed for
     bobcat itself).
  3. Build a sample_id -> split map from that result.
  4. Concat tree's own train+val+test, filter to text_hash values present in
     the common pool (drops the 78,965 tree-only captions, from ALL three
     splits, not just train), then join against the sample_id -> split map to
     assign each remaining row to train/val/test — same split as bobcat's
     matching image, by construction.

Output — NEW files, originals untouched (standing rule):
  data/datasets/coco_single_caption_nlc_matched_{train,val,test}.parquet (bobcat)
  data/datasets/coco_single_caption_nlc_tree_no_type_matched_{train,val,test}.parquet (tree)

Run on the cluster from PROJECT_DIR:
  python -m qnlp.scripts.coco_multi_caption.unify_splits
  python -m qnlp.scripts.coco_multi_caption.unify_splits --verify-only  # rerun checks on existing output
"""

import argparse
from pathlib import Path

import polars as pl

from qnlp.core.data_engine.dataset_creator.dataset_generator import split_by_groups
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="unify_splits")

DATASETS_PATH = Path("data/datasets")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42

BOBCAT_SOURCE = "coco_single_caption_nlc"
TREE_SOURCE = "coco_single_caption_nlc_tree_no_type"
BOBCAT_OUT = "coco_single_caption_nlc_matched"
TREE_OUT = "coco_single_caption_nlc_tree_no_type_matched"

EXPECTED_SCHEMA_COLS = {"sample_id", "local_image_path", "processed_text", "text_hash", "diagram", "symbols"}


def _load_pool(base_name: str) -> pl.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = DATASETS_PATH / f"{base_name}_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Expected original split file not found: {path}")
        frames.append(pl.read_parquet(path))
    return pl.concat(frames, how="vertical_relaxed")


def _write(df: pl.DataFrame, base_name: str, split: str) -> Path:
    path = DATASETS_PATH / f"{base_name}_{split}.parquet"
    tmp = path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    tmp.rename(path)
    logger.info(f"Wrote {df.height} rows -> {path}")
    return path


def build() -> None:
    logger.info(f"Loading bobcat pool ({BOBCAT_SOURCE}_{{train,val,test}}.parquet)...")
    bobcat_pool = _load_pool(BOBCAT_SOURCE)
    logger.info(f"Bobcat pool: {bobcat_pool.height} rows.")

    logger.info(f"Splitting bobcat pool via split_by_groups (ratios={RATIOS}, seed={SEED}, group=sample_id)...")
    bobcat_train, bobcat_val, bobcat_test = split_by_groups(
        bobcat_pool, ratios=RATIOS, seed=SEED, group_column="sample_id"
    )
    logger.info(f"Bobcat matched: train={bobcat_train.height} val={bobcat_val.height} test={bobcat_test.height}")

    # sample_id -> split map, derived from bobcat's own split result (the
    # authoritative shared assignment every parser's matched files must agree with).
    split_map = pl.concat(
        [
            bobcat_train.select("sample_id").unique().with_columns(pl.lit("train").alias("split")),
            bobcat_val.select("sample_id").unique().with_columns(pl.lit("val").alias("split")),
            bobcat_test.select("sample_id").unique().with_columns(pl.lit("test").alias("split")),
        ]
    )

    logger.info(f"Loading tree pool ({TREE_SOURCE}_{{train,val,test}}.parquet)...")
    tree_pool = _load_pool(TREE_SOURCE)
    logger.info(f"Tree pool: {tree_pool.height} rows.")

    common_hashes = bobcat_pool["text_hash"].unique()
    tree_common = tree_pool.filter(pl.col("text_hash").is_in(common_hashes))
    logger.info(
        f"Tree pool restricted to common-pool text_hash: {tree_pool.height} -> {tree_common.height} rows "
        f"(dropped {tree_pool.height - tree_common.height} tree-only rows)."
    )

    tree_joined = tree_common.join(split_map, on="sample_id", how="inner")
    n_unmapped = tree_common.height - tree_joined.height
    if n_unmapped:
        # Edge case: a common-pool caption text (by text_hash) attached to a
        # DIFFERENT sample_id in tree's independently-ingested data than in
        # bobcat's — text_hash alone doesn't capture that. Inner join safely
        # drops these rather than crashing or guessing a split; logged loudly
        # since the plan doc's structural-integrity checks would otherwise
        # need to explain a row-count mismatch.
        logger.warning(
            f"{n_unmapped} tree rows had a common-pool text_hash but a sample_id absent from "
            "bobcat's split map (same caption text under a different image?) — dropped, not assigned "
            "a split. Investigate before trusting the matched files if this is nonzero."
        )

    for split in ("train", "val", "test"):
        tree_split_df = tree_joined.filter(pl.col("split") == split).drop("split")
        _write(tree_split_df, TREE_OUT, split)

    _write(bobcat_train, BOBCAT_OUT, "train")
    _write(bobcat_val, BOBCAT_OUT, "val")
    _write(bobcat_test, BOBCAT_OUT, "test")

    logger.info("Phase 0 build complete. Run --verify-only to check the output before using it.")


def verify() -> bool:
    """Checks A-D from the plan doc's Phase 0 verification plan (structural
    integrity, cross-parser consistency, data fidelity, reproducibility).
    Check E (end-to-end smoke test through run.py/run_frozen.py) and F
    (submit-script ML_DATASET_NAME grep) are NOT automatable here — run those
    separately before Phase C. Returns True iff every check passed."""
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        status = "PASS" if passed else "FAIL"
        logger.info(f"[{status}] {label}" + (f" — {detail}" if detail else ""))
        if not passed:
            ok = False

    bobcat_pool = _load_pool(BOBCAT_SOURCE)
    tree_pool = _load_pool(TREE_SOURCE)

    bobcat_matched = {s: pl.read_parquet(DATASETS_PATH / f"{BOBCAT_OUT}_{s}.parquet") for s in ("train", "val", "test")}
    tree_matched = {s: pl.read_parquet(DATASETS_PATH / f"{TREE_OUT}_{s}.parquet") for s in ("train", "val", "test")}

    # A1: no row loss/duplication
    bobcat_total = sum(df.height for df in bobcat_matched.values())
    check(
        "A1 bobcat row count == full pool",
        bobcat_total == bobcat_pool.height,
        f"{bobcat_total} vs {bobcat_pool.height}",
    )
    tree_total = sum(df.height for df in tree_matched.values())
    tree_common_expected = tree_pool.filter(pl.col("text_hash").is_in(bobcat_pool["text_hash"].unique())).height
    check(
        "A1 tree row count <= common-pool-restricted tree rows",
        tree_total <= tree_common_expected,
        f"{tree_total} vs <= {tree_common_expected} (some may be dropped by the sample_id-unmapped edge case)",
    )

    # A2: text_hash subset/equality
    bobcat_matched_hashes = pl.concat([df.select("text_hash") for df in bobcat_matched.values()])["text_hash"]
    bobcat_orig_hashes = set(bobcat_pool["text_hash"].unique().to_list())
    check(
        "A2 bobcat matched text_hash == original pool text_hash",
        set(bobcat_matched_hashes.unique().to_list()) == bobcat_orig_hashes,
    )
    tree_matched_hashes = set(
        pl.concat([df.select("text_hash") for df in tree_matched.values()])["text_hash"].unique().to_list()
    )
    tree_orig_hashes = set(tree_pool["text_hash"].unique().to_list())
    check("A2 tree matched text_hash subset of original tree pool", tree_matched_hashes.issubset(tree_orig_hashes))

    # A3: tree-only captions absent from ALL tree matched files
    tree_only_hashes = tree_orig_hashes - bobcat_orig_hashes
    check("A3 tree-only captions absent from matched files", tree_matched_hashes.isdisjoint(tree_only_hashes))

    # B4/B6: sample_id -> split maps agree, no sample_id in >1 split, per parser
    def sample_id_split_map(matched: dict[str, pl.DataFrame]) -> tuple[dict[str, str], bool]:
        m: dict[str, str] = {}
        dupe = False
        for split, df in matched.items():
            for sid in df["sample_id"].unique().to_list():
                if sid in m:
                    dupe = True
                m[sid] = split
        return m, dupe

    bobcat_map, bobcat_dupe = sample_id_split_map(bobcat_matched)
    tree_map, tree_dupe = sample_id_split_map(tree_matched)
    check("B6 no sample_id in >1 split (bobcat)", not bobcat_dupe)
    check("B6 no sample_id in >1 split (tree)", not tree_dupe)

    common_sample_ids = set(bobcat_map) & set(tree_map)
    mismatched = [sid for sid in common_sample_ids if bobcat_map[sid] != tree_map[sid]]
    check(
        "B4 sample_id->split agrees across parsers",
        len(mismatched) == 0,
        f"{len(mismatched)}/{len(common_sample_ids)} mismatched" if mismatched else "",
    )

    # B5: spot-check ~20 random multi-caption sample_ids land in the same split
    import random

    multi_cap_ids = bobcat_pool.group_by("sample_id").len().filter(pl.col("len") > 1)["sample_id"].to_list()
    sample = random.Random(0).sample(multi_cap_ids, min(20, len(multi_cap_ids)))
    spot_ok = all(
        sid in bobcat_map and sid in tree_map and bobcat_map[sid] == tree_map[sid] for sid in sample if sid in tree_map
    )
    check("B5 spot-check 20 multi-caption sample_ids agree", spot_ok)

    # C7: byte-identical diagram/symbols for a random sample of shared text_hashes
    def spot_check_payload(orig_pool: pl.DataFrame, matched: dict[str, pl.DataFrame], n: int = 50) -> bool:
        orig_by_hash = {row["text_hash"]: (row["diagram"], row["symbols"]) for row in orig_pool.iter_rows(named=True)}
        all_matched = pl.concat([df.select("text_hash", "diagram", "symbols") for df in matched.values()])
        sample_hashes = random.Random(1).sample(all_matched["text_hash"].to_list(), min(n, all_matched.height))
        matched_by_hash = {
            row["text_hash"]: (row["diagram"], row["symbols"])
            for row in all_matched.filter(pl.col("text_hash").is_in(sample_hashes)).iter_rows(named=True)
        }
        return all(matched_by_hash.get(h) == orig_by_hash.get(h) for h in sample_hashes)

    check("C7 bobcat diagram/symbols byte-identical (50-sample)", spot_check_payload(bobcat_pool, bobcat_matched))
    check("C7 tree diagram/symbols byte-identical (50-sample)", spot_check_payload(tree_pool, tree_matched))

    # C8: schema match
    for label, matched, orig in (("bobcat", bobcat_matched, bobcat_pool), ("tree", tree_matched, tree_pool)):
        for split, df in matched.items():
            missing = EXPECTED_SCHEMA_COLS - set(df.columns)
            check(f"C8 {label} {split} schema has expected columns", not missing, f"missing: {missing}")
            extra_or_wrong_dtype = {
                c for c in EXPECTED_SCHEMA_COLS if c in df.columns and df.schema[c] != orig.schema[c]
            }
            check(f"C8 {label} {split} dtypes match original", not extra_or_wrong_dtype, str(extra_or_wrong_dtype))

    # D9: reproducibility — rerun split_by_groups twice, assert identical assignment
    _, _, _ = split_by_groups(bobcat_pool, ratios=RATIOS, seed=SEED, group_column="sample_id")
    t1, v1, te1 = split_by_groups(bobcat_pool, ratios=RATIOS, seed=SEED, group_column="sample_id")
    t2, v2, te2 = split_by_groups(bobcat_pool, ratios=RATIOS, seed=SEED, group_column="sample_id")
    ids1 = (set(t1["sample_id"]), set(v1["sample_id"]), set(te1["sample_id"]))
    ids2 = (set(t2["sample_id"]), set(v2["sample_id"]), set(te2["sample_id"]))
    check("D9 split_by_groups reproducible across repeated calls", ids1 == ids2)

    logger.info("=" * 60)
    logger.info("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see above")
    logger.info(
        "NOT automated here (do separately before Phase C): "
        "E. run a few batches through run.py/run_frozen.py against the matched dataset names; "
        "F. grep the sweep's submit scripts to confirm ML_DATASET_NAME points at *_matched, not the originals."
    )
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 0: unify bobcat/tree_no_type train/val/test splits.")
    ap.add_argument("--verify-only", action="store_true", help="Skip building, just run verification checks.")
    args = ap.parse_args()

    if not args.verify_only:
        build()
    ok = verify()
    if not ok:
        raise SystemExit(1)
