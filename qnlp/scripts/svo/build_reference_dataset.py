"""
DISCOCLIP_REPRODUCTION_PLAN.md's E1: run OUR pipeline directly on
discoclip's own released train/val/test CSVs. Same rows, same splits,
same vocabulary as the reference (no word-frequency filtering here — the
reference's own preprocessing already produced these files, so every
caption they kept survives, unfiltered) — only our compilation code
differs from theirs. Closes the reproduction outright if this reaches
~0.83; if it stays near our own numbers, a real implementation
divergence survives independent of data coverage or split protocol.

Reads `data/svo/discoclip_reference/{train,val,test}.csv` (copied
verbatim from `~/Desktop/Dev/discoclip/data/processed/svo_probes/`),
resolves `pos_image_id`/`neg_image_id` against our own already-downloaded
images (Phase 5.4: 99.97% coverage — the reference is a near-total
subset of what we hold), compiles each split's `corrected_sentence`
through the same steps `qnlp/preprocessing_pipelines/svo/pipeline.py`
uses (RemoveTrailingDotsStep -> LemmatizeStep -> CCGCompilerStep ->
SymbolLemmatizeStep -> UnifyEinsumRankStep), into a **dedicated,
isolated LMDB** (not the project's shared or lemmafix store — this is a
one-off ~9k-row external dataset, not part of our own atlas), and emits
the same schema `prepare_datasets.py` produces so `run.py`/`run_frozen.py`
need no changes — only `SVO_ML_DATASET_SUFFIX` selects this variant.

Also builds the matching SVO-Swap set directly from the reference's own
95-row `svo_probes_swapped.csv` (true=`corrected_sentence`,
false=`swapped_sentence`, already a clean grammatical subject/object
swap — verified by inspection, no reparsing of `sentence` needed).

Usage:
    python -m qnlp.scripts.svo.build_reference_dataset
"""

import orjson
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.processing.common_steps import RemoveTrailingDotsStep
from qnlp.core.data_engine.processing.compiler_step import CCGCompilerStep
from qnlp.core.data_engine.processing.conform_rank_step import UnifyEinsumRankStep
from qnlp.core.data_engine.processing.lemmatize_step import LemmatizeStep
from qnlp.core.data_engine.processing.symbol_lemmatize_step import _relabel_symbols
from qnlp.utils.logging import setup_logger

logger = setup_logger(log_name="build_reference_dataset")

REF_DIR = constants.atlases_path.parent / "svo" / "discoclip_reference"
IMAGE_DIRS = [
    constants.atlases_path.parent / "svo" / "raw" / "images",
    constants.atlases_path.parent / "svo" / "raw" / "images_old",
]
REF_LMDB_PATH = constants.lmdb_path.parent / "sentence_mapping_e1_ref"
OUTPUT_SUFFIX = "_e1_refdata"


def _resolve_image_path(image_id: str) -> str | None:
    for image_dir in IMAGE_DIRS:
        candidate = image_dir / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate)
    return None


def _load_split(name: str) -> pl.DataFrame:
    df = pl.read_csv(REF_DIR / f"{name}.csv")
    df = df.rename({"corrected_sentence": "processed_text"})
    all_ids = set(df["pos_image_id"].cast(pl.String).to_list()) | set(df["neg_image_id"].cast(pl.String).to_list())
    path_map = {image_id: _resolve_image_path(image_id) for image_id in all_ids}
    df = df.with_columns(
        pl.col("pos_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("pos_local_image_path"),
        pl.col("neg_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("neg_local_image_path"),
    )
    n_before = len(df)
    df = df.filter(pl.col("pos_local_image_path").is_not_null() & pl.col("neg_local_image_path").is_not_null())
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"{name}: {n_dropped}/{n_before} rows dropped (image not found locally)")
    df = df.with_columns(
        pl.arange(0, len(df))
        .cast(pl.String)
        .alias("sample_id")
        .map_elements(lambda i: f"{name}_{i}", return_dtype=pl.String)
    )
    return df


def _compile(df: pl.DataFrame) -> pl.DataFrame:
    """Run RemoveTrailingDotsStep -> LemmatizeStep -> CCGCompilerStep ->
    SymbolLemmatizeStep -> UnifyEinsumRankStep, matching svo_pipeline's
    exact step sequence, against a dedicated isolated LMDB."""
    df = RemoveTrailingDotsStep(text_column="processed_text").process(df)
    df = LemmatizeStep(text_column="processed_text").process(df)
    ccg_step = CCGCompilerStep(
        lmdb_path=REF_LMDB_PATH,
        bond_dim=constants.bond_dim,
        embedding_dim=constants.embedding_dim,
        max_workers=4,
        worker_batch_size=200,
        cache_path=str(constants.bobcat_cache_path),
    )
    df = ccg_step.process(df)
    ccg_step.teardown()
    df = df.with_columns(
        pl.col("compiled_bytes").map_elements(_relabel_symbols, return_dtype=pl.Binary).alias("compiled_bytes")
    )
    df = UnifyEinsumRankStep().process(df)
    return df


def _extract_diagram_symbols(df: pl.DataFrame) -> pl.DataFrame:
    def _get(field: str):
        def fn(raw: bytes | None):
            if raw is None:
                return None
            payload = orjson.loads(raw)
            if payload.get("error"):
                return None
            val = payload.get(field)
            return orjson.dumps(val).decode() if field == "symbols" else val

        return fn

    df = df.with_columns(
        pl.col("compiled_bytes").map_elements(_get("diagram"), return_dtype=pl.String).alias("diagram"),
        pl.col("compiled_bytes").map_elements(_get("symbols"), return_dtype=pl.String).alias("symbols"),
    )
    before = len(df)
    df = df.filter(pl.col("diagram").is_not_null() & pl.col("symbols").is_not_null())
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped}/{before} rows with a compile failure.")
    return df


def _build_train_split(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(["sample_id", "pos_local_image_path", "processed_text", "text_hash", "diagram", "symbols"]).rename(
        {"pos_local_image_path": "local_image_path"}
    )


def _build_probes_split(df: pl.DataFrame) -> pl.DataFrame:
    cols = [
        "sample_id",
        "pos_local_image_path",
        "neg_local_image_path",
        "diagram",
        "symbols",
        "subj_neg",
        "verb_neg",
        "obj_neg",
        "subj",
        "verb",
        "obj",
    ]
    return df.select(cols).rename(
        {"pos_local_image_path": "true_local_image_path", "neg_local_image_path": "false_local_image_path"}
    )


def _build_probes() -> None:
    datasets_path = constants.datasets_path
    datasets_path.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        df = _load_split(split_name)
        logger.info(f"{split_name}: {len(df)} rows with both images resolved locally")
        df = _compile(df)
        df = _extract_diagram_symbols(df)
        logger.info(f"{split_name}: {len(df)} rows survive compilation")

        if split_name == "train":
            out = _build_train_split(df)
            out_path = datasets_path / f"svo_train{OUTPUT_SUFFIX}.parquet"
            out.write_parquet(out_path)
            logger.info(f"{out_path.name}: {len(out)} rows")

        probes_out = _build_probes_split(df)
        probes_path = datasets_path / f"svo_{split_name}_probes{OUTPUT_SUFFIX}.parquet"
        probes_out.write_parquet(probes_path)
        logger.info(f"{probes_path.name}: {len(probes_out)} rows")


def _build_swap() -> None:
    df = pl.read_csv(REF_DIR / "svo_probes_swapped.csv")
    all_ids = set(df["pos_image_id"].cast(pl.String).to_list())
    path_map = {image_id: _resolve_image_path(image_id) for image_id in all_ids}
    df = df.with_columns(
        pl.col("pos_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("local_image_path")
    )
    n_before = len(df)
    df = df.filter(pl.col("local_image_path").is_not_null())
    if n_before - len(df):
        logger.warning(f"svo-swap: {n_before - len(df)}/{n_before} rows dropped (image not found locally)")

    df = df.with_columns(
        pl.arange(0, len(df))
        .cast(pl.String)
        .alias("sample_id")
        .map_elements(lambda i: f"swap_{i}", return_dtype=pl.String)
    )

    true_df = df.select(["sample_id", "corrected_sentence"]).rename({"corrected_sentence": "processed_text"})
    false_df = df.select(["sample_id", "swapped_sentence"]).rename({"swapped_sentence": "processed_text"})

    true_df = _extract_diagram_symbols(_compile(true_df)).rename({"diagram": "true_diagram", "symbols": "true_symbols"})
    false_df = _extract_diagram_symbols(_compile(false_df)).rename(
        {"diagram": "false_diagram", "symbols": "false_symbols"}
    )

    out = (
        df.select(["sample_id", "local_image_path"])
        .join(true_df.select(["sample_id", "true_diagram", "true_symbols"]), on="sample_id", how="inner")
        .join(false_df.select(["sample_id", "false_diagram", "false_symbols"]), on="sample_id", how="inner")
    )
    out_path = constants.datasets_path / f"svo_swap_eval{OUTPUT_SUFFIX}.parquet"
    out.write_parquet(out_path)
    logger.info(f"{out_path.name}: {len(out)} rows (of {n_before} candidates)")


def run() -> None:
    _build_probes()
    _build_swap()


if __name__ == "__main__":
    run()
