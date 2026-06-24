"""Load the SugarCrepe++ dataset (AsphyXIA/sugarcrepepp) into an Atlas.

All HF columns are stored as lists of length 1 — images and captions are
extracted with list.first() before ingestion.

Each row has two positive captions. We expand into two contrastive pairs:
  (positive_caption_1, negative_caption)
  (positive_caption_2, negative_caption)

Usage:
    HF_TOKEN=hf_... python -m qnlp.scripts.load_sugarcrepepp_to_atlas
"""

import os

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas
from qnlp.core.data_engine.atlas.hf_utils import fetch_hf_batch_lazily, save_images_and_clear_df

HF_PARQUET = "hf://datasets/AsphyXIA/sugarcrepepp/data/test-*.parquet"
ATLAS_NAME = "sugarcrepepp"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME


def run() -> None:
    storage_options = {"token": os.environ["HF_TOKEN"]}

    if ATLAS_DIR.exists():
        atlas = Atlas.load_atlas(ATLAS_DIR / "metadata.json")
        print(f"Resuming atlas '{ATLAS_NAME}' from cursor {atlas.cursor_location}.")
    else:
        atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=HF_PARQUET, image_column="images")

    df = fetch_hf_batch_lazily(HF_PARQUET, cursor_location=0, n_to_fetch=100_000, storage_options=storage_options)
    if df.is_empty():
        print("No data to ingest.")
        return

    df = save_images_and_clear_df(
        df=df,
        image_column="images",
        image_file_path_column=None,
        image_storage_path=atlas.image_path,
    )

    neg = pl.col("negative_caption").list.first().alias("false_caption")
    pairs1 = df.select(
        [
            pl.col("positive_caption_1").list.first().alias("true_caption"),
            neg,
            pl.col("local_image_path"),
        ]
    )
    pairs2 = df.select(
        [
            pl.col("positive_caption_2").list.first().alias("true_caption"),
            neg,
            pl.col("local_image_path"),
        ]
    )
    expanded = pl.concat([pairs1, pairs2], how="vertical")

    atlas.ingest_dataframe(expanded)
    print(f"Ingested {len(expanded)} pairs ({len(df)} original rows × 2 positives).")


if __name__ == "__main__":
    run()
