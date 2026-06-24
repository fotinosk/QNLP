"""Load the SugarCrepe++ dataset (AsphyXIA/sugarcrepepp) into an Atlas.

Each row has two positive captions (positive_caption_1, positive_caption_2) and one
negative (negative_caption). We expand each row into two contrastive pairs so both
positives are evaluated:
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
_MANIFEST_SCHEMA = {
    "true_caption": pl.String,
    "false_caption": pl.String,
    "local_image_path": pl.String,
}


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

    # Save images to disk, replacing 'images' bytes column with 'local_image_path'
    df = save_images_and_clear_df(
        df=df,
        image_column="images",
        image_file_path_column=None,
        image_storage_path=atlas.image_path,
    )

    # Expand each row into two pairs: one per positive caption
    pairs1 = df.select(
        [
            pl.col("positive_caption_1").alias("true_caption"),
            pl.col("negative_caption").alias("false_caption"),
            pl.col("local_image_path"),
        ]
    )
    pairs2 = df.select(
        [
            pl.col("positive_caption_2").alias("true_caption"),
            pl.col("negative_caption").alias("false_caption"),
            pl.col("local_image_path"),
        ]
    )
    expanded = pl.concat([pairs1, pairs2], how="vertical")

    atlas.ingest_dataframe(expanded.cast(_MANIFEST_SCHEMA))
    print(f"Ingested {len(expanded)} pairs ({len(df)} original rows × 2 positives).")


if __name__ == "__main__":
    run()
