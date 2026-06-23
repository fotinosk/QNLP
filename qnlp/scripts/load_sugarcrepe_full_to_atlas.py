"""Load the full SugarCrepe dataset (AsphyXIA/sugarcrepe, all subsets) into an Atlas.

Images are stored in the HuggingFace parquet as structs {"bytes": ..., "path": ...}.
Caption columns: `pos` (positive/true) and `neg` (negative/false).

Usage:
    python -m qnlp.scripts.load_sugarcrepe_full_to_atlas
"""

import io
import shutil

import polars as pl
from PIL import Image

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

HF_PARQUET = "hf://datasets/AsphyXIA/sugarcrepe/data/test-*.parquet"
ATLAS_NAME = "sugarcrepe_full"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME
_MANIFEST_SCHEMA = {
    "true_caption": pl.String,
    "false_caption": pl.String,
    "local_image_path": pl.String,
}


def run() -> None:
    if ATLAS_DIR.exists():
        shutil.rmtree(ATLAS_DIR)

    df = pl.read_parquet(HF_PARQUET)

    atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=HF_PARQUET, image_column="image")
    image_dir = atlas.image_path
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in enumerate(df.iter_rows(named=True)):
        true_caption = row["pos"]
        false_caption = row["neg"]

        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = (image_dir / f"{ATLAS_NAME}_{i}.jpg").resolve()
        image.save(image_path)

        rows.append(
            {
                "true_caption": true_caption,
                "false_caption": false_caption,
                "local_image_path": str(image_path),
            }
        )

    atlas.ingest_dataframe(pl.DataFrame(rows, schema=_MANIFEST_SCHEMA))
    print(f"Atlas '{ATLAS_NAME}' created with {len(rows)} rows.")


if __name__ == "__main__":
    run()
