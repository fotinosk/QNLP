from pathlib import Path

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

SVO_DIR = Path("data/svo/raw").resolve()
SVO_CSV = SVO_DIR / "svo_probes.csv"
IMAGE_DIR = SVO_DIR / "images"

ATLAS_METADATA = constants.atlases_path / "svo" / "metadata.json"


def run() -> None:
    if ATLAS_METADATA.exists():
        atlas = Atlas.load_atlas(ATLAS_METADATA)
    else:
        atlas = Atlas.create_atlas(name="svo", source_path_or_url=str(SVO_DIR))

    df = pl.read_csv(SVO_CSV)

    # Resolve both image paths to absolute — never copy
    df = df.with_columns(
        (pl.lit(str(IMAGE_DIR) + "/") + pl.col("pos_image_id").cast(pl.String) + pl.lit(".jpg")).alias(
            "pos_local_image_path"
        ),
        (pl.lit(str(IMAGE_DIR) + "/") + pl.col("neg_image_id").cast(pl.String) + pl.lit(".jpg")).alias(
            "neg_local_image_path"
        ),
    )

    atlas.ingest_dataframe(df)
    print(f"SVO atlas ready at {ATLAS_METADATA}")


if __name__ == "__main__":
    run()
