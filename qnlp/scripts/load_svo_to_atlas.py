import shutil
from pathlib import Path

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

SVO_DIR = Path("data/svo/raw").resolve()
SVO_CSV = SVO_DIR / "svo_probes_corrected.csv"
IMAGE_DIRS = [SVO_DIR / "images", SVO_DIR / "images_old"]

ATLAS_METADATA = constants.atlases_path / "svo" / "metadata.json"


def _resolve_image_path(image_id: str) -> str | None:
    """Return the first existing path for this image id across IMAGE_DIRS, else None."""
    for image_dir in IMAGE_DIRS:
        candidate = image_dir / f"{image_id}.jpg"
        if candidate.exists():
            return str(candidate)
    return None


def run() -> None:
    if ATLAS_METADATA.exists():
        # Stale atlas was built from the uncorrected CSV/columns — rebuild from scratch.
        shutil.rmtree(ATLAS_METADATA.parent)

    atlas = Atlas.create_atlas(name="svo", source_path_or_url=str(SVO_DIR))

    df = pl.read_csv(SVO_CSV)

    all_ids = set(df["pos_image_id"].cast(pl.String).to_list()) | set(df["neg_image_id"].cast(pl.String).to_list())
    path_map = {image_id: _resolve_image_path(image_id) for image_id in all_ids}

    df = df.with_columns(
        pl.col("pos_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("pos_local_image_path"),
        pl.col("neg_image_id").cast(pl.String).replace_strict(path_map, default=None).alias("neg_local_image_path"),
    )

    n_before = len(df)
    df = df.filter(pl.col("pos_local_image_path").is_not_null() & pl.col("neg_local_image_path").is_not_null())
    print(f"SVO: {n_before} rows -> {len(df)} rows with both images available on disk")

    atlas.ingest_dataframe(df)
    print(f"SVO atlas ready at {ATLAS_METADATA}")


if __name__ == "__main__":
    run()
