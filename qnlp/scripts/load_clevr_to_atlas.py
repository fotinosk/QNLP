import io
import shutil

import orjson
import polars as pl
from PIL import Image

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

HF_PARQUET = {
    "train": "hf://datasets/dpdl-benchmark/clevr/data/train-*.parquet",
    "test": "hf://datasets/dpdl-benchmark/clevr/data/test-*.parquet",
}

ATLAS_NAME = "clevr"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME

_MANIFEST_SCHEMA = {
    "image_id": pl.String,
    "split": pl.String,
    "local_image_path": pl.String,
    "n_objects": pl.Int32,
    "color": pl.List(pl.Int32),
    "shape": pl.List(pl.Int32),
    "material": pl.List(pl.Int32),
    "size": pl.List(pl.Int32),
    "coords_3d": pl.String,  # JSON: [[x, y, z], ...] one entry per object
    "pixel_coords": pl.String,  # JSON: [[x, y, depth], ...] one entry per object
}


def _decode_image(image_field) -> Image.Image:
    """HF parquet stores images as a struct {bytes, path} or raw bytes."""
    if isinstance(image_field, dict):
        raw = image_field["bytes"]
    else:
        raw = image_field
    return Image.open(io.BytesIO(raw)).convert("RGB")


def run() -> None:
    if ATLAS_DIR.exists():
        print(f"Removing existing atlas at {ATLAS_DIR}")
        shutil.rmtree(ATLAS_DIR)

    atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=list(HF_PARQUET.values())[0], image_column="image")
    image_dir = atlas.image_path
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, glob in HF_PARQUET.items():
        print(f"Reading {split} split from {glob} ...")
        df = pl.read_parquet(glob)

        for idx, row in enumerate(df.iter_rows(named=True)):
            image_id = f"clevr_{split}_{idx:06d}"
            image_path = (image_dir / f"{image_id}.jpg").resolve()

            _decode_image(row["image"]).save(image_path)

            objects = row["objects"]
            n_objects = len(objects["color"])

            rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "local_image_path": str(image_path),
                    "n_objects": n_objects,
                    "color": objects["color"],
                    "shape": objects["shape"],
                    "material": objects["material"],
                    "size": objects["size"],
                    "coords_3d": orjson.dumps(objects["3d_coords"]).decode(),
                    "pixel_coords": orjson.dumps(objects["pixel_coords"]).decode(),
                }
            )

        print(f"  {split}: {len(df)} scenes processed.")

    manifest = pl.DataFrame(rows, schema=_MANIFEST_SCHEMA)
    atlas.ingest_dataframe(manifest)

    print(f"CLEVR atlas ready at {atlas.metadata_location} ({len(manifest)} records).")
    print(manifest.group_by("split", "n_objects").len().sort("split", "n_objects"))


if __name__ == "__main__":
    run()
