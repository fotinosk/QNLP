import shutil

import orjson
import polars as pl
from datasets import load_dataset

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

HF_REPO = "dpdl-benchmark/clevr"
ATLAS_NAME = "clevr"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME
SPLITS = ["train", "test"]

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


def run() -> None:
    if ATLAS_DIR.exists():
        print(f"Removing existing atlas at {ATLAS_DIR}")
        shutil.rmtree(ATLAS_DIR)

    atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=HF_REPO, image_column="image")
    image_dir = atlas.image_path
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in SPLITS:
        ds = load_dataset(HF_REPO, split=split)

        for idx, ex in enumerate(ds):
            image_id = f"clevr_{split}_{idx:06d}"
            image_path = (image_dir / f"{image_id}.jpg").resolve()
            ex["image"].convert("RGB").save(image_path)

            n_objects = len(ex["color"])

            rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "local_image_path": str(image_path),
                    "n_objects": n_objects,
                    "color": ex["color"],
                    "shape": ex["shape"],
                    "material": ex["material"],
                    "size": ex["size"],
                    "coords_3d": orjson.dumps(ex["3d_coords"]).decode(),
                    "pixel_coords": orjson.dumps(ex["pixel_coords"]).decode(),
                }
            )

    df = pl.DataFrame(rows, schema=_MANIFEST_SCHEMA)
    atlas.ingest_dataframe(df)

    print(f"CLEVR atlas ready at {atlas.metadata_location} ({len(df)} records).")
    print(df.group_by("split", "n_objects").len().sort("split", "n_objects"))


if __name__ == "__main__":
    run()
