import shutil

import polars as pl
from datasets import load_dataset

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

HF_REPO = "facebook/winoground"
ATLAS_NAME = "winoground"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME

_MANIFEST_SCHEMA = {
    "id": pl.Int32,
    "caption_0": pl.String,
    "caption_1": pl.String,
    "tag": pl.String,
    "secondary_tag": pl.String,
    "num_main_preds": pl.Int32,
    "collapsed_tag": pl.String,
    "local_image_0_path": pl.String,
    "local_image_1_path": pl.String,
    "sample_id": pl.String,
}


def run() -> None:
    if ATLAS_DIR.exists():
        print(f"Removing existing atlas at {ATLAS_DIR}")
        shutil.rmtree(ATLAS_DIR)

    ds = load_dataset(HF_REPO, split="test")

    atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=HF_REPO, image_column=["image_0", "image_1"])
    image_dir = atlas.image_path
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ex in ds:
        i = ex["id"]

        image_0_path = (image_dir / f"winoground_{i}_0.jpg").resolve()
        image_1_path = (image_dir / f"winoground_{i}_1.jpg").resolve()
        ex["image_0"].convert("RGB").save(image_0_path)
        ex["image_1"].convert("RGB").save(image_1_path)

        rows.append(
            {
                "id": i,
                "caption_0": ex["caption_0"],
                "caption_1": ex["caption_1"],
                "tag": ex["tag"],
                "secondary_tag": ex["secondary_tag"],
                "num_main_preds": ex["num_main_preds"],
                "collapsed_tag": ex["collapsed_tag"],
                "local_image_0_path": str(image_0_path),
                "local_image_1_path": str(image_1_path),
                "sample_id": f"winoground_{i}",
            }
        )

    atlas.ingest_dataframe(pl.DataFrame(rows, schema=_MANIFEST_SCHEMA))
    print(f"Winoground atlas ready at {atlas.metadata_location} ({len(rows)} examples).")


if __name__ == "__main__":
    run()
