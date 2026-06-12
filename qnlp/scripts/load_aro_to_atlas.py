import json
from pathlib import Path

import orjson
import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

# Combined ARO (Visual Genome Relation + Attribution) with pre-defined splits.
ARO_COMBINED_DIR = Path("data/aro/processed/combined")
ARO_IMAGE_DIR = Path("data/aro/raw/images")
ATLAS_METADATA = constants.atlases_path / "aro" / "metadata.json"

SPLITS = ["train", "val", "test"]

_MANIFEST_SCHEMA = {
    "image_id": pl.Int64,
    "true_caption": pl.String,
    "false_caption": pl.String,
    "local_image_path": pl.String,
    "split": pl.String,
    "task": pl.String,
    "obj1_name": pl.String,
    "obj2_name": pl.String,
    "relation_name": pl.String,
    "attributes": pl.String,
}


def _load_split(split: str) -> pl.DataFrame:
    records = json.load(open(ARO_COMBINED_DIR / f"{split}.json"))
    image_root = ARO_IMAGE_DIR.resolve()

    rows = []
    for r in records:
        rows.append(
            {
                "image_id": r["image_id"],
                "true_caption": r["true_caption"],
                "false_caption": r["false_caption"],
                # Absolute path — the atlas references images in place (never copies).
                "local_image_path": str(image_root / r["image_path"]),
                "split": split,
                # relation records have relation_name; attribution records have attributes
                "task": "relation" if r.get("relation_name") else "attribution",
                "obj1_name": r.get("obj1_name"),
                "obj2_name": r.get("obj2_name"),
                "relation_name": r.get("relation_name"),
                "attributes": orjson.dumps(r.get("attributes")).decode() if r.get("attributes") else None,
            }
        )
    return pl.DataFrame(rows, schema=_MANIFEST_SCHEMA)


def run() -> None:
    if ATLAS_METADATA.exists():
        print(f"ARO atlas already exists at {ATLAS_METADATA}. Delete the directory to rebuild.")
        return

    atlas = Atlas.create_atlas(
        name="aro",
        source_path_or_url=str(ARO_COMBINED_DIR.resolve()),
        image_column="local_image_path",
        image_file_path_column=None,
    )

    # Ingest all splits in one manifest; the `split` column preserves ARO's
    # pre-defined train/val/test partition for later dataset creation.
    df = pl.concat([_load_split(s) for s in SPLITS], how="vertical")
    atlas.ingest_dataframe(df)

    print(f"ARO atlas ready at {ATLAS_METADATA} ({len(df)} records).")
    print(df.group_by("split", "task").len().sort("split", "task"))


if __name__ == "__main__":
    run()
