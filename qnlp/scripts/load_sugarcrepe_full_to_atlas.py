"""Load the full SugarCrepe dataset (AsphyXIA/sugarcrepe, all subsets) into an Atlas.

Images are stored in the HuggingFace parquet as structs {"bytes": ..., "path": ...}.
Caption columns: `pos` (positive/true) and `neg` (negative/false).

Usage:
    HF_TOKEN=hf_... python -m qnlp.scripts.load_sugarcrepe_full_to_atlas
"""

import os

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

HF_PARQUET = "hf://datasets/AsphyXIA/sugarcrepe/data/test-*.parquet"
ATLAS_NAME = "sugarcrepe_full"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME


def run() -> None:
    storage_options = {"token": os.environ["HF_TOKEN"]}

    if ATLAS_DIR.exists():
        atlas = Atlas.load_atlas(ATLAS_DIR / "metadata.json")
        print(f"Resuming atlas '{ATLAS_NAME}' from cursor {atlas.cursor_location}.")
    else:
        atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=HF_PARQUET, image_column="images")

    atlas.ingest_data_from_remote(
        n=100_000,
        storage_options=storage_options,
        column_rename={"positive_caption": "true_caption", "negative_caption": "false_caption"},
    )


if __name__ == "__main__":
    run()
