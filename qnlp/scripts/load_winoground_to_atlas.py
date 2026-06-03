from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

HF_SOURCE = "hf://datasets/facebook/winoground/data/test-00000-of-00001.parquet"
ATLAS_METADATA = constants.atlases_path / "winoground" / "metadata.json"


def run() -> None:
    if ATLAS_METADATA.exists():
        atlas = Atlas.load_atlas(ATLAS_METADATA)
    else:
        atlas = Atlas.create_atlas(
            name="winoground",
            source_path_or_url=HF_SOURCE,
            image_column=["image_0", "image_1"],
        )

    atlas.ingest_data_from_remote(n=400)
    print(f"Winoground atlas ready at {ATLAS_METADATA}")


if __name__ == "__main__":
    run()
