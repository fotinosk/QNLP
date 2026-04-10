import os

import pandas as pd

ARO_DATA_PATH = "data/aro/processed/combined"
ARO_IMAGE_PATH_PREFIX = "data/aro/raw/images/"
WINO_DATA_PATH = "data/winoground/processed"

COMBINED_OUTPUT_PATH = "data/aro_and_wino/processed"


def _add_dir_prefix_to_aro_images(df) -> pd.DataFrame:
    df["image_path"] = ARO_IMAGE_PATH_PREFIX + df["image_path"].astype(str)
    return df


if __name__ == "__main__":
    sets = ["train", "val", "test"]
    os.makedirs(COMBINED_OUTPUT_PATH, exist_ok=True)

    for ds in sets:
        aro_ds = pd.read_json(f"{ARO_DATA_PATH}/{ds}.json")
        wino_ds = pd.read_json(f"{WINO_DATA_PATH}/{ds}.json")

        aro_processed = pd.read_json(f"{ARO_DATA_PATH}/{ds}_processed_512.jsonl", lines=True)
        wino_processed = pd.read_json(f"{WINO_DATA_PATH}/{ds}_processed_512.jsonl", lines=True)

        aro_ds = _add_dir_prefix_to_aro_images(aro_ds)

        combined_ds = pd.concat([aro_ds, wino_ds], ignore_index=True)
        combined_processed = pd.concat([aro_processed, wino_processed], ignore_index=True)

        combined_ds.to_json(f"{COMBINED_OUTPUT_PATH}/{ds}.json", orient="records")
        combined_processed.to_json(f"{COMBINED_OUTPUT_PATH}/{ds}_processed_512.jsonl", lines=True, orient="records")
