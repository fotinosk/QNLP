"""Create an atlas for a SugarCrepe subset (evaluation-only benchmark).

SugarCrepe examples are (image, positive caption, hard-negative caption) triples
— structurally the same as ARO — so the same flatten/compile/ContrastivePair
machinery applies downstream.

The HuggingFaceM4 parquet stores the two candidate captions in `tested_labels`
and the index of the correct one in `true_label`, a ClassLabel whose names are
the positive-caption vocabulary. We decode it here into explicit
`true_caption` / `false_caption` columns; the generic HF ingest can't, because
polars drops the ClassLabel mapping.

Starts with the swap-attribute subset; other subsets share the schema, so only
SUBSET needs changing.
"""

import shutil

import polars as pl
from datasets import load_dataset

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.atlas import Atlas

SUBSET = "swap_att"
HF_REPO = f"HuggingFaceM4/SugarCrepe_{SUBSET}"
ATLAS_NAME = f"sugarcrepe_{SUBSET}"
ATLAS_DIR = constants.atlases_path / ATLAS_NAME

_MANIFEST_SCHEMA = {
    "true_caption": pl.String,
    "false_caption": pl.String,
    "local_image_path": pl.String,
}


def run() -> None:
    # The atlas is fully regenerable from HF; rebuild from scratch each run.
    if ATLAS_DIR.exists():
        print(f"Removing existing atlas at {ATLAS_DIR}")
        shutil.rmtree(ATLAS_DIR)

    ds = load_dataset(HF_REPO, split="test")
    positive_vocab = ds.features["true_label"]  # ClassLabel: index -> positive caption

    atlas = Atlas.create_atlas(name=ATLAS_NAME, source_path_or_url=HF_REPO, image_column="image")
    image_dir = atlas.image_path
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_bad = 0
    for i, ex in enumerate(ds):
        positive = positive_vocab.int2str(ex["true_label"])
        negatives = [c for c in ex["tested_labels"] if c != positive]
        if len(negatives) != 1:
            n_bad += 1
            continue

        # SugarCrepe images repeat across examples and carry no stable id, so save
        # one file per example with a unique name.
        image_path = (image_dir / f"{ATLAS_NAME}_{i}.jpg").resolve()
        ex["image"].convert("RGB").save(image_path)

        rows.append({"true_caption": positive, "false_caption": negatives[0], "local_image_path": str(image_path)})

    atlas.ingest_dataframe(pl.DataFrame(rows, schema=_MANIFEST_SCHEMA))
    print(f"SugarCrepe '{SUBSET}' atlas ready at {atlas.metadata_location} ({len(rows)} examples, {n_bad} skipped).")


if __name__ == "__main__":
    run()
