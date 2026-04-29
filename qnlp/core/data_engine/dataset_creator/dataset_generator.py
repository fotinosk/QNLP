"""
Generates dataset(s) from derived manifests
we need to make sure that we can create val sets even after the trian set has been created?

For example i created a train test val set on aro and wino and now want to create val on wino only.
This means I need to exclude all the wino samples

create a global sample id? Does it already exists? yes sample_id
"""

from pathlib import Path

import numpy as np
import polars as pl


def create_dataset(
    dataset_name: str,
    manifests: list[Path],
    num_rows: int,
    output_schema: pl.Schema,
    dataset_composition: list[float] | None = None,
    exclude_sample_ids: list[str] | None = None,
):
    """
    Read provided derived manifests lazily
    Excludes sample ids if provided
    Creates dataset of num_rows rows with the dataset_composition relative sizes
    Enforces output schema and writes
    """
    if dataset_composition:
        assert len(manifests) == len(dataset_composition)
        assert sum(dataset_composition) == 1.0
    else:
        dataset_composition = [1 / len(manifests)] * len(manifests)

    manifest_data = [pl.scan_parquet(manifest, row_index_name="index") for manifest in manifests]
    datasets = []

    # TODO: remove sample ids
    for ratio, df in zip(dataset_composition, manifest_data):
        num_rows_from_df = int(ratio * num_rows)
        manifest_size = df.select(pl.col("index")).max().collect()[0, "index"]
        subsample_indices = np.random.choice(manifest_size, num_rows_from_df, replace=False)
        subsample_df = df.filter(pl.col("index").is_in(subsample_indices))
        datasets.append(subsample_df)

    dataset = pl.concat(datasets)
    # TODO: add einsum repr from db and conform to schema
    print(dataset.head().collect())


def create_train_test_val_datasets():
    # ensures that there is no overlap between the data
    # dedupes on sample_id
    pass


if __name__ == "__main__":
    create_dataset(
        dataset_name="test_dataset",
        manifests=[Path("data/atlases/coco/derived_test")],
        num_rows=100,
        output_schema=pl.Schema({"local_local_path": pl.String, "processed_text": pl.String}),
    )
