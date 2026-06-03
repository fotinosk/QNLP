import concurrent.futures
from pathlib import Path

import polars as pl


def fetch_hf_batch_lazily(
    hf_parquet_glob: str,
    cursor_location: int,
    n_to_fetch: int,
) -> pl.DataFrame:
    print(f"Scanning {hf_parquet_glob} for rows {cursor_location} to {cursor_location + n_to_fetch}...")

    batch_df = pl.scan_parquet(hf_parquet_glob).slice(cursor_location, n_to_fetch).collect()
    return batch_df


def _write_single_file(image_data, original_filepath: str | None, storage_dir: Path) -> str:
    if isinstance(image_data, dict):
        img_bytes: bytes = image_data.get("bytes") or b""
        if original_filepath is None:
            original_filepath = image_data.get("path") or "unknown.jpg"
    else:
        img_bytes = image_data

    filename = Path(original_filepath or "unknown.jpg").name
    local_filepath = storage_dir / filename

    with open(local_filepath, "wb") as f:
        f.write(img_bytes)
    return str(local_filepath)


def save_images_and_clear_df(
    df: pl.DataFrame,
    image_column: str | list[str],
    image_file_path_column: str | list[str] | None,
    image_storage_path: Path,
) -> pl.DataFrame:
    """Saves image bytes concurrently and returns the lightweight manifest.

    Single image_column (str) → adds ``local_image_path`` column (backward compatible).
    Multiple image_columns (list) → adds ``local_{col}_path`` for each column.
    image_file_path_column=None → path extracted from the image struct's ``path`` field.
    """
    is_single = isinstance(image_column, str)

    if is_single:
        image_columns = [image_column]
        file_path_columns = [image_file_path_column]
        output_columns = ["local_image_path"]
    else:
        image_columns = image_column
        if image_file_path_column is None:
            file_path_columns = [None] * len(image_columns)
        elif isinstance(image_file_path_column, str):
            file_path_columns = [image_file_path_column] * len(image_columns)
        else:
            file_path_columns = image_file_path_column
        output_columns = [f"local_{col}_path" for col in image_columns]

    absolute_storage_dir = image_storage_path.resolve()
    absolute_storage_dir.mkdir(parents=True, exist_ok=True)

    for img_col, fp_col, out_col in zip(image_columns, file_path_columns, output_columns):
        if img_col not in df.columns:
            continue

        image_data_list = df[img_col].to_list()
        filepath_list = df[fp_col].to_list() if (fp_col and fp_col in df.columns) else [None] * len(image_data_list)

        local_paths = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_write_single_file, img, path, absolute_storage_dir)
                for img, path in zip(image_data_list, filepath_list)
            ]
            for future in futures:
                local_paths.append(future.result())

        df = df.with_columns(pl.Series(out_col, local_paths)).drop(img_col)

    return df


if __name__ == "__main__":
    fetch_hf_batch_lazily(
        hf_parquet_glob="hf://datasets/Multimodal-Fatima/COCO_captions_train/data/train-*-of-*.parquet",
        cursor_location=1000,
        n_to_fetch=100,
    )
