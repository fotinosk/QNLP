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


def _write_single_file(image_data: bytes, original_filepath: str, storage_dir: Path) -> str:
    original_filename = Path(original_filepath).name
    img_bytes = image_data.get("bytes") if isinstance(image_data, dict) else image_data

    local_filepath = storage_dir / original_filename

    with open(local_filepath, "wb") as f:
        f.write(img_bytes)
    return str(local_filepath)


def save_images_and_clear_df(
    df: pl.DataFrame,
    image_column: str,
    image_file_path_column: str,
    image_storage_path: Path,
) -> pl.DataFrame:
    """Saves image bytes concurrently and returns the lightweight manifest."""

    if df.is_empty() or image_column not in df.columns or image_file_path_column not in df.columns:
        return df

    absolute_storage_dir = image_storage_path.resolve()
    absolute_storage_dir.mkdir(parents=True, exist_ok=True)

    image_data_list = df[image_column].to_list()
    filepath_list = df[image_file_path_column].to_list()

    local_paths = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_write_single_file, img, path, absolute_storage_dir)
            for img, path in zip(image_data_list, filepath_list)
        ]

        for future in futures:
            local_paths.append(future.result())

    return df.with_columns(pl.Series("local_image_path", local_paths)).drop(image_column)


if __name__ == "__main__":
    fetch_hf_batch_lazily(
        hf_parquet_glob="hf://datasets/Multimodal-Fatima/COCO_captions_train/data/train-*-of-*.parquet",
        cursor_location=1000,
        n_to_fetch=100,
    )
