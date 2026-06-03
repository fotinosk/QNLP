from pathlib import Path

import polars as pl


def read_local_manifest(manifest_path: str | Path) -> pl.DataFrame:
    """Read a manifest file. Supports .parquet, .csv, .json, .jsonl."""
    path = Path(manifest_path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    elif suffix == ".csv":
        return pl.read_csv(path)
    elif suffix in (".json", ".jsonl"):
        return pl.read_ndjson(path)
    else:
        raise ValueError(f"Unsupported manifest format '{suffix}'. Use .parquet, .csv, .json, or .jsonl.")


def resolve_image_paths(
    df: pl.DataFrame,
    image_path_column: str | None,
    image_dir: str | Path | None,
    image_filename_column: str | None,
) -> pl.DataFrame:
    """
    Add a `local_image_path` column of resolved absolute paths. Images are never copied.

    Two modes:
      - Full path: `image_path_column` is a column of absolute (or resolvable) paths.
      - Filename + dir: `image_filename_column` contains bare filenames; `image_dir` is the base.
    """
    if image_path_column is not None:
        paths = [str(Path(p).resolve()) for p in df[image_path_column].to_list()]
    elif image_dir is not None and image_filename_column is not None:
        base = Path(image_dir).resolve()
        paths = [str(base / fname) for fname in df[image_filename_column].to_list()]
    else:
        raise ValueError(
            "Provide either `image_path_column` (column of full paths) "
            "or both `image_dir` and `image_filename_column`."
        )

    return df.with_columns(pl.Series("local_image_path", paths))
