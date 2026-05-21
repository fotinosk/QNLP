import json
from pathlib import Path
from typing import Self

import polars as pl

from qnlp.constants import constants
from qnlp.core.data_engine.atlas.hf_utils import fetch_hf_batch_lazily, save_images_and_clear_df
from qnlp.core.data_engine.atlas.local_utils import read_local_manifest, resolve_image_paths

ATLAS_DIR = Path.cwd() / constants.atlases_path


class Atlas:
    def __init__(
        self,
        name: str,
        source_path_or_url: str,
        data_manifest_location: Path,
        cursor_location: int,
        image_column: str = "image",
        image_file_path_column: str = "filepath",
    ):
        self.name = name
        self.remote_location = source_path_or_url
        self.data_manifest_location = Path(data_manifest_location)
        self.cursor_location = cursor_location

        self.image_column = image_column
        self.image_file_path_column = image_file_path_column

        # Paths
        self.metadata_location = self.data_manifest_location.parent / "metadata.json"
        self.image_path = self.data_manifest_location.parent / "raw_images"

        # Load existing manifest or start fresh
        if self.data_manifest_location.exists():
            self.manifest = pl.read_parquet(self.data_manifest_location)
        else:
            self.manifest = pl.DataFrame()

        self._is_hf = self.remote_location.startswith("hf://")

    @classmethod
    def load_atlas(cls, atlas_metadata_location: str | Path) -> Self:
        """Loads an existing Atlas from its metadata JSON file."""
        with open(atlas_metadata_location, "r") as f:
            atlas_metadata = json.load(f)

        return cls(
            name=atlas_metadata["name"],
            source_path_or_url=atlas_metadata["source_path_or_url"],
            data_manifest_location=Path(atlas_metadata["data_manifest_location"]),
            cursor_location=atlas_metadata["cursor_location"],
            image_column=atlas_metadata.get("image_column", "image"),
            image_file_path_column=atlas_metadata.get("image_file_path_column", "filepath"),
        )

    @classmethod
    def create_atlas(
        cls,
        name: str,
        source_path_or_url: str,
        image_column: str = "image",
        image_file_path_column: str = "filepath",
    ) -> Self:
        """Creates a new, blank Atlas and sets up the directory structure."""
        atlas_dir = ATLAS_DIR / name

        if atlas_dir.exists():
            raise FileExistsError(
                f"An atlas named '{name}' already exists at {atlas_dir}. "
                "Use `Atlas.load_atlas()` to resume it, or delete the directory to start fresh."
            )

        atlas_dir.mkdir(parents=True, exist_ok=False)
        data_manifest_location = atlas_dir / "data_manifest.parquet"

        instance = cls(
            name=name,
            source_path_or_url=source_path_or_url,
            data_manifest_location=data_manifest_location,
            cursor_location=0,
            image_column=image_column,
            image_file_path_column=image_file_path_column,
        )

        instance.image_path.mkdir(parents=True, exist_ok=True)
        instance._save_metadata()

        return instance

    def _save_metadata(self) -> None:
        """Persists the cursor and configuration to disk with soft atomicity."""
        state = {
            "name": self.name,
            "source_path_or_url": self.remote_location,
            "data_manifest_location": str(self.data_manifest_location),
            "cursor_location": self.cursor_location,
            "image_column": self.image_column,
            "image_file_path_column": self.image_file_path_column,
        }

        temp_path = self.metadata_location.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(state, f, indent=4)
        temp_path.replace(self.metadata_location)

    def ingest_data_from_local(
        self,
        manifest_path: str | Path,
        image_path_column: str | None = None,
        image_dir: str | Path | None = None,
        image_filename_column: str | None = None,
        n: int | None = None,
    ) -> None:
        """
        Ingest from a local manifest file. Images are never copied — local_image_path
        stores resolved absolute pointers to the original files.

        Args:
            manifest_path: Path to .parquet, .csv, .json, or .jsonl manifest.
            image_path_column: Column in the manifest containing full image paths.
            image_dir: Base directory when paths are bare filenames.
            image_filename_column: Column of bare filenames (used with image_dir).
            n: Max rows to ingest this call. None = ingest all remaining rows.
        """
        df = read_local_manifest(manifest_path)

        # Apply cursor and optional row limit
        remaining = df.slice(self.cursor_location)
        if n is not None:
            remaining = remaining.head(n)

        if remaining.is_empty():
            print("No more data to ingest.")
            return

        df = resolve_image_paths(
            remaining,
            image_path_column=image_path_column,
            image_dir=image_dir,
            image_filename_column=image_filename_column,
        )

        n_rows = len(df)
        sample_ids = [f"{self.name}_{i}" for i in range(self.cursor_location, self.cursor_location + n_rows)]
        df = df.with_columns(pl.Series("sample_id", sample_ids))

        if self.manifest.is_empty():
            self.manifest = df
        else:
            self.manifest = pl.concat([self.manifest, df], how="diagonal")

        self.manifest.write_parquet(self.data_manifest_location)
        self.cursor_location += n_rows
        self._save_metadata()
        print(f"Successfully ingested {n_rows} records. Cursor at {self.cursor_location}.")

    def ingest_dataframe(self, df: pl.DataFrame) -> None:
        """
        Ingest a pre-prepared DataFrame. The caller is responsible for resolving
        all image paths to absolute paths. Images are never copied — paths are
        stored as-is. Useful for multi-image datasets (e.g. SVO, Winoground)
        where a single local_image_path column is insufficient.
        """
        n_rows = len(df)
        sample_ids = [f"{self.name}_{i}" for i in range(self.cursor_location, self.cursor_location + n_rows)]
        df = df.with_columns(pl.Series("sample_id", sample_ids))

        if self.manifest.is_empty():
            self.manifest = df
        else:
            self.manifest = pl.concat([self.manifest, df], how="diagonal")

        self.manifest.write_parquet(self.data_manifest_location)
        self.cursor_location += n_rows
        self._save_metadata()
        print(f"Successfully ingested {n_rows} records. Cursor at {self.cursor_location}.")

    def ingest_data_from_remote(self, n: int = 100) -> None:
        if not self._is_hf:
            raise NotImplementedError("Loading for local files not implemented yet")

        new_dataset = fetch_hf_batch_lazily(
            hf_parquet_glob=self.remote_location, cursor_location=self.cursor_location, n_to_fetch=n
        )

        if new_dataset.is_empty():
            print("No more data to ingest.")
            return

        processed = save_images_and_clear_df(
            df=new_dataset,
            image_column=self.image_column,
            image_file_path_column=self.image_file_path_column,
            image_storage_path=self.image_path,
        )

        n_rows = len(processed)
        sample_ids = [f"{self.name}_{i}" for i in range(self.cursor_location, self.cursor_location + n_rows)]
        processed = processed.with_columns(pl.Series("sample_id", sample_ids))

        if self.manifest.is_empty():
            self.manifest = processed
        else:
            self.manifest = pl.concat([self.manifest, processed], how="vertical")

        self.manifest.write_parquet(self.data_manifest_location)

        self.cursor_location += n
        self._save_metadata()
        print(f"Successfully ingested {n_rows} records. Cursor at {self.cursor_location}.")
