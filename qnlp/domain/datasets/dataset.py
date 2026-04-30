from pathlib import Path
from typing import Any, Callable

import orjson
import polars as pl
import torch
import torchvision.io
from lambeq.backend.symbol import Symbol
from torch.utils.data import Dataset


class VLMDataset(Dataset):
    """
    Generic Vision-Language Dataset backed by an enriched parquet file.

    Loads all non-image data into RAM at init. Images are read from disk
    per __getitem__. No LMDB access at runtime.

    Args:
        parquet_path: Path to the enriched parquet produced by create_dataset
            or create_train_val_test_datasets.
        image_columns: Columns containing absolute image paths. Each is loaded
            as an image tensor in __getitem__.
        compiled_columns: List of (diagram_col, symbols_col, output_key) triples.
            In __getitem__, diagram and deserialized symbols are bundled into a
            (diagram_str, [Symbol, ...]) tuple under output_key, matching the
            interface expected by the tensor network models.
        image_transform: Optional transform applied to each loaded image tensor.

    Example — single caption (COCO style):
        VLMDataset(path, compiled_columns=[("diagram", "symbols", "caption")])
        # __getitem__ returns: {image, caption: (diagram, symbols), sample_id, ...}

    Example — contrastive pair (ARO style):
        VLMDataset(
            path,
            compiled_columns=[
                ("true_diagram", "true_symbols", "true_caption"),
                ("false_diagram", "false_symbols", "false_caption"),
            ],
        )
        # __getitem__ returns: {image, true_caption: (...), false_caption: (...), sample_id}
    """

    def __init__(
        self,
        parquet_path: str | Path,
        image_columns: list[str] | None = None,
        compiled_columns: list[tuple[str, str, str]] | None = None,
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.df = pl.read_parquet(parquet_path)
        self.image_columns = image_columns or ["local_image_path"]
        self.compiled_columns = compiled_columns or [("diagram", "symbols", "caption")]
        self.image_transform = image_transform

        # Columns consumed by compiled_columns — excluded from the raw dict output
        self._compiled_raw_cols = {col for diag_col, sym_col, _ in self.compiled_columns for col in (diag_col, sym_col)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.row(idx, named=True)

        result: dict[str, Any] = {}

        # Load images
        for col in self.image_columns:
            img = torchvision.io.read_image(row[col]).float().div(255.0)
            if self.image_transform is not None:
                img = self.image_transform(img)
            result[col] = img

        # Bundle compiled (diagram, symbols) pairs
        for diag_col, sym_col, output_key in self.compiled_columns:
            diagram = row[diag_col]
            raw_symbols = row[sym_col]
            symbols = _deserialize_symbols(raw_symbols)
            result[output_key] = (diagram, symbols)

        # Pass through all remaining columns
        for col, val in row.items():
            if col not in self.image_columns and col not in self._compiled_raw_cols:
                result[col] = val

        return result


def _deserialize_symbols(raw: str | list | None) -> list[Symbol]:
    """Deserialize symbols from JSON string or raw list to Symbol objects."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = orjson.loads(raw)
    return [Symbol(**entry) if isinstance(entry, dict) else entry for entry in raw]
