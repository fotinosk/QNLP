from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from qnlp.domain.datasets.dataset import VLMDataset


def vlm_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate a list of VLMDataset samples into a batch dict.

    Tensors are stacked. Everything else (strings, tuples, lists) is
    kept as a list — the model unpacks what it needs.
    """
    keys = batch[0].keys()
    result = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def get_dataloaders(
    train_parquet: str | Path,
    val_parquet: str | Path,
    test_parquet: str | Path,
    batch_size: int,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    image_columns: list[str] | None = None,
    compiled_columns: list[tuple[str, str, str]] | None = None,
    num_workers: int = 0,
) -> tuple[list[DataLoader], list[VLMDataset]]:
    """
    Build train/val/test DataLoaders from enriched parquet files.

    Returns [[train_loader, val_loader, test_loader], [train_ds, val_ds, test_ds]].

    Args:
        train_parquet: Path to the train split parquet.
        val_parquet: Path to the val split parquet.
        test_parquet: Path to the test split parquet.
        batch_size: Batch size for all loaders.
        train_transform: Image transform applied during training.
        val_transform: Image transform applied during validation and testing.
        image_columns: Passed to VLMDataset (default ["local_image_path"]).
        compiled_columns: Passed to VLMDataset (default [("diagram", "symbols", "caption")]).
        num_workers: DataLoader worker count.
    """
    train_ds = VLMDataset(
        train_parquet,
        image_columns=image_columns,
        compiled_columns=compiled_columns,
        image_transform=train_transform,
    )
    val_ds = VLMDataset(
        val_parquet,
        image_columns=image_columns,
        compiled_columns=compiled_columns,
        image_transform=val_transform,
    )
    test_ds = VLMDataset(
        test_parquet,
        image_columns=image_columns,
        compiled_columns=compiled_columns,
        image_transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=vlm_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vlm_collate_fn,
        num_workers=num_workers,
    )

    return [[train_loader, val_loader, test_loader], [train_ds, val_ds, test_ds]]
