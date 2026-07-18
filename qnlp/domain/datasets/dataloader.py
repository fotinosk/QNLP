from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Subset

from qnlp.domain.datasets.dataset import VLMDataset
from qnlp.domain.datasets.topology_bucket_sampler import TopologyBucketSampler


def _diagrams_for(dataset: VLMDataset | Subset, diagram_col: str) -> list[str]:
    """Diagram string per row, in the dataset's own index space — works for a
    plain VLMDataset or a Subset wrapping one (e.g. the test_size-capped test set)."""
    if isinstance(dataset, Subset):
        base = dataset.dataset.df[diagram_col].to_list()
        return [base[i] for i in dataset.indices]
    return dataset.df[diagram_col].to_list()


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
    compiled_columns: list[tuple] | None = None,
    num_workers: int = 4,
    use_non_linear_contractions: bool = False,
    test_size: int | None = None,
    topology_bucketing: bool = False,
    min_bucket_size: int = 64,
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
        topology_bucketing: If True, batch by shared diagram topology via
            TopologyBucketSampler (see that module) so EinsumModel's batched
            same-topology fast path engages for linear-mode training. Only
            supported for single-compiled-column datasets (e.g. COCO's
            (diagram, symbols, caption) — not ARO-style true/false pairs).
            Applied to all three splits — but val/test are ALWAYS built with
            tail_fraction=1.0 (full coverage, no truncation), hardcoded here,
            independent of anything train-side experimentation adds later.
        min_bucket_size: Passed to TopologyBucketSampler for every split.
    """
    train_ds = VLMDataset(
        train_parquet,
        image_columns=image_columns,
        compiled_columns=compiled_columns,
        image_transform=train_transform,
        use_non_linear_contractions=use_non_linear_contractions,
    )
    val_ds = VLMDataset(
        val_parquet,
        image_columns=image_columns,
        compiled_columns=compiled_columns,
        image_transform=val_transform,
        use_non_linear_contractions=use_non_linear_contractions,
    )
    test_ds = VLMDataset(
        test_parquet,
        image_columns=image_columns,
        compiled_columns=compiled_columns,
        image_transform=val_transform,
        use_non_linear_contractions=use_non_linear_contractions,
    )

    test_dataset = Subset(test_ds, range(min(test_size, len(test_ds)))) if test_size is not None else test_ds

    worker_kwargs = dict(
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if topology_bucketing:
        cols = compiled_columns or [("diagram", "symbols", "caption")]
        if len(cols) != 1:
            raise ValueError(
                "topology_bucketing only supports single-compiled-column datasets "
                f"(e.g. COCO's (diagram, symbols, caption)); got {len(cols)} compiled_columns."
            )
        diagram_col = cols[0][0]

        # tail_fraction is hardcoded to 1.0 for every split here (not threaded
        # through from a parameter) — val/test must never truncate, and train
        # stays lossless too until a dedicated train-only knob is added later.
        train_sampler = TopologyBucketSampler(
            _diagrams_for(train_ds, diagram_col),
            batch_size=batch_size,
            min_bucket_size=min_bucket_size,
            tail_fraction=1.0,
            shuffle=True,
        )
        val_sampler = TopologyBucketSampler(
            _diagrams_for(val_ds, diagram_col),
            batch_size=batch_size,
            min_bucket_size=min_bucket_size,
            tail_fraction=1.0,
            shuffle=False,
        )
        test_sampler = TopologyBucketSampler(
            _diagrams_for(test_dataset, diagram_col),
            batch_size=batch_size,
            min_bucket_size=min_bucket_size,
            tail_fraction=1.0,
            shuffle=False,
        )

        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=vlm_collate_fn, **worker_kwargs)
        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=vlm_collate_fn, **worker_kwargs)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=vlm_collate_fn, **worker_kwargs)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=vlm_collate_fn,
            **worker_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=vlm_collate_fn,
            **worker_kwargs,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=vlm_collate_fn,
            **worker_kwargs,
        )

    return [[train_loader, val_loader, test_loader], [train_ds, val_ds, test_ds]]
