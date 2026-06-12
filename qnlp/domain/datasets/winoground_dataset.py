from pathlib import Path
from typing import Any, Callable

import orjson
import polars as pl
import torch
import torchvision.io
from torch.utils.data import Dataset

from qnlp.domain.datasets.dataset import _deserialize_symbols


class WinogroundDataset(Dataset):
    """
    Dataset for the Winoground benchmark.

    Each parquet row is one Winoground item: 2 images + 2 captions forming a
    2x2 grid where (image_0, caption_0) and (image_1, caption_1) are the
    correct pairings.

    Two modes:

        "train" — yields 2 samples per item, one per correct image<->caption pair:
            index 2k   -> (image_0, true=caption_0, false=caption_1)
            index 2k+1 -> (image_1, true=caption_1, false=caption_0)
            __len__ = 2 * N

        "eval"  — yields 1 item per row with all 4 combinations so that
            text/image/group Winoground scores can be computed.
            __len__ = N

    Expected parquet columns (produced by WinogroundPairStrategy):
        pair_id, local_image_0_path, local_image_1_path,
        cap0_diagram, cap0_symbols, cap1_diagram, cap1_symbols
    """

    def __init__(
        self,
        parquet_path: str | Path,
        mode: str = "train",
        image_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_non_linear_contractions: bool = False,
    ):
        assert mode in ("train", "eval"), f"mode must be 'train' or 'eval', got '{mode}'"
        self.mode = mode
        self.image_transform = image_transform
        self.use_non_linear_contractions = use_non_linear_contractions
        self.df = pl.read_parquet(parquet_path)

        if use_non_linear_contractions and "cap0_path" not in self.df.columns:
            raise ValueError(
                "use_non_linear_contractions=True but parquet has no 'cap0_path' column. "
                "Recreate the dataset with compute_contraction_paths=True."
            )

    def __len__(self) -> int:
        return 2 * len(self.df) if self.mode == "train" else len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB).float().div(255.0)
        if self.image_transform is not None:
            img = self.image_transform(img)
        return img

    def _get_caption(self, row: dict[str, Any], index: int) -> tuple:
        diagram = row[f"cap{index}_diagram"]
        symbols = _deserialize_symbols(row[f"cap{index}_symbols"])
        if self.use_non_linear_contractions:
            raw_path = row[f"cap{index}_path"]
            path = [tuple(step) for step in orjson.loads(raw_path)] if raw_path else None
            return (diagram, symbols, path)
        return (diagram, symbols)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.mode == "train":
            item_idx = idx // 2
            which = idx % 2
            row = self.df.row(item_idx, named=True)

            if which == 0:
                image = self._load_image(row["local_image_0_path"])
                true_caption = self._get_caption(row, 0)
                false_caption = self._get_caption(row, 1)
            else:
                image = self._load_image(row["local_image_1_path"])
                true_caption = self._get_caption(row, 1)
                false_caption = self._get_caption(row, 0)

            return {
                "image": image,
                "true_caption": true_caption,
                "false_caption": false_caption,
                "pair_id": row["pair_id"],
            }

        else:  # eval
            row = self.df.row(idx, named=True)
            return {
                "image_0": self._load_image(row["local_image_0_path"]),
                "image_1": self._load_image(row["local_image_1_path"]),
                "caption_0": self._get_caption(row, 0),
                "caption_1": self._get_caption(row, 1),
                "pair_id": row["pair_id"],
            }


def winoground_train_collate_fn(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "true_captions": [item["true_caption"] for item in batch],
        "false_captions": [item["false_caption"] for item in batch],
        "pair_ids": [item["pair_id"] for item in batch],
    }


def winoground_eval_collate_fn(batch: list[dict]) -> dict:
    return {
        "images_0": torch.stack([item["image_0"] for item in batch]),
        "images_1": torch.stack([item["image_1"] for item in batch]),
        "captions_0": [item["caption_0"] for item in batch],
        "captions_1": [item["caption_1"] for item in batch],
        "pair_ids": [item["pair_id"] for item in batch],
    }
