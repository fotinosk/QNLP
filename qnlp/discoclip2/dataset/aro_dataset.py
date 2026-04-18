import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import torch
from lambeq.backend.symbol import Symbol
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def create_train_val_test_split(
    vga_data_path,
    vgr_data_path,
    vga_save_path,
    vgr_save_path,
    train_size=0.7,
    random_state=42,
):
    """
    Create a train/val/test split from the given data paths.

    Args:
        vga_data_path (str): Path to the VGA data.
        vgr_data_path (str): Path to the VGR data.
        train_size (float): Proportion of the dataset to include in the train split.
    Returns:
        tuple: Paths for train, validation, and test datasets.
    """

    df_vga = pd.read_json(vga_data_path)
    df_vgr = pd.read_json(vgr_data_path)
    images = list(set(df_vga["image_id"].tolist() + df_vgr["image_id"].tolist()))

    images_train, images_val_test = train_test_split(images, test_size=1 - train_size, random_state=random_state)
    images_val, images_test = train_test_split(images_val_test, test_size=0.5, random_state=random_state)

    print(
        f"Train/Val/Test split created with {len(images_train)} train, {len(images_val)} val, "
        f"and {len(images_test)} test images."
    )

    train_df_vgr = df_vgr[df_vgr["image_id"].isin(images_train)]
    val_df_vgr = df_vgr[df_vgr["image_id"].isin(images_val)]
    test_df_vgr = df_vgr[df_vgr["image_id"].isin(images_test)]

    print(
        f"Train/Val/Test split created with {len(train_df_vgr)} train, {len(val_df_vgr)} val, "
        f"and {len(test_df_vgr)} test samples in VGR dataset."
    )

    train_df_vga = df_vga[df_vga["image_id"].isin(images_train)]
    val_df_vga = df_vga[df_vga["image_id"].isin(images_val)]
    test_df_vga = df_vga[df_vga["image_id"].isin(images_test)]

    print(
        f"Train/Val/Test split created with {len(train_df_vga)} train, {len(val_df_vga)} val, "
        f"and {len(test_df_vga)} test samples in VGA dataset."
    )

    # make sure the save directories exist
    os.makedirs(vgr_save_path, exist_ok=True)
    os.makedirs(vga_save_path, exist_ok=True)

    train_df_vgr.to_json(os.path.join(vgr_save_path, "train.json"), orient="records")
    val_df_vgr.to_json(os.path.join(vgr_save_path, "val.json"), orient="records")
    test_df_vgr.to_json(os.path.join(vgr_save_path, "test.json"), orient="records")

    print(f"VGR datasets saved to {vgr_save_path}")

    train_df_vga.to_json(os.path.join(vga_save_path, "train.json"), orient="records")
    val_df_vga.to_json(os.path.join(vga_save_path, "val.json"), orient="records")
    test_df_vga.to_json(os.path.join(vga_save_path, "test.json"), orient="records")

    print(f"VGA datasets saved to {vga_save_path}")


def aro_tn_collate_fn(batch):
    images = []
    true_captions = []
    false_captions = []
    indices = []
    image_names = []

    for el in batch:
        true_captions.append(el["true_caption"])
        false_captions.append(el["false_caption"])
        indices.append(el["index"])
        images.append(el["image"])
        image_names.append(el["image_name"])
    return {
        "images": torch.stack(images),
        "true_captions": true_captions,
        "false_captions": false_captions,
        "indices": indices,
        "image_names": image_names,
    }


class ProcessedARODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        return_images: bool = False,
        image_processing_fn: Callable = lambda x: x,
    ):
        self.return_images = return_images
        self.process_image = image_processing_fn

        # Use pathlib for cross-platform compatibility
        data_path = Path(data_path)
        raw_dataset = pd.read_json(data_path)

        processed_file_name = data_path.parent / f"{data_path.stem}_processed_512.jsonl"
        if not processed_file_name.exists():
            raise FileNotFoundError(f"Processed file not found: {processed_file_name}")

        self.processed_dataset = pd.read_json(processed_file_name, lines=True)

        # Build text map efficiently
        self.text_map = {
            row["caption"]: self.remove_shape((row["diagram"], row["symbols"]))
            for row in self.processed_dataset.to_dict("records")
        }

        # Filter out 2D captions from text_map
        self.text_map = {
            caption: processed
            for caption, processed in self.text_map.items()
            if self._is_1d_output(processed[0])  # processed[0] is the einsum expression
        }

        # Filter and track dropped rows
        valid_mask = raw_dataset["true_caption"].isin(self.text_map.keys()) & raw_dataset["false_caption"].isin(
            self.text_map.keys()
        )

        dropped_count = (~valid_mask).sum()
        if dropped_count > 0:
            logging.info(f"Dropped {dropped_count} rows with missing or 2D captions from {data_path}")

        # Keep single source of truth - filtered dataframe
        self.dataset = raw_dataset[valid_mask].reset_index(drop=True)

        if len(self.dataset) == 0:
            raise ValueError(f"No valid data found in {data_path}")

        # Pre-compute processed captions as new columns
        self.dataset["true_caption_processed"] = self.dataset["true_caption"].map(self.text_map)
        self.dataset["false_caption_processed"] = self.dataset["false_caption"].map(self.text_map)

        # Optional: Extract symbols if needed elsewhere (lazy evaluation)
        self._symbols_cache = None
        self._sizes_cache = None

        logging.info(f"Initialized dataset for {data_path}. Final size: {len(self.dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataset.iloc[idx]
        img_name = row["image_path"]

        try:
            if self.return_images:
                image = self.process_image(Image.open(img_name).convert("RGB"))
            else:
                image = img_name
        except Exception as e:
            logging.error(f"Failed to load image {img_name} at index {idx}: {e}")
            raise

        return {
            "image": image,
            "true_caption": row["true_caption_processed"],
            "false_caption": row["false_caption_processed"],
            "index": idx,
            "image_name": img_name,
        }

    @staticmethod
    def _is_1d_output(einsum_expr: str) -> bool:
        """
        Check if the einsum expression has a 1D output.
        Examples:
        - "jb,bd,df,fh,h,jk,km,m->j" -> True (1D output: 'j')
        - "ab,bc,cd->ad" -> False (2D output: 'ad')
        - "ij,jk->ik" -> False (2D output: 'ik')
        """
        if "->" not in einsum_expr:
            return True  # No explicit output, assume 1D

        output_part = einsum_expr.split("->")[1].strip()
        # Count letters only (ignore special characters if any)
        output_dims = re.findall(r"[a-zA-Z]", output_part)
        return len(output_dims) == 1

    @property
    def symbols(self) -> List[Symbol]:
        """Lazy evaluation of symbols list if needed."""
        if self._symbols_cache is None:
            self._symbols_cache = []
            for row in self.processed_dataset["symbols"]:
                for x in row:
                    self._symbols_cache.append(Symbol(**x[0]))
        return self._symbols_cache

    @property
    def sizes(self) -> List[Any]:
        """Lazy evaluation of sizes list if needed."""
        if self._sizes_cache is None:
            self._sizes_cache = []
            for row in self.processed_dataset["symbols"]:
                for x in row:
                    self._sizes_cache.append(x[1])
        return self._sizes_cache

    @staticmethod
    def remove_shape(einsum_input: Tuple[str, List[Tuple[Dict, Any]]]) -> Tuple[str, List[Symbol]]:
        einsum_expr, symbol_size_list = einsum_input
        return (einsum_expr, [Symbol(**sym) for sym, _ in symbol_size_list])

    def get_raw_dataframe(self) -> pd.DataFrame:
        """Return the underlying dataframe for debugging/inspection."""
        return self.dataset.copy()

    def to_dict_records(self) -> List[Dict]:
        """Convert dataset to list of dicts for faster access if needed."""
        return self.dataset.to_dict("records")


if __name__ == "__main__":
    ds = ProcessedARODataset(data_path="data/aro/processed/combined/test.json")
    print(ds.__getitem__(100))
