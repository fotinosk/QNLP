import os
from typing import Tuple, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from lambeq.backend.symbol import Symbol

from sklearn.model_selection import train_test_split

from qnlp.discoclip2.models.image_model import preprocess



def create_train_val_test_split(vga_data_path, vgr_data_path,
                                vga_save_path, vgr_save_path, 
                                train_size=0.7, random_state=42):
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
    images = list(set(df_vga['image_id'].tolist() + df_vgr['image_id'].tolist()))

    images_train, images_val_test = train_test_split(images, test_size=1-train_size, random_state=random_state)
    images_val, images_test = train_test_split(images_val_test, test_size=0.5, random_state=random_state)

    print(f"Train/Val/Test split created with {len(images_train)} train, {len(images_val)} val, and {len(images_test)} test images.")

    train_df_vgr = df_vgr[df_vgr['image_id'].isin(images_train)]
    val_df_vgr = df_vgr[df_vgr['image_id'].isin(images_val)]
    test_df_vgr = df_vgr[df_vgr['image_id'].isin(images_test)]

    print (f"Train/Val/Test split created with {len(train_df_vgr)} train, {len(val_df_vgr)} val, and {len(test_df_vgr)} test samples in VGR dataset.")

    train_df_vga = df_vga[df_vga['image_id'].isin(images_train)]
    val_df_vga = df_vga[df_vga['image_id'].isin(images_val)]
    test_df_vga = df_vga[df_vga['image_id'].isin(images_test)]

    print (f"Train/Val/Test split created with {len(train_df_vga)} train, {len(val_df_vga)} val, and {len(test_df_vga)} test samples in VGA dataset.")

    # make sure the save directories exist
    os.makedirs(vgr_save_path, exist_ok=True)
    os.makedirs(vga_save_path, exist_ok=True)

    train_df_vgr.to_json(os.path.join(vgr_save_path, 'train.json'), orient='records')
    val_df_vgr.to_json(os.path.join(vgr_save_path, 'val.json'), orient='records')
    test_df_vgr.to_json(os.path.join(vgr_save_path, 'test.json'), orient='records')

    print(f"VGR datasets saved to {vgr_save_path}")

    train_df_vga.to_json(os.path.join(vga_save_path, 'train.json'), orient='records')
    val_df_vga.to_json(os.path.join(vga_save_path, 'val.json'), orient='records')
    test_df_vga.to_json(os.path.join(vga_save_path, 'test.json'), orient='records')

    print(f"VGA datasets saved to {vga_save_path}")


def aro_tn_collate_fn(batch):
    images  = []
    true_captions = []
    false_captions = []
    indices = []
    image_names = []
        
    for el in batch:
        true_captions.append(el['true_caption'])
        false_captions.append(el['false_caption'])
        indices.append(el['index'])
        images.append(el['image'])
        image_names.append(el['image_name'])
    return {
        "images": torch.stack(images),
        "true_captions": true_captions,
        "false_captions": false_captions,
        "indices": indices,
        "image_names": image_names
    }


class ProcessedARODataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 image_dir_path: str | None = None,
                 return_images: bool = False
        ):
        self.return_images = return_images
        self.image_path = image_dir_path
        
        raw_dataset = pd.read_json(data_path)
        dir_data_path, file_name = data_path.rsplit("/", 1)
        file_name = file_name.split(".")[0]
        processed_file_name = f"{dir_data_path}/{file_name}_processed_512.jsonl"
        self.processed_dataset = pd.read_json(processed_file_name, lines=True)
        
        self.text_map = {}        
        for _, row in self.processed_dataset.iterrows():
            self.text_map[row["caption"]] = self.remove_shape((row['diagram'], row['symbols']))

        valid_mask = (
            raw_dataset['true_caption'].isin(self.text_map.keys()) & 
            raw_dataset['false_caption'].isin(self.text_map.keys())
        )
        
        self.dataset = raw_dataset[valid_mask].reset_index(drop=True)

        self.image_paths = self.dataset['image_path'].to_list()

        self.true_captions = [self.text_map[c] for c in self.dataset['true_caption']]
        self.false_captions = [self.text_map[c] for c in self.dataset['false_caption']]

        self.symbols = []
        self.sizes = []
        for row in self.processed_dataset['symbols']:
            for x in row:
                self.symbols.append(Symbol(**x[0]))
                self.sizes.append(x[1])
        
        print(f"Initialized dataset for {data_path}. Final size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        if self.return_images:
            img_path = f"{self.image_path}/{img_name}"
            
            # Using a context manager or just Image.open
            image = preprocess(Image.open(img_path).convert("RGB"))
        else: 
            image = img_name
        
        return {
            "image": image,
            "true_caption": self.true_captions[idx],
            "false_caption": self.false_captions[idx],
            "index": idx,
            "image_name": img_name
        }

    @staticmethod
    def remove_shape(einsum_input) -> Tuple[str, List[Symbol]]:
        einsum_expr, symbol_size_list = einsum_input
        return (
            einsum_expr, 
            [Symbol(**sym) for sym, _ in symbol_size_list]
        )

