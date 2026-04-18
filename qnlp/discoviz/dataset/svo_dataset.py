import os
from typing import Callable, List, Tuple

import pandas as pd
import torch
from lambeq.backend.symbol import Symbol
from PIL import Image
from torch.utils.data import Dataset


def svo_tn_collate_fn(batch):
    pos_images = []
    neg_images = []
    sentences = []

    for el in batch:
        pos_images.append(el["pos_image"])
        neg_images.append(el["neg_image"])
        sentences.append(el["sentence"])
    return {
        "pos_images": torch.stack(pos_images),
        "neg_images": torch.stack(neg_images),
        "sentences": sentences,
    }


class ProcessedSVODataset(Dataset):
    def __init__(self, data_path: str, image_dir_path: str | None = None, image_processing_fn: Callable = lambda x: x):
        self.image_path = image_dir_path
        self.process_image = image_processing_fn

        raw_dataset = pd.read_csv(data_path)
        dir_data_path, file_name = data_path.rsplit("/", 1)
        file_name = file_name.split(".")[0]
        processed_file_name = f"{dir_data_path}/{file_name}_processed_512.jsonl"
        valid_images = [int(x.strip(".jpg")) for x in os.listdir(image_dir_path)]

        self.processed_dataset = pd.read_json(processed_file_name, lines=True)
        self.dataset = raw_dataset

        self.dataset = self.dataset[
            self.dataset["pos_image_id"].isin(valid_images) & self.dataset["neg_image_id"].isin(valid_images)
        ]

        self.text_map = {}
        for _, row in self.processed_dataset.iterrows():
            self.text_map[row["caption"]] = self.remove_shape((row["diagram"], row["symbols"]))

        self.sentences = [self.text_map[c] for c in self.dataset["corrected_sentence"]]
        self.pos_images = self.dataset["pos_image_id"].to_list()
        self.neg_images = self.dataset["neg_image_id"].to_list()

        self.symbols = []
        self.sizes = []
        for row in self.processed_dataset["symbols"]:
            for x in row:
                self.symbols.append(Symbol(**x[0]))
                self.sizes.append(x[1])

        print(f"Initialized dataset for {data_path}. Final size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pos_image_path = f"{self.image_path}/{self.pos_images[idx]}.jpg"
        neg_image_path = f"{self.image_path}/{self.neg_images[idx]}.jpg"

        return {
            "pos_image": self.process_image(Image.open(pos_image_path).convert("RGB")),
            "neg_image": self.process_image(Image.open(neg_image_path).convert("RGB")),
            "sentence": self.sentences[idx],
        }

    @staticmethod
    def remove_shape(einsum_input) -> Tuple[str, List[Symbol]]:
        einsum_expr, symbol_size_list = einsum_input
        return (einsum_expr, [Symbol(**sym) for sym, _ in symbol_size_list])


if __name__ == "__main__":
    from torchvision.transforms.v2 import functional as F

    from qnlp.discoviz.models.image_model import val_preprocess

    ds = ProcessedSVODataset(
        data_path="data/svo/processed/test.csv",
        image_dir_path="data/svo/raw/images",
        image_processing_fn=val_preprocess,
    )
    item = ds.__getitem__(100)
    print(item["sentence"])
    F.to_pil_image(item["pos_image"]).show()
    F.to_pil_image(item["neg_image"]).show()
