import os
from typing import Tuple, List
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from diskcache import Cache

from lambeq.backend.tensor import Dim
from lambeq.backend.symbol import Symbol

from lambeq import AtomicType, Rewriter

from sklearn.model_selection import train_test_split


from qnlp.discoclip2.parser.cached_bobcat import CachedBobcatParser
from qnlp.discoclip2.parser.asnsatz import CustomMPSAnsatz
from qnlp.discoclip2.models.bobcat_text_processor import BobcatTextProcessor


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


class ARODataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 text_transform=None,
                 crop: bool = True,
                 progress = True):
        self.dataset = pd.read_json(data_path)
        self.crop = crop
        self.progress = progress

        required_columns = ['true_caption', 'false_caption', 'image_path']
        if not all(col in self.dataset.columns for col in required_columns):
            raise ValueError(f"ARODataset must contain the following columns: {required_columns}")

        if text_transform is not None:
            print("Precomputing text")
            self.precompute_text(text_transform)

    def precompute_text(self, text_transform):
        # build a map from text to einsum inputs
        self.text_map = {}
        symbol_size_set = set()

        captions = list(set(self.dataset['true_caption'].tolist() + self.dataset['false_caption'].tolist()))
        batch_size = 6

        for i in tqdm(range(0, len(captions), batch_size), disable=not self.progress):
            batch_captions = captions[i:i + batch_size]

            einsum_inputs = text_transform(batch_captions, suppress_exceptions=True)['einsum_inputs']
            for caption, einsum_input in zip(batch_captions, einsum_inputs):
                self.text_map[caption] = einsum_input
                if einsum_input is not None:
                    _, symbol_size_list = einsum_input
                    for symbol, size in symbol_size_list:
                        symbol_size_set.add((symbol, size))
        self.symbols, self.sizes = zip(*symbol_size_set) if symbol_size_set else ([], [])
                    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.crop:
            x = self.dataset.iloc[idx]['bbox_x']
            y = self.dataset.iloc[idx]['bbox_y']
            w = self.dataset.iloc[idx]['bbox_w']
            h = self.dataset.iloc[idx]['bbox_h']
            image = f'{self.dataset.iloc[idx]["image_path"].split(".")[0]}_{x}_{y}_{w}_{h}.jpg'
        else:
            image = self.dataset.iloc[idx]['image_path']

        true_caption = self.dataset.iloc[idx]['true_caption']
        false_caption = self.dataset.iloc[idx]['false_caption']

        true_caption = self.text_map.get(true_caption, None)
        false_caption = self.text_map.get(false_caption, None)

        def remove_shape(einsum_input):
            if einsum_input is None:
                return None
            einsum_expr, symbol_size_list = einsum_input
            return (einsum_expr, [sym for sym, _ in symbol_size_list])
        true_caption = remove_shape(true_caption)
        false_caption = remove_shape(false_caption)

        return {
            "image": image,
            "true_caption": true_caption,
            "false_caption": false_caption,
            "index": idx
        }
    
    def state_dict(self):
        return {
            'text_map': self.text_map,
            'symbols': self.symbols,
            'sizes': self.sizes
        }
    
    def load_state_dict(self, state_dict):
        self.text_map = state_dict['text_map']
        self.symbols = state_dict['symbols']
        self.sizes = state_dict['sizes']

def aro_tn_collate_fn(batch):
    valid = [item['true_caption'] is not None and item['false_caption'] is not None for item in batch]
    # print the number of invalid samples if any
    if not all(valid):
        print(f"Found {sum(not v for v in valid)} invalid samples")

    images = [item['image'] for item, v in zip(batch, valid) if v]
    true_captions = [item['true_caption'] for item, v in zip(batch, valid) if v]
    false_captions = [item['false_caption'] for item, v in zip(batch, valid) if v]
    indices = [item['index'] for item, v in zip(batch, valid) if v]

    return {
        "images": images,
        "true_captions": true_captions,
        "false_captions": false_captions,
        "indices": indices
    }


def get_aro_dataset(
    data_path: str, dim: int, bond_dim: int, progress: bool = True
) -> ARODataset:
    
    cache_dir = os.path.join(os.path.dirname(data_path), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache = Cache(cache_dir, size_limit=10 * 2**30) # 10GB
   
    cache_key = os.path.join(
        os.path.basename(data_path),
        f"bobcat_dim{dim}_bond{bond_dim}"
    )
    
    state_dict = cache.get(cache_key)
    if state_dict is not None:
        print(f"Loading cached dataset from {cache_key}")

        dataset = ARODataset(data_path=data_path, text_transform=None)
        dataset.load_state_dict(state_dict)
        return dataset

    ansatz = CustomMPSAnsatz(
        {
            AtomicType.SENTENCE: Dim(dim),
            AtomicType.NOUN: Dim(dim),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(dim),
        },
        bond_dim=bond_dim,
    )

    rules = [
        "auxiliary",
        "connector",
        "determiner",
        "postadverb",
        "preadverb",
        "prepositional_phrase",
        "coordination",
        "object_rel_pronoun",
        "subject_rel_pronoun",
    ]
    rewriter = Rewriter(rules)
    bobcat_parser = CachedBobcatParser()
    text_transform = BobcatTextProcessor(
        ccg_parser=bobcat_parser,
        ansatz=ansatz,
        rewriter=rewriter,
    )
    
    dataset = ARODataset(
        data_path=data_path, text_transform=text_transform, progress=progress
    )

    print(f"Caching dataset to {cache_key}")
    cache.set(cache_key, dataset.state_dict())
    cache.close()
    return dataset

class ProcessedARODataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                ):
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
                # self.symbols.append(x[0])
                self.sizes.append(x[1])
        
        assert isinstance(self.true_captions[0][0], str)
        assert isinstance(self.true_captions[0][1][0], Symbol)
        assert isinstance(self.symbols[0], Symbol)
        
        print(f"Initialized dataset for {data_path}. Final size: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Optimized: Just simple list indexing. No pandas, no string splits.
        return {
            "image": self.image_paths[idx],
            "true_caption": self.true_captions[idx],
            "false_caption": self.false_captions[idx],
            "index": idx
        }

    @staticmethod
    def remove_shape(einsum_input) -> Tuple[str, List[Symbol]]:
        einsum_expr, symbol_size_list = einsum_input
        return (
            einsum_expr, 
            [Symbol(**sym) for sym, _ in symbol_size_list]
        )

