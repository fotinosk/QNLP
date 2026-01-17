import pandas as pd

from lambeq.backend.symbol import Symbol
from torch.utils.data import Dataset


class ARODataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 crop: bool = True,
                 progress = True):
        self.dataset = pd.read_json(data_path)
        self.crop = crop
        self.progress = progress
        
        
        dir_data_path, file_name = data_path.rsplit("/", 1)
        file_name = file_name.split(".")[0]
        processed_file_name = f"{dir_data_path}/processed_{file_name}.jsonl"
        self.processed_dataset = pd.read_json(processed_file_name, lines=True)
        
        self.text_map = self.processed_dataset.set_index("caption")['diagram'].to_dict()
        
        self.symbols = []
        self.sizes = []
        
        for row in self.processed_dataset['symbols']:
            for x in row:
                self.symbols.append(Symbol(**x[0]))
                self.sizes.append(x[1])
                
        print(f"Initialized dataset for {data_path}")

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

        true_caption = self.remove_shape(true_caption)
        false_caption = self.remove_shape(false_caption)

        return {
            "image": image,
            "true_caption": true_caption,
            "false_caption": false_caption,
            "index": idx
        }
    
    @staticmethod
    def remove_shape(einsum_input):
        if einsum_input is None:
            return None
        einsum_expr, symbol_size_list = einsum_input
        return (einsum_expr, [sym for sym, _ in symbol_size_list])
    
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