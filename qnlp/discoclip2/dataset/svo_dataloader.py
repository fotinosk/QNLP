from typing import Callable

from torch.utils.data import DataLoader

from qnlp.discoclip2.dataset.svo_dataset import ProcessedSVODataset, svo_tn_collate_fn

TRAIN_DATA_PATH = "data/svo/processed/train.csv"
VAL_DATA_PATH = "data/svo/processed/val.csv"
TEST_DATA_PATH = "data/svo/processed/test.csv"
IMAGES_PATH = "data/svo/raw/images"


def get_svo_dataloader(batch_size: int, train_process_function: Callable, val_process_function: Callable):
    train_ds = ProcessedSVODataset(
        data_path=TRAIN_DATA_PATH, image_dir_path=IMAGES_PATH, image_processing_fn=train_process_function
    )
    val_ds = ProcessedSVODataset(
        data_path=VAL_DATA_PATH, image_dir_path=IMAGES_PATH, image_processing_fn=val_process_function
    )
    test_ds = ProcessedSVODataset(
        data_path=TEST_DATA_PATH, image_dir_path=IMAGES_PATH, image_processing_fn=val_process_function
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=svo_tn_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=svo_tn_collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=svo_tn_collate_fn,
    )

    return [[train_loader, val_loader, test_loader], [train_ds, val_ds, test_ds]]
