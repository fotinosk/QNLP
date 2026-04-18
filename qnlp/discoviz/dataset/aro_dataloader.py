from typing import Callable

from torch.utils.data import DataLoader

from qnlp.discoviz.dataset.aro_dataset import ProcessedARODataset, aro_tn_collate_fn

# TRAIN_DATA_PATH = "data/aro/processed/combined/train.json"
# VAL_DATA_PATH = "data/aro/processed/combined/val.json"
# TEST_DATA_PATH = "data/aro/processed/combined/test.json"
# IMAGES_PATH = "data/aro/raw/images/"

TRAIN_DATA_PATH = "data/aro_and_wino/processed/train.json"
VAL_DATA_PATH = "data/aro_and_wino/processed/val.json"
TEST_DATA_PATH = "data/aro_and_wino/processed/test.json"


def get_aro_dataloader(
    batch_size: int,
    train_process_function: Callable,
    val_process_function: Callable,
    return_images: bool = True,
):
    train_ds = ProcessedARODataset(
        data_path=TRAIN_DATA_PATH,
        return_images=return_images,
        image_processing_fn=train_process_function,
    )
    val_ds = ProcessedARODataset(
        data_path=VAL_DATA_PATH,
        return_images=return_images,
        image_processing_fn=val_process_function,
    )
    test_ds = ProcessedARODataset(
        data_path=TEST_DATA_PATH,
        return_images=return_images,
        image_processing_fn=val_process_function,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=aro_tn_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=aro_tn_collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=aro_tn_collate_fn,
    )

    return [[train_loader, val_loader, test_loader], [train_ds, val_ds, test_ds]]
