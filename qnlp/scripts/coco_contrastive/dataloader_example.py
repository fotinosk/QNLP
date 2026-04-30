from torchvision import transforms

from qnlp.constants import constants
from qnlp.domain.datasets.dataloader import get_dataloaders

DATASETS_PATH = constants.datasets_path

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":
    loaders, datasets = get_dataloaders(
        train_parquet=DATASETS_PATH / "coco_contrastive_train.parquet",
        val_parquet=DATASETS_PATH / "coco_contrastive_val.parquet",
        test_parquet=DATASETS_PATH / "coco_contrastive_test.parquet",
        batch_size=8,
        train_transform=train_transform,
        val_transform=val_transform,
        compiled_columns=[
            ("true_diagram", "true_symbols", "true_caption"),
            ("false_diagram", "false_symbols", "false_caption"),
        ],
    )

    train_loader, val_loader, test_loader = loaders
    train_ds, val_ds, test_ds = datasets

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Inspect one raw sample
    sample = train_ds[0]
    print("\nSample keys:", list(sample.keys()))
    print("Image shape:", sample["local_image_path"].shape)
    true_diagram, true_symbols = sample["true_caption"]
    false_diagram, false_symbols = sample["false_caption"]
    print("True diagram:", true_diagram[:60], "...")
    print("False diagram:", false_diagram[:60], "...")
    print("True symbols:", true_symbols[:2])
    print("False symbols:", false_symbols[:2])

    # Inspect one batch
    batch = next(iter(train_loader))
    print("\nBatch keys:", list(batch.keys()))
    print("Batch image shape:", batch["local_image_path"].shape)
    print("Batch true_captions (first):", batch["true_caption"][0][0][:60], "...")
    print("Batch false_captions (first):", batch["false_caption"][0][0][:60], "...")
