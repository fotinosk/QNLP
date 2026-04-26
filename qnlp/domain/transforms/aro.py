import torch
from torchvision.transforms import v2


def create_aro_image_transforms(image_size: int):
    preprocess = v2.Compose(
        [
            v2.RandomCrop(image_size, padding=4, padding_mode="reflect"),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            v2.RandomHorizontalFlip(p=0.5),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_preprocess = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return preprocess, val_preprocess
