import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class PhaseEncode:
    def __call__(self, x):
        angle = x * (torch.pi / 2)
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=0)


def get_mnist_loaders(batch_size=64, img_size=32, root="./data", with_phase_embedding=False):
    """
    Returns (train_loader, test_loader) for MNIST with specified batch size and image size.
    """
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if with_phase_embedding:
        transform_list.append(PhaseEncode())

    transform = transforms.Compose(transform_list)
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_caltech_sample_loaders(batch_size=32, img_size=224, root="./data"):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    # Downloads the whole dataset (~1.2GB) very quickly
    full_dataset = torchvision.datasets.Caltech256(root=root, download=True, transform=transform)

    # Use a Subset to keep it light
    train_size = 5000
    test_size = 600
    train_dataset, test_dataset, _ = torch.utils.data.random_split(
        full_dataset,
        [train_size, test_size, len(full_dataset) - train_size - test_size],
    )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
