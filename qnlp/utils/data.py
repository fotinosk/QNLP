import torchvision
import torchvision.transforms as transforms
import torch
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
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
