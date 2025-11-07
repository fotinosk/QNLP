from torchvision import datasets

dataset = datasets.CIFAR10(root="../image_encoding/test_images", train=True, download=True)
