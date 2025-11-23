import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tensornetwork as tn
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from cached_mps import FeatureMap, CachedMPS

# --- 0. CONFIGURATION ---
# MUST match the training configuration
tn.set_default_backend("pytorch")
IMG_SIZE = 10 
N_PIXELS = IMG_SIZE * IMG_SIZE
FEATURE_DIM = 2
BOND_DIM = 16
NUM_CLASSES = 10
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_model(model_dir):
    path = os.path.join(model_dir, "mps_best_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No best model found at {path}")
    
    print(f"Loading weights from: {path}")
    checkpoint = torch.load(path, map_location=device)
    
    model = CachedMPS(N_PIXELS, FEATURE_DIM, BOND_DIM, NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (Epoch {checkpoint.get('epoch', 'Unknown')}, Loss {checkpoint.get('loss', 'Unknown'):.4f})")
    return model

def evaluate(model, test_loader):
    print("\n--- Starting Evaluation on Test Set ---")
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    acc = 100 * correct / total
    print(f'\nOverall Accuracy: {acc:.2f}%')
    
    print('\nPer-Class Accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            print(f'Digit {i}: {100 * class_correct[i] / class_total[i]:.1f}%')

    return acc

def visualize_predictions(model, test_loader, num_images=5):
    """Shows a few images with their predicted labels."""
    print(f"\n--- Visualizing {num_images} Predictions ---")
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Move to CPU for plotting
    images = images.cpu()
    predicted = predicted.cpu()
    labels = labels.cpu()
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        # Reshape 10x10 image
        img = images[i].reshape(IMG_SIZE, IMG_SIZE)
        ax.imshow(img, cmap='gray')
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predicted[i]}\nTrue: {labels[i]}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate MPS Model")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the timestamped checkpoint directory")
    args = parser.parse_args()

    # Load Data
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    try:
        model = load_best_model(args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run Eval
    evaluate(model, test_loader)
    
    # Visual Sanity Check
    visualize_predictions(model, test_loader)

if __name__ == '__main__':
    main()