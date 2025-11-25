import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from tree_tn2 import PatchTTN 

# ==========================================
# 1. Configuration (Must match training)
# ==========================================
BATCH_SIZE = 256
BOND_DIM = 64          
CP_RANK = 16           
PATCH_SIZE = 4         
MODEL_PATH = "mnist_ttn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

print(f"Running evaluation on device: {DEVICE}")


def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Run training script first.")
        return

    # Data Loader
    transform = transforms.Compose([
        transforms.Resize((32,32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model Load
    model = PatchTTN(
        img_size=32,
        patch_size=PATCH_SIZE,
        bond_dim=BOND_DIM,
        cp_rank=CP_RANK,
        dropout=0.0 # Disable dropout for eval
    ).to(DEVICE)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    print("\nStarting Evaluation...")
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n" + "="*30)
    print(f"Overall Accuracy: {100 * correct / total:.2f}%")
    print("="*30)
    print("Accuracy per class:")
    for i in range(10):
        if class_total[i] > 0:
            print(f"Digit {i}: {100 * class_correct[i] / class_total[i]:.2f}%")
        else:
            print(f"Digit {i}: N/A")
    print("="*30)

if __name__ == "__main__":
    evaluate()