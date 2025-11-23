import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import tensornetwork as tn
import numpy as np

# --- 0. Setup ---
tn.set_default_backend("pytorch")

# --- 1. "Toy" Hyperparameters ---
IMG_SIZE = 4           # Resize 28x28 -> 4x4
N_PIXELS = IMG_SIZE**2 # Chain length = 16 (Much easier than 784)
FEATURE_DIM = 2        # [cos, sin]
BOND_DIM = 4           # Small bond dim is enough for this simple task
NUM_CLASSES = 2        # Only digits 0 and 1
BATCH_SIZE = 32
LEARNING_RATE = 0.01   # Higher LR for small problems
EPOCHS = 10

device = torch.device("mps") # CPU is fast enough for this tiny scale

# --- 2. The Feature Map (Same as before) ---
class FeatureMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.tensor(torch.pi / 2.0))

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.cat([torch.cos(self.factor * x), torch.sin(self.factor * x)], dim=-1)

# --- 3. MPS Classifier (Same logic, better init) ---
class MPSClassifier(nn.Module):
    def __init__(self, input_dim, feature_dim, bond_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.feature_map = FeatureMap()

        cores = []
        
        # STRATEGY: Initialize closer to Identity to allow gradient flow
        # 1. First Core (Feature, 1, Bond)
        cores.append(nn.Parameter(torch.randn(feature_dim, 1, bond_dim) * 0.1))

        # 2. Middle Cores (Feature, Bond, Bond)
        for _ in range(1, input_dim - 1):
            # Initialize somewhat close to identity to let info pass through
            core = torch.eye(bond_dim).unsqueeze(0).repeat(feature_dim, 1, 1) 
            noise = torch.randn(feature_dim, bond_dim, bond_dim) * 0.1
            cores.append(nn.Parameter(core + noise))

        # 3. Last Core (Feature, Bond, Classes)
        cores.append(nn.Parameter(torch.randn(feature_dim, bond_dim, num_classes) * 0.1))

        self.cores = nn.ParameterList(cores)

    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)
        features = self.feature_map(x_flat) 
        
        # Start Contraction
        v_0 = features[:, 0, :] 
        A_0 = self.cores[0]
        
        # Initial Accumulator M: [Batch, Bond]
        M = tn.ncon([v_0, A_0.squeeze(1)], [[-1, 1], [1, -2]])

        for i in range(1, self.input_dim - 1):
            v_i = features[:, i, :]
            A_i = self.cores[i]
            
            # Contract Feature with Core -> [Batch, Left, Right]
            core_contracted = tn.ncon([v_i, A_i], [[-1, 1], [1, -2, -3]])
            
            # Contract Accumulator with Core -> [Batch, Right]
            M = tn.ncon([M, core_contracted], [[-1, 1], [-1, 1, -2]])
            
            # Normalize to keep values sane
            M_norm = torch.norm(M, p='fro', dim=1, keepdim=True)
            M = M / (M_norm + 1e-8)

        # Final Core
        v_N = features[:, -1, :]
        A_N = self.cores[-1]
        final_contracted = tn.ncon([v_N, A_N], [[-1, 1], [1, -2, -3]])
        
        # Result -> [Batch, Classes]
        logits = tn.ncon([M, final_contracted], [[-1, 1], [-1, 1, -2]])
        return logits

def main():
    # --- Data Prep: Resize to 4x4 and filter only 0s and 1s ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize 28x28 -> 4x4
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Filter dataset to only keep digits 0 and 1
    idx = (full_dataset.targets == 0) | (full_dataset.targets == 1)
    full_dataset.targets = full_dataset.targets[idx]
    full_dataset.data = full_dataset.data[idx]
    
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Input Features: {N_PIXELS} (4x4 image)")
    print("Training Binary Classifier (0 vs 1)...")

    model = MPSClassifier(N_PIXELS, FEATURE_DIM, BOND_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping (Helpful for TNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f} | Acc = {100 * correct/total:.1f}%")

if __name__ == '__main__':
    main()