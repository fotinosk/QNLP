import tensornetwork as tn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from qnlp.utils.data import get_mnist_loaders
from qnlp.utils.feature import FeatureMap

tn.set_default_backend("pytorch")

# --- Hyperparameters ---
# We can use the full 28x28 now because Sweeping solves the gradient depth issue
N_PIXELS = 784
FEATURE_DIM = 2
BOND_DIM = 10  # Kept small for speed in this demo
NUM_CLASSES = 10
BATCH_SIZE = 256  # Larger batch size helps sweeping stability
SWEEP_EPOCHS = 2  # We only need a couple of sweeps
STEPS_PER_CORE = 5  # How many gradient steps to take on *each* core per visit

device = torch.device("mps")


# --- MPS Model ---
class MPSClassifier(nn.Module):
    def __init__(self, input_dim, feature_dim, bond_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.feature_map = FeatureMap()

        cores = []
        # Initialize cores roughly normalized to keep signals stable
        # Core 0
        cores.append(nn.Parameter(torch.randn(feature_dim, 1, bond_dim) * 0.1))
        # Middle Cores
        for _ in range(1, input_dim - 1):
            # Initialize close to Identity to allow signal flow
            base = torch.eye(bond_dim).unsqueeze(0).repeat(feature_dim, 1, 1)
            noise = torch.randn(feature_dim, bond_dim, bond_dim) * 0.01
            cores.append(nn.Parameter(base + noise))
        # Last Core
        cores.append(nn.Parameter(torch.randn(feature_dim, bond_dim, num_classes) * 0.1))

        self.cores = nn.ParameterList(cores)

    def forward(self, x):
        # Standard contraction (same as before) for prediction
        x_flat = x.view(-1, self.input_dim)
        features = self.feature_map(x_flat)

        v_0 = features[:, 0, :]
        A_0 = self.cores[0]
        M = tn.ncon([v_0, A_0.squeeze(1)], [[-1, 1], [1, -2]])

        for i in range(1, self.input_dim - 1):
            v_i = features[:, i, :]
            A_i = self.cores[i]
            core_contracted = tn.ncon([v_i, A_i], [[-1, 1], [1, -2, -3]])
            M = tn.ncon([M, core_contracted], [[-1, 1], [-1, 1, -2]])

            # Heavy normalization needed for long chains
            M = M / (torch.norm(M, dim=1, keepdim=True) + 1e-8)

        v_N = features[:, -1, :]
        A_N = self.cores[-1]
        final = tn.ncon([v_N, A_N], [[-1, 1], [1, -2, -3]])
        logits = tn.ncon([M, final], [[-1, 1], [-1, 1, -2]])
        return logits


def train_sweep():
    # Setup Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader, test_loader = get_mnist_loaders(batch_size=BATCH_SIZE, transform=transform)

    model = MPSClassifier(N_PIXELS, FEATURE_DIM, BOND_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Sweeping Optimization on {device}...")
    print(f"Chain Length: {N_PIXELS} nodes")

    # --- THE SWEEP LOOP ---
    for epoch in range(SWEEP_EPOCHS):
        print(f"\n--- Sweep Epoch {epoch+1} ---")

        # We iterate through every core in the chain (0 to 783)
        for core_idx in range(N_PIXELS):
            # 1. FREEZE EVERYTHING
            for param in model.parameters():
                param.requires_grad = False

            # 2. UNFREEZE CURRENT CORE
            current_core = model.cores[core_idx]
            current_core.requires_grad = True

            # Create a local optimizer just for this core
            # We use a higher LR because the problem is local and simpler
            optimizer = optim.Adam([current_core], lr=0.005)

            # 3. LOCAL TRAINING LOOP
            # We take a batch of data and update *only* this core
            # In a real rigorous DMRG, we would project the whole dataset first,
            # but here we just run a few mini-batches.

            data_iter = iter(train_loader)
            for step in range(STEPS_PER_CORE):
                try:
                    images, labels = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    images, labels = next(data_iter)

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if core_idx % 50 == 0:
                print(f"Optimized Core {core_idx}/{N_PIXELS} - Loss: {loss.item():.4f}")

    print("Sweeping finished.")


if __name__ == "__main__":
    train_sweep()
