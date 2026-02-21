import os
import sys
from datetime import datetime

import tensornetwork as tn
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot

from qnlp.utils.feature import FeatureMap

# --- 0. CONFIGURATION ---
sys.setrecursionlimit(10000)
tn.set_default_backend("pytorch")
device = torch.device("mps")

IMG_SIZE = 10
N_PIXELS = IMG_SIZE * IMG_SIZE
FEATURE_DIM = 2
BOND_DIM = 16  # Max Bond Dimension
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 5
SVD_THRESHOLD = 1e-4

# --- 1. MODEL CLASSES ---


class CachedMPS(nn.Module):
    def __init__(self, input_dim, feature_dim, bond_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.feature_map = FeatureMap()
        self.cores = nn.ParameterList()

        # Init with noise
        self.cores.append(nn.Parameter(torch.randn(feature_dim, 1, bond_dim) * 0.1))
        for _ in range(1, input_dim - 1):
            self.cores.append(nn.Parameter(torch.randn(feature_dim, bond_dim, bond_dim) * 0.1))
        self.cores.append(nn.Parameter(torch.randn(feature_dim, bond_dim, num_classes) * 0.1))

    def forward(self, x):
        x_flat = x.view(-1, self.input_dim)
        features = self.feature_map(x_flat)

        v_0 = features[:, 0, :]
        A_0 = self.cores[0]
        # Squeeze dummy dim 1
        M = tn.ncon([v_0, A_0.squeeze(1)], [[-1, 1], [1, -2]])

        for i in range(1, self.input_dim - 1):
            v_i = features[:, i, :]
            A_i = self.cores[i]
            M = tn.ncon([M, v_i, A_i], [[-1, 1], [-1, 2], [2, 1, -2]])
            M = M / (torch.norm(M, dim=1, keepdim=True) + 1e-8)

        v_N = features[:, -1, :]
        A_N = self.cores[-1]
        final = tn.ncon([v_N, A_N], [[-1, 1], [1, -2, -3]])
        logits = tn.ncon([M, final], [[-1, 1], [-1, 1, -2]])
        return logits

    def precompute_right_env(self, features):
        self.R_cache = [None] * self.input_dim
        v_last = features[:, -1, :]
        A_last = self.cores[-1]
        curr = tn.ncon([v_last, A_last], [[-1, 1], [1, -2, -3]])
        self.R_cache[-1] = curr

        for i in range(self.input_dim - 2, -1, -1):
            v_i = features[:, i, :]
            A_i = self.cores[i]
            prev_R = self.R_cache[i + 1]
            if i == self.input_dim - 2:
                curr = tn.ncon([v_i, A_i, prev_R], [[-1, 1], [1, -2, 3], [-1, 3, -3]])
            else:
                curr = tn.ncon([v_i, A_i, prev_R], [[-1, 1], [1, -2, 3], [-1, 3, -3]])
            curr = curr / (torch.norm(curr, dim=1, keepdim=True) + 1e-8)
            self.R_cache[i] = curr

    def update_L_cache(self, index, features):
        v_i = features[:, index, :]
        core = self.cores[index]
        if index == 0:
            new_L = tn.ncon([v_i, core.squeeze(1)], [[-1, 1], [1, -2]])
        else:
            prev_L = self.L_cache[index - 1]
            new_L = tn.ncon([prev_L, v_i, core], [[-1, 1], [-1, 2], [2, 1, -2]])
        new_L = new_L / (torch.norm(new_L, dim=1, keepdim=True) + 1e-8)
        if not hasattr(self, "L_cache"):
            self.L_cache = [None] * self.input_dim
        self.L_cache[index] = new_L


# --- 2. TWO-SITE LOGIC ---


def get_two_site_loss(mps, i, B_tensor, features, labels, criterion):
    v_left = features[:, i, :]
    v_right = features[:, i + 1, :]

    parts = [v_left, v_right]
    schemes = [[-1, 1], [-1, 3]]

    # Handle Left Env
    if i == 0:
        # At i=0, B has shape [F, 1, F, R]. Squeeze the dummy dim 1.
        B_eff = B_tensor.squeeze(1)
        parts.append(B_eff)
        # B indices: [Feature1 (1), Feature2 (3), RightBond (4)]
        schemes.append([1, 3, 4])
    else:
        parts.append(mps.L_cache[i - 1])  # L: [-1, 2]
        parts.append(B_tensor)
        # B indices: [F1 (1), LeftBond (2), F2 (3), RightBond (4)]
        schemes.append([-1, 2])
        schemes.append([1, 2, 3, 4])

    # Handle Right Env
    if i == mps.input_dim - 2:
        # Last site: B contains Class index at pos 4 (or 3 if squeezed)
        if i == 0:
            # schemes[-1] (B) is [1, 3, -2]
            schemes[-1] = [1, 3, -2]
        else:
            # schemes[-1] (B) is [1, 2, 3, -2]
            schemes[-1] = [1, 2, 3, -2]
    else:
        R = mps.R_cache[i + 2]  # R: [-1, 4, -2]
        parts.append(R)
        schemes.append([-1, 4, -2])

    logits = tn.ncon(parts, schemes)
    return criterion(logits, labels)


def two_site_step(mps, i, features, labels, criterion, writer, global_step):
    core_left = mps.cores[i].data
    core_right = mps.cores[i + 1].data

    # Merge into Bond Tensor B
    # Left: [F1, L, Bond] -- Right: [F2, Bond, R]
    B_data = tn.ncon([core_left, core_right], [[-1, -2, 1], [-3, 1, -4]])

    # Optimize B
    B_param = B_data.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([B_param], lr=0.05)

    optimizer.zero_grad()
    loss = get_two_site_loss(mps, i, B_param, features, labels, criterion)
    loss.backward()
    optimizer.step()

    # SVD & Truncation
    with torch.no_grad():
        F1, L, F2, R = B_param.shape
        matrix = B_param.reshape(F1 * L, F2 * R)

        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if global_step % 100 == 0:
            writer.add_histogram(f"SVD/Bond_{i}", S, global_step)

        # Soft Truncation
        S[S < SVD_THRESHOLD] = 0.0

        # Limit Max Rank
        k = min(S.shape[0], BOND_DIM)
        U, S, Vh = U[:, :k], S[:k], Vh[:k, :]

        S_sqrt = torch.diag(torch.sqrt(S))

        # Reconstruct
        new_left = (U @ S_sqrt).reshape(F1, L, k)
        new_right = (S_sqrt @ Vh).reshape(k, F2, R).permute(1, 0, 2)

        # FIX: Assign new Parameters to handle shape changes
        mps.cores[i] = nn.Parameter(new_left)
        mps.cores[i + 1] = nn.Parameter(new_right)

    return loss.item()


# --- 3. UTILS ---


def create_visualizations(model, sample_input, save_dir):
    print("\n--- Generating Visualizations ---")
    try:
        model.to("cpu")
        y = model(sample_input.to("cpu"))
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render(os.path.join(save_dir, "mps_structure"))
        print("   -> TorchViz saved.")
        model.to(device)
    except Exception as e:
        print(f"   -> Skipped Viz: {e}")


def get_run_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"MPS_TwoSite_{ts}"
    return os.path.join("checkpoints", name), name


# --- 4. MAIN ---


def main():
    run_dir, run_name = get_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"--> Training: {run_name}")

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.13,), (0.3,)),
        ]
    )
    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    mps = CachedMPS(N_PIXELS, FEATURE_DIM, BOND_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    # Visualize initial structure
    create_visualizations(mps, torch.randn(BATCH_SIZE, IMG_SIZE, IMG_SIZE).to(device), run_dir)

    global_step = 0
    print("\n--- Starting Two-Site DMRG ---")

    for epoch in range(EPOCHS):
        mps.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).long()
            features = mps.feature_map(images.view(-1, N_PIXELS))

            # Prepare Cache
            with torch.no_grad():
                mps.precompute_right_env(features)
                mps.L_cache = [None] * N_PIXELS

            # Sweep
            for i in range(N_PIXELS - 1):
                step_loss = two_site_step(mps, i, features, labels, criterion, writer, global_step)

                # Update Cache with the NEW core
                with torch.no_grad():
                    mps.update_L_cache(i, features)

                if global_step % 100 == 0:
                    writer.add_scalar("Loss/train", step_loss, global_step)
                global_step += 1

            total_loss += step_loss

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {step_loss:.4f}")

        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")
        torch.save(mps.state_dict(), os.path.join(run_dir, "mps_latest.pth"))

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()
