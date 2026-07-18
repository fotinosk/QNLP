import time

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

# =====================================================================
# Device Declaration (16 Qubits Noiseless)
# =====================================================================
num_qubits = 16
dev = qml.device("default.qubit", wires=num_qubits)


# =====================================================================
# IQP Ansatz Unitary
# =====================================================================
def iqp_ansatz(wires, weights):
    # weights shape: [len(wires), 2]
    for w in wires:
        qml.Hadamard(wires=w)
    # RZ rotations
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
    # Ising ZZ couplings (ring layout)
    if len(wires) > 1:
        for i in range(len(wires)):
            w1 = wires[i]
            w2 = wires[(i + 1) % len(wires)]
            qml.IsingZZ(weights[i, 1], wires=[w1, w2])
    # Hadamard projection
    for w in wires:
        qml.Hadamard(wires=w)


# =====================================================================
# State Prep: Multi-Axis Encoding (3 features per qubit)
# =====================================================================
def multi_axis_prep(inputs, wires):
    # inputs shape: [Batch, len(wires) * 3]
    for i, w in enumerate(wires):
        qml.RX(inputs[:, 3 * i], wires=w)
        qml.RY(inputs[:, 3 * i + 1], wires=w)
        qml.RZ(inputs[:, 3 * i + 2], wires=w)


# =====================================================================
# QNode Scaffolding
# =====================================================================
@qml.qnode(dev, interface="torch")
def qttn_noiseless(inputs, weights):
    # inputs shape: [Batch, 48] (16 qubits * 3 features/qubit)
    # weights shape: [5, 4, 2]

    # State Prep: Load 48 features on 16 qubits
    multi_axis_prep(inputs, range(16))

    # Layer 1: 16 patch qubits -> 4 parent qubits
    for b in range(4):
        iqp_ansatz(range(4 * b, 4 * b + 4), weights[b])

    # Layer 2: 4 parent qubits -> 4 root expectations
    iqp_ansatz([0, 4, 8, 12], weights[4])

    return [qml.expval(qml.PauliZ(w)) for w in [0, 4, 8, 12]]


# =====================================================================
# PyTorch Classifier Model (32x32 inputs)
# =====================================================================
class Overlapping32x32QTTN(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch embedding: 16 patches of size 8x8x3 = 192 features
        # Map 192 features per patch -> 3 angles
        self.patch_embed = nn.Linear(192, 3)

        # IQP ansatz weights: 5 nodes in the tree, each acts on 4 wires (2 params/qubit)
        self.weights = nn.Parameter(torch.randn(5, 4, 2) * 0.1)
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Unfold 32x32 image to 16 patches of size 8x8
        x = x.unfold(2, 8, 8).unfold(3, 8, 8)  # [B, 3, 4, 4, 8, 8]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 4, 4, 3, 8, 8]
        x = x.view(batch_size, 16, 192)  # [B, 16, 192]

        # Patch embedding to angle space
        features = torch.tanh(self.patch_embed(x)) * np.pi  # [B, 16, 3]
        features = features.view(batch_size, 48)  # [B, 48]

        # Execute vectorized 16-qubit QNode
        q_out = qttn_noiseless(features, self.weights)
        q_embeddings = torch.stack(q_out, dim=-1).float()  # [B, 4]

        return self.head(q_embeddings)


# =====================================================================
# Main Training Loop
# =====================================================================
def run_32x32_proof():
    epochs = 20
    batch_size = 32

    print("=============================================================")
    print("Training 32x32 Canvas Overlapping Mode Feature Binding Proof")
    print("Image Canvas: 32x32 | Qubits: 16 | Train Samples: 512 | Epochs: 20")
    print("=============================================================")

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=512, test_samples=128, img_size=32, mode="overlapping", seed=42
    )

    model = Overlapping32x32QTTN()
    optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    t_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_imgs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        train_acc = 100.0 * correct / total
        dt = time.time() - t0

        # Evaluate validation set every epoch
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for val_imgs, val_labels in test_loader:
                val_outputs = model(val_imgs)
                val_preds = val_outputs.argmax(dim=1)
                val_correct += val_preds.eq(val_labels).sum().item()
        val_acc = 100.0 * val_correct / len(test_loader.dataset)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Time: {dt:.1f}s"
        )

    total_dt = time.time() - t_start
    print(f"\nTraining completed in {total_dt/60.0:.1f} minutes.")
    print("=============================================================")
    print(f"Final Validation Accuracy: {val_acc:.1f}%")
    print("Color-Only limit: 50.0% | Shape-Only limit: 50.0%")
    if val_acc > 50.0:
        print("SUCCESS: Model successfully bound both color and shape attributes!")
    else:
        print("FAIL: Model did not exceed the single-attribute boundary.")
    print("=============================================================")


if __name__ == "__main__":
    run_32x32_proof()
