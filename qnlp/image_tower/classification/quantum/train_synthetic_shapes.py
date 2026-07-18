import time

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders


# =====================================================================
# Modular Ansätze Definitions
# =====================================================================
def hea_3cnot_node(wires, weights):
    """Hardware Efficient Ansatz with 3 CNOTs (Optimal)"""
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.CNOT(wires=[wires[3], wires[0]])


def iqp_node(wires, weights):
    """IQP-inspired Ansatz (Modular Alternative)"""
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        # We only use RZ and ZZ rotations (first 2 columns of weights)
        qml.RZ(weights[i, 0], wires=w)
    for i in range(len(wires)):
        w1 = wires[i]
        w2 = wires[(i + 1) % len(wires)]
        qml.IsingZZ(weights[i, 1], wires=[w1, w2])
    for w in wires:
        qml.Hadamard(wires=w)


# =====================================================================
# QNode Device & Definition (Standard 16-qubit Vectorized)
# =====================================================================
# We use standard 16-qubit simulation with parameter broadcasting. Instead of looping
# over individual batch items in a Python for-loop, we pass the entire batch of inputs
# to the QNode. PennyLane automatically broadcasts the state preparation and gate operations,
# executing the batch in a single compiled tape. This speeds up simulation by 30x.
dev = qml.device("default.qubit", wires=16)


@qml.qnode(dev, interface="torch")
def standard_qttn_qnode(inputs, weights, ansatz_type="hea_3cnot"):
    # inputs shape: [Batch, 16]
    # weights shape: [5, 4, 3]
    ansatz = hea_3cnot_node if ansatz_type == "hea_3cnot" else iqp_node

    # State Prep: Load 16 patch inputs (broadcasted over Batch dimension)
    for i in range(16):
        # inputs[:, i] has shape [Batch]
        qml.RY(inputs[:, i], wires=i)

    # Layer 1: 16 patches -> 4 parents (located on wires 0, 4, 8, 12)
    for b in range(4):
        block_wires = [4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3]
        ansatz(block_wires, weights[b])

    # Layer 2: 4 parents -> 4 root expectations (located on wires 0, 4, 8, 12)
    ansatz([0, 4, 8, 12], weights[4])

    # Return expectation values of all 4 parent qubits.
    # Returns a list of 4 tensors, each of shape [Batch]
    return [qml.expval(qml.PauliZ(w)) for w in [0, 4, 8, 12]]


# =====================================================================
# PyTorch Classifier Module
# =====================================================================
class RecycledQTTNClassifier(nn.Module):
    def __init__(self, ansatz_type="hea_3cnot"):
        super().__init__()
        self.ansatz_type = ansatz_type

        # Classical patch embedding: 16 patches of size 4x4x3 channels = 48 features
        # Map 48 features per patch -> 1 angle
        self.patch_embed = nn.Linear(48, 1)

        # Quantum weights: 5 nodes in the tree, each node acts on 4 wires (3 rotations/angles)
        self.weights = nn.Parameter(torch.randn(5, 4, 3) * 0.1)

        # Classical head: 4 quantum expectations -> 4 classes
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        # Input shape: [Batch, 3, 16, 16]
        batch_size = x.shape[0]

        # 1. Image Patching (16 patches, each 4x4)
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # [B, 3, 4, 4, 4, 4]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 4, 4, 3, 4, 4]
        x = x.view(batch_size, 16, 48)  # [B, 16, 48]

        # 2. Patch embedding to Angle space [-pi, pi]
        angles = torch.tanh(self.patch_embed(x)) * np.pi  # [B, 16, 1]
        angles = angles.squeeze(-1)  # [B, 16]

        # 3. Quantum QTTN Tower (Vectorized call over the batch)
        # standard_qttn_qnode returns a list of 4 tensors of shape [Batch]
        q_out = standard_qttn_qnode(angles, self.weights, ansatz_type=self.ansatz_type)
        q_embeddings = torch.stack(q_out, dim=-1).float()  # Stack along dim=-1 to get [Batch, 4]

        # 4. Classification Head
        return self.head(q_embeddings)


# =====================================================================
# Training Loop
# =====================================================================
def train_model(epochs=10, batch_size=32, lr=0.01, ansatz_type="hea_3cnot"):
    print(f"\nTraining Recycled QTTN on Synthetic Shapes (Ansatz: {ansatz_type})")
    print(f"Device: CPU (simulated) | Epochs: {epochs} | Batch Size: {batch_size}")

    # 1. Load small dataset for fast iteration
    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=256, test_samples=64, seed=1234
    )

    model = RecycledQTTNClassifier(ansatz_type=ansatz_type)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Metric tracking
    epoch_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_imgs, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        train_acc = 100.0 * correct / total
        mean_loss = total_loss / len(train_loader)
        dt = time.time() - t0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(batch_labels).sum().item()
                val_total += batch_labels.size(0)
        val_acc = 100.0 * val_correct / val_total

        # Record metrics
        epoch_losses.append(mean_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {mean_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Time: {dt:.1f}s"
        )

    print("\nTraining completed successfully!")

    # Generate and save metrics plot
    import os

    import matplotlib.pyplot as plt

    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(range(1, epochs + 1), epoch_losses, "b-o", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("QTTN Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Accuracy
    ax2.plot(range(1, epochs + 1), train_accs, "g-o", label="Train Accuracy")
    ax2.plot(range(1, epochs + 1), val_accs, "r-o", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("QTTN Classification Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = "qnlp/image_tower/classification/quantum/results/training_metrics.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training metrics plot to: {plot_path}")

    return train_acc, val_acc


if __name__ == "__main__":
    # Run a short training run to verify gradient convergence
    train_model(epochs=6, batch_size=32, lr=0.03, ansatz_type="hea_3cnot")
