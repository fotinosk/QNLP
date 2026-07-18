import time

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

# =====================================================================
# Device Declaration (4 Qubits Noiseless)
# =====================================================================
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)


# =====================================================================
# IQP Ansatz Unitary on 4 Qubits
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
    # inputs shape: [Batch, 12] (4 qubits * 3 features/qubit)
    # weights shape: [4, 2]
    multi_axis_prep(inputs, range(4))
    iqp_ansatz(range(4), weights)
    return [qml.expval(qml.PauliZ(w)) for w in range(4)]


# =====================================================================
# PyTorch Classifier Model (8x8 inputs)
# =====================================================================
class OverlappingProofQTTN(nn.Module):
    def __init__(self):
        super().__init__()
        # Patch embedding: 4 patches of size 4x4x3 = 48 features
        # Map 48 features per patch -> 3 angles
        self.patch_embed = nn.Linear(48, 3)
        self.weights = nn.Parameter(torch.randn(4, 2) * 0.1)
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Unfold 8x8 image to 4 patches of size 4x4
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # [B, 3, 2, 2, 4, 4]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 2, 2, 3, 4, 4]
        x = x.view(batch_size, 4, 48)  # [B, 4, 48]

        # Patch embedding to angle space
        features = torch.tanh(self.patch_embed(x)) * np.pi  # [B, 4, 3]
        features = features.view(batch_size, 12)  # [B, 12]

        # Execute vectorized QNode
        q_out = qttn_noiseless(features, self.weights)
        q_embeddings = torch.stack(q_out, dim=-1).float()  # [B, 4]

        return self.head(q_embeddings)


# =====================================================================
# Main Training Loop
# =====================================================================
def run_proof():
    epochs = 25
    batch_size = 32

    print("=============================================================")
    print("Training Overlapping Mode (Feature Binding) Proof Model")
    print("Image Canvas: 8x8 | Qubits: 4 | Train Samples: 512 | Epochs: 25")
    print("=============================================================")

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=512, test_samples=128, img_size=8, mode="overlapping", seed=42
    )

    model = OverlappingProofQTTN()
    optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(epochs):
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

        # Eval test set periodically
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for val_imgs, val_labels in test_loader:
                    val_outputs = model(val_imgs)
                    val_preds = val_outputs.argmax(dim=1)
                    val_correct += val_preds.eq(val_labels).sum().item()
            val_acc = 100.0 * val_correct / len(test_loader.dataset)
            print(
                f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
            )
        else:
            print(
                f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}%"
            )

    dt = time.time() - t0
    print(f"\nTraining completed in {dt:.1f} seconds.")


if __name__ == "__main__":
    run_proof()
