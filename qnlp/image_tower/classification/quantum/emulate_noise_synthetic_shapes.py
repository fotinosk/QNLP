import os

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

# =====================================================================
# Device & Noisy/Noiseless QNode Definitions (4 Qubits)
# =====================================================================
num_qubits = 4
dev_noiseless = qml.device("default.qubit", wires=num_qubits)
dev_noisy = qml.device("default.mixed", wires=num_qubits)


def hea_ansatz_node(wires, weights, p_noise=0.0):
    """HEA ansatz with depolarizing noise after every gate"""
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Entangling gates (CNOTs)
    qml.CNOT(wires=[wires[1], wires[0]])
    if p_noise > 0:
        qml.DepolarizingChannel(p_noise, wires=wires[0])
        qml.DepolarizingChannel(p_noise, wires=wires[1])

    qml.CNOT(wires=[wires[2], wires[0]])
    if p_noise > 0:
        qml.DepolarizingChannel(p_noise, wires=wires[0])
        qml.DepolarizingChannel(p_noise, wires=wires[2])

    qml.CNOT(wires=[wires[3], wires[0]])
    if p_noise > 0:
        qml.DepolarizingChannel(p_noise, wires=wires[0])
        qml.DepolarizingChannel(p_noise, wires=wires[3])


@qml.qnode(dev_noiseless, interface="torch")
def qttn_noiseless(inputs, weights):
    # inputs shape: [Batch, 4]
    # weights shape: [4, 3]
    for i in range(num_qubits):
        qml.RY(inputs[:, i], wires=i)
    hea_ansatz_node(range(num_qubits), weights, p_noise=0.0)
    return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]


@qml.qnode(dev_noisy, interface="torch")
def qttn_noisy(inputs, weights, p_noise):
    # inputs shape: [Batch, 4]
    # weights shape: [4, 3]
    for i in range(num_qubits):
        qml.RY(inputs[:, i], wires=i)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=i)
    hea_ansatz_node(range(num_qubits), weights, p_noise=p_noise)
    return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]


# =====================================================================
# PyTorch Classifier Model
# =====================================================================
class SimpleQTTNClassifier(nn.Module):
    def __init__(self, mode="noiseless", p_noise=0.0):
        super().__init__()
        self.mode = mode
        self.p_noise = p_noise

        # Patch embedding: 4 patches, each 4x4x3 = 48 features -> 1 angle
        self.patch_embed = nn.Linear(48, 1)
        # Weights: 1 node, 4 wires, 3 rotations
        self.weights = nn.Parameter(torch.randn(4, 3) * 0.1)
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Patch unfolding: 8x8 images -> 4 patches of size 4x4
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # [B, 3, 2, 2, 4, 4]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 2, 2, 3, 4, 4]
        x = x.view(batch_size, 4, 48)  # [B, 4, 48]

        angles = torch.tanh(self.patch_embed(x)) * np.pi
        angles = angles.squeeze(-1)  # [B, 4]

        if self.mode == "noiseless":
            q_out = qttn_noiseless(angles, self.weights)
        else:
            q_out = qttn_noisy(angles, self.weights, self.p_noise)

        q_embeddings = torch.stack(q_out, dim=-1).float()
        return self.head(q_embeddings)


# =====================================================================
# Training & Noise Sweeping Loop
# =====================================================================
def run_emulation():
    print("====================================================")
    print("Stage 1: Training Noiseless QTTN on 8x8 Shapes...")
    print("====================================================")

    # 1. Load 8x8 synthetic shapes dataloaders
    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=32, train_samples=512, test_samples=128, img_size=8, seed=42
    )

    model = SimpleQTTNClassifier(mode="noiseless")
    optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train for 8 epochs
    epochs = 8
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
        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}%"
        )

    # Evaluate clean model on test set
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            outputs = model(batch_imgs)
            preds = outputs.argmax(dim=1)
            val_correct += preds.eq(batch_labels).sum().item()
    clean_acc = 100.0 * val_correct / len(test_loader.dataset)
    print(f"\nNoiseless Test Accuracy: {clean_acc:.1f}%")

    print("\n====================================================")
    print("Stage 2: Sweeping Depolarizing Noise (Emulation)...")
    print("====================================================")

    noise_rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    noisy_accuracies = []

    # Switch model to noisy mode
    model.mode = "noisy"

    for p in noise_rates:
        model.p_noise = p
        val_correct = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(batch_labels).sum().item()

        acc = 100.0 * val_correct / len(test_loader.dataset)
        noisy_accuracies.append(acc)
        print(f"Noise Rate (p): {p:.3f} | Test Accuracy: {acc:.1f}%")

    # Plot noise degradation curve
    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(noise_rates, noisy_accuracies, "r-o", linewidth=2, label="QTTN accuracy")
    plt.axhline(y=25.0, color="b", linestyle="--", label="Random Guess (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("QTTN Noise Tolerance Curve (Depolarizing Noise)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = "qnlp/image_tower/classification/quantum/results/noise_tolerance_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved noise degradation curve to: {plot_path}")

    return noise_rates, noisy_accuracies


if __name__ == "__main__":
    run_emulation()
