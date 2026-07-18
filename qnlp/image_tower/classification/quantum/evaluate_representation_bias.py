import os
import time

import matplotlib.pyplot as plt
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
class BiasEvaluationQTTN(nn.Module):
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
# Training Harness
# =====================================================================
def train_eval_mode(mode="overlapping", epochs=6, batch_size=32):
    print(f"\nTraining Model on Mode: {mode.upper()}...")

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=128, test_samples=32, img_size=32, mode=mode, seed=42
    )

    model = BiasEvaluationQTTN()
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

    dt = time.time() - t0

    # Test Evaluation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for batch_imgs, batch_labels in test_loader:
            outputs = model(batch_imgs)
            preds = outputs.argmax(dim=1)
            val_correct += preds.eq(batch_labels).sum().item()

    val_acc = 100.0 * val_correct / len(test_loader.dataset)
    print(f"Finished Mode: {mode.upper()} | Test Acc: {val_acc:.1f}% | Time: {dt:.1f}s")
    return val_acc


# =====================================================================
# Main Scaffolding
# =====================================================================
def run_bias_experiments():
    modes = ["color_only", "shape_only", "overlapping"]
    accuracies = {}

    print("=============================================================")
    print("Evaluating QTTN Representation Bias on 32x32 Images")
    print("=============================================================")

    for mode in modes:
        acc = train_eval_mode(mode=mode, epochs=6, batch_size=32)
        accuracies[mode] = acc

    # Generate and save bar chart comparison
    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)
    plt.figure(figsize=(8, 5))

    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    x_labels = ["Color-Only", "Shape-Only\n(Grayscale)", "Overlapping\n(Feature Binding)"]
    y_values = [accuracies[m] for m in modes]

    bars = plt.bar(x_labels, y_values, color=colors, width=0.5, edgecolor="black", linewidth=1)

    # Annotate heights
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.ylim(0, 110)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("QTTN Accuracy Across Shape vs. Color Modes (32x32 Canvas)", fontsize=13, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)

    plot_path = "qnlp/image_tower/classification/quantum/results/representation_bias_results.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved representation bias comparison plot to: {plot_path}")

    print("\n" + "=" * 70)
    print("REPRESENTATION BIAS BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Color-Only Test Accuracy:            {accuracies['color_only']:.1f}%")
    print(f"Shape-Only (Grayscale) Accuracy:     {accuracies['shape_only']:.1f}%")
    print(f"Overlapping (Feature Binding) Acc:   {accuracies['overlapping']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_bias_experiments()
