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
# Device Declarations (4 Qubits)
# =====================================================================
num_qubits = 4
dev_noiseless = qml.device("default.qubit", wires=num_qubits)
dev_noisy = qml.device("default.mixed", wires=num_qubits)


# =====================================================================
# Ansatz Block Definitions (HEA, IQP, ALT)
# =====================================================================
def hea_ansatz_node(wires, weights, p_noise=0.0):
    """HEA: Rotations (RZ-RY-RZ) + Central controller CNOTs"""
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Entanglement: CNOTs connecting all children to parent (wire 0)
    for i in range(1, len(wires)):
        qml.CNOT(wires=[wires[i], wires[0]])
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=wires[0])
            qml.DepolarizingChannel(p_noise, wires=wires[i])


def iqp_ansatz_node(wires, weights, p_noise=0.0):
    """IQP: Hadamard prep + RZ + IsingZZ couplings + Hadamard projection"""
    # Hadamard prep
    for w in wires:
        qml.Hadamard(wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # RZ rotations
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Ising ZZ couplings
    for i in range(len(wires)):
        w1 = wires[i]
        w2 = wires[(i + 1) % len(wires)]
        qml.IsingZZ(weights[i, 1], wires=[w1, w2])
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w1)
            qml.DepolarizingChannel(p_noise, wires=w2)

    # Hadamard projection
    for w in wires:
        qml.Hadamard(wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)


def alt_ansatz_node(wires, weights, p_noise=0.0):
    """ALT: Alternating layers of RY and brickwork CNOTs"""
    # Layer 1 Rotations
    for i, w in enumerate(wires):
        qml.RY(weights[i, 0], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Layer 1 CNOTs: [0, 1] and [2, 3]
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[2], wires[3]])
    if p_noise > 0:
        qml.DepolarizingChannel(p_noise, wires=wires[0])
        qml.DepolarizingChannel(p_noise, wires=wires[1])
        qml.DepolarizingChannel(p_noise, wires=wires[2])
        qml.DepolarizingChannel(p_noise, wires=wires[3])

    # Layer 2 Rotations
    for i, w in enumerate(wires):
        qml.RY(weights[i, 1], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Layer 2 CNOTs: [1, 2] and [3, 0]
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[3], wires[0]])
    if p_noise > 0:
        qml.DepolarizingChannel(p_noise, wires=wires[1])
        qml.DepolarizingChannel(p_noise, wires=wires[2])
        qml.DepolarizingChannel(p_noise, wires=wires[3])
        qml.DepolarizingChannel(p_noise, wires=wires[0])


# =====================================================================
# QNode Scaffolding
# =====================================================================
@qml.qnode(dev_noiseless, interface="torch")
def qnode_noiseless(inputs, weights, ansatz_type):
    for i in range(num_qubits):
        qml.RY(inputs[:, i], wires=i)

    if ansatz_type == "hea":
        hea_ansatz_node(range(num_qubits), weights, p_noise=0.0)
    elif ansatz_type == "iqp":
        iqp_ansatz_node(range(num_qubits), weights, p_noise=0.0)
    elif ansatz_type == "alt":
        alt_ansatz_node(range(num_qubits), weights, p_noise=0.0)

    return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]


@qml.qnode(dev_noisy, interface="torch")
def qnode_noisy(inputs, weights, p_noise, ansatz_type):
    for i in range(num_qubits):
        qml.RY(inputs[:, i], wires=i)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=i)

    if ansatz_type == "hea":
        hea_ansatz_node(range(num_qubits), weights, p_noise=p_noise)
    elif ansatz_type == "iqp":
        iqp_ansatz_node(range(num_qubits), weights, p_noise=p_noise)
    elif ansatz_type == "alt":
        alt_ansatz_node(range(num_qubits), weights, p_noise=p_noise)

    return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]


# =====================================================================
# PyTorch Classifier Model
# =====================================================================
class BenchmarkedQTTNClassifier(nn.Module):
    def __init__(self, ansatz_type="hea", mode="noiseless", p_noise=0.0):
        super().__init__()
        self.ansatz_type = ansatz_type
        self.mode = mode
        self.p_noise = p_noise

        # Patch embedding: 4 patches, each 4x4x3 = 48 features -> 1 angle
        self.patch_embed = nn.Linear(48, 1)

        # Initialize weights based on ansatz parameter counts
        if ansatz_type == "hea":
            # 4 wires, 3 rotations each = 12 parameters
            self.weights = nn.Parameter(torch.randn(4, 3) * 0.1)
        else:
            # iqp and alt use 8 parameters per block
            self.weights = nn.Parameter(torch.randn(4, 2) * 0.1)

        self.head = nn.Linear(num_qubits, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Unfold 8x8 image to 4 patches of size 4x4
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # [B, 3, 2, 2, 4, 4]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 2, 2, 3, 4, 4]
        x = x.view(batch_size, 4, 48)  # [B, 4, 48]

        angles = torch.tanh(self.patch_embed(x)) * np.pi
        angles = angles.squeeze(-1)  # [B, 4]

        if self.mode == "noiseless":
            q_out = qnode_noiseless(angles, self.weights, self.ansatz_type)
        else:
            q_out = qnode_noisy(angles, self.weights, self.p_noise, self.ansatz_type)

        q_embeddings = torch.stack(q_out, dim=-1).float()
        return self.head(q_embeddings)


# =====================================================================
# Analytical Complexity Metrics
# =====================================================================
def get_complexity_metrics(ansatz_type):
    """Return (Qubits, Parameters, 1-qubit gates, 2-qubit gates, Depth)"""
    if ansatz_type == "hea":
        return 4, 12, 12, 3, 6
    elif ansatz_type == "iqp":
        return 4, 8, 12, 4, 7
    elif ansatz_type == "alt":
        return 4, 8, 8, 4, 4
    return 4, 0, 0, 0, 0


# =====================================================================
# Benchmarking Execution Scaffolding
# =====================================================================
def benchmark_all(epochs=8, batch_size=32):
    ansatze = ["hea", "iqp", "alt"]
    noise_rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Load 8x8 shapes
    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=512, test_samples=128, img_size=8, seed=42
    )

    results = {}

    for ansatz in ansatze:
        print(f"\n--- Benchmarking Ansatz: {ansatz.upper()} ---")

        # 1. Train model noiselessly
        model = BenchmarkedQTTNClassifier(ansatz_type=ansatz, mode="noiseless")
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
            # print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}%")

        training_time = time.time() - t0

        # Clean evaluation
        model.eval()
        clean_correct = 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                clean_correct += preds.eq(batch_labels).sum().item()
        clean_acc = 100.0 * clean_correct / len(test_loader.dataset)
        print(f"Noiseless Test Accuracy: {clean_acc:.1f}% (Train Time: {training_time:.1f}s)")

        # 2. Noisy Sweeps
        model.mode = "noisy"
        noisy_accs = []
        p_crit = 0.20  # default if it never drops below 50%

        for p in noise_rates:
            model.p_noise = p
            val_correct = 0
            with torch.no_grad():
                for batch_imgs, batch_labels in test_loader:
                    outputs = model(batch_imgs)
                    preds = outputs.argmax(dim=1)
                    val_correct += preds.eq(batch_labels).sum().item()
            acc = 100.0 * val_correct / len(test_loader.dataset)
            noisy_accs.append(acc)

            # Record critical noise threshold where accuracy drops below 50%
            if acc < 50.0 and p_crit == 0.20:
                p_crit = p

        results[ansatz] = {
            "clean_acc": clean_acc,
            "noisy_accs": noisy_accs,
            "p_crit": p_crit,
            "complexity": get_complexity_metrics(ansatz),
        }

    # =====================================================================
    # Plotting Combined Curves
    # =====================================================================
    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = {"hea": "red", "iqp": "blue", "alt": "green"}
    markers = {"hea": "o", "iqp": "s", "alt": "^"}

    for ansatz in ansatze:
        plt.plot(
            noise_rates,
            results[ansatz]["noisy_accs"],
            color=colors[ansatz],
            marker=markers[ansatz],
            linestyle="-",
            linewidth=2.5,
            label=f"{ansatz.upper()} (Noiseless: {results[ansatz]['clean_acc']:.1f}%)",
        )

    plt.axhline(y=25.0, color="gray", linestyle="--", label="Random Guess (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Quantum Ansatz Noise Resilience Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plot_path = "qnlp/image_tower/classification/quantum/results/ansatz_comparison_noise.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved combined comparison plot to: {plot_path}")

    # =====================================================================
    # Output Comparative Markdown Table
    # =====================================================================
    print("\n" + "=" * 80)
    print("COMPARATIVE ANSATZ PERFORMANCE & RESOURCE METRICS")
    print("=" * 80)
    print(
        "| Ansatz Type | Parameters | 1-Qubit Gates | 2-Qubit Gates | Gate Depth | Noiseless Acc | Critical Noise (p_crit) |"
    )
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ansatz in ansatze:
        metrics = results[ansatz]["complexity"]
        clean_acc = results[ansatz]["clean_acc"]
        p_crit = results[ansatz]["p_crit"]
        p_crit_str = f"{p_crit:.3f}" if p_crit < 0.20 else ">0.200"
        print(
            f"| {ansatz.upper()} | {metrics[1]} | {metrics[2]} | {metrics[3]} | {metrics[4]} | {clean_acc:.1f}% | {p_crit_str} |"
        )
    print("=" * 80)


if __name__ == "__main__":
    benchmark_all(epochs=8, batch_size=32)
