import os

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

# =====================================================================
# Device Declarations (4 Wires to bypass density matrix simulation walls)
# =====================================================================
num_qubits = 4
dev_noiseless = qml.device("default.qubit", wires=num_qubits)
dev_noisy = qml.device("default.mixed", wires=num_qubits)


# =====================================================================
# Variational Ansatz Unitaries (Parameterized by Wires)
# =====================================================================
def hea_ansatz(wires, weights, p_noise=0.0):
    """HEA: Rotations (RZ-RY-RZ) + Central controller CNOTs"""
    # weights shape: [len(wires), 3]
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


def iqp_ansatz(wires, weights, p_noise=0.0):
    """IQP: Hadamard prep + RZ + IsingZZ couplings + Hadamard projection"""
    # weights shape: [len(wires), 2]
    for w in wires:
        qml.Hadamard(wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # RZ rotations
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Ising ZZ couplings (ring layout for len(wires) > 1)
    if len(wires) > 1:
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


def alt_ansatz(wires, weights, p_noise=0.0):
    """ALT: Alternating layers of RY and brickwork CNOTs"""
    # weights shape: [len(wires), 2]
    # Layer 1 Rotations
    for i, w in enumerate(wires):
        qml.RY(weights[i, 0], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Layer 1 CNOTs: adjacent pairs
    if len(wires) > 1:
        for i in range(0, len(wires) - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            if p_noise > 0:
                qml.DepolarizingChannel(p_noise, wires=wires[i])
                qml.DepolarizingChannel(p_noise, wires=wires[i + 1])

    # Layer 2 Rotations
    for i, w in enumerate(wires):
        qml.RY(weights[i, 1], wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)

    # Layer 2 CNOTs: offset adjacent pairs
    if len(wires) > 2:
        for i in range(1, len(wires) - 1, 2):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            if p_noise > 0:
                qml.DepolarizingChannel(p_noise, wires=wires[i])
                qml.DepolarizingChannel(p_noise, wires=wires[i + 1])
        # Ring wrap CNOT
        qml.CNOT(wires=[wires[-1], wires[0]])
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=wires[-1])
            qml.DepolarizingChannel(p_noise, wires=wires[0])
    elif len(wires) == 2:
        qml.CNOT(wires=[wires[1], wires[0]])
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=wires[0])
            qml.DepolarizingChannel(p_noise, wires=wires[1])


# =====================================================================
# Data Encoding State Preps (Supports both 1D and 2D tensors)
# =====================================================================
def angle_prep(inputs, wires, p_noise=0.0):
    for i, w in enumerate(wires):
        val = inputs[i] if inputs.ndim == 1 else inputs[:, i]
        qml.RY(val, wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)


def multi_axis_prep(inputs, wires, p_noise=0.0):
    for i, w in enumerate(wires):
        val_x = inputs[3 * i] if inputs.ndim == 1 else inputs[:, 3 * i]
        val_y = inputs[3 * i + 1] if inputs.ndim == 1 else inputs[:, 3 * i + 1]
        val_z = inputs[3 * i + 2] if inputs.ndim == 1 else inputs[:, 3 * i + 2]
        qml.RX(val_x, wires=w)
        qml.RY(val_y, wires=w)
        qml.RZ(val_z, wires=w)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w)


def amplitude_prep(inputs, wires, p_noise=0.0):
    qml.AmplitudeEmbedding(inputs, wires=wires, normalize=True)
    if p_noise > 0:
        for w in wires:
            qml.DepolarizingChannel(p_noise, wires=w)


def zz_feature_map_prep(inputs, wires, p_noise=0.0):
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        val = inputs[i] if inputs.ndim == 1 else inputs[:, i]
        qml.RZ(val, wires=w)
    if len(wires) > 1:
        for i in range(len(wires)):
            w1 = wires[i]
            w2 = wires[(i + 1) % len(wires)]
            val1 = inputs[i] if inputs.ndim == 1 else inputs[:, i]
            val2 = inputs[(i + 1) % len(wires)] if inputs.ndim == 1 else inputs[:, (i + 1) % len(wires)]
            qml.IsingZZ(val1 * val2, wires=[w1, w2])
    if p_noise > 0:
        for w in wires:
            qml.DepolarizingChannel(p_noise, wires=w)


# =====================================================================
# QNode Scaffolding
# =====================================================================
def qttn_circuit(inputs, weights, p_noise, encoding, ansatz_type):
    ansatz = hea_ansatz if ansatz_type == "hea" else (iqp_ansatz if ansatz_type == "iqp" else alt_ansatz)

    if encoding == "angle":
        angle_prep(inputs, range(4), p_noise)
        ansatz(range(4), weights, p_noise)
        return [qml.expval(qml.PauliZ(w)) for w in range(4)]

    elif encoding == "zz_map":
        zz_feature_map_prep(inputs, range(4), p_noise)
        ansatz(range(4), weights, p_noise)
        return [qml.expval(qml.PauliZ(w)) for w in range(4)]

    elif encoding == "amplitude":
        amplitude_prep(inputs, range(2), p_noise)
        ansatz(range(2), weights, p_noise)
        # Pad expectations to length 4 to match output layer size
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.Identity(0)),
            qml.expval(qml.Identity(1)),
        ]

    elif encoding == "multi_axis":
        multi_axis_prep(inputs, range(4), p_noise)
        ansatz(range(4), weights, p_noise)
        return [qml.expval(qml.PauliZ(w)) for w in range(4)]


@qml.qnode(dev_noiseless, interface="torch")
def noiseless_qttn_qnode(inputs, weights, encoding, ansatz_type):
    return qttn_circuit(inputs, weights, 0.0, encoding, ansatz_type)


@qml.qnode(dev_noisy, interface="torch")
def noisy_qttn_qnode(inputs, weights, p_noise, encoding, ansatz_type):
    return qttn_circuit(inputs, weights, p_noise, encoding, ansatz_type)


# =====================================================================
# PyTorch Classifier Model
# =====================================================================
class ScopedQTTNClassifier(nn.Module):
    def __init__(self, encoding="angle", ansatz_type="hea", mode="noiseless", p_noise=0.0):
        super().__init__()
        self.encoding = encoding
        self.ansatz_type = ansatz_type
        self.mode = mode
        self.p_noise = p_noise

        # Patch embedding: 4 patches, each 4x4x3 = 48 features
        if encoding in ["angle", "zz_map"]:
            # 1 feature per patch -> 4 features total
            self.patch_embed = nn.Linear(48, 1)
        elif encoding == "amplitude":
            # 1 feature per patch -> 4 features total (amplitude prepped into 2 qubits)
            self.patch_embed = nn.Linear(48, 1)
        elif encoding == "multi_axis":
            # 3 features per patch -> 12 features total (multi-axis prepped into 4 qubits)
            self.patch_embed = nn.Linear(48, 3)

        n_params = 3 if ansatz_type == "hea" else 2
        w_size = 2 if encoding == "amplitude" else 4
        self.weights = nn.Parameter(torch.randn(w_size, n_params) * 0.1)
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Unfold 8x8 image to 4 patches of size 4x4
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)  # [B, 3, 2, 2, 4, 4]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 2, 2, 3, 4, 4]
        x = x.view(batch_size, 4, 48)  # [B, 4, 48]

        features = torch.tanh(self.patch_embed(x)) * np.pi  # [B, 4, F_per_patch]

        # Flatten patches depending on encoding style
        if self.encoding in ["angle", "zz_map", "amplitude"]:
            features = features.view(batch_size, 4)  # [B, 4]
        elif self.encoding == "multi_axis":
            features = features.view(batch_size, 12)  # [B, 12]

        if self.mode == "noiseless":
            q_out = noiseless_qttn_qnode(features, self.weights, self.encoding, self.ansatz_type)
            q_embeddings = torch.stack(q_out, dim=-1).float()  # [B, 4]
        else:
            # Noisy evaluation: run sample-by-sample to bypass the PennyLane default.mixed parameter broadcasting bug on RZ
            q_embeddings = []
            for i in range(batch_size):
                q_out = noisy_qttn_qnode(features[i], self.weights, self.p_noise, self.encoding, self.ansatz_type)
                q_embeddings.append(torch.stack(q_out))
            q_embeddings = torch.stack(q_embeddings).float()  # [B, 4]

        return self.head(q_embeddings)


# =====================================================================
# Analytical Complexity Metrics
# =====================================================================
def get_resource_complexity(encoding, ansatz):
    n_qubits = {"angle": 4, "zz_map": 4, "amplitude": 2, "multi_axis": 4}[encoding]
    n_params = (n_qubits * 3) if ansatz == "hea" else (n_qubits * 2)
    return n_qubits, n_params


# =====================================================================
# Benchmarking Harvester
# =====================================================================
def harvest_sweeps():
    encodings = ["angle", "multi_axis", "amplitude", "zz_map"]
    ansatze = ["hea", "iqp", "alt"]
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=32, train_samples=512, test_samples=128, img_size=8, seed=42
    )

    results = {}

    print("\nStarting Multi-Dimensional Encoding & Ansatz Sweep (Optimized)...")
    print("=" * 65)

    for enc in encodings:
        results[enc] = {}
        for ans in ansatze:
            print(f"Running Config: Encoding={enc.upper()} | Ansatz={ans.upper()}")

            model = ScopedQTTNClassifier(encoding=enc, ansatz_type=ans, mode="noiseless")
            optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            # Train for 8 epochs noiselessly
            for epoch in range(8):
                model.train()
                for batch_imgs, batch_labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_imgs)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

            # Clean eval
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_imgs, batch_labels in test_loader:
                    outputs = model(batch_imgs)
                    preds = outputs.argmax(dim=1)
                    correct += preds.eq(batch_labels).sum().item()
            clean_acc = 100.0 * correct / len(test_loader.dataset)

            # Noisy eval sweep
            model.mode = "noisy"
            noisy_accs = []
            p_crit = 0.20  # default

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
                if acc < 50.0 and p_crit == 0.20:
                    p_crit = p

            qubits, params = get_resource_complexity(enc, ans)

            results[enc][ans] = {
                "qubits": qubits,
                "params": params,
                "clean_acc": clean_acc,
                "noisy_accs": noisy_accs,
                "p_crit": p_crit,
            }
            print(f" -> Noiseless Acc: {clean_acc:.1f}% | p_crit: {p_crit:.3f}")

    # =====================================================================
    # Plotting Combined Comparison
    # =====================================================================
    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)
    plt.figure(figsize=(12, 8))

    colors = {"hea": "red", "iqp": "blue", "alt": "green"}
    linestyles = {"angle": "-", "multi_axis": "--", "amplitude": "-.", "zz_map": ":"}

    for enc in encodings:
        for ans in ansatze:
            plt.plot(
                noise_rates,
                results[enc][ans]["noisy_accs"],
                color=colors[ans],
                linestyle=linestyles[enc],
                marker="o",
                alpha=0.7,
                label=f"{enc.upper()} + {ans.upper()}",
            )

    plt.axhline(y=25.0, color="gray", linestyle="--", label="Random (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Multi-Dimensional Encoding & Ansatz Noise Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    plot_path = "qnlp/image_tower/classification/quantum/results/encoding_ansatz_sweep.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved combined comparative plot to: {plot_path}")

    # =====================================================================
    # Comparative Markdown Table
    # =====================================================================
    print("\n" + "=" * 80)
    print("COMPARATIVE ENCODING & ANSATZ SWEEP MATRIX")
    print("=" * 80)
    print("| Encoding | Ansatz | Qubits | Parameters | Noiseless Acc | Critical Noise (p_crit) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for enc in encodings:
        for ans in ansatze:
            res = results[enc][ans]
            p_crit_str = f"{res['p_crit']:.3f}" if res["p_crit"] < 0.20 else ">0.200"
            print(
                f"| {enc.upper()} | {ans.upper()} | {res['qubits']} | {res['params']} | {res['clean_acc']:.1f}% | {p_crit_str} |"
            )
    print("=" * 80)


if __name__ == "__main__":
    harvest_sweeps()
