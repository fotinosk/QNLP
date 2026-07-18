import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch


# =====================================================================
# Variational IQP Ansatz
# =====================================================================
def iqp_ansatz(wires, weights):
    # weights shape: [len(wires), 2]
    for w in wires:
        qml.Hadamard(wires=w)
    # RZ rotations
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
    # Ising ZZ couplings
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
        qml.RX(inputs[3 * i], wires=w)
        qml.RY(inputs[3 * i + 1], wires=w)
        qml.RZ(inputs[3 * i + 2], wires=w)


# =====================================================================
# Device & QNode Definitions for System Sizes N
# =====================================================================
def build_qttn_qnode(N):
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="torch")
    def qttn_circuit(inputs, weights_l1, weights_l2=None):
        # inputs shape: [N * 3]
        multi_axis_prep(inputs, range(N))

        if N == 4:
            # 1 Layer: 1 block of 4 qubits
            iqp_ansatz(range(4), weights_l1)
        elif N == 9:
            # 2 Layers: 3 blocks of 3 qubits -> 1 block of 3 qubits
            for b in range(3):
                iqp_ansatz(range(3 * b, 3 * b + 3), weights_l1[b])
            iqp_ansatz([0, 3, 6], weights_l2)
        elif N == 16:
            # 2 Layers: 4 blocks of 4 qubits -> 1 block of 4 qubits
            for b in range(4):
                iqp_ansatz(range(4 * b, 4 * b + 4), weights_l1[b])
            iqp_ansatz([0, 4, 8, 12], weights_l2)
        elif N == 20:
            # 2 Layers: 4 blocks of 5 qubits -> 1 block of 4 qubits
            for b in range(4):
                iqp_ansatz(range(5 * b, 5 * b + 5), weights_l1[b])
            iqp_ansatz([0, 5, 10, 15], weights_l2)

        # Local expectation value measurement at the root
        return qml.expval(qml.PauliZ(0))

    return qttn_circuit


# =====================================================================
# Barren Plateau Tester
# =====================================================================
def run_barren_plateau_sweep():
    sizes = [4, 9, 16, 20]
    trials = 100
    variances = []

    print("=============================================================")
    print("Executing Barren Plateau Gradient Variance Scaling Sweep (Safe)")
    print(f"System Sizes (Qubits): {sizes} | Trials per Size: {trials}")
    print("=============================================================")

    for N in sizes:
        print(f"Running System Size: N = {N} qubits (Hilbert space dimension: {2**N})...")
        qnode = build_qttn_qnode(N)
        gradients = []

        t0 = time.time()
        for t in range(trials):
            # 1. Random Input features
            inputs = torch.randn(N * 3) * np.pi

            # 2. Random parameter weights (initialized as leaf tensors)
            if N == 4:
                weights_l1 = (torch.rand(4, 2) * 2 * np.pi).requires_grad_(True)
                weights_l2 = None
            elif N == 9:
                weights_l1 = (torch.rand(3, 3, 2) * 2 * np.pi).requires_grad_(True)
                weights_l2 = (torch.rand(3, 2) * 2 * np.pi).requires_grad_(True)
            elif N == 16:
                weights_l1 = (torch.rand(4, 4, 2) * 2 * np.pi).requires_grad_(True)
                weights_l2 = (torch.rand(4, 2) * 2 * np.pi).requires_grad_(True)
            elif N == 20:
                weights_l1 = (torch.rand(4, 5, 2) * 2 * np.pi).requires_grad_(True)
                weights_l2 = (torch.rand(4, 2) * 2 * np.pi).requires_grad_(True)

            # 3. Forward pass
            out = qnode(inputs, weights_l1, weights_l2)

            # 4. Backward pass
            out.backward()

            # 5. Extract gradient of leaf parameter weights_l1[0, 0, 0] or weights_l1[0, 0]
            if N == 4:
                grad_val = weights_l1.grad[0, 0].item()
            else:
                grad_val = weights_l1.grad[0, 0, 0].item()

            gradients.append(grad_val)

        # Calculate statistical variance
        var = np.var(gradients)
        variances.append(var)
        dt = time.time() - t0
        print(f" -> N = {N} | Gradient Mean: {np.mean(gradients):.2e} | Variance: {var:.2e} | Time: {dt:.1f}s")

    # =====================================================================
    # Plotting Scaling curves (Log-Log and Semi-Log)
    # =====================================================================
    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)

    plt.figure(figsize=(12, 5))

    # 1. Semi-log plot (checks for exponential barren plateau decay: Var ~ e^-N)
    plt.subplot(1, 2, 1)
    plt.semilogy(sizes, variances, marker="o", color="red", label="QTTN Leaf")
    plt.xlabel("Qubit Count ($N$)", fontsize=12)
    plt.ylabel("Gradient Variance $\\text{Var}[\\partial_{\\theta} \\mathcal{L}]$", fontsize=12)
    plt.title("Semi-Log Scale (Exponential Check)", fontsize=13)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    # 2. Log-log plot (checks for polynomial decay: Var ~ 1/Poly(N))
    plt.subplot(1, 2, 2)
    plt.loglog(sizes, variances, marker="o", color="blue", label="QTTN Leaf")
    ref_x = np.array(sizes)
    ref_y = variances[0] * (sizes[0] / ref_x) ** 2
    plt.loglog(ref_x, ref_y, linestyle="--", color="gray", label="Reference $1/N^2$")

    plt.xlabel("Qubit Count ($N$)", fontsize=12)
    plt.ylabel("Gradient Variance $\\text{Var}[\\partial_{\\theta} \\mathcal{L}]$", fontsize=12)
    plt.title("Log-Log Scale (Polynomial Check)", fontsize=13)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_path = "qnlp/image_tower/classification/quantum/results/barren_plateau_scaling.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"\nSaved scaling plots to: {plot_path}")

    # Print Markdown Summary Table
    print("\n" + "=" * 50)
    print("GRADIENT VARIANCE SCALING MATRIX")
    print("=" * 50)
    print("| Qubits (N) | Image size (H x W) | Hilbert Dimension | Variance |")
    print("| :--- | :--- | :--- | :--- |")
    for i, N in enumerate(sizes):
        if N == 4:
            h_w = "8x8"
        elif N == 9:
            h_w = "12x12"
        elif N == 16:
            h_w = "16x16"
        elif N == 20:
            h_w = "20x20"
        print(f"| {N} | {h_w} | {2**N:,} | {variances[i]:.2e} |")
    print("=" * 50)


if __name__ == "__main__":
    run_barren_plateau_sweep()
