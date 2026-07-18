import os

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders


# =====================================================================
# Variational IQP Blocks (Helper functions)
# =====================================================================
def iqp_block_2q(wires, weights, p_noise=0.0):
    # weights shape: [3] -> RZ, RZ, ZZ
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    qml.RZ(weights[0], wires=wires[0])
    qml.RZ(weights[1], wires=wires[1])
    qml.IsingZZ(weights[2], wires=[wires[0], wires[1]])
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])
    if p_noise > 0:
        qml.DepolarizingChannel(p_noise, wires=wires[0])
        qml.DepolarizingChannel(p_noise, wires=wires[1])


def iqp_block_any(wires, weights, p_noise=0.0):
    # weights shape: [len(wires), 2] -> RZ, ZZ
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
    if len(wires) > 1:
        for i in range(len(wires)):
            w1 = wires[i]
            w2 = wires[(i + 1) % len(wires)]
            qml.IsingZZ(weights[i, 1], wires=[w1, w2])
    for w in wires:
        qml.Hadamard(wires=w)
    if p_noise > 0:
        for w in wires:
            qml.DepolarizingChannel(p_noise, wires=w)


# =====================================================================
# State Prep: Multi-Axis Encoding (3 features per qubit)
# =====================================================================
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


# =====================================================================
# QNode Circuits (QTTN, MPS, MERA)
# =====================================================================
def build_circuits(N, dev):
    @qml.qnode(dev, interface="torch")
    def qttn_circuit(inputs, weights_l1, weights_l2, p_noise):
        multi_axis_prep(inputs, range(N), p_noise)
        # Layer 1: N/4 blocks of 4 qubits
        for b in range(N // 4):
            iqp_block_any(range(4 * b, 4 * b + 4), weights_l1[b], p_noise)
        # Layer 2: contract parent qubits
        parent_wires = list(range(0, N, 4))
        iqp_block_any(parent_wires, weights_l2, p_noise)
        return [qml.expval(qml.PauliZ(w)) for w in parent_wires]

    @qml.qnode(dev, interface="torch")
    def mps_circuit(inputs, weights, p_noise):
        multi_axis_prep(inputs, range(N), p_noise)
        # 1D linear chain of 2-qubit IQP blocks
        for i in range(N - 1):
            iqp_block_2q([i, i + 1], weights[i], p_noise)
        # Return expectations for boundary qubits (for N qubits, return N//4 parent equivalents to align outputs)
        out_wires = list(range(0, N, 4))
        return [qml.expval(qml.PauliZ(w)) for w in out_wires]

    @qml.qnode(dev, interface="torch")
    def mera_circuit(inputs, weights_dis, weights_l1, weights_l2, p_noise):
        multi_axis_prep(inputs, range(N), p_noise)
        # 1. Disentangler Layer (act across block boundaries)
        if N == 4:
            iqp_block_2q([1, 2], weights_dis[0], p_noise)
        else:
            for i, w_idx in enumerate(range(3, N - 1, 4)):
                iqp_block_2q([w_idx, w_idx + 1], weights_dis[i], p_noise)
        # 2. Tree Layer 1
        for b in range(N // 4):
            iqp_block_any(range(4 * b, 4 * b + 4), weights_l1[b], p_noise)
        # 3. Tree Layer 2
        parent_wires = list(range(0, N, 4))
        iqp_block_any(parent_wires, weights_l2, p_noise)
        return [qml.expval(qml.PauliZ(w)) for w in parent_wires]

    return qttn_circuit, mps_circuit, mera_circuit


# =====================================================================
# PyTorch Topology Models (For 8x8 inputs, N=4 patches)
# =====================================================================
class TopologyClassifier(nn.Module):
    def __init__(self, topology="qttn", mode="noiseless", p_noise=0.0):
        super().__init__()
        self.topology = topology
        self.mode = mode
        self.p_noise = p_noise

        # 4 patches of size 4x4x3 = 48 features -> 3 features per qubit
        self.patch_embed = nn.Linear(48, 3)

        # Setup devices
        self.dev_noiseless = qml.device("default.qubit", wires=4)
        self.dev_noisy = qml.device("default.mixed", wires=4)

        # Build QNodes
        q_qttn_noiseless, q_mps_noiseless, q_mera_noiseless = build_circuits(4, self.dev_noiseless)
        q_qttn_noisy, q_mps_noisy, q_mera_noisy = build_circuits(4, self.dev_noisy)

        self.q_noiseless = {"qttn": q_qttn_noiseless, "mps": q_mps_noiseless, "mera": q_mera_noiseless}[topology]
        self.q_noisy = {"qttn": q_qttn_noisy, "mps": q_mps_noisy, "mera": q_mera_noisy}[topology]

        # Weights definition
        if topology == "qttn":
            self.weights_l1 = nn.Parameter(torch.randn(1, 4, 2) * 0.1)
            self.weights_l2 = nn.Parameter(torch.randn(1, 2) * 0.1)  # 1 parent wire measured
        elif topology == "mps":
            self.weights = nn.Parameter(torch.randn(3, 3) * 0.1)  # 3 blocks of 2-qubit unitaries
        elif topology == "mera":
            self.weights_dis = nn.Parameter(torch.randn(1, 3) * 0.1)  # 1 disentangler block
            self.weights_l1 = nn.Parameter(torch.randn(1, 4, 2) * 0.1)
            self.weights_l2 = nn.Parameter(torch.randn(1, 2) * 0.1)

        self.head = nn.Linear(1, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        # Unfold 8x8 image to 4 patches of size 4x4
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(batch_size, 4, 48)

        features = torch.tanh(self.patch_embed(x)) * np.pi
        features = features.view(batch_size, 12)

        if self.mode == "noiseless":
            if self.topology == "qttn":
                q_out = self.q_noiseless(features, self.weights_l1, self.weights_l2, 0.0)
            elif self.topology == "mps":
                q_out = self.q_noiseless(features, self.weights, 0.0)
            elif self.topology == "mera":
                q_out = self.q_noiseless(features, self.weights_dis, self.weights_l1, self.weights_l2, 0.0)
            q_embeddings = torch.stack(q_out, dim=-1).float()
        else:
            # Sample-by-sample noisy evaluation to bypass PennyLane mixed-state batching bug
            q_embeddings = []
            for i in range(batch_size):
                if self.topology == "qttn":
                    q_out = self.q_noisy(features[i], self.weights_l1, self.weights_l2, self.p_noise)
                elif self.topology == "mps":
                    q_out = self.q_noisy(features[i], self.weights, self.p_noise)
                elif self.topology == "mera":
                    q_out = self.q_noisy(features[i], self.weights_dis, self.weights_l1, self.weights_l2, self.p_noise)
                q_embeddings.append(torch.stack(q_out))
            q_embeddings = torch.stack(q_embeddings).float()

        return self.head(q_embeddings)


# =====================================================================
# Experiment A: Barren Plateau Scaling Sweep
# =====================================================================
def run_barren_plateau_experiment(sizes, trials=100):
    print("\nStarting Barren Plateau Sweep (QTTN vs. MPS vs. MERA)...")
    print("=" * 65)

    results = {"qttn": [], "mps": [], "mera": []}

    for N in sizes:
        print(f"Testing system size: N = {N} qubits...")
        dev_bp = qml.device("default.qubit", wires=N)
        q_qttn, q_mps, q_mera = build_circuits(N, dev_bp)

        # Trial Loop
        g_qttn, g_mps, g_mera = [], [], []
        for t in range(trials):
            inputs = torch.randn(N * 3) * np.pi

            # Setup weights as leaf parameters
            # QTTN weights
            w_qttn_l1 = (torch.rand(N // 4, 4, 2) * 2 * np.pi).requires_grad_(True)
            w_qttn_l2 = (torch.rand(N // 4, 2) * 2 * np.pi).requires_grad_(True)
            # MPS weights
            w_mps = (torch.rand(N - 1, 3) * 2 * np.pi).requires_grad_(True)
            # MERA weights
            n_dis = 1 if N <= 8 else (N // 4 - 1)
            w_mera_dis = (torch.rand(n_dis, 3) * 2 * np.pi).requires_grad_(True)
            w_mera_l1 = (torch.rand(N // 4, 4, 2) * 2 * np.pi).requires_grad_(True)
            w_mera_l2 = (torch.rand(N // 4, 2) * 2 * np.pi).requires_grad_(True)

            # Forward + Backward
            out_qttn = q_qttn(inputs, w_qttn_l1, w_qttn_l2, 0.0)
            out_qttn[0].backward()
            g_qttn.append(w_qttn_l1.grad[0, 0, 0].item())

            out_mps = q_mps(inputs, w_mps, 0.0)
            out_mps[0].backward()
            g_mps.append(w_mps.grad[0, 0].item())

            out_mera = q_mera(inputs, w_mera_dis, w_mera_l1, w_mera_l2, 0.0)
            out_mera[0].backward()
            g_mera.append(w_mera_l1.grad[0, 0, 0].item())

        results["qttn"].append(np.var(g_qttn))
        results["mps"].append(np.var(g_mps))
        results["mera"].append(np.var(g_mera))

        print(
            f" -> QTTN Var: {results['qttn'][-1]:.2e} | MPS Var: {results['mps'][-1]:.2e} | MERA Var: {results['mera'][-1]:.2e}"
        )

    # Plot BP scaling comparison
    os.makedirs("qnlp/image_tower/classification/quantum/results", exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Log-Log scale check
    plt.loglog(sizes, results["qttn"], marker="o", color="red", label="QTTN (Hierarchical)")
    plt.loglog(sizes, results["mps"], marker="x", color="blue", label="MPS (1D Chain)")
    plt.loglog(sizes, results["mera"], marker="s", color="green", label="MERA (Disentangled)")

    plt.xlabel("Qubit Count ($N$)", fontsize=12)
    plt.ylabel("Gradient Variance $\\text{Var}[\\partial_{\\theta} \\mathcal{L}]$", fontsize=12)
    plt.title("Barren Plateau Scaling Comparison (Log-Log Scale)", fontsize=13, fontweight="bold")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plot_path = "qnlp/image_tower/classification/quantum/results/topology_barren_plateaus.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Barren Plateau comparison plot to: {plot_path}")
    return results


# =====================================================================
# Main Benchmark Sweep Execution
# =====================================================================
def run_comparative_benchmark():
    # 1. Barren Plateau sweep (N=4, 8, 12, 16, 20)
    sizes = [4, 8, 12, 16, 20]
    bp_results = run_barren_plateau_experiment(sizes, trials=100)

    # 2. Noiseless training & Noisy evaluation sweep
    topologies = ["qttn", "mps", "mera"]
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    eval_accs = {}

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=32, train_samples=512, test_samples=128, img_size=8, mode="overlapping", seed=42
    )

    print("\nStarting Noiseless training & Noisy sweeps (Overlapping mode)...")
    print("=" * 65)

    for topo in topologies:
        print(f"Training Model: Topology={topo.upper()}...")
        model = TopologyClassifier(topology=topo, mode="noiseless")
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

        # Clean evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(batch_labels).sum().item()
        clean_acc = 100.0 * correct / len(test_loader.dataset)

        # Noisy evaluation sweep
        model.mode = "noisy"
        topo_accs = []
        for p in noise_rates:
            model.p_noise = p
            val_correct = 0
            with torch.no_grad():
                for batch_imgs, batch_labels in test_loader:
                    outputs = model(batch_imgs)
                    preds = outputs.argmax(dim=1)
                    val_correct += preds.eq(batch_labels).sum().item()
            acc = 100.0 * val_correct / len(test_loader.dataset)
            topo_accs.append(acc)

        eval_accs[topo] = {"clean_acc": clean_acc, "noisy_accs": topo_accs}
        print(f" -> Topology: {topo.upper()} | Noiseless Acc: {clean_acc:.1f}%")

    # Plot Noise Resilience Comparison
    plt.figure(figsize=(8, 5))
    colors = {"qttn": "red", "mps": "blue", "mera": "green"}
    labels = {"qttn": "QTTN (Hierarchical)", "mps": "MPS (1D Chain)", "mera": "MERA (Disentangled)"}

    for topo in topologies:
        plt.plot(noise_rates, eval_accs[topo]["noisy_accs"], color=colors[topo], marker="o", label=labels[topo])

    plt.axhline(y=25.0, color="gray", linestyle="--", label="Random (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)", fontsize=12)
    plt.ylabel("Test Accuracy (%)", fontsize=12)
    plt.title("Noise Resilience Sweep Across Tensor Topologies", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()

    noise_plot_path = "qnlp/image_tower/classification/quantum/results/topology_noise_resilience.png"
    plt.savefig(noise_plot_path)
    plt.close()
    print(f"Saved Noise Resilience comparison plot to: {noise_plot_path}")

    # Print Markdown Summary Matrix
    print("\n" + "=" * 80)
    print("COMPARATIVE TOPOLOGY BENCHMARK MATRIX")
    print("=" * 80)
    print("| Topology | Active Qubits | Parameter count | Noiseless Acc | BP Variance (N=20) | Noise Acc (p=0.10) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    # Parameter counts for 4 qubits:
    # QTTN: weights_l1 (1,4,2) + weights_l2 (1,2) = 10 params.
    # MPS: weights (3,3) = 9 params.
    # MERA: weights_dis (1,3) + weights_l1 (1,4,2) + weights_l2 (1,2) = 13 params.
    param_counts = {"qttn": 10, "mps": 9, "mera": 13}

    for topo in topologies:
        res = eval_accs[topo]
        print(
            f"| {labels[topo]} | 4 | {param_counts[topo]} | {res['clean_acc']:.1f}% | {bp_results[topo][-1]:.2e} | {res['noisy_accs'][3]:.1f}% |"
        )
    print("=" * 80)


if __name__ == "__main__":
    run_comparative_benchmark()
