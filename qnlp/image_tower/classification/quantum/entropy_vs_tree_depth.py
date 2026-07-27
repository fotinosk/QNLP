"""
Question A.2: Entanglement Entropy vs. Classical TTN Area-Law Bound.

llm/quantum_investigation_roadmap.md Section 2, Question A.2: can we bound the
entanglement entropy of the QTTN's quantum state at each level of the tree,
and does it match the area-law/tree-entropy bounds of a classical TTN?

Extends ansatz_diagnostics.py's single-node CNOT-to-entropy diagnostic
(2026-07-17, entropy ~0.51 for the 3-CNOT "all children to parent" config at
a single node) to the FULL multi-level tree. Each node in this QTTN passes
exactly 1 surviving qubit to its parent (bond dimension chi=2), so the
area-law prediction is: entanglement entropy across ANY internal tree bond is
bounded by log2(chi) = ln(2) ~= 0.693 nats, REGARDLESS of how many leaf
patches feed into that subtree. A volume-law state would instead show entropy
growing with subtree size. This script measures the root qubit's entropy at
tree depth 1 (4 patches, 1 level) and depth 2 (16 patches, 2 levels) using
random weights/inputs (matching ansatz_diagnostics.py's diagnostic style --
this characterizes the architecture's structural entropy bound, independent
of any specific trained model) and checks whether it stays flat near ln(2)
rather than growing with tree size.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

RESULTS_DIR = "qnlp/image_tower/classification/quantum/results"
LN2 = np.log(2)

# Same ansatz as ansatz_diagnostics.py's "3 CNOTs (All children to parent)" config,
# identified there as the minimal density achieving ~73% of max entropy transfer.
CNOT_PAIRS = [(1, 0), (2, 0), (3, 0)]


def quad_node_unitary(wires, weights, cnot_pairs=CNOT_PAIRS):
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)
    for ctrl, tgt in cnot_pairs:
        qml.CNOT(wires=[wires[ctrl], wires[tgt]])


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))


# =====================================================================
# Depth 1: 4 patches -> 1 root (single quad-node)
# =====================================================================
dev_d1 = qml.device("default.qubit", wires=4)


@qml.qnode(dev_d1)
def circuit_depth1(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(4), rotation="Y")
    quad_node_unitary(range(4), weights)
    return qml.density_matrix(wires=[0])


# =====================================================================
# Depth 2: 16 patches -> 4 nodes -> 1 root
# =====================================================================
dev_d2 = qml.device("default.qubit", wires=16)


@qml.qnode(dev_d2)
def circuit_depth2(inputs, weights_l1, weights_l2):
    for i in range(16):
        qml.RY(inputs[i], wires=i)
    for b in range(4):
        block_wires = [4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3]
        quad_node_unitary(block_wires, weights_l1[b])
    quad_node_unitary([0, 4, 8, 12], weights_l2)
    return qml.density_matrix(wires=[0])


def run_experiment(num_samples=150, seed=42):
    rng = np.random.default_rng(seed)
    results = {}

    print("Depth 1 (4 patches, 1 level)...")
    entropies_d1 = []
    for _ in range(num_samples):
        inputs = rng.uniform(0, 2 * np.pi, size=4)
        weights = rng.uniform(0, 2 * np.pi, size=(4, 3))
        rho = circuit_depth1(inputs, weights)
        entropies_d1.append(von_neumann_entropy(rho))
    results[1] = {"num_leaves": 4, "mean": float(np.mean(entropies_d1)), "std": float(np.std(entropies_d1))}

    print("Depth 2 (16 patches, 2 levels)...")
    entropies_d2 = []
    for _ in range(num_samples):
        inputs = rng.uniform(0, 2 * np.pi, size=16)
        weights_l1 = rng.uniform(0, 2 * np.pi, size=(4, 4, 3))
        weights_l2 = rng.uniform(0, 2 * np.pi, size=(4, 3))
        rho = circuit_depth2(inputs, weights_l1, weights_l2)
        entropies_d2.append(von_neumann_entropy(rho))
    results[2] = {"num_leaves": 16, "mean": float(np.mean(entropies_d2)), "std": float(np.std(entropies_d2))}

    print("\n" + "=" * 60)
    print("RESULTS: Root-Qubit Entanglement Entropy vs. Tree Depth")
    print("=" * 60)
    print(f"Area-law bound (log2(chi=2) = ln(2)): {LN2:.4f} nats")
    for depth, r in results.items():
        pct_of_bound = 100 * r["mean"] / LN2
        print(
            f"Depth {depth} ({r['num_leaves']:2d} leaf patches): "
            f"entropy = {r['mean']:.4f} +/- {r['std']:.4f} ({pct_of_bound:.1f}% of area-law bound)"
        )

    # ---- Plot ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    depths = list(results.keys())
    means = [results[d]["mean"] for d in depths]
    stds = [results[d]["std"] for d in depths]
    leaves = [results[d]["num_leaves"] for d in depths]

    plt.figure(figsize=(7, 5))
    plt.errorbar(
        depths, means, yerr=stds, fmt="-o", color="tab:blue", capsize=5, linewidth=2, label="Root-qubit entropy"
    )
    plt.axhline(y=LN2, color="red", linestyle="--", label=f"Area-law bound: ln(2) = {LN2:.3f}")
    plt.xticks(depths, [f"Depth {d}\n({n} leaves)" for d, n in zip(depths, leaves)])
    plt.ylabel("Von Neumann Entropy (nats)")
    plt.title("Root-Qubit Entanglement Entropy vs. Tree Depth\n(flat = area-law; growing = volume-law)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "entropy_vs_tree_depth.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved plot to: {plot_path}")

    return results


if __name__ == "__main__":
    run_experiment(num_samples=150)
