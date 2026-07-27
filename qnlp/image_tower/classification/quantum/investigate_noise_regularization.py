"""
Question E.2 + Question F: Entropy Propagation Through Partial Trace, and
Depolarizing Noise as Implicit Regularization.

llm/quantum_investigation_roadmap.md Section 2:
  Question E.2: How does noise propagate through the partial-trace/pooling
    operations? Does discarding qubits "wash away" noise, or does it
    propagate mixed-state entropy to the root of the tree?
  Question F: Test accuracy has twice been observed to *improve* under
    depolarizing noise (topology benchmark: 42.2% clean -> 53.9% at p=0.10;
    synthetic-shapes sweep: 85.9% -> 91.4% at p=0.005). Why, and can this be
    scheduled deliberately (train noisy, evaluate clean)?

Part A (E.2): measure Von Neumann entropy at the level-1 survivor qubit and
the root qubit as depolarizing noise p increases, to see whether entropy
picked up at level 1 keeps accumulating through level 2's partial trace, or
whether each pooling step "resets" some of it.

Part B (F): train the baseline hierarchical QTTN three ways -- noiseless
throughout, trained under a fixed noise rate, evaluated only on clean
(p=0) test data -- to directly test whether noise injected during training
acts as a regularizer that improves clean generalization (the noise-as-
dropout hypothesis), rather than just happening to help at eval time.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.image_tower.classification.quantum.investigate_quantum_residuals import HierarchicalQTTNClassifier
from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

RESULTS_DIR = "qnlp/image_tower/classification/quantum/results"
NUM_QUBITS = 4
NUM_LAYERS = 2
CNOT_PAIRS = [(1, 0), (2, 0), (3, 0)]  # matches ansatz_diagnostics.py's "3 CNOTs" config


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
# Part A: Entropy propagation through partial trace, vs. noise rate
# =====================================================================
# NOTE: default.mixed is a density-matrix simulator -- memory scales as
# O(4^N), not O(2^N) like default.qubit. A genuine 16-qubit noisy density
# matrix needs ~68GB and is infeasible (confirmed empirically -- OOM-killed).
# Every other noisy-circuit experiment in this project has stayed at 4-5
# qubits for exactly this reason. For this diagnostic we use 8 total qubits:
# two level-1 quad-nodes (wires 0-3 and 4-7, survivors at 0 and 4), then a
# level-2 quad-node applied to [0, 4, 1, 5] -- the two official survivors
# plus two of level-1's *discarded* qubits (not reset, so they still carry
# whatever state level 1's noisy processing left them in). This is a
# simplification of the true 4-ary branching (kept small for tractability)
# but is a genuine, coherent two-level circuit suitable for testing whether
# entropy generated at level 1 propagates to level 2 or gets washed away.
dev_level1 = qml.device("default.mixed", wires=4)
dev_root = qml.device("default.mixed", wires=8)


@qml.qnode(dev_level1)
def circuit_level1(inputs, weights, p_noise):
    for i in range(4):
        qml.RY(inputs[i], wires=i)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=i)
    quad_node_unitary(range(4), weights)
    if p_noise > 0:
        for w in range(4):
            qml.DepolarizingChannel(p_noise, wires=w)
    return qml.density_matrix(wires=[0])


@qml.qnode(dev_root)
def circuit_root(inputs, weights_l1, weights_l2, p_noise):
    for i in range(8):
        qml.RY(inputs[i], wires=i)
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=i)
    for b in range(2):
        bw = [4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3]
        quad_node_unitary(bw, weights_l1[b])
        if p_noise > 0:
            for w in bw:
                qml.DepolarizingChannel(p_noise, wires=w)
    quad_node_unitary([0, 4, 1, 5], weights_l2)
    if p_noise > 0:
        for w in [0, 4, 1, 5]:
            qml.DepolarizingChannel(p_noise, wires=w)
    return qml.density_matrix(wires=[0])


def run_entropy_propagation(num_samples=100, noise_rates=(0.0, 0.02, 0.05, 0.10, 0.15, 0.20), seed=42):
    rng = np.random.default_rng(seed)
    results = {"level1": [], "root": []}

    for p in noise_rates:
        print(f"Entropy propagation @ p={p:.3f} ...", flush=True)
        ent_l1, ent_root = [], []
        for _ in range(num_samples):
            inputs4 = rng.uniform(0, 2 * np.pi, size=4)
            weights4 = rng.uniform(0, 2 * np.pi, size=(4, 3))
            rho_l1 = circuit_level1(inputs4, weights4, p)
            ent_l1.append(von_neumann_entropy(rho_l1))

            inputs8 = rng.uniform(0, 2 * np.pi, size=8)
            weights_l1 = rng.uniform(0, 2 * np.pi, size=(2, 4, 3))
            weights_l2 = rng.uniform(0, 2 * np.pi, size=(4, 3))
            rho_root = circuit_root(inputs8, weights_l1, weights_l2, p)
            ent_root.append(von_neumann_entropy(rho_root))

        results["level1"].append({"p": p, "mean": float(np.mean(ent_l1)), "std": float(np.std(ent_l1))})
        results["root"].append({"p": p, "mean": float(np.mean(ent_root)), "std": float(np.std(ent_root))})

    print("\n" + "=" * 70)
    print("ENTROPY PROPAGATION: Level-1 Survivor vs. Root, vs. Noise Rate")
    print("=" * 70)
    print(f"{'p':>6} | {'Level-1 entropy':>20} | {'Root entropy':>20} | {'Root - L1 (propagated)':>22}")
    for r1, r2 in zip(results["level1"], results["root"]):
        delta = r2["mean"] - r1["mean"]
        print(
            f"{r1['p']:6.3f} | {r1['mean']:8.4f} +/- {r1['std']:.4f} | {r2['mean']:8.4f} +/- {r2['std']:.4f} | {delta:+.4f}"
        )

    # Delta from clean baseline at each depth -- how much does noise ADD to entropy
    # at each level, relative to noiseless, and does that increment grow or shrink
    # by the time it reaches the root (i.e. does the second partial trace "reset"
    # some of what noise added at level 1)?
    l1_clean, root_clean = results["level1"][0]["mean"], results["root"][0]["mean"]
    print(
        f"\n{'p':>6} | {'L1 increase over clean':>22} | {'Root increase over clean':>24} | {'Ratio (root/L1 increase)':>24}"
    )
    for r1, r2 in zip(results["level1"][1:], results["root"][1:]):
        d_l1 = r1["mean"] - l1_clean
        d_root = r2["mean"] - root_clean
        ratio = d_root / d_l1 if d_l1 > 1e-9 else float("nan")
        print(f"{r1['p']:6.3f} | {d_l1:22.4f} | {d_root:24.4f} | {ratio:24.2f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    ps = [r["p"] for r in results["level1"]]
    l1_means = [r["mean"] for r in results["level1"]]
    l1_stds = [r["std"] for r in results["level1"]]
    root_means = [r["mean"] for r in results["root"]]
    root_stds = [r["std"] for r in results["root"]]
    plt.errorbar(ps, l1_means, yerr=l1_stds, fmt="-o", label="Level-1 survivor qubit", capsize=4)
    plt.errorbar(ps, root_means, yerr=root_stds, fmt="-s", label="Root qubit (after 2 levels)", capsize=4)
    plt.axhline(y=np.log(2), color="gray", linestyle="--", alpha=0.5, label="Max entropy: ln(2)")
    plt.xlabel("Depolarizing Noise Rate ($p$)")
    plt.ylabel("Von Neumann Entropy (nats)")
    plt.title("Entropy Propagation Through Partial Trace vs. Noise Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "entropy_propagation_vs_noise.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved plot to: {plot_path}")

    return results


# =====================================================================
# Part B: Train under noise, evaluate clean -- is noise a real regularizer?
# =====================================================================
def train_variant(train_p_noise, seed, epochs=15, batch_size=32, lr=0.03):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=256, test_samples=64, img_size=16, seed=seed
    )

    model = HierarchicalQTTNClassifier(mode="baseline")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses, train_accs, clean_val_accs = [], [], []
    tag = f"train_p={train_p_noise}"

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_imgs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_imgs, p_noise=train_p_noise)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(batch_labels).sum().item()
            total += batch_labels.size(0)

        mean_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Always evaluate CLEAN (p=0), regardless of training noise rate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs, p_noise=0.0)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(batch_labels).sum().item()
                val_total += batch_labels.size(0)
        clean_val_acc = 100.0 * val_correct / val_total

        losses.append(mean_loss)
        train_accs.append(train_acc)
        clean_val_accs.append(clean_val_acc)

        print(
            f"[{tag} | seed={seed}] Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Loss: {mean_loss:.4f} | Train Acc: {train_acc:.1f}% | Clean Val Acc: {clean_val_acc:.1f}%"
        )

    return {"losses": losses, "train_accs": train_accs, "clean_val_accs": clean_val_accs}


def run_noise_scheduling_experiment(seeds=(0, 1, 2, 3, 4), epochs=15, train_noise_rates=(0.0, 0.02, 0.05)):
    results = {p: [] for p in train_noise_rates}
    for p in train_noise_rates:
        print(f"\n{'='*60}\nTraining with train_p_noise={p} ({len(seeds)} seeds)\n{'='*60}")
        for seed in seeds:
            results[p].append(train_variant(p, seed, epochs=epochs))

    summary = {}
    for p in train_noise_rates:
        final_accs = [r["clean_val_accs"][-1] for r in results[p]]
        peak_accs = [max(r["clean_val_accs"]) for r in results[p]]
        val_curve = np.array([r["clean_val_accs"] for r in results[p]])
        summary[p] = {
            "final_acc_mean": float(np.mean(final_accs)),
            "final_acc_std": float(np.std(final_accs)),
            "peak_acc_mean": float(np.mean(peak_accs)),
            "val_curve_mean": val_curve.mean(axis=0).tolist(),
            "val_curve_std": val_curve.std(axis=0).tolist(),
        }

    print("\n" + "=" * 70)
    print(f"NOISE-SCHEDULING SUMMARY ({len(seeds)} seeds, always evaluated clean)")
    print("=" * 70)
    for p in train_noise_rates:
        s = summary[p]
        print(
            f"train_p={p:.3f} | Final clean acc: {s['final_acc_mean']:.1f}% +/- {s['final_acc_std']:.1f} | Peak: {s['peak_acc_mean']:.1f}%"
        )

    plt.figure(figsize=(8, 5))
    epochs_range = range(1, epochs + 1)
    colors = {0.0: "tab:blue", 0.02: "tab:green", 0.05: "tab:orange"}
    for p in train_noise_rates:
        mean = np.array(summary[p]["val_curve_mean"])
        std = np.array(summary[p]["val_curve_std"])
        c = colors.get(p, None)
        plt.plot(epochs_range, mean, "-o", label=f"train_p={p}", color=c)
        plt.fill_between(epochs_range, mean - std, mean + std, alpha=0.15, color=c)
    plt.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    plt.xlabel("Epoch")
    plt.ylabel("Clean Test Accuracy (%)")
    plt.title(f"Noise-Scheduled Training, Evaluated Clean ({len(seeds)} seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "noise_scheduling_training.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved plot to: {plot_path}")

    return summary


if __name__ == "__main__":
    entropy_results = run_entropy_propagation(num_samples=100)

    scheduling_summary = run_noise_scheduling_experiment(seeds=(0, 1, 2, 3, 4), epochs=15)

    all_results = {"entropy_propagation": entropy_results, "noise_scheduling": scheduling_summary}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "noise_regularization_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to: {os.path.join(RESULTS_DIR, 'noise_regularization_results.json')}")
