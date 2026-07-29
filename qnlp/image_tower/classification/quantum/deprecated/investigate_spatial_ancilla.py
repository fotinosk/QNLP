"""
Question C: Spatial Positional Encoding -- Explicit vs. Implicit.

llm/quantum_investigation_roadmap.md Section 2, Question C: quantum_image_embedding_model_hea.py
allocates a 5th "ancilla" qubit per patch to explicitly encode learned 2D
spatial position (RX/RY on a dedicated wire), on the theory that a QTTN's
fixed hierarchical wiring only *implicitly* encodes geometry (patch i always
interacts with the same siblings). This script tests whether the explicit
ancilla actually buys anything over that implicit topology-based encoding.

Adaptation for tractability: quantum_image_embedding_model_hea.py's design
(5 qubits/patch x 16 patches = 80 qubits, one ancilla per *individual* patch)
is a circuit-drawing demo, not set up for batched training. We reuse the
efficient hierarchical QTTN pattern from investigate_ancilla_residuals.py
(16 patches -> 4 nodes -> 1 root, weights SHARED across the 4 level-1 nodes,
classical expectation-value readout between levels) and add the ancilla at
the *block* granularity instead of the individual-patch granularity: each
level-1 node gets a 5th wire encoding a learned embedding of *which of the 4
spatial quadrants* (top-left/top-right/bottom-left/bottom-right) this node's
4 patches belong to. Level 2 (root) has only one node, so "position" is not
meaningful there -- ancilla is only added at level 1.

Because level-1 weights are shared across the 4 blocks, the fixed tree
topology already gives the network *some* implicit position signal for free
(level 2 sees the 4 level-1 outputs in a fixed slot order, so "which slot"
already encodes quadrant identity) -- the real question this ablation tests
is whether adding an *explicit* per-quadrant embedding on top of that
implicit ordering signal helps further.

Only Question C sub-question 1 (ablation: with vs. without ancilla) and
sub-question 3 (qubit cost) are testable here. Sub-question 2 (relational
task sensitivity) needs CLEVR's multi-object relative-position labels and
is deferred to Phase 2.

Stage 1 (Simulation): train with/without ancilla, 5 seeds, noiseless.
Stage 2 (Emulation): depolarizing noise sweep, averaged across all trained
seed models (per the multi-seed-averaging lesson from the mixed_channel
retest -- see research_log.md 2026-07-26).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

RESULTS_DIR = "qnlp/image_tower/classification/quantum/results"
NUM_QUBITS = 4
NUM_LAYERS = 2
VARIANTS = ["no_ancilla", "with_ancilla"]
COLORS = {"no_ancilla": "tab:blue", "with_ancilla": "tab:orange"}


# =====================================================================
# Level-1 Quantum Node: 4 patches -> 1 node output, optional spatial ancilla
# =====================================================================
class Level1Node(nn.Module):
    def __init__(self, use_ancilla=False):
        super().__init__()
        self.use_ancilla = use_ancilla

        if use_ancilla:
            # Learned embedding of "which spatial quadrant" (4 possible positions),
            # matching quantum_image_embedding_model_hea.py's spatial_ancilla_encoding.
            self.weights = nn.Parameter(torch.randn(NUM_LAYERS, NUM_QUBITS + 1, 3) * 0.1)
            self.pos_weights = nn.Parameter(torch.randn(4, 2) * 0.1)
            n_wires_clean, n_wires_noisy = NUM_QUBITS + 1, NUM_QUBITS + 1
        else:
            self.weights = nn.Parameter(torch.randn(NUM_LAYERS, NUM_QUBITS, 3) * 0.1)
            n_wires_clean, n_wires_noisy = NUM_QUBITS, NUM_QUBITS

        dev_clean = qml.device("default.qubit", wires=n_wires_clean)
        dev_noisy = qml.device("default.mixed", wires=n_wires_noisy)

        if use_ancilla:

            @qml.qnode(dev_clean, interface="torch")
            def circuit_clean(inputs, weights, pos_angles):
                for i in range(NUM_QUBITS):
                    qml.RY(inputs[:, i], wires=i)
                qml.RX(pos_angles[:, 0], wires=NUM_QUBITS)
                qml.RY(pos_angles[:, 1], wires=NUM_QUBITS)
                qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS + 1))
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev_noisy, interface="torch")
            def circuit_noisy(inputs, weights, pos_angles, p_noise):
                for i in range(NUM_QUBITS):
                    qml.RY(inputs[:, i], wires=i)
                    qml.DepolarizingChannel(p_noise, wires=i)
                qml.RX(pos_angles[:, 0], wires=NUM_QUBITS)
                qml.RY(pos_angles[:, 1], wires=NUM_QUBITS)
                qml.DepolarizingChannel(p_noise, wires=NUM_QUBITS)
                qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS + 1))
                for w in range(NUM_QUBITS + 1):
                    qml.DepolarizingChannel(p_noise, wires=w)
                return qml.expval(qml.PauliZ(0))

        else:

            @qml.qnode(dev_clean, interface="torch")
            def circuit_clean(inputs, weights):
                for i in range(NUM_QUBITS):
                    qml.RY(inputs[:, i], wires=i)
                qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev_noisy, interface="torch")
            def circuit_noisy(inputs, weights, p_noise):
                for i in range(NUM_QUBITS):
                    qml.RY(inputs[:, i], wires=i)
                    qml.DepolarizingChannel(p_noise, wires=i)
                qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
                for w in range(NUM_QUBITS):
                    qml.DepolarizingChannel(p_noise, wires=w)
                return qml.expval(qml.PauliZ(0))

        self._circuit_clean = circuit_clean
        self._circuit_noisy = circuit_noisy

    def forward(self, x, p_noise=0.0):
        # x: [Batch, 4 nodes, 4 features]
        batch, nodes, k = x.shape
        x_flat = x.reshape(batch * nodes, k)

        if self.use_ancilla:
            # quadrant index 0..3 repeats per batch item, matching node order
            quadrant_idx = torch.arange(nodes).repeat(batch)
            pos_angles = self.pos_weights[quadrant_idx]  # [Batch*nodes, 2]
            q_out = (
                self._circuit_noisy(x_flat, self.weights, pos_angles, p_noise)
                if p_noise > 0
                else self._circuit_clean(x_flat, self.weights, pos_angles)
            )
        else:
            q_out = (
                self._circuit_noisy(x_flat, self.weights, p_noise)
                if p_noise > 0
                else self._circuit_clean(x_flat, self.weights)
            )

        return q_out.view(batch, nodes, 1).float()


# =====================================================================
# Level-2 Quantum Node: 4 level-1 outputs -> 1 root (no ancilla -- N/A at root)
# =====================================================================
class Level2Node(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(NUM_LAYERS, NUM_QUBITS, 3) * 0.1)
        dev_clean = qml.device("default.qubit", wires=NUM_QUBITS)
        dev_noisy = qml.device("default.mixed", wires=NUM_QUBITS)

        @qml.qnode(dev_clean, interface="torch")
        def circuit_clean(inputs, weights):
            for i in range(NUM_QUBITS):
                qml.RY(inputs[:, i], wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_noisy, interface="torch")
        def circuit_noisy(inputs, weights, p_noise):
            for i in range(NUM_QUBITS):
                qml.RY(inputs[:, i], wires=i)
                qml.DepolarizingChannel(p_noise, wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
            for w in range(NUM_QUBITS):
                qml.DepolarizingChannel(p_noise, wires=w)
            return qml.expval(qml.PauliZ(0))

        self._circuit_clean = circuit_clean
        self._circuit_noisy = circuit_noisy

    def forward(self, x, p_noise=0.0):
        batch, nodes, k = x.shape
        x_flat = x.reshape(batch * nodes, k)
        q_out = (
            self._circuit_noisy(x_flat, self.weights, p_noise)
            if p_noise > 0
            else self._circuit_clean(x_flat, self.weights)
        )
        return q_out.view(batch, nodes, 1).float()


# =====================================================================
# Hierarchical QTTN Classifier (16 patches -> 4 nodes -> 1 root)
# =====================================================================
class HierarchicalQTTNClassifier(nn.Module):
    def __init__(self, use_ancilla=False, img_size=16, patch_size=4):
        super().__init__()
        self.grid_dim = img_size // patch_size
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, 1)
        self.level1 = Level1Node(use_ancilla=use_ancilla)
        self.level2 = Level2Node()
        self.head = nn.Linear(1, 4)

    @staticmethod
    def _group_2x2(grid):
        b, h, w = grid.shape
        grid = grid.view(b, h // 2, 2, w // 2, 2)
        grid = grid.permute(0, 1, 3, 2, 4).contiguous()
        return grid.view(b, (h // 2) * (w // 2), 4)

    def forward(self, x, p_noise=0.0):
        b = x.shape[0]
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, 16, 48)

        angles = torch.tanh(self.patch_embed(x)) * np.pi
        angles = angles.squeeze(-1).view(b, self.grid_dim, self.grid_dim)

        x1 = self._group_2x2(angles)  # [B, 4, 4]
        n1 = self.level1(x1, p_noise=p_noise)  # [B, 4, 1]

        x2 = n1.squeeze(-1).view(b, 1, 4)
        n2 = self.level2(x2, p_noise=p_noise)

        return self.head(n2.squeeze(1))


# =====================================================================
# Training
# =====================================================================
def train_variant(use_ancilla, seed, epochs=15, batch_size=32, lr=0.03):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=256, test_samples=64, img_size=16, seed=seed
    )

    model = HierarchicalQTTNClassifier(use_ancilla=use_ancilla)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses, train_accs, val_accs = [], [], []
    tag = "with_ancilla" if use_ancilla else "no_ancilla"

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
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

        mean_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(batch_labels).sum().item()
                val_total += batch_labels.size(0)
        val_acc = 100.0 * val_correct / val_total

        losses.append(mean_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"[{tag} | seed={seed}] Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Loss: {mean_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
        )

    return {
        "model": model,
        "test_loader": test_loader,
        "losses": losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }


def convergence_epoch(val_accs, threshold=50.0):
    for i, acc in enumerate(val_accs):
        if acc >= threshold:
            return i + 1
    return None


def noise_sweep(model, test_loader, noise_rates):
    model.eval()
    accs = []
    with torch.no_grad():
        for p in noise_rates:
            correct, total = 0, 0
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs, p_noise=p)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(batch_labels).sum().item()
                total += batch_labels.size(0)
            acc = 100.0 * correct / total
            accs.append(acc)
    return accs


# =====================================================================
# Main experiment
# =====================================================================
def run_experiment(seeds=(0, 1, 2, 3, 4), epochs=15):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {v: [] for v in VARIANTS}

    for use_ancilla, tag in [(False, "no_ancilla"), (True, "with_ancilla")]:
        print(f"\n{'='*60}\nTraining variant: {tag} ({len(seeds)} seeds)\n{'='*60}")
        for seed in seeds:
            results[tag].append(train_variant(use_ancilla, seed, epochs=epochs))

    summary = {}
    for tag in VARIANTS:
        losses = np.array([r["losses"] for r in results[tag]])
        val_accs = np.array([r["val_accs"] for r in results[tag]])
        train_accs = np.array([r["train_accs"] for r in results[tag]])
        conv_epochs = [convergence_epoch(r["val_accs"]) for r in results[tag]]

        summary[tag] = {
            "loss_mean": losses.mean(axis=0).tolist(),
            "loss_std": losses.std(axis=0).tolist(),
            "val_acc_mean": val_accs.mean(axis=0).tolist(),
            "val_acc_std": val_accs.std(axis=0).tolist(),
            "train_acc_mean": train_accs.mean(axis=0).tolist(),
            "final_val_acc_per_seed": val_accs[:, -1].tolist(),
            "final_val_acc_mean": float(val_accs[:, -1].mean()),
            "final_val_acc_std": float(val_accs[:, -1].std()),
            "peak_val_acc_mean": float(val_accs.max(axis=1).mean()),
            "convergence_epoch_to_50pct": conv_epochs,
            "num_params": sum(p.numel() for p in results[tag][0]["model"].parameters()),
        }

    print("\n" + "=" * 60)
    print(f"STAGE 1 SUMMARY (Noiseless, mean over {len(seeds)} seeds)")
    print("=" * 60)
    for tag in VARIANTS:
        s = summary[tag]
        print(
            f"{tag:14s} | Final Val Acc: {s['final_val_acc_mean']:.1f}% (+/-{s['final_val_acc_std']:.1f}) | "
            f"Peak Val Acc: {s['peak_val_acc_mean']:.1f}% | Params: {s['num_params']} | "
            f"Convergence epoch (>=50% val): {s['convergence_epoch_to_50pct']}"
        )

    # ---- Plot Stage 1 comparison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, epochs + 1)
    for tag in VARIANTS:
        loss_mean = np.array(summary[tag]["loss_mean"])
        loss_std = np.array(summary[tag]["loss_std"])
        ax1.plot(epochs_range, loss_mean, "-o", color=COLORS[tag], label=tag)
        ax1.fill_between(epochs_range, loss_mean - loss_std, loss_mean + loss_std, color=COLORS[tag], alpha=0.15)

        val_mean = np.array(summary[tag]["val_acc_mean"])
        val_std = np.array(summary[tag]["val_acc_std"])
        ax2.plot(epochs_range, val_mean, "-o", color=COLORS[tag], label=tag)
        ax2.fill_between(epochs_range, val_mean - val_std, val_mean + val_std, color=COLORS[tag], alpha=0.15)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Loss: With vs. Without Spatial Ancilla")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Val Accuracy: With vs. Without Spatial Ancilla")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "spatial_ancilla_comparison_training.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved training comparison plot to: {plot_path}")

    # ---- Stage 2: Noise sweep, averaged across ALL trained seed models ----
    print("\n" + "=" * 60)
    print("STAGE 2: Depolarizing Noise Sweep (averaged across all seeds)")
    print("=" * 60)
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    noise_curves = {tag: [] for tag in VARIANTS}
    for tag in VARIANTS:
        print(f"\n[{tag}]")
        for i, r in enumerate(results[tag]):
            accs = noise_sweep(r["model"], r["test_loader"], noise_rates)
            noise_curves[tag].append(accs)
            print(f"  seed={seeds[i]}: {[round(a,1) for a in accs]}")

    noise_summary = {}
    for tag in VARIANTS:
        curves = np.array(noise_curves[tag])
        noise_summary[tag] = {"mean": curves.mean(axis=0).tolist(), "std": curves.std(axis=0).tolist()}

    print("\nNoise sweep summary (mean +/- std across seeds):")
    for tag in VARIANTS:
        ns = noise_summary[tag]
        for p, m, s in zip(noise_rates, ns["mean"], ns["std"]):
            print(f"  [{tag}] p={p:.3f} | {m:.1f}% +/- {s:.1f}")

    plt.figure(figsize=(8, 5))
    for tag in VARIANTS:
        mean = np.array(noise_summary[tag]["mean"])
        std = np.array(noise_summary[tag]["std"])
        plt.plot(noise_rates, mean, "-o", color=COLORS[tag], label=tag, linewidth=2)
        plt.fill_between(noise_rates, mean - std, mean + std, color=COLORS[tag], alpha=0.15)
    plt.axhline(y=25.0, color="gray", linestyle="--", label="Random (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"Noise Tolerance: With vs. Without Spatial Ancilla (mean +/- std, {len(seeds)} seeds)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    noise_plot_path = os.path.join(RESULTS_DIR, "spatial_ancilla_comparison_noise.png")
    plt.savefig(noise_plot_path)
    plt.close()
    print(f"\nSaved noise comparison plot to: {noise_plot_path}")

    summary["noise_rates"] = noise_rates
    summary["noise_summary"] = noise_summary
    summary["seeds"] = list(seeds)
    results_path = os.path.join(RESULTS_DIR, "spatial_ancilla_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved raw results to: {results_path}")

    return summary


if __name__ == "__main__":
    run_experiment(seeds=(0, 1, 2, 3, 4), epochs=15)
