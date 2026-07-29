"""
Quantum-Native Residual Connection Investigation for the QTTN Image Tower.

Compares three variants of a hierarchical QTTN classifier's per-node ansatz:
  - baseline:      standard StronglyEntanglingLayers, standard init (no residual)
  - reupload:      data re-uploading residual (Method 1) -- the node input is
                    re-encoded partway through the ansatz, so the tree level's
                    raw signal is re-injected rather than only reachable through
                    compounded unitaries. Same total layer count as baseline.
  - near_identity:  near-identity ansatz initialization (Method 2) -- identical
                    circuit to baseline, but rotation weights are initialized
                    ~10x smaller so each node starts close to a pass-through
                    unitary and learns a perturbation from there (the VQC
                    analogue of classical residual-network warm-start init).

Both methods are genuinely in-circuit: unlike the classical-bypass residual
prototyped and discarded earlier, neither adds a classical value computed
before the quantum circuit runs. See llm/research_log.md and
llm/quantum_investigation_roadmap.md (Section 5, "Quantum-Native Residual
Connections") for context.

Stage 1 (Simulation): train all three variants noiseless (default.qubit),
3 seeds each, compare loss/accuracy convergence.
Stage 2 (Emulation): sweep depolarizing noise (default.mixed) on the trained
weights to see whether either mechanism changes noise robustness.
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
VARIANTS = ["baseline", "reupload", "near_identity"]
COLORS = {"baseline": "tab:blue", "reupload": "tab:green", "near_identity": "tab:orange"}


# =====================================================================
# Quantum Quad Node (single tree level: 4 children -> 1 parent scalar)
# =====================================================================
class QuantumQuadNode(nn.Module):
    """One QTTN tree level, 4 classical child features -> 1 expectation value.

    mode="baseline":      RY encode once, StronglyEntanglingLayers(2 layers), measure.
    mode="reupload":      RY encode, 1 ansatz layer, RY re-encode (same inputs),
                           1 more ansatz layer, measure. Same total layer count
                           as baseline -- isolates the re-injection effect.
    mode="near_identity": identical circuit to baseline, weights initialized
                           ~10x smaller (near-identity unitary at t=0).
    """

    def __init__(self, mode="baseline"):
        super().__init__()
        assert mode in VARIANTS
        self.mode = mode
        self.reupload = mode == "reupload"

        init_scale = 0.01 if mode == "near_identity" else 0.1
        self.weights = nn.Parameter(torch.randn(NUM_LAYERS, NUM_QUBITS, 3) * init_scale)

        dev_clean = qml.device("default.qubit", wires=NUM_QUBITS)
        dev_noisy = qml.device("default.mixed", wires=NUM_QUBITS)
        reupload = self.reupload

        def _body(inputs, weights, p_noise):
            noisy = p_noise > 0
            for i in range(NUM_QUBITS):
                qml.RY(inputs[:, i], wires=i)
                if noisy:
                    qml.DepolarizingChannel(p_noise, wires=i)

            if reupload:
                qml.StronglyEntanglingLayers(weights[0:1], wires=range(NUM_QUBITS))
                if noisy:
                    for w in range(NUM_QUBITS):
                        qml.DepolarizingChannel(p_noise, wires=w)
                # Data re-uploading: re-inject the original node input
                for i in range(NUM_QUBITS):
                    qml.RY(inputs[:, i], wires=i)
                    if noisy:
                        qml.DepolarizingChannel(p_noise, wires=i)
                qml.StronglyEntanglingLayers(weights[1:2], wires=range(NUM_QUBITS))
                if noisy:
                    for w in range(NUM_QUBITS):
                        qml.DepolarizingChannel(p_noise, wires=w)
            else:
                qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
                if noisy:
                    for w in range(NUM_QUBITS):
                        qml.DepolarizingChannel(p_noise, wires=w)

            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev_clean, interface="torch")
        def circuit_clean(inputs, weights):
            return _body(inputs, weights, 0.0)

        @qml.qnode(dev_noisy, interface="torch")
        def circuit_noisy(inputs, weights, p_noise):
            return _body(inputs, weights, p_noise)

        self._circuit_clean = circuit_clean
        self._circuit_noisy = circuit_noisy

    def forward(self, x, p_noise=0.0):
        # x: [Batch, Nodes, 4]
        batch, nodes, k = x.shape
        x_flat = x.reshape(batch * nodes, k)

        if p_noise > 0:
            q_out = self._circuit_noisy(x_flat, self.weights, p_noise)
        else:
            q_out = self._circuit_clean(x_flat, self.weights)

        return q_out.view(batch, nodes, 1).float()


# =====================================================================
# Hierarchical QTTN Classifier (16 patches -> 4 nodes -> 1 root)
# =====================================================================
class HierarchicalQTTNClassifier(nn.Module):
    def __init__(self, mode="baseline", img_size=16, patch_size=4):
        super().__init__()
        self.grid_dim = img_size // patch_size  # 4 (4x4 patch grid)
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, 1)
        self.level1 = QuantumQuadNode(mode=mode)
        self.level2 = QuantumQuadNode(mode=mode)
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

        x2 = n1.squeeze(-1).view(b, 1, 4)  # [B, 1, 4]
        n2 = self.level2(x2, p_noise=p_noise)  # [B, 1, 1]

        root = n2.squeeze(1)  # [B, 1]
        return self.head(root)


# =====================================================================
# Training
# =====================================================================
def train_variant(mode, seed, epochs=15, batch_size=32, lr=0.03):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=256, test_samples=64, img_size=16, seed=seed
    )

    model = HierarchicalQTTNClassifier(mode=mode)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses, train_accs, val_accs = [], [], []

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
            f"[{mode} | seed={seed}] Epoch {epoch+1:02d}/{epochs:02d} | "
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


# =====================================================================
# Stage 2: Noise sweep on trained weights
# =====================================================================
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
            print(f"  p={p:.3f} | Test Acc: {acc:.1f}%")
    return accs


# =====================================================================
# Main experiment
# =====================================================================
def run_experiment(seeds=(0, 1, 2), epochs=15):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {v: [] for v in VARIANTS}

    for mode in VARIANTS:
        print(f"\n{'='*60}\nTraining variant: {mode}\n{'='*60}")
        for seed in seeds:
            results[mode].append(train_variant(mode, seed, epochs=epochs))

    # ---- Aggregate Stage 1 curves ----
    summary = {}
    for mode in VARIANTS:
        losses = np.array([r["losses"] for r in results[mode]])
        val_accs = np.array([r["val_accs"] for r in results[mode]])
        train_accs = np.array([r["train_accs"] for r in results[mode]])
        conv_epochs = [convergence_epoch(r["val_accs"]) for r in results[mode]]

        summary[mode] = {
            "loss_mean": losses.mean(axis=0).tolist(),
            "loss_std": losses.std(axis=0).tolist(),
            "val_acc_mean": val_accs.mean(axis=0).tolist(),
            "val_acc_std": val_accs.std(axis=0).tolist(),
            "train_acc_mean": train_accs.mean(axis=0).tolist(),
            "final_val_acc_mean": float(val_accs[:, -1].mean()),
            "final_val_acc_std": float(val_accs[:, -1].std()),
            "peak_val_acc_mean": float(val_accs.max(axis=1).mean()),
            "convergence_epoch_to_50pct": conv_epochs,
        }

    print("\n" + "=" * 60)
    print("STAGE 1 SUMMARY (Noiseless, mean over seeds)")
    print("=" * 60)
    for mode in VARIANTS:
        s = summary[mode]
        print(
            f"{mode:14s} | Final Val Acc: {s['final_val_acc_mean']:.1f}% (+/-{s['final_val_acc_std']:.1f}) | "
            f"Peak Val Acc: {s['peak_val_acc_mean']:.1f}% | "
            f"Convergence epoch (>=50% val): {s['convergence_epoch_to_50pct']}"
        )

    # ---- Plot Stage 1 comparison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs_range = range(1, epochs + 1)

    for mode in VARIANTS:
        loss_mean = np.array(summary[mode]["loss_mean"])
        loss_std = np.array(summary[mode]["loss_std"])
        ax1.plot(epochs_range, loss_mean, "-o", color=COLORS[mode], label=mode)
        ax1.fill_between(epochs_range, loss_mean - loss_std, loss_mean + loss_std, color=COLORS[mode], alpha=0.15)

        val_mean = np.array(summary[mode]["val_acc_mean"])
        val_std = np.array(summary[mode]["val_acc_std"])
        ax2.plot(epochs_range, val_mean, "-o", color=COLORS[mode], label=mode)
        ax2.fill_between(epochs_range, val_mean - val_std, val_mean + val_std, color=COLORS[mode], alpha=0.15)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Loss: Baseline vs. Re-upload vs. Near-Identity Init")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Val Accuracy: Baseline vs. Re-upload vs. Near-Identity Init")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "quantum_residual_comparison_training.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved training comparison plot to: {plot_path}")

    # ---- Stage 2: Noise sweep (seed=0 model of each variant) ----
    print("\n" + "=" * 60)
    print("STAGE 2: Depolarizing Noise Sweep (Emulation)")
    print("=" * 60)
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    noise_results = {}
    for mode in VARIANTS:
        print(f"\n[{mode}] (seed=0 model)")
        model = results[mode][0]["model"]
        test_loader = results[mode][0]["test_loader"]
        noise_results[mode] = noise_sweep(model, test_loader, noise_rates)

    plt.figure(figsize=(8, 5))
    for mode in VARIANTS:
        plt.plot(noise_rates, noise_results[mode], "-o", color=COLORS[mode], label=mode, linewidth=2)
    plt.axhline(y=25.0, color="gray", linestyle="--", label="Random (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Noise Tolerance: Baseline vs. Re-upload vs. Near-Identity Init (seed=0 models)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    noise_plot_path = os.path.join(RESULTS_DIR, "quantum_residual_comparison_noise.png")
    plt.savefig(noise_plot_path)
    plt.close()
    print(f"\nSaved noise comparison plot to: {noise_plot_path}")

    # ---- Save raw results ----
    summary["noise_rates"] = noise_rates
    summary["noise_results"] = noise_results
    results_path = os.path.join(RESULTS_DIR, "quantum_residual_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved raw results to: {results_path}")

    return summary


if __name__ == "__main__":
    run_experiment(seeds=(0, 1, 2), epochs=15)
