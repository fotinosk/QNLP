"""
Ancilla/Channel-Based Quantum Residual Connections for the QTTN Image Tower.

Tests Methods 3-5 from llm/quantum_investigation_roadmap.md Section 5
("Quantum-Native Residual Connections") against a plain baseline. Methods 1-2
(data re-uploading, near-identity init) were tested separately and found to
trade expressivity for noise fragility (re-uploading) or do nothing
(near-identity init) -- see research_log.md 2026-07-26. These remaining
methods use an ancilla qubit to realize the residual mechanistically inside
the circuit, rather than duplicating the noisy encode step.

Each tree node gains a 5th "mixing" ancilla wire (system: 0-3, ancilla: 4):
  - baseline:       no ancilla, plain StronglyEntanglingLayers (control).
  - mixed_channel:  (Method 3) RY(mix_angle) on ancilla, controlled-U on
                    system, ancilla traced out (not measured/returned). This
                    realizes the incoherent Kraus mixture
                    rho -> (1-lambda) rho + lambda * U rho U^dagger,
                    lambda = sin^2(mix_angle/2), a genuine quantum channel
                    (not classical arithmetic on measured values).
  - lcu:            (Methods 4+5) RY(mix_angle), controlled-U, RY(-mix_angle)
                    uncompute, then postselect the ancilla on |0>: a standard
                    single-ancilla LCU "PREPARE-SELECT-PREPARE-dagger" block
                    encoding of alpha*I + beta*U, coherent (not a classical
                    mixture). "Full LCU" (Method 5) additionally reports the
                    postselection success probability -- the real resource
                    overhead LCU pays on hardware that mixed_channel and
                    baseline don't.

Known limitation (2026-07-26): PennyLane's default.mixed device does not
support qml.measure(postselect=...) via the deferred-measurement transform
in this installed version (0.43.2) -- it requires a Projector gate the
transform doesn't provide for density-matrix devices. So `lcu` is evaluated
Stage 1 (noiseless) only; `baseline` and `mixed_channel` get both stages.

Stage 1 (Simulation): train all variants noiseless (default.qubit), 3 seeds.
Stage 2 (Emulation): sweep depolarizing noise (default.mixed) for baseline
and mixed_channel only.
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
VARIANTS = ["baseline", "mixed_channel", "lcu"]
COLORS = {"baseline": "tab:blue", "mixed_channel": "tab:purple", "lcu": "tab:red"}
NOISY_VARIANTS = ["baseline", "mixed_channel"]  # lcu excluded: see module docstring


# =====================================================================
# Quantum Quad Node (single tree level: 4 children -> 1 parent scalar)
# =====================================================================
class QuantumQuadNode(nn.Module):
    def __init__(self, mode="baseline"):
        super().__init__()
        assert mode in VARIANTS
        self.mode = mode
        self.weights = nn.Parameter(torch.randn(NUM_LAYERS, NUM_QUBITS, 3) * 0.1)

        if mode in ("mixed_channel", "lcu"):
            # Start near-identity (small mixing angle -> mostly pass-through),
            # consistent with the residual-connection motivation.
            self.mix_angle = nn.Parameter(torch.randn(()) * 0.1)
            n_wires_clean = NUM_QUBITS + 1
            n_wires_noisy = NUM_QUBITS + 2  # deferred-measurement transform needs a spare wire
        else:
            n_wires_clean = NUM_QUBITS
            n_wires_noisy = NUM_QUBITS

        dev_clean = qml.device("default.qubit", wires=n_wires_clean)
        dev_noisy = qml.device("default.mixed", wires=n_wires_noisy)

        def _encode(inputs, p_noise):
            for i in range(NUM_QUBITS):
                qml.RY(inputs[:, i], wires=i)
                if p_noise > 0:
                    qml.DepolarizingChannel(p_noise, wires=i)

        def _ansatz(weights, p_noise):
            qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
            if p_noise > 0:
                for w in range(NUM_QUBITS):
                    qml.DepolarizingChannel(p_noise, wires=w)

        if mode == "baseline":

            @qml.qnode(dev_clean, interface="torch")
            def circuit_clean(inputs, weights):
                _encode(inputs, 0.0)
                _ansatz(weights, 0.0)
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev_noisy, interface="torch")
            def circuit_noisy(inputs, weights, p_noise):
                _encode(inputs, p_noise)
                _ansatz(weights, p_noise)
                return qml.expval(qml.PauliZ(0))

        elif mode == "mixed_channel":

            @qml.qnode(dev_clean, interface="torch")
            def circuit_clean(inputs, weights, mix_angle):
                _encode(inputs, 0.0)
                qml.RY(mix_angle, wires=NUM_QUBITS)
                qml.ctrl(qml.StronglyEntanglingLayers, control=NUM_QUBITS)(weights, wires=range(NUM_QUBITS))
                return qml.expval(qml.PauliZ(0))  # ancilla traced out (not returned)

            @qml.qnode(dev_noisy, interface="torch")
            def circuit_noisy(inputs, weights, mix_angle, p_noise):
                _encode(inputs, p_noise)
                qml.RY(mix_angle, wires=NUM_QUBITS)
                qml.ctrl(qml.StronglyEntanglingLayers, control=NUM_QUBITS)(weights, wires=range(NUM_QUBITS))
                if p_noise > 0:
                    for w in range(NUM_QUBITS):
                        qml.DepolarizingChannel(p_noise, wires=w)
                return qml.expval(qml.PauliZ(0))

        else:  # lcu

            @qml.qnode(dev_clean, interface="torch")
            def circuit_clean(inputs, weights, mix_angle):
                _encode(inputs, 0.0)
                qml.RY(mix_angle, wires=NUM_QUBITS)
                qml.ctrl(qml.StronglyEntanglingLayers, control=NUM_QUBITS)(weights, wires=range(NUM_QUBITS))
                qml.RY(-mix_angle, wires=NUM_QUBITS)
                qml.measure(NUM_QUBITS, postselect=0)
                return qml.expval(qml.PauliZ(0))

            @qml.qnode(dev_clean, interface="torch")
            def success_prob_circuit(inputs, weights, mix_angle):
                # Diagnostic only (Method 5): raw P(ancilla=0) BEFORE postselection,
                # i.e. the fraction of trials LCU would keep on real hardware.
                _encode(inputs, 0.0)
                qml.RY(mix_angle, wires=NUM_QUBITS)
                qml.ctrl(qml.StronglyEntanglingLayers, control=NUM_QUBITS)(weights, wires=range(NUM_QUBITS))
                qml.RY(-mix_angle, wires=NUM_QUBITS)
                return qml.probs(wires=NUM_QUBITS)

            circuit_noisy = None  # not supported, see module docstring
            self._success_prob_circuit = success_prob_circuit

        self._circuit_clean = circuit_clean
        self._circuit_noisy = circuit_noisy

    def forward(self, x, p_noise=0.0):
        batch, nodes, k = x.shape
        x_flat = x.reshape(batch * nodes, k)

        if self.mode == "baseline":
            q_out = (
                self._circuit_noisy(x_flat, self.weights, p_noise)
                if p_noise > 0
                else self._circuit_clean(x_flat, self.weights)
            )
        elif self.mode == "mixed_channel":
            q_out = (
                self._circuit_noisy(x_flat, self.weights, self.mix_angle, p_noise)
                if p_noise > 0
                else self._circuit_clean(x_flat, self.weights, self.mix_angle)
            )
        else:  # lcu
            if p_noise > 0:
                raise NotImplementedError("lcu variant does not support noisy emulation (see module docstring)")
            q_out = self._circuit_clean(x_flat, self.weights, self.mix_angle)

        return q_out.view(batch, nodes, 1).float()

    def success_probability(self, x):
        assert self.mode == "lcu"
        batch, nodes, k = x.shape
        x_flat = x.reshape(batch * nodes, k)
        probs = self._success_prob_circuit(x_flat, self.weights, self.mix_angle)  # [B*nodes, 2]
        return probs[:, 0].mean().item()  # P(ancilla=0), averaged


# =====================================================================
# Hierarchical QTTN Classifier (16 patches -> 4 nodes -> 1 root)
# =====================================================================
class HierarchicalQTTNClassifier(nn.Module):
    def __init__(self, mode="baseline", img_size=16, patch_size=4):
        super().__init__()
        self.mode = mode
        self.grid_dim = img_size // patch_size
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

    def _angles(self, x):
        b = x.shape[0]
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, 16, 48)
        angles = torch.tanh(self.patch_embed(x)) * np.pi
        return angles.squeeze(-1).view(b, self.grid_dim, self.grid_dim)

    def forward(self, x, p_noise=0.0):
        b = x.shape[0]
        angles = self._angles(x)
        x1 = self._group_2x2(angles)
        n1 = self.level1(x1, p_noise=p_noise)
        x2 = n1.squeeze(-1).view(b, 1, 4)
        n2 = self.level2(x2, p_noise=p_noise)
        return self.head(n2.squeeze(1))

    def lcu_success_probabilities(self, x):
        b = x.shape[0]
        angles = self._angles(x)
        x1 = self._group_2x2(angles)
        p1 = self.level1.success_probability(x1)
        n1 = self.level1(x1)
        x2 = n1.squeeze(-1).view(b, 1, 4)
        p2 = self.level2.success_probability(x2)
        return p1, p2


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


def lcu_diagnostics(model, test_loader):
    model.eval()
    p1s, p2s = [], []
    with torch.no_grad():
        for batch_imgs, _ in test_loader:
            p1, p2 = model.lcu_success_probabilities(batch_imgs)
            p1s.append(p1)
            p2s.append(p2)
    return {"level1_success_prob": float(np.mean(p1s)), "level2_success_prob": float(np.mean(p2s))}


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

    # ---- LCU-specific diagnostic (Method 5): postselection success probability ----
    print("\n" + "=" * 60)
    print("LCU DIAGNOSTIC: Postselection Success Probability (seed=0 model)")
    print("=" * 60)
    lcu_diag = lcu_diagnostics(results["lcu"][0]["model"], results["lcu"][0]["test_loader"])
    print(f"  Level 1 P(ancilla=0): {lcu_diag['level1_success_prob']:.3f}")
    print(f"  Level 2 P(ancilla=0): {lcu_diag['level2_success_prob']:.3f}")

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
    ax1.set_title("Loss: Baseline vs. Mixed-Channel vs. LCU")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy (%)")
    ax2.set_title("Val Accuracy: Baseline vs. Mixed-Channel vs. LCU")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "ancilla_residual_comparison_training.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved training comparison plot to: {plot_path}")

    # ---- Stage 2: Noise sweep (baseline + mixed_channel only) ----
    print("\n" + "=" * 60)
    print("STAGE 2: Depolarizing Noise Sweep (Emulation) -- lcu excluded, see docstring")
    print("=" * 60)
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    noise_results = {}
    for mode in NOISY_VARIANTS:
        print(f"\n[{mode}] (seed=0 model)")
        model = results[mode][0]["model"]
        test_loader = results[mode][0]["test_loader"]
        noise_results[mode] = noise_sweep(model, test_loader, noise_rates)

    plt.figure(figsize=(8, 5))
    for mode in NOISY_VARIANTS:
        plt.plot(noise_rates, noise_results[mode], "-o", color=COLORS[mode], label=mode, linewidth=2)
    plt.axhline(y=25.0, color="gray", linestyle="--", label="Random (25%)")
    plt.xlabel("Depolarizing Noise Rate ($p$)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Noise Tolerance: Baseline vs. Mixed-Channel (seed=0 models; lcu N/A)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    noise_plot_path = os.path.join(RESULTS_DIR, "ancilla_residual_comparison_noise.png")
    plt.savefig(noise_plot_path)
    plt.close()
    print(f"\nSaved noise comparison plot to: {noise_plot_path}")

    summary["noise_rates"] = noise_rates
    summary["noise_results"] = noise_results
    summary["lcu_diagnostic"] = lcu_diag
    results_path = os.path.join(RESULTS_DIR, "ancilla_residual_comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved raw results to: {results_path}")

    return summary


if __name__ == "__main__":
    run_experiment(seeds=(0, 1, 2), epochs=15)
