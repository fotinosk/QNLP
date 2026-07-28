"""
Task R1 (roadmap Section 7): remove the scalar-readout bottleneck and find a
configuration that demonstrably binds both attributes on 16x16 overlapping
synthetic shapes, instead of sitting at the 50% single-attribute shortcut
ceiling. See qttn_core.py for the shared model and research_log.md
2026-07-27 "Code Audit" for why this is necessary before any further
ablation (residuals, spatial ancilla, topology) can be trusted.

Method: change one thing at a time.
  Sweep 1 -- fix encoding=scalar_ry, vary readout in {scalar, root_multi_pauli,
             level1_survivors}. 5 seeds x 15 epochs, 256 train / 64 test.
  Sweep 2 -- fix the winning readout, vary encoding in {scalar_ry, multi_axis}.
             5 seeds.

Acceptance criterion: >= 70% mean final val accuracy over 5 seeds for at
least one configuration (demonstrated already: the 2026-07-17 4-dim-readout
run hit 75% at this exact size/dataset). If nothing clears 70%, this script
also runs the capacity fallback (1024 train samples / 30 epochs) on the best
config found, per the roadmap's escalation order.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.image_tower.classification.quantum.qttn_core import HierarchicalQTTNClassifier
from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

RESULTS_DIR = "qnlp/image_tower/classification/quantum/results"
GATE_THRESHOLD = 70.0
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 15


def train_config(readout, encoding, seed, epochs=EPOCHS, train_samples=256, test_samples=64, batch_size=32, lr=0.03):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=train_samples, test_samples=test_samples, img_size=16, seed=seed
    )

    model = HierarchicalQTTNClassifier(readout=readout, encoding=encoding)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    val_accs = []
    for epoch in range(epochs):
        model.train()
        for batch_imgs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(batch_labels).sum().item()
                total += batch_labels.size(0)
        val_acc = 100.0 * correct / total
        val_accs.append(val_acc)
        print(
            f"  [readout={readout} encoding={encoding} seed={seed}] epoch {epoch+1:02d}/{epochs} val_acc={val_acc:.1f}%"
        )

    return val_accs


def run_sweep(configs, tag, seeds=SEEDS, **train_kwargs):
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    summary = {}
    for readout, encoding in configs:
        key = f"readout={readout},encoding={encoding}"
        print(f"\n--- {key} ---")
        all_val_accs = []
        for seed in seeds:
            all_val_accs.append(train_config(readout, encoding, seed, **train_kwargs))
        val_accs = np.array(all_val_accs)  # [n_seeds, epochs]
        final = val_accs[:, -1]
        summary[key] = {
            "readout": readout,
            "encoding": encoding,
            "val_acc_mean_curve": val_accs.mean(axis=0).tolist(),
            "final_val_acc_per_seed": final.tolist(),
            "final_val_acc_mean": float(final.mean()),
            "final_val_acc_std": float(final.std()),
            "peak_val_acc_mean": float(val_accs.max(axis=1).mean()),
            "num_params": sum(
                p.numel() for p in HierarchicalQTTNClassifier(readout=readout, encoding=encoding).parameters()
            ),
        }
        print(
            f"{key}: final val acc {summary[key]['final_val_acc_mean']:.1f}%"
            f" +/- {summary[key]['final_val_acc_std']:.1f}"
        )
    return summary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    # Sweep 1: readout, encoding fixed at scalar_ry
    sweep1_configs = [(r, "scalar_ry") for r in ("scalar", "root_multi_pauli", "level1_survivors")]
    results["sweep1_readout"] = run_sweep(sweep1_configs, "SWEEP 1: readout ablation (encoding=scalar_ry)")

    best_readout_key = max(results["sweep1_readout"], key=lambda k: results["sweep1_readout"][k]["final_val_acc_mean"])
    best_readout = results["sweep1_readout"][best_readout_key]["readout"]
    print(
        f"\nBest readout from Sweep 1: {best_readout}"
        f" ({results['sweep1_readout'][best_readout_key]['final_val_acc_mean']:.1f}%)"
    )

    # Sweep 2: encoding, readout fixed at Sweep 1 winner
    sweep2_configs = [(best_readout, e) for e in ("scalar_ry", "multi_axis")]
    results["sweep2_encoding"] = run_sweep(sweep2_configs, f"SWEEP 2: encoding ablation (readout={best_readout})")

    best_key_overall = max(
        {**results["sweep1_readout"], **results["sweep2_encoding"]},
        key=lambda k: {**results["sweep1_readout"], **results["sweep2_encoding"]}[k]["final_val_acc_mean"],
    )
    all_configs = {**results["sweep1_readout"], **results["sweep2_encoding"]}
    best = all_configs[best_key_overall]
    gate_passed = best["final_val_acc_mean"] >= GATE_THRESHOLD

    print(
        f"\n{'='*70}\nBEST CONFIG: {best_key_overall} -> "
        f"{best['final_val_acc_mean']:.1f}% +/- {best['final_val_acc_std']:.1f}"
    )
    print(f"GATE (>=70% over 5 seeds): {'PASSED' if gate_passed else 'FAILED'}\n{'='*70}")

    results["gate"] = {
        "threshold": GATE_THRESHOLD,
        "best_config": best_key_overall,
        "best_final_val_acc_mean": best["final_val_acc_mean"],
        "passed": gate_passed,
    }

    if not gate_passed:
        print("\nGate failed. Running capacity fallback (1024 train samples / 30 epochs) on best config...")
        readout, encoding = best["readout"], best["encoding"]
        fallback_accs = []
        for seed in SEEDS:
            fallback_accs.append(train_config(readout, encoding, seed, epochs=30, train_samples=1024, test_samples=64))
        fallback_accs = np.array(fallback_accs)
        final = fallback_accs[:, -1]
        results["capacity_fallback"] = {
            "readout": readout,
            "encoding": encoding,
            "train_samples": 1024,
            "epochs": 30,
            "final_val_acc_per_seed": final.tolist(),
            "final_val_acc_mean": float(final.mean()),
            "final_val_acc_std": float(final.std()),
            "passed": float(final.mean()) >= GATE_THRESHOLD,
        }
        print(
            f"Capacity fallback: {results['capacity_fallback']['final_val_acc_mean']:.1f}% "
            f"+/- {results['capacity_fallback']['final_val_acc_std']:.1f} "
            f"({'PASSED' if results['capacity_fallback']['passed'] else 'STILL FAILED'})"
        )

    results_path = os.path.join(RESULTS_DIR, "r1_remediation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    return results


if __name__ == "__main__":
    main()
