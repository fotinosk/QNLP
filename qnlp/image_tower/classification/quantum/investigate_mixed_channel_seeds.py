"""
Extended-seed comparison: baseline vs. mixed_channel (Method 3).

The 3-seed run in investigate_ancilla_residuals.py showed mixed_channel
matching or beating baseline on 2/3 seeds (both clean and under noise), but
1/3 seeds hit a dead-gradient trap (loss pinned at ln(4) for all 15 epochs).
That's too few seeds to tell a real ~1-in-3 failure rate apart from
coincidence, and the noise sweep only used a single seed=0 model per variant.

This script reruns baseline vs. mixed_channel on 10 seeds to:
  1. Get a tighter estimate of the dead-gradient failure rate.
  2. Report mean/std both including and excluding failed runs, so the
     "does it work when it works" question and the "is it reliable" question
     don't get conflated into one misleading average (as in the 3-seed run).
  3. Average the noise sweep across all trained seeds instead of a single
     seed=0 model, per the caveat noted in research_log.md 2026-07-26.

See llm/research_log.md and llm/quantum_investigation_roadmap.md (Section 5,
item 13 "Stabilize mixed_channel training") for context.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from qnlp.image_tower.classification.quantum.investigate_ancilla_residuals import (
    RESULTS_DIR,
    convergence_epoch,
    noise_sweep,
    train_variant,
)

VARIANTS = ["baseline", "mixed_channel"]
COLORS = {"baseline": "tab:blue", "mixed_channel": "tab:purple"}
SEEDS = tuple(range(10))
FAIL_THRESHOLD = 35.0  # final val acc below this (near random=25%) counts as a failed run


def run_extended_comparison(seeds=SEEDS, epochs=15):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {v: [] for v in VARIANTS}

    for mode in VARIANTS:
        print(f"\n{'='*60}\nTraining variant: {mode} ({len(seeds)} seeds)\n{'='*60}")
        for seed in seeds:
            results[mode].append(train_variant(mode, seed, epochs=epochs))

    # ---- Per-seed table + failure classification ----
    summary = {}
    for mode in VARIANTS:
        final_accs = [r["val_accs"][-1] for r in results[mode]]
        peak_accs = [max(r["val_accs"]) for r in results[mode]]
        conv_epochs = [convergence_epoch(r["val_accs"]) for r in results[mode]]
        failed = [i for i, acc in enumerate(final_accs) if acc < FAIL_THRESHOLD]

        converged_final = [acc for i, acc in enumerate(final_accs) if i not in failed]

        summary[mode] = {
            "final_accs_per_seed": final_accs,
            "peak_accs_per_seed": peak_accs,
            "convergence_epoch_per_seed": conv_epochs,
            "failed_seed_indices": failed,
            "failure_rate": len(failed) / len(seeds),
            "final_acc_mean_all": float(np.mean(final_accs)),
            "final_acc_std_all": float(np.std(final_accs)),
            "final_acc_mean_converged_only": float(np.mean(converged_final)) if converged_final else None,
            "final_acc_std_converged_only": float(np.std(converged_final)) if converged_final else None,
            "peak_acc_mean_all": float(np.mean(peak_accs)),
        }

    print("\n" + "=" * 60)
    print(f"EXTENDED SUMMARY ({len(seeds)} seeds, noiseless)")
    print("=" * 60)
    for mode in VARIANTS:
        s = summary[mode]
        print(f"\n[{mode}]")
        print(f"  Per-seed final val acc: {[f'{a:.1f}' for a in s['final_accs_per_seed']]}")
        print(
            f"  Failed seeds (final acc < {FAIL_THRESHOLD}%): {s['failed_seed_indices']} "
            f"({s['failure_rate']*100:.0f}% failure rate)"
        )
        print(f"  Final acc, ALL seeds:       {s['final_acc_mean_all']:.1f}% ± {s['final_acc_std_all']:.1f}")
        if s["final_acc_mean_converged_only"] is not None:
            print(
                f"  Final acc, CONVERGED only:  {s['final_acc_mean_converged_only']:.1f}% "
                f"± {s['final_acc_std_converged_only']:.1f} (n={len(seeds) - len(s['failed_seed_indices'])})"
            )
        print(f"  Peak acc, ALL seeds mean:   {s['peak_acc_mean_all']:.1f}%")

    # ---- Plot: per-seed final val acc bar comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seeds))
    width = 0.35
    for i, mode in enumerate(VARIANTS):
        offset = (i - 0.5) * width
        accs = summary[mode]["final_accs_per_seed"]
        bar_colors = [COLORS[mode] if a >= FAIL_THRESHOLD else "lightgray" for a in accs]
        ax.bar(x + offset, accs, width, label=mode, color=bar_colors, edgecolor=COLORS[mode])
    ax.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax.axhline(y=FAIL_THRESHOLD, color="red", linestyle=":", alpha=0.5, label=f"Failure threshold ({FAIL_THRESHOLD}%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_ylabel("Final Validation Accuracy (%)")
    ax.set_title(f"Per-Seed Final Val Acc: Baseline vs. Mixed-Channel ({len(seeds)} seeds)\n(grayed bars = failed run)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "mixed_channel_extended_seeds_bar.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved per-seed bar plot to: {plot_path}")

    # ---- Noise sweep averaged over ALL seeds, and over CONVERGED seeds only ----
    print("\n" + "=" * 60)
    print("NOISE SWEEP (averaged across all trained seed models)")
    print("=" * 60)
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    noise_curves_all_seeds = {mode: [] for mode in VARIANTS}

    for mode in VARIANTS:
        failed_idx = set(summary[mode]["failed_seed_indices"])
        for i, r in enumerate(results[mode]):
            print(f"\n[{mode} | seed={seeds[i]}]" + (" (FAILED run)" if i in failed_idx else ""))
            accs = noise_sweep(r["model"], r["test_loader"], noise_rates)
            noise_curves_all_seeds[mode].append(accs)

    noise_summary = {}
    for mode in VARIANTS:
        curves = np.array(noise_curves_all_seeds[mode])  # [n_seeds, n_noise_rates]
        failed_idx = summary[mode]["failed_seed_indices"]
        converged_mask = np.array([i not in failed_idx for i in range(len(seeds))])

        noise_summary[mode] = {
            "mean_all_seeds": curves.mean(axis=0).tolist(),
            "std_all_seeds": curves.std(axis=0).tolist(),
            "mean_converged_only": curves[converged_mask].mean(axis=0).tolist() if converged_mask.any() else None,
            "std_converged_only": curves[converged_mask].std(axis=0).tolist() if converged_mask.any() else None,
        }

    print("\n" + "=" * 60)
    print("NOISE SWEEP SUMMARY")
    print("=" * 60)
    for mode in VARIANTS:
        ns = noise_summary[mode]
        print(f"\n[{mode}]")
        for p, m_all, m_conv in zip(noise_rates, ns["mean_all_seeds"], ns["mean_converged_only"] or []):
            print(f"  p={p:.3f} | ALL seeds: {m_all:.1f}% | CONVERGED only: {m_conv:.1f}%")

    # ---- Plot: noise sweep, both framings ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for mode in VARIANTS:
        ns = noise_summary[mode]
        mean_all = np.array(ns["mean_all_seeds"])
        std_all = np.array(ns["std_all_seeds"])
        ax1.plot(noise_rates, mean_all, "-o", color=COLORS[mode], label=mode)
        ax1.fill_between(noise_rates, mean_all - std_all, mean_all + std_all, color=COLORS[mode], alpha=0.15)

        if ns["mean_converged_only"] is not None:
            mean_conv = np.array(ns["mean_converged_only"])
            std_conv = np.array(ns["std_converged_only"])
            ax2.plot(noise_rates, mean_conv, "-o", color=COLORS[mode], label=mode)
            ax2.fill_between(noise_rates, mean_conv - std_conv, mean_conv + std_conv, color=COLORS[mode], alpha=0.15)

    ax1.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax1.set_xlabel("Depolarizing Noise Rate ($p$)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title(f"Noise Sweep: ALL {len(seeds)} Seeds (incl. failed runs)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(y=25.0, color="gray", linestyle="--", alpha=0.5, label="Random (25%)")
    ax2.set_xlabel("Depolarizing Noise Rate ($p$)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Noise Sweep: CONVERGED Seeds Only")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    noise_plot_path = os.path.join(RESULTS_DIR, "mixed_channel_extended_seeds_noise.png")
    plt.savefig(noise_plot_path)
    plt.close()
    print(f"\nSaved extended noise sweep plot to: {noise_plot_path}")

    # ---- Save raw results ----
    summary["noise_rates"] = noise_rates
    summary["noise_summary"] = noise_summary
    summary["seeds"] = list(seeds)
    results_path = os.path.join(RESULTS_DIR, "mixed_channel_extended_seeds_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved raw results to: {results_path}")

    return summary


if __name__ == "__main__":
    run_extended_comparison(seeds=SEEDS, epochs=15)
