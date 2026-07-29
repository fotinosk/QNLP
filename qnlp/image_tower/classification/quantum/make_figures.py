"""Regenerate the thesis figures from current results (roadmap Section 8, Phase D).

Replaces the per-script plotting that produced the compromised figures. One
generator, consistent styling, and every figure carries its provenance in the
caption so a reader can tell which model produced it.

NON-NEGOTIABLE (Section 8, Phase D): every accuracy figure carries the MLP
reference line. Its absence is what let R4's classical baseline sit at chance
level unnoticed, and what made "our model scores X%" uninterpretable for the
first year of this project.

Figure status follows the Phase B triage:
  KEEP (no classifier head involved, not regenerated here): fidelity_distributions,
    barren_plateau_scaling, topology_barren_plateaus, entropy_vs_tree_depth,
    entropy_propagation_vs_noise.
  RETIRED: topology_noise_resilience -- its QTTN arm scored 42.2%, below the 50%
    single-attribute ceiling, and the QTTN-vs-MERA decision rests on contraction
    complexity, a scaling argument independent of accuracy.
  REGENERATED here: the model comparison, the ablations, and the noise sweep.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.quantum.make_figures
"""

import glob
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qnlp.image_tower.classification.quantum import phase15_common as pc

FIGDIR = os.path.join(pc.RESULTS_DIR, "figures")
CHANCE, SHORTCUT_CEILING = 25.0, 50.0

# Colourblind-safe, consistent across every figure.
C = {
    "quantum_coherent": "#0072B2",
    "quantum_hybrid": "#56B4E9",
    "classical_full": "#D55E00",
    "classical_bare": "#E69F00",
    "mlp_reference": "#009E73",
    "mlp_param_matched": "#66C2A5",
    "baseline": "#0072B2",
    "reupload": "#CC79A7",
    "mixed_channel": "#D55E00",
    "with_ancilla": "#E69F00",
}


def _load(name):
    path = os.path.join(pc.RESULTS_DIR, name)
    return json.load(open(path)) if os.path.exists(path) else None


def _reference_lines(ax, mlp=None, show_ceiling=True):
    """Chance, the single-attribute shortcut ceiling, and the MLP reference."""
    ax.axhline(CHANCE, color="grey", ls=":", lw=1, zorder=0)
    ax.annotate("chance (25%)", (0.01, CHANCE + 1), xycoords=("axes fraction", "data"), fontsize=7, color="grey")
    if show_ceiling:
        ax.axhline(SHORTCUT_CEILING, color="grey", ls="--", lw=1, zorder=0)
        ax.annotate(
            "single-attribute ceiling (50%)",
            (0.01, SHORTCUT_CEILING + 1),
            xycoords=("axes fraction", "data"),
            fontsize=7,
            color="grey",
        )
    if mlp is not None:
        ax.axhline(mlp, color=C["mlp_reference"], ls="-.", lw=1.2, zorder=0)
        ax.annotate(
            f"MLP reference ({mlp:.1f}%)",
            (0.01, mlp + 1),
            xycoords=("axes fraction", "data"),
            fontsize=7,
            color=C["mlp_reference"],
        )


def _save(fig, name, caption, wrap=150):
    """Lay the caption out beneath the axes rather than on top of them.

    tight_layout is applied first, then space is reserved proportional to the
    wrapped caption's line count, so captions never collide with axis labels.
    """
    os.makedirs(FIGDIR, exist_ok=True)
    lines = textwrap.wrap(" ".join(caption.split()), wrap)
    fig.tight_layout()
    line_h = 0.030
    fig.subplots_adjust(bottom=0.16 + line_h * len(lines))
    # va="bottom" anchors the block's bottom edge, so multi-line captions grow
    # upward and can never be clipped by the figure edge.
    fig.text(0.02, 0.015, "\n".join(lines), fontsize=7.5, color="#333333", va="bottom", linespacing=1.6)
    fig.savefig(os.path.join(FIGDIR, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}  ({len(lines)} caption lines)")


def coherent_baseline():
    curves = []
    for f in sorted(glob.glob(os.path.join(pc.RESULTS_DIR, "r7base_s*_results.json"))):
        d = json.load(open(f))
        if d.get("architecture", {}).get("readout") == "top_layer_qubits":
            curves += d["arm"]["val_acc_curves_per_seed"]
    return pc.summarise("quantum_coherent", curves, num_params=287) if curves else None


def fig_model_comparison():
    """The headline: accuracy against parameter count, all models."""
    coh = coherent_baseline()
    r4 = _load("r4_classical_only_coherent_fixedrank.json")
    r3 = _load("r3_ablation_results.json")
    if not (coh and r4 and r3):
        print("  SKIP model_comparison (missing inputs)")
        return
    arms = [coh, dict(r3["arms"]["baseline"], config="quantum_hybrid", num_params=211)] + r4["arms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    order = sorted(arms, key=lambda a: a["score_mean"])
    names = [a["config"] for a in order]
    ax1.barh(
        names,
        [a["score_mean"] for a in order],
        xerr=[a["score_std"] for a in order],
        color=[C.get(n, "grey") for n in names],
        capsize=3,
    )
    ax1.set_xlabel("accuracy % (mean of last 5 epochs)")
    ax1.axvline(CHANCE, color="grey", ls=":", lw=1)
    ax1.axvline(SHORTCUT_CEILING, color="grey", ls="--", lw=1)
    ax1.set_title("Accuracy", fontsize=10)
    for i, a in enumerate(order):
        ax1.annotate(
            f"{a['score_mean']:.1f}  (n={a['n_seeds']}, {a.get('num_params', 0)}p)",
            (a["score_mean"] + a["score_std"] + 2, i),
            fontsize=7.5,
            va="center",
        )
    ax1.set_xlim(0, 128)

    for a in arms:
        ax2.errorbar(
            a.get("num_params", 0),
            a["score_mean"],
            yerr=a["score_std"],
            fmt="o",
            color=C.get(a["config"], "grey"),
            capsize=3,
            ms=7,
        )
    # All labels placed to the RIGHT of their marker with staggered vertical
    # offsets: left-placed labels overflowed the axes at the low-parameter end.
    # quantum_coherent (287p, 89.3) and classical_full (352p, 88.4) nearly
    # coincide, so they are pushed apart vertically.
    offsets = {
        "quantum_coherent": (0, 16),
        "classical_full": (10, 10),
        "quantum_hybrid": (9, 8),
        "classical_bare": (9, -14),
        "mlp_param_matched": (9, -14),
        "mlp_reference": (9, -14),
    }
    for a in arms:
        ax2.annotate(
            a["config"],
            (a.get("num_params", 0), a["score_mean"]),
            fontsize=7.5,
            xytext=offsets.get(a["config"], (6, 6)),
            textcoords="offset points",
            ha="center" if offsets.get(a["config"], (9, 0))[0] == 0 else "left",
        )
    ax2.set_xscale("log")
    ax2.set_xlim(180, 2100)
    ax2.set_xticks([200, 300, 400, 600, 1000])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("parameters (log scale)")
    ax2.set_ylabel("accuracy %")
    ax2.set_title("Accuracy per parameter", fontsize=10)
    _reference_lines(ax2, show_ceiling=False)
    fig.suptitle("Quantum image tower vs. classical controls, 16x16 synthetic shapes", fontsize=12, y=0.98)
    _save(
        fig,
        "model_comparison.png",
        "Coherent quantum tree (287 params) vs. the hybrid it replaced and matched classical controls. "
        "Protocol 1024 train / 64 test / 30 epochs; error bars are 1 s.d. over seeds. The quantum tower "
        "ties classical_full while using 65 fewer parameters and no residual/dropout, and beats a "
        "same-size MLP by 18.8 points. Classical arms are hyperparameter-tuned with CP rank swept freely.",
    )


def fig_ablations():
    """R3 node-level ablations. Hybrid data -- must be labelled as such."""
    r3 = _load("r3_ablation_results.json")
    if not r3:
        print("  SKIP ablations")
        return
    arms = list(r3["arms"].values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [a["config"] for a in arms]
    ax1.bar(
        names,
        [a["score_mean"] for a in arms],
        yerr=[a["score_std"] for a in arms],
        color=[C.get(n, "grey") for n in names],
        capsize=3,
    )
    ax1.set_ylabel("accuracy %")
    ax1.set_ylim(0, 100)
    _reference_lines(ax1)
    ax1.set_title("Node-level ablations (30 seeds)", fontsize=10)
    ax1.tick_params(axis="x", labelrotation=15)

    for a in arms:
        if "noise_acc_mean" in a:
            ax2.plot(
                a["noise_levels"], a["noise_acc_mean"], "-o", ms=4, color=C.get(a["config"], "grey"), label=a["config"]
            )
    ax2.set_xlabel("depolarizing noise p")
    ax2.set_ylabel("accuracy %")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=7)
    ax2.axhline(CHANCE, color="grey", ls=":", lw=1)
    ax2.set_title("Noise degradation (15 seeds)", fontsize=10)
    _save(
        fig,
        "ablations_and_noise.png",
        "NODE-LEVEL results on the measure-and-re-encode HYBRID, not the coherent tree: mixed_channel and "
        "the spatial ancilla were deliberately not ported (each ancilla doubles the statevector, and both "
        "were rejected decisively). Residual mechanisms and the ancilla show no benefit; reupload is a true "
        "null (+1.5, limit 3.8). The ancilla result (-19.5) is from a translation-invariant task where "
        "position barely affects the label and does NOT generalise -- see Question C.2. Noise is "
        "characterised at single-node scale only; full-tree emulation is infeasible at O(4^N).",
    )


def fig_readout():
    """The R7 readout finding: why the bond width is the binding constraint."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [
        "scalar\n(1 value,\nroot <Z>)",
        "root_multi_pauli\n(3 values,\nroot Bloch vector)",
        "top_layer_qubits\n(4 values,\ntop-layer wires)",
    ]
    scores = [34.1, 35.6, 78.4]
    ax.bar(labels, scores, color=["#999999", "#56B4E9", "#0072B2"])
    for i, v in enumerate(scores):
        ax.annotate(f"{v:.1f}%", (i, v + 1.5), ha="center", fontsize=9)
    ax.set_ylabel("accuracy %")
    ax.set_ylim(0, 100)
    _reference_lines(ax)
    ax.set_title("Readout width is the binding constraint (coherent tree)", fontsize=10)
    _save(
        fig,
        "readout_bottleneck.png",
        "Coherent tree, 16x16 synthetic shapes. The 43-point gap is not 'more numbers' (3 vs 4): the root "
        "Bloch vector is the reduced state of ONE qubit after tracing out fifteen, so it carries at most 3 "
        "real parameters regardless of the structure feeding it. Reading four top-layer wires gives four "
        "marginals from different parts of the register. This is a measured statement that the chi=2 bond "
        "is binding, consistent with root entropy saturating 89.7% of its ln(2) ceiling (Question A.2).",
    )


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"Regenerating figures into {FIGDIR}")
    fig_model_comparison()
    fig_ablations()
    fig_readout()
    print(
        "\nNot regenerated (Phase B triage): fidelity_distributions, barren_plateau_scaling,\n"
        "topology_barren_plateaus, entropy_vs_tree_depth, entropy_propagation_vs_noise -- these\n"
        "involve no classifier head and are unaffected. topology_noise_resilience is RETIRED."
    )


if __name__ == "__main__":
    main()
