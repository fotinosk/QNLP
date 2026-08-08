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


ANSATZ_C = {"strongly_entangling": "#E69F00", "iqp": "#0072B2"}


def _r2_arms():
    """R2's encoding x ansatz sweep, keyed by (encoding, ansatz)."""
    r2 = _load("r2_ansatz_results.json")
    if not r2:
        return None, None
    return r2, {tuple(a["config"].split("+")): a for a in r2["arms"]}


def _mlp_ref():
    r4 = _load("r4_classical_only_coherent_fixedrank.json")
    if not r4:
        return None
    return {a["config"]: a for a in r4["arms"]}


def fig_encoding_ansatz():
    """Replaces encoding_ansatz_sweep.png.

    The retired figure swept ANGLE/MULTI_AXIS/AMPLITUDE/ZZ_MAP x HEA/IQP/ALT at
    8x8 with ONE seed per config. HEA and ALT are not in the shipped codebase --
    qttn_core implements {strongly_entangling, iqp} -- so that figure benchmarks
    an architecture that does not exist. This plots R2 instead: the same
    question at 16x16, 21 seeds per arm, on the protocol of record.
    """
    _, arms = _r2_arms()
    mlp = _mlp_ref()
    if not arms:
        print("  SKIP encoding_ansatz (missing r2_ansatz_results.json)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    encodings, ansatze = ("scalar_ry", "multi_axis"), ("strongly_entangling", "iqp")
    width, xs = 0.36, range(len(encodings))
    for k, ansatz in enumerate(ansatze):
        pos = [x + (k - 0.5) * width for x in xs]
        ax1.bar(
            pos,
            [arms[(e, ansatz)]["score_mean"] for e in encodings],
            width,
            yerr=[arms[(e, ansatz)]["score_std"] for e in encodings],
            label=ansatz,
            color=ANSATZ_C[ansatz],
            capsize=3,
        )
        for x, e in zip(pos, encodings):
            a = arms[(e, ansatz)]
            ax1.annotate(f"{a['score_mean']:.1f}", (x, a["score_mean"] + a["score_std"] + 1.5), ha="center", fontsize=8)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([f"{e}\n({arms[(e, 'iqp')].get('num_params', '?')}p)" for e in encodings])
    ax1.set_ylabel("accuracy % (mean of last 5 epochs)")
    ax1.set_ylim(0, 122)
    ax1.legend(fontsize=8, loc="upper center", ncol=2, frameon=False)
    ax1.set_title("Encoding x ansatz, 21 seeds each", fontsize=10)
    _reference_lines(ax1, mlp=mlp["mlp_reference"]["score_mean"] if mlp else None)

    # Effect sizes against their resolution limits -- the point of the figure.
    cmps = [
        (
            "encoding\n(multi_axis - scalar_ry,\nat iqp)",
            pc.compare(arms[("scalar_ry", "iqp")], arms[("multi_axis", "iqp")]),
        ),
        (
            "ansatz\n(iqp - strongly_ent.,\nat multi_axis)",
            pc.compare(arms[("multi_axis", "strongly_entangling")], arms[("multi_axis", "iqp")]),
        ),
    ]
    ys = range(len(cmps))
    for y, (_, c) in zip(ys, cmps):
        m = c["min_detectable_effect"]
        ax2.barh(y, 2 * m, left=-m, height=0.45, color="#CCCCCC", zorder=1)
        ax2.plot([c["difference"]], [y], "D", ms=9, color="#0072B2" if c["resolved"] else "#888888", zorder=3)
        # Left-aligned at a fixed x so long labels can never overflow the axis.
        ax2.annotate(
            f"{c['difference']:+.1f}  (limit {m:.1f}) -- {'RESOLVED' if c['resolved'] else 'not resolved'}",
            (-5.7, y + 0.30),
            ha="left",
            fontsize=8,
            color="#0072B2" if c["resolved"] else "#666666",
        )
    ax2.axvline(0, color="black", lw=1, zorder=2)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([n for n, _ in cmps], fontsize=8)
    ax2.set_xlabel("difference in accuracy (pts)")
    ax2.set_xlim(-6, 16)
    ax2.set_ylim(-0.6, len(cmps) - 0.25)
    ax2.set_title("Effect size vs resolution limit", fontsize=10)

    _save(
        fig,
        "encoding_ansatz.png",
        "REPLACES results/encoding_ansatz_sweep.png, which swept HEA/IQP/ALT at 8x8 with one seed per config; "
        "HEA and ALT are not implemented in qttn_core, so that figure benchmarked an architecture that does "
        "not exist. Task R2, hybrid tree, 16x16 synthetic shapes, 1024 train / 64 test / 30 epochs, 21 seeds "
        "per arm, unpaired Welch. Grey bands on the right are the minimum detectable effect: the ENCODING "
        "choice is resolved (+12.9, limit 3.7) and comes from using all three of a qubit's rotation "
        "parameters where scalar_ry uses one; the ANSATZ choice is NOT (+1.6, limit 4.8). IQP is adopted for "
        "shallower gate depth, not for accuracy and not for noise tolerance.",
    )


def fig_ansatz_comparison():
    """Replaces ansatz_comparison_noise.png.

    The retired figure compared HEA/IQP/ALT under depolarizing noise at 8x8,
    single seed, and its IQP curve contradicts the Metrics Record (which logs
    p_crit > 0.200 for the same config). The project is noiseless-only by scope
    decision and noise resilience was never a selection criterion, so the
    replacement answers the question that actually decided the ansatz.
    """
    _, arms = _r2_arms()
    if not arms:
        print("  SKIP ansatz_comparison (missing r2_ansatz_results.json)")
        return
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cmp_ = pc.compare(arms[("multi_axis", "strongly_entangling")], arms[("multi_axis", "iqp")])

    for ansatz in ("strongly_entangling", "iqp"):
        a = arms[("multi_axis", ansatz)]
        curves = np.array(a["val_acc_curves_per_seed"])
        ep, mu, sd = np.arange(1, curves.shape[1] + 1), curves.mean(0), curves.std(0, ddof=1)
        ax1.plot(ep, mu, "-", lw=1.8, color=ANSATZ_C[ansatz], label=f"{ansatz} ({a['score_mean']:.1f}%)")
        ax1.fill_between(ep, mu - sd, mu + sd, color=ANSATZ_C[ansatz], alpha=0.16, lw=0)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy %")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_title("multi_axis encoding, mean +/- 1 s.d. over 21 seeds", fontsize=10)
    _reference_lines(ax1)

    for k, ansatz in enumerate(("strongly_entangling", "iqp")):
        a = arms[("multi_axis", ansatz)]
        pts = a["score_per_seed"]
        ax2.scatter(
            np.random.default_rng(0).normal(k, 0.055, len(pts)), pts, s=26, color=ANSATZ_C[ansatz], alpha=0.75, zorder=3
        )
        ax2.hlines(a["score_mean"], k - 0.22, k + 0.22, color="black", lw=2, zorder=4)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["strongly_entangling", "iqp"], fontsize=9)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylabel("accuracy % (mean of last 5 epochs)")
    ax2.set_ylim(0, 100)
    ax2.set_title(
        f"Per-seed spread: {cmp_['difference']:+.1f} pts, limit {cmp_['min_detectable_effect']:.1f} -> NOT RESOLVED",
        fontsize=10,
    )
    _reference_lines(ax2, show_ceiling=True)

    _save(
        fig,
        "ansatz_comparison.png",
        "REPLACES results/ansatz_comparison_noise.png, which is RETIRED for three reasons: it benchmarks "
        "HEA/IQP/ALT (HEA and ALT are not in qttn_core), it is single-seed at 8x8, and its IQP curve "
        "contradicts the Metrics Record, which logs p_crit > 0.200 for the same MULTI_AXIS+IQP config while "
        "the figure shows it crossing 50% near p=0.12. Its flat HEA curve is also not evidence of "
        "robustness: depolarizing noise contracts expectation values multiplicatively and preserves argmax "
        "order, so a decision can be noise-invariant while the representation is destroyed. This "
        "replacement is NOISELESS, matching the project's scope. The two ansatze are statistically tied; "
        "IQP's spread is tighter (s.d. 4.9 vs 9.7) and it is adopted for shallower gate depth.",
    )


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"Regenerating figures into {FIGDIR}")
    fig_model_comparison()
    fig_ablations()
    fig_readout()
    fig_encoding_ansatz()
    fig_ansatz_comparison()
    print(
        "\nNot regenerated (Phase B triage): fidelity_distributions, barren_plateau_scaling,\n"
        "topology_barren_plateaus, entropy_vs_tree_depth, entropy_propagation_vs_noise -- these\n"
        "involve no classifier head and are unaffected. topology_noise_resilience is RETIRED."
    )


if __name__ == "__main__":
    main()
