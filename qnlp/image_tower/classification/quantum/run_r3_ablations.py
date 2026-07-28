"""Task R3 (roadmap Section 7): re-run the decisive ablations at restored capacity.

Both verdicts being re-tested were reached on the scalar-readout model, in a
50-57% band against a 50% single-attribute shortcut ceiling (research_log.md
2026-07-27 "Code Audit"), so they are currently PROVISIONAL:

  R3a -- spatial ancilla (Question C.1). Old verdict "not better, mildly worse"
         rested on 52.8 +/- 5.7 vs 55.0 +/- 3.5 at n=5, well inside noise.
  R3b -- the two residual mechanisms that showed any signal: `reupload`
         (noiseless expressivity gain, collapsed under noise) and
         `mixed_channel` (small noiseless edge, ~10% dead-gradient failures).
         `near_identity` and `lcu`/`lcu-lite` are not re-run -- see roadmap R3b.

Design follows R1b's measurements (research_log.md 2026-07-28):
  - architecture of record: readout=root_multi_pauli, encoding=multi_axis,
    ansatz=iqp (phase15_common.ARCH; readout from R1/R1b, rest from R2)
  - protocol of record: 1024 train / 64 test / 30 epochs
  - runs scored by mean of last 5 epochs
  - UNPAIRED comparison (cross-variant seed correlation measured at -0.22)
  - 58 seeds/arm by default -> resolves ~3-point effects. The old 10-seed plan
    resolved only ~8-point effects, while these mechanisms move 2-3 points;
    at 10 seeds this would have produced another inconclusive null.

Noise sweeps are multi-seed averaged (20 by default), never a single-seed spot
check -- that is what produced the 2026-07-26 mixed-channel retraction.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r3_ablations
"""

import argparse

import numpy as np
import torch

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import HierarchicalQTTNClassifier
from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

# Architecture of record from R1/R1b (readout) and R2 (encoding, ansatz).
ARCH = pc.ARCH
NOISE_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
# Below this score a run is treated as a dead-gradient failure rather than a
# poor result (25% is the random baseline for 4 classes).
FAILURE_THRESHOLD = 35.0

ARMS = {
    # name: kwargs to HierarchicalQTTNClassifier
    "baseline": {},
    "reupload": {"mode": "reupload"},
    "mixed_channel": {"mode": "mixed_channel"},
    "with_ancilla": {"use_ancilla": True},
}


def noise_sweep(model, seed, levels=NOISE_LEVELS, test_samples=64, batch_size=32):
    _, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size, train_samples=8, test_samples=test_samples, img_size=16, seed=seed
    )
    accs = []
    model.eval()
    with torch.no_grad():
        for p in levels:
            correct = total = 0
            for imgs, labels in test_loader:
                out = model(imgs, p_noise=p) if p > 0 else model(imgs)
                correct += out.argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)
            accs.append(100.0 * correct / total)
    return accs


def _checkpoint(name, out_prefix, curves, noise_curves, seeds_done):
    """Write partial results after every seed.

    The first R3 run produced a single JSON only at exit, so a crash at seed 57
    of arm 4 would have discarded hours of compute. It also buffered all output
    (a `conda run` artifact), leaving no way to see progress. Both fixed here.
    """
    pc.save(
        {
            "arm": name,
            "seeds_done": seeds_done,
            "partial": True,
            "val_acc_curves_per_seed": curves,
            "noise_curves": noise_curves,
        },
        f"{out_prefix}_{name}_partial.json",
    )


def run_arm(name, kwargs, seeds, n_noise_seeds=20, out_prefix="r3"):
    """Trains every seed; noise-sweeps the first `n_noise_seeds` of them.

    The accuracy comparison is what needs 58 seeds (R1b power analysis), and the
    noise sweep costs ~21s/seed because multi_axis forces row-by-row evaluation
    on default.mixed (see qttn_core._run_noisy). 20 seeds is still multi-seed
    averaging -- the 2026-07-26 retraction came from single-seed spot checks --
    and the noise-response effects being looked for are much larger than the
    2-3 point accuracy effects.
    """
    print(f"\n--- arm: {name} ({len(seeds)} seeds, {n_noise_seeds} noise-swept) ---", flush=True)
    curves, noise_curves = [], []
    for s in seeds:
        curve, model = pc.train_run(lambda: HierarchicalQTTNClassifier(**ARCH, **kwargs), seed=s)
        curves.append(curve)
        score = sum(curve[-pc.SCORE_LAST_K :]) / pc.SCORE_LAST_K
        if s < n_noise_seeds:
            noise_curves.append(noise_sweep(model, s))
        print(
            f"  seed {s:>2}: score={score:.1f}%  final={curve[-1]:.1f}%"
            + ("  FAILED" if score < FAILURE_THRESHOLD else ""),
            flush=True,
        )
        _checkpoint(name, out_prefix, curves, noise_curves, s + 1)

    arm = pc.summarise(
        name,
        curves,
        num_params=sum(p.numel() for p in HierarchicalQTTNClassifier(**ARCH, **kwargs).parameters()),
        **kwargs,
    )
    scores = np.array(arm["score_per_seed"])
    failed = scores < FAILURE_THRESHOLD
    arm["n_failed"] = int(failed.sum())
    arm["failure_rate"] = float(failed.mean())
    if (~failed).any():
        arm["score_mean_converged_only"] = float(scores[~failed].mean())
        arm["score_std_converged_only"] = float(scores[~failed].std(ddof=1)) if (~failed).sum() > 1 else 0.0
    if noise_curves:
        nc = np.array(noise_curves)
        arm["noise_levels"] = list(NOISE_LEVELS)
        arm["noise_acc_mean"] = nc.mean(axis=0).tolist()
        arm["noise_acc_std"] = nc.std(axis=0, ddof=1).tolist() if nc.shape[0] > 1 else [0.0] * nc.shape[1]
        # `failed` covers every trained seed; only the first n_noise_seeds were swept.
        swept_ok = ~failed[: nc.shape[0]]
        arm["n_noise_seeds"] = int(nc.shape[0])
        if swept_ok.any():
            arm["noise_acc_mean_converged_only"] = nc[swept_ok].mean(axis=0).tolist()
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seeds-per-arm",
        type=int,
        default=30,
        help="30 resolves ~2.6-pt effects at the architecture of record's measured "
        "std of 4.94 (R2). The earlier default of 58 came from R1b's std of 8.2, "
        "which was measured on a different config and over-provisions this one. "
        "Each arm reports the resolution it actually achieved; top up if an arm's "
        "variance turns out higher (checkpoints make resuming cheap).",
    )
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument(
        "--noise-seeds", type=int, default=20, help="How many of the trained seeds also get a noise sweep (0 = none)."
    )
    ap.add_argument("--out", default="r3", help="Output filename prefix (one per parallel worker).")
    args = ap.parse_args()
    seeds = list(range(args.seeds_per_arm))

    arms = {name: run_arm(name, ARMS[name], seeds, args.noise_seeds, args.out) for name in args.arms}

    base = arms.get("baseline")
    comparisons, decisions = [], {}
    if base is not None:
        for name, arm in arms.items():
            if name == "baseline":
                continue
            c = pc.compare(base, arm)
            comparisons.append(c)
            question = "Question C.1 (spatial ancilla)" if name == "with_ancilla" else f"Residual mechanism: {name}"
            if not c["resolved"]:
                verdict = (
                    f"NO EVIDENCE OF BENEFIT. Difference {c['difference']:+.1f} pts is inside "
                    f"this design's {c['min_detectable_effect']:.1f}-pt resolution limit "
                    f"({c['n_seeds']} seeds). The provisional 'do not use' rule is upheld as "
                    f"'no demonstrated benefit at effects >= {c['min_detectable_effect']:.1f} pts' "
                    f"-- state the limit, do not claim the mechanism is harmful."
                )
            elif c["difference"] > 0:
                verdict = (
                    f"BENEFIT CONFIRMED, +{c['difference']:.1f} pts (resolvable limit "
                    f"{c['min_detectable_effect']:.1f}). This FLIPS the provisional verdict -- "
                    f"update the binding rule in quantum_implementation_plan.md and roadmap "
                    f"Section 5."
                )
            else:
                verdict = (
                    f"HARMFUL CONFIRMED, {c['difference']:.1f} pts (resolvable limit "
                    f"{c['min_detectable_effect']:.1f}). The provisional rejection is upheld "
                    f"with real evidence."
                )
            decisions[name] = {"question": question, "verdict": verdict, "failure_rate": arm.get("failure_rate")}

    pc.print_arms(
        f"R3: ablations @ {ARCH['encoding']}+{ARCH['ansatz']}, 1024/30, "
        f"{args.seeds_per_arm} seeds, scored on last-{pc.SCORE_LAST_K} mean",
        list(arms.values()),
    )
    print(f"\n{'arm':<20}{'failures':>12}{'converged-only score':>24}")
    for name, a in arms.items():
        co = a.get("score_mean_converged_only")
        print(f"{name:<20}{a['n_failed']:>4}/{a['n_seeds']:<7}" f"{(f'{co:.1f}' if co is not None else 'n/a'):>24}")
    if comparisons:
        pc.print_comparisons(comparisons)
        print()
        for name, d in decisions.items():
            print(f"\n[{name}] {d['question']}\n  {d['verdict']}")

    if args.noise_seeds:
        print(f"\n{f'noise sweep (mean, {args.noise_seeds} seeds)':<24}" + "".join(f"{p:>8}" for p in NOISE_LEVELS))
        for name, a in arms.items():
            if "noise_acc_mean" in a:
                print(f"{name:<24}" + "".join(f"{v:>8.1f}" for v in a["noise_acc_mean"]))

    pc.save(
        {
            "architecture": ARCH,
            "protocol": pc.PROTOCOL,
            "seeds_per_arm": args.seeds_per_arm,
            "score_metric": f"mean of last {pc.SCORE_LAST_K} epochs",
            "comparison_type": "unpaired (cross-variant seed corr measured at -0.22 in R1b)",
            "n_noise_seeds": args.noise_seeds,
            "arms": arms,
            "comparisons": comparisons,
            "decisions": decisions,
        },
        f"{args.out}_ablation_results.json",
    )


if __name__ == "__main__":
    main()
