"""Merge the per-arm R3 workers into one result set and apply the verdicts.

R3 was originally one sequential process: 4 arms x 58 seeds plus noise sweeps,
single-threaded on an 11-core machine, with all output buffered by `conda run`
so progress was invisible and a late crash would have discarded everything.

It is now run as one worker per arm (see the launch command in research_log.md),
which is ~4x faster, checkpoints after every seed, and streams progress. This
script merges the per-arm JSONs and performs the cross-arm comparisons that a
single-arm worker cannot do on its own.

Run:  conda run -n qnlp python -m qnlp.image_tower.classification.quantum.combine_r3
"""

import argparse
import glob
import json
import os

from qnlp.image_tower.classification.quantum import phase15_common as pc


def load_arms(prefix):
    arms = {}
    for path in sorted(glob.glob(os.path.join(pc.RESULTS_DIR, f"{prefix}_*_ablation_results.json"))):
        with open(path) as f:
            payload = json.load(f)
        for arm in payload["arms"].values():
            arms[arm["config"]] = arm
    return arms


def verdict_for(name, cmp_result, arm):
    """Phrase the outcome so a null cannot be mistaken for a refutation."""
    question = "Question C.1 (spatial ancilla)" if name == "with_ancilla" else f"Residual mechanism: {name}"
    d, mde = cmp_result["difference"], cmp_result["min_detectable_effect"]

    if not cmp_result["resolved"]:
        text = (
            f"NO EVIDENCE OF BENEFIT. Difference {d:+.1f} pts is inside this design's "
            f"{mde:.1f}-pt resolution limit ({cmp_result['n_seeds']} seeds). Report as "
            f"'no demonstrated benefit at effects >= {mde:.1f} pts' -- NOT as evidence the "
            f"mechanism is harmful, and not as a general rule beyond this task."
        )
    elif d > 0:
        text = (
            f"BENEFIT CONFIRMED, +{d:.1f} pts (limit {mde:.1f}). This FLIPS the provisional "
            f"verdict -- update the binding rule in quantum_implementation_plan.md and roadmap "
            f"Section 5."
        )
    else:
        text = (
            f"HARMFUL CONFIRMED, {d:.1f} pts (limit {mde:.1f}). The provisional rejection is "
            f"upheld, now with real evidence."
        )

    if name == "with_ancilla":
        text += (
            " SCOPE CAVEAT: this task is translation-invariant single-object classification, "
            "where position is nearly irrelevant to the label, so a null here is close to "
            "guaranteed and says little about positional encoding in general. It also tests a "
            "per-quadrant ancilla (4 positions), not the original per-patch design (16). The "
            "discriminating test is CLEVR's relational task (Question C.2), where position IS "
            "the label -- run it there with and without the ancilla."
        )
    return {"question": question, "verdict": text, "failure_rate": arm.get("failure_rate")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="r3p")
    ap.add_argument("--out", default="r3_ablation_results.json")
    args = ap.parse_args()

    arms = load_arms(args.prefix)
    if not arms:
        raise SystemExit(f"No worker outputs matching {args.prefix}_*_ablation_results.json")
    missing = [a for a in ("baseline", "reupload", "mixed_channel", "with_ancilla") if a not in arms]
    if missing:
        print(f"WARNING: workers still running or failed for: {missing}")
    if "baseline" not in arms:
        raise SystemExit("baseline arm is required for comparisons")

    base = arms["baseline"]
    comparisons, decisions = [], {}
    for name, arm in arms.items():
        if name == "baseline":
            continue
        c = pc.compare(base, arm)
        comparisons.append(c)
        decisions[name] = verdict_for(name, c, arm)

    pc.print_arms(
        f"R3 (combined): {pc.ARCH['encoding']}+{pc.ARCH['ansatz']}, " f"1024/30, scored on last-{pc.SCORE_LAST_K} mean",
        list(arms.values()),
    )
    print(f"\n{'arm':<20}{'seeds':>7}{'failures':>11}{'converged-only':>17}")
    for name, a in arms.items():
        co = a.get("score_mean_converged_only")
        print(
            f"{name:<20}{a['n_seeds']:>7}{a['n_failed']:>4}/{a['n_seeds']:<6}"
            f"{(f'{co:.1f}' if co is not None else 'n/a'):>17}"
        )
    pc.print_comparisons(comparisons)
    for name, d in decisions.items():
        print(f"\n[{name}] {d['question']}\n  {d['verdict']}")

    if any("noise_acc_mean" in a for a in arms.values()):
        levels = next(a["noise_levels"] for a in arms.values() if "noise_levels" in a)
        print(f"\n{'noise sweep (mean over swept seeds)':<26}" + "".join(f"{p:>8}" for p in levels))
        for name, a in arms.items():
            if "noise_acc_mean" in a:
                print(f"{name:<26}" + "".join(f"{v:>8.1f}" for v in a["noise_acc_mean"]))

    pc.save(
        {
            "architecture": pc.ARCH,
            "protocol": pc.PROTOCOL,
            "score_metric": f"mean of last {pc.SCORE_LAST_K} epochs",
            "comparison_type": "unpaired (cross-variant seed correlation measured at -0.22 in R1b)",
            "source": f"parallel per-arm workers, prefix {args.prefix}",
            "arms": arms,
            "comparisons": comparisons,
            "decisions": decisions,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
