"""Task R2 (roadmap Section 7): reconcile the ansatz-of-record with the code.

research_log.md's Current Status and the 2026-07-17 encoding/ansatz benchmark
both state the selected architecture is Multi-Axis Encoding + IQP. The
July-26/27 scripts actually ran RY + StronglyEntanglingLayers. Every recent
architecture decision was therefore made on a different circuit than the one
the project says it selected.

The original selection also came from a 12-config sweep at 8x8 with one seed
per config, choosing Multi-Axis+IQP (95.3%) over Multi-Axis+HEA (96.9%) on
parameter-efficiency grounds -- a weak basis for a decision the thesis rests on.

This re-runs {scalar_ry, multi_axis} x {strongly_entangling, iqp} at 16x16 on
R1's protocol of record (1024 train / 30 epochs), with readout fixed at
root_multi_pauli (R1/R1b winner).

Seeds: 21 by default, which per R1b's power analysis resolves ~5-point
differences. That is the right granularity here -- R2 is choosing between
architectures, and a difference smaller than 5 points would not justify
overriding the documented choice either way. Use --seeds-per-arm 58 for
3-point resolution if the result lands ambiguous.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r2_ansatz
"""

import argparse
import itertools

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import HierarchicalQTTNClassifier

READOUT = "root_multi_pauli"
ENCODINGS = ("scalar_ry", "multi_axis")
ANSATZE = ("strongly_entangling", "iqp")

# What the documentation currently claims is the architecture of record.
DOCUMENTED = ("multi_axis", "iqp")
# What the July-26/27 code actually ran.
AS_CODED = ("scalar_ry", "strongly_entangling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-arm", type=int, default=21)
    args = ap.parse_args()
    seeds = range(args.seeds_per_arm)

    arms = []
    for encoding, ansatz in itertools.product(ENCODINGS, ANSATZE):
        name = f"{encoding}+{ansatz}"
        print(f"\n--- {name} ({args.seeds_per_arm} seeds) ---", flush=True)
        curves = []
        for s in seeds:
            c, model = pc.train_run(
                lambda: HierarchicalQTTNClassifier(readout=READOUT, encoding=encoding, ansatz=ansatz),
                seed=s,
            )
            curves.append(c)
            print(f"  seed {s:>2}: last5={sum(c[-5:])/5:.1f}%  final={c[-1]:.1f}%", flush=True)
        n_params = sum(
            p.numel()
            for p in HierarchicalQTTNClassifier(readout=READOUT, encoding=encoding, ansatz=ansatz).parameters()
        )
        arms.append(pc.summarise(name, curves, encoding=encoding, ansatz=ansatz, num_params=n_params))

    by_name = {a["config"]: a for a in arms}
    best = max(arms, key=lambda a: a["score_mean"])
    documented = by_name[f"{DOCUMENTED[0]}+{DOCUMENTED[1]}"]
    as_coded = by_name[f"{AS_CODED[0]}+{AS_CODED[1]}"]

    cmps = [pc.compare(as_coded, documented)]
    cmps += [pc.compare(a, best) for a in arms if a["config"] != best["config"]]

    pc.print_arms(f"R2: encoding x ansatz @ 16x16, readout={READOUT}, " f"1024/30, {args.seeds_per_arm} seeds", arms)
    pc.print_comparisons(cmps)

    doc_vs_best = pc.compare(documented, best)
    if best["config"] == documented["config"]:
        decision = (
            "CONFIRM_DOCUMENTED",
            f"{documented['config']} is the best arm. The documented choice "
            "(Multi-Axis + IQP) is confirmed at 16x16 -- port it into qttn_core "
            "as the default and update the July-26/27 scripts, which ran "
            "scalar_ry+strongly_entangling.",
        )
    elif not doc_vs_best["resolved"]:
        decision = (
            "KEEP_DOCUMENTED_UNRESOLVED",
            f"{best['config']} scored highest but its edge over the documented "
            f"{documented['config']} ({doc_vs_best['difference']:+.1f} pts) is inside "
            f"the {doc_vs_best['min_detectable_effect']:.1f}-pt resolution limit. No "
            "evidence to override the documented choice -- keep Multi-Axis + IQP and "
            "record that the alternatives are statistically tied.",
        )
    else:
        decision = (
            "UPDATE_DOCUMENTATION",
            f"{best['config']} beats the documented {documented['config']} by "
            f"{doc_vs_best['difference']:+.1f} pts, above the "
            f"{doc_vs_best['min_detectable_effect']:.1f}-pt resolution limit. Update the "
            "stated architecture of record in research_log.md Current Status and the "
            "roadmap -- do not leave documentation and code disagreeing.",
        )

    print(
        f"\nBEST ARM: {best['config']} ({best['score_mean']:.1f} +/- {best['score_std']:.1f}, "
        f"{best['num_params']} params)"
    )
    print(f"DECISION: {decision[0]}\n{decision[1]}")

    pc.save(
        {
            "readout": READOUT,
            "seeds_per_arm": args.seeds_per_arm,
            "protocol": pc.PROTOCOL,
            "arms": arms,
            "comparisons": cmps,
            "best": best["config"],
            "documented_choice": f"{DOCUMENTED[0]}+{DOCUMENTED[1]}",
            "as_coded_july": f"{AS_CODED[0]}+{AS_CODED[1]}",
            "documented_vs_best": doc_vs_best,
            "decision": {"branch": decision[0], "explanation": decision[1]},
        },
        "r2_ansatz_results.json",
    )


if __name__ == "__main__":
    main()
