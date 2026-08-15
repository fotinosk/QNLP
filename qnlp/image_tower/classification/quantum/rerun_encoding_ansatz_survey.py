"""Re-run the encoding x ansatz survey multi-seed, noiseless only.

The original survey (`benchmark_encodings_ansatze.py`) ran 4 encodings x 3
ansatze at 8x8 on a single 4-qubit node with ONE seed per config, and produced
only a PNG -- no JSON. Two things rest on it and neither is currently
supportable from data on disk:

  1. The thesis (S4.3.1) quotes multi_axis+IQP at 95.3% as the leading
     candidate, but `run_r2_ansatz.py:8-10` records that multi_axis+HEA scored
     96.9% in the same sweep -- i.e. HEA led and IQP was adopted on
     parameter-efficiency grounds. The figure shows both bars.
  2. The ALT ansatz appears in the survey and then never again; there is no
     record of how it placed.

This re-runs the same grid on the project's scoring convention (per-epoch
curves, `pc.summarise` -> mean of last SCORE_LAST_K epochs) so the figure can be
regenerated from data and both points can be stated factually.

Scope: noiseless only. The noise axis was scoped out of the thesis, and the
original noisy sweep is single-node anyway.

Run:
    conda run -n qnlp python -m \
        qnlp.image_tower.classification.quantum.rerun_encoding_ansatz_survey \
        --seeds-per-arm 10
"""

import argparse
import itertools
import json
import os
import time

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.benchmark_encodings_ansatze import (
    ScopedQTTNClassifier,
    get_resource_complexity,
)

ENCODINGS = ("angle", "multi_axis", "amplitude", "zz_map")
ANSATZE = ("hea", "iqp", "alt")

# The original survey's protocol, preserved so the re-run is comparable to the
# figure it replaces: 8x8 crops, a single 4-qubit node, 512/128, 8 epochs.
PROTOCOL = {
    "train_samples": 512,
    "test_samples": 128,
    "img_size": 8,
    "epochs": 8,
    "batch_size": 32,
    "lr": 0.03,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-arm", type=int, default=10)
    ap.add_argument("--out", default="encoding_ansatz_survey_rerun_results.json")
    args = ap.parse_args()

    arms = []
    t_start = time.time()

    for enc, ans in itertools.product(ENCODINGS, ANSATZE):
        name = f"{enc}+{ans}"
        qubits, params = get_resource_complexity(enc, ans)
        t0 = time.time()

        curves = []
        for seed in range(args.seeds_per_arm):
            curve, _ = pc.train_run(
                lambda e=enc, a=ans: ScopedQTTNClassifier(encoding=e, ansatz_type=a, mode="noiseless"),
                seed,
                **PROTOCOL,
            )
            curves.append(curve)

        arm = pc.summarise(name, curves, encoding=enc, ansatz=ans, num_params=params, qubits=qubits)
        arms.append(arm)
        print(
            f"{name:<24} score {arm['score_mean']:5.1f} +/-{arm['score_std']:4.1f}   "
            f"final {arm['final_val_acc_mean']:5.1f}   {params:>2}p {qubits}q   "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    pc.print_arms(
        f"Encoding x ansatz survey re-run @ 8x8 single node, "
        f"{PROTOCOL['train_samples']}/{PROTOCOL['epochs']}, {args.seeds_per_arm} seeds",
        arms,
    )

    # Every pairwise comparison, so the ansatz question can be read off directly.
    comparisons = []
    for a, b in itertools.combinations(arms, 2):
        diff = a["score_mean"] - b["score_mean"]
        limit = pc.mde_unpaired(a["score_std"], b["score_std"], a["n_seeds"], b["n_seeds"])
        comparisons.append(
            {
                "a": a["config"],
                "b": b["config"],
                "diff": float(diff),
                "limit": float(limit),
                "resolved": bool(abs(diff) > limit),
            }
        )

    print(f"\n{'ansatz comparisons at fixed encoding':<52}{'diff':>9}{'limit':>9}  verdict")
    for enc in ENCODINGS:
        for x, y in itertools.combinations(ANSATZE, 2):
            c = next(c for c in comparisons if {c["a"], c["b"]} == {f"{enc}+{x}", f"{enc}+{y}"})
            verdict = "RESOLVED" if c["resolved"] else "unresolved"
            print(f"{c['a'] + ' vs ' + c['b']:<52}{c['diff']:>+9.1f}{c['limit']:>9.1f}  {verdict}")

    out_path = os.path.join(pc.RESULTS_DIR, args.out)
    with open(out_path, "w") as f:
        json.dump(
            {
                "task": "encoding_ansatz_survey_rerun",
                "scope": "noiseless only; single 4-qubit node; 8x8",
                "protocol": PROTOCOL,
                "seeds_per_arm": args.seeds_per_arm,
                "score_last_k": pc.SCORE_LAST_K,
                "arms": arms,
                "comparisons": comparisons,
                "wall_seconds": time.time() - t_start,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}  [{time.time() - t_start:.0f}s total]")


if __name__ == "__main__":
    main()
