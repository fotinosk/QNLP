"""Merge sharded CLEVR workers into one result set and apply the comparisons.

A CLEVR quantum seed costs ~62 min (measured 2026-07-30), so C2/C3/C4 are run as
one worker per seed rather than one long sequential process. Each worker
therefore sees ONE seed and cannot compute anything cross-arm: with n=1 the
variance is unmeasured and `pc.compare` correctly refuses to resolve anything.
**This script is where the actual result is produced.**

Generic over C2/C3/C4 so the three tasks do not each grow their own combiner --
forked analysis is the same drift that forked training loops caused in Phase 1.

Verdict wording is imported from the runners, not re-written here, so the
sharded and single-process paths cannot disagree about what the same numbers
mean.

Run:
  python -m qnlp.image_tower.classification.clevr.combine_clevr \
      --prefix c2 --seeds 0 1 2 --arms top_layer_qubits top_layer_multi_pauli
"""

import argparse
import json
import os

import numpy as np

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.clevr.run_c2_readout import READOUTS, readout_verdict
from qnlp.image_tower.classification.quantum import phase15_common as pc


def load_curves(prefix, arm, seeds):
    """Collect one seed's curves from each worker's checkpoint.

    Workers checkpoint after every seed as `{out}_{arm}_partial.json` with
    `curves` = a list of {head: [per-epoch accuracy]}. Sharded one seed per
    worker, each file holds exactly one entry.
    """
    curves, missing, found_params = [], [], 0
    for s in seeds:
        path = os.path.join(pc.RESULTS_DIR, f"{prefix}_s{s}_{arm}_partial.json")
        if not os.path.exists(path):
            missing.append(s)
            continue
        with open(path) as f:
            payload = json.load(f)
        curves.extend(payload["curves"])
        # Written by the runners since 2026-08-02. Older checkpoints lack it and
        # fall back to --params, then to 0.
        if payload.get("num_params"):
            found_params = payload["num_params"]
    return curves, missing, found_params


def collapsed_seeds(curves, heads, margin=2.5):
    """Which seeds failed to train at all?

    A seed whose EVERY head finished at that head's own chance floor did not
    learn anything; averaging it into the arm hides a training failure inside a
    plausible-looking mean and inflates the variance, which then makes every
    comparison unresolvable. Phase 1 reported these explicitly ("1 of 3 seeds hit
    a dead-gradient trap", "0/30 failures") and CLEVR must too -- the arm mean
    alone cannot show it, because two healthy seeds pull the average clear of
    chance.

    Reported, never silently dropped: a collapse rate IS the result when it
    differs between arms.

    TWO FAILURE MODES, AND THEY NEED DIFFERENT FIXES. A seed that ends at chance
    may never have left it, or may have LEARNED THE TASK AND THEN LOST IT. C4
    (2026-08-02) had both: of `quantum_none`'s 7 collapses, 3 peaked at 35-50%
    mid-run -- one reached 50.2% at epoch 12, above the classical baseline's mean
    -- then fell back to ~24% for the last 10 epochs, while 4 never moved at all.
    Pooled into one number they look like a single "unstable optimiser" problem;
    split, the first group says the learning rate is too high (or wants a
    schedule) and the second says initialisation. Returns `diverged` per seed.
    """
    out = []
    for i, c in enumerate(curves):
        scores = {h: float(np.mean(c[h][-pc.SCORE_LAST_K :])) for h in heads}
        if all(pc.is_chance_level(scores[h], n_classes=k, margin=margin) for h, k in heads.items()):
            peaks = {h: float(np.max(c[h])) for h in heads}
            # "Diverged" = some head got clear of its own chance rate at some
            # point and did not stay there. The 1.5x factor keeps ordinary
            # epoch-to-epoch noise around the floor from being read as learning.
            diverged = any(peaks[h] > 1.5 * cc.chance_rate(k) for h, k in heads.items())
            out.append({"index": i, "scores": scores, "peaks": peaks, "diverged": diverged})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="e.g. c2 -- workers wrote c2_s0_*, c2_s1_*, ...")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="Arm names, baseline FIRST.")
    ap.add_argument("--task", default="objects", choices=["objects", "relations"])
    ap.add_argument("--img-size", type=int, default=16)
    ap.add_argument("--params", type=int, nargs="+", default=None, help="Parameter count per arm, in --arms order.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    heads = cc.TASK_HEADS[args.task]
    floors = cc.majority_baselines(task=args.task, img_size=args.img_size, seed=args.seeds[0])
    params = dict(zip(args.arms, args.params)) if args.params else {}

    arms_by_name, collapse = {}, {}
    for arm in args.arms:
        curves, missing, ckpt_params = load_curves(args.prefix, arm, args.seeds)
        if missing:
            print(f"WARNING: {arm}: no checkpoint for seed(s) {missing} -- combining {len(curves)} seeds.")
        if not curves:
            raise SystemExit(f"No checkpoints found for arm {arm!r} under prefix {args.prefix!r}.")
        dead = collapsed_seeds(curves, heads)
        collapse[arm] = {
            "n_collapsed": len(dead),
            "n_seeds": len(curves),
            "seed_index": [d["index"] for d in dead],
            "n_diverged": sum(d["diverged"] for d in dead),
            "diverged_index": [d["index"] for d in dead if d["diverged"]],
            "peak_of_collapsed": {d["index"]: round(max(d["peaks"].values()), 1) for d in dead},
        }
        arms_by_name[arm] = cc.summarise_heads(
            arm,
            curves,
            heads,
            # The checkpoint's own count wins over --params: it cannot be typed in wrong.
            num_params=ckpt_params or params.get(arm, 0),
            img_size=args.img_size,
            collapse=collapse[arm],
        )

    cc.print_head_table(
        f"{args.prefix.upper()}: {args.task}, {args.img_size}x{args.img_size}, "
        f"{arms_by_name[args.arms[0]][list(heads)[0]]['n_seeds']} seeds",
        arms_by_name,
        heads,
        floors=floors,
    )

    # Per-seed training collapses. Printed before anything else is interpreted,
    # because a differing collapse rate invalidates a comparison of the means.
    if any(v["n_collapsed"] for v in collapse.values()):
        print("\n⚠️  TRAINING COLLAPSES (every head at its own chance floor):")
        for arm, v in collapse.items():
            split = (
                f"  [{v['n_diverged']} diverged after learning, {v['n_collapsed'] - v['n_diverged']} never left chance]"
            )
            print(f"    {arm:<24} {v['n_collapsed']}/{v['n_seeds']} seeds collapsed  {v['seed_index'] or ''}{split}")
            if v["n_diverged"]:
                print(f"      peak reached by each collapsed seed: {v['peak_of_collapsed']}")
        print(
            "    A collapsed seed drags the mean down and inflates the variance, so comparisons below\n"
            "    may be unresolvable for reasons that have nothing to do with the arms. If the rate\n"
            "    DIFFERS between arms that is itself the finding (optimisation stability, not\n"
            "    accuracy); if it is similar in both, fix the optimiser before reading the means.\n"
            "    THE SPLIT MATTERS MORE THAN THE TOTAL: a seed that reached 50% and fell back to\n"
            "    chance is a learning-rate/schedule problem, one that never moved is initialisation.\n"
            "    Report the converged-seed subgroup as a DIAGNOSTIC, never as an arm -- it is\n"
            "    post-hoc conditioning (Phase 1 precedent: R3's 'converged only: 55.6 +/- 5.2')."
        )

    # A head at ITS OWN chance floor is a broken run, not a result. Warn rather
    # than abort: a combiner that refuses to print is worse than one that flags.
    for arm, summary in arms_by_name.items():
        try:
            cc.assert_not_chance_level_per_head(arm, {h: summary[h]["score_mean"] for h in heads}, heads)
        except AssertionError as e:
            print(f"\n⚠️  {e}")

    result = {
        "prefix": args.prefix,
        "task": args.task,
        "img_size": args.img_size,
        "seeds": args.seeds,
        "majority_floors": floors,
        "num_params": params,
        "arms": arms_by_name,
        "comparisons": {},
    }

    baseline = args.arms[0]
    for arm in args.arms[1:]:
        cmps = cc.compare_heads(arms_by_name[baseline], arms_by_name[arm], heads)
        cc.print_head_comparisons(f"{baseline} -> {arm}", cmps, heads)
        result["comparisons"][arm] = cmps

        # Report what this design could actually have detected, and what it would
        # take. A null without this is not a finding (standing requirement 2).
        print(f"{'':<12}{'seeds needed for the observed difference':>50}")
        for h in heads:
            c = cmps[h]
            need = pc.seeds_needed(c["pooled_std"], max(abs(c["difference"]), 0.5))
            print(
                f"  {h:<10} observed {c['difference']:>+6.1f}  resolves {c['min_detectable_effect']:>5.1f}  "
                f"-> would need ~{need} seeds/arm"
            )

        # Gate the C2 verdict on the ARMS being the two readouts, not on the
        # prefix string -- a naming convention is not a reliable signal, and a
        # test prefix silently suppressed the verdict entirely.
        if {baseline, arm} <= set(READOUTS):
            n = arms_by_name[baseline][list(heads)[0]]["n_seeds"]
            verdict = readout_verdict(baseline, arm, cmps, heads, params or {baseline: 0, arm: 0}, n)
            print(f"\nVERDICT: {verdict}")
            result["verdict"] = verdict

    cc.save(result, args.out or f"{args.prefix}_combined_results.json")


if __name__ == "__main__":
    main()
