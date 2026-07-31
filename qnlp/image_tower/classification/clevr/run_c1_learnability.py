"""Task C1: the learnability gate. Run this BEFORE any quantum run.

The single highest-value experiment in Phase 2, and it costs minutes.

Train ONLY the classical MLP reference on every attribute at every resolution.
If an MLP cannot learn an attribute at a given resolution, no quantum model will,
and a quantum null there measures the DATA, not the architecture. That is the R4
lesson applied before spending compute rather than after.

The output table settles three things at once:
  * which resolution to use  -- this IS the deferred C5 decision, made against
    real data rather than in the abstract;
  * which attributes are in scope -- drop any that are unlearnable even
    classically, and say in the thesis that resolution, not architecture, is the
    limit;
  * the ceiling every later quantum result should be read against.

GATE: do not start C2 or C3 until this table exists.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.clevr.run_c1_learnability
"""

import argparse

import numpy as np

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.quantum.run_r4_classical_control import MLPReference

PATCH_FOR = {16: 4, 32: 8, 64: 16}  # keeps the 4x4 patch grid at every resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="objects", choices=["objects", "relations"])
    ap.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=16, help="MLP width. 16 is R4's `mlp_reference`.")
    ap.add_argument("--epochs", type=int, default=cc.CLEVR_PROTOCOL["epochs"])
    ap.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03],
        help="Learning rates to sweep PER RESOLUTION. Do not collapse this to one value -- see below.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    heads = cc.TASK_HEADS[args.task]
    print(
        f"C1 learnability gate | task={args.task} | resolutions={args.resolutions} | "
        f"seeds={args.seeds} | MLP hidden={args.hidden}\n"
        f"Classical only -- this measures the DATA, not the architecture.",
        flush=True,
    )

    results = {}
    for res in args.resolutions:
        floors = cc.majority_baselines(task=args.task, img_size=res, seed=args.seeds[0])
        make = lambda: MLPReference(  # noqa: E731
            hidden=args.hidden, img_size=res, patch_size=PATCH_FOR[res], n_classes=heads
        )
        n_params = sum(p.numel() for p in make().parameters())

        # SWEEP THE LEARNING RATE PER RESOLUTION. This is not optional tuning.
        # The first version of this script hardcoded lr=0.01 and reported that
        # 64x64 was unlearnable on ALL FOUR heads -- every score exactly at its
        # majority-class floor. At lr=0.003 the same model scores 91.9 / 75.6 /
        # 77.2 / 97.5. The "resolution limit" was an optimisation artifact, and
        # 32x32's huge seed variance (+/-17.9) was the same effect with only
        # some seeds collapsing.
        #
        # This is Phase 1's standing requirement 4 ("sweep, never derive, any
        # capacity parameter") and the R4 lesson: an untuned baseline is not a
        # measurement. It matters doubly here, because C1 GATES everything --
        # a false "unlearnable" would drop a head from the thesis on the
        # strength of one learning rate.
        best = None
        for lr in args.lrs:
            curves = []
            for s in args.seeds:
                c, _ = cc.train_run_multihead(make, seed=s, task=args.task, img_size=res, epochs=args.epochs, lr=lr)
                curves.append(c)
            mean = float(np.mean([[cc.score_of(c[h]) for h in heads] for c in curves]))
            floor_mean = float(np.mean([floors[h] for h in heads]))
            print(
                f"  {res}x{res} lr={lr:<6}: mean over heads {mean:5.1f}% "
                f"(floor {floor_mean:.1f}%)" + ("  <-- best so far" if best is None or mean > best[0] else ""),
                flush=True,
            )
            if best is None or mean > best[0]:
                best = (mean, lr, curves)
        _, best_lr, curves = best
        for s, c in zip(args.seeds, curves):
            print(
                f"    {res}x{res} lr={best_lr} seed {s}: " + "  ".join(f"{h}={cc.score_of(c[h]):.1f}%" for h in heads),
                flush=True,
            )

        arms = cc.summarise_heads(f"mlp_{res}", curves, heads, num_params=n_params, img_size=res, lr=best_lr)
        results[res] = {"arms": arms, "majority_floors": floors, "num_params": n_params, "lr": best_lr}

    print(f"\n{'=' * 78}\nC1: MLP reference accuracy by resolution and attribute\n{'=' * 78}")
    print(f"{'resolution':<12}{'params':>8}{'lr':>7}  " + "".join(f"{h:>16}" for h in heads))
    for res in args.resolutions:
        r = results[res]
        cells = "".join(f"{r['arms'][h]['score_mean']:>10.1f} +/-{r['arms'][h]['score_std']:>4.1f}" for h in heads)
        print(f"{str(res) + 'x' + str(res):<12}{r['num_params']:>8}{r['lr']:>7}  {cells}")
    print(
        f"{'majority':<12}{'':>8}{'':>7}  "
        + "".join(f"{results[args.resolutions[0]]['majority_floors'][h]:>16.1f}" for h in heads)
    )
    print(f"{'chance':<12}{'':>8}{'':>7}  " + "".join(f"{cc.chance_rate(k):>16.1f}" for k in heads.values()))
    print(
        "\nNOTE: each row is the BEST learning rate from the sweep -- deliberately an optimistic "
        "ceiling.\nThe gate question is 'can a classical model learn this AT ALL', so an untuned "
        "collapse must not\nbe reported as a data limit. A single fixed lr made 64x64 look "
        "unlearnable on all four heads."
    )

    # Verdicts. "Learnable" means the MLP clears the majority-class floor by a
    # margin that this many seeds can actually resolve -- not merely "above
    # chance", which an unbalanced head can manage by predicting the prior.
    print("\nVerdicts (margin over the majority-class floor, vs the resolution limit):")
    verdicts = {}
    for res in args.resolutions:
        r = results[res]
        for h in heads:
            arm = r["arms"][h]
            margin = arm["score_mean"] - r["majority_floors"][h]
            limit = cc.mde_unpaired(arm["score_std"], arm["score_std"], arm["n_seeds"], arm["n_seeds"])
            # With one seed the variance is unmeasured, so the limit is infinite
            # and NOTHING can be called unlearnable. Say "undetermined" rather
            # than "NOT learnable" -- a 1-seed diagnostic that reports a negative
            # verdict is exactly the kind of over-claim R5 had to walk back.
            undetermined = arm["n_seeds"] < 2 or not np.isfinite(limit)
            ok = (not undetermined) and margin > limit
            verdicts[f"{res}_{h}"] = {
                "score": arm["score_mean"],
                "floor": r["majority_floors"][h],
                "margin": float(margin),
                "limit": float(limit),
                "learnable": bool(ok),
                "undetermined": bool(undetermined),
            }
            label = "UNDETERMINED (needs >=2 seeds)" if undetermined else ("LEARNABLE" if ok else "NOT learnable")
            print(
                f"  {res:>3}x{res:<3} {h:<9} {arm['score_mean']:>6.1f}%  floor {r['majority_floors'][h]:>5.1f}%  "
                f"margin {margin:>+6.1f} (resolves {limit:.1f})  -> {label}"
            )

    # Pick the SMALLEST resolution that is not resolvably worse than the best one
    # on any head -- not the argmax of the mean.
    #
    # Argmax-on-mean chooses on noise: 32x32 beat 16x16 by 0.4 pts averaged over
    # heads, against per-head resolution limits of 5-9 pts. Acting on that would
    # have moved the whole phase to a larger resolution for no measurable gain,
    # while incurring the C5 route-(a) caveat (the circuit still sees 16 qubits,
    # so the extra pixels are absorbed by the classical encoder) and a cost model
    # that was measured at 16x16. When resolutions tie, the smaller one is
    # strictly better: cheaper, already validated in Phase 1, and it needs no
    # caveat in the write-up.
    top = max(args.resolutions, key=lambda r: np.mean([results[r]["arms"][h]["score_mean"] for h in heads]))
    best = top
    for res in sorted(args.resolutions):
        if all(
            not cc.compare(results[res]["arms"][h], results[top]["arms"][h])["resolved"]
            or results[res]["arms"][h]["score_mean"] >= results[top]["arms"][h]["score_mean"]
            for h in heads
        ):
            best = res
            break
    if best != top:
        print(
            f"\nNOTE: {top}x{top} has the highest mean, but {best}x{best} is not resolvably worse on "
            f"any head. Choosing the smaller resolution -- cheaper, Phase-1-validated, and it avoids "
            f"the C5 route-(a) caveat. Do not read the mean difference as a real effect."
        )
    unlearnable = [
        h for h in heads if not verdicts[f"{best}_{h}"]["learnable"] and not verdicts[f"{best}_{h}"]["undetermined"]
    ]
    if any(verdicts[f"{best}_{h}"]["undetermined"] for h in heads):
        print("\nWARNING: some heads are UNDETERMINED (fewer than 2 seeds). This run cannot gate C2/C3.")
    print(
        f"\nRECOMMENDATION: run C2/C3 at {best}x{best} (patch_size={PATCH_FOR[best]}).\n"
        + (
            f"  DROP these heads, or report them as data-limited rather than architecture-limited: {unlearnable}\n"
            if unlearnable
            else "  All heads are learnable at this resolution; keep all four.\n"
        )
        + (
            "  NOTE this is C5 route (a): the circuit still sees 16 qubits, so the extra pixels are\n"
            "  absorbed by the classical patch encoder. That is RESOLUTION scaling, not quantum\n"
            "  scaling, and the thesis must say so.\n"
            if best != 16
            else "  16x16 suffices, so C5 closes with no scaling work needed.\n"
        )
    )

    cc.save(
        {
            "task": args.task,
            "protocol": {**cc.CLEVR_PROTOCOL, "epochs": args.epochs, "lr": 0.01},
            "seeds": args.seeds,
            "mlp_hidden": args.hidden,
            "by_resolution": {
                str(r): {
                    "num_params": results[r]["num_params"],
                    "majority_floors": results[r]["majority_floors"],
                    "arms": results[r]["arms"],
                }
                for r in args.resolutions
            },
            "verdicts": verdicts,
            "recommended_resolution": best,
            "unlearnable_at_recommended": unlearnable,
        },
        args.out or f"c1_learnability_{args.task}.json",
    )


if __name__ == "__main__":
    main()
