"""Task R7 (roadmap Section 7): re-baseline on the coherent quantum tree.

Everything from R1 to R4 was measured on a measure-and-re-encode hybrid: each
node its own 4-qubit device, level-1 nodes returning a single <Z>, and those
classical scalars re-encoded as rotation angles into a separate root circuit.
That model has zero entanglement across tree levels and an inter-level bond of
one real number. See research_log.md 2026-07-28 "Code Audit #2".

This re-runs on `CoherentQTTNClassifier` -- one device, survivors passed as
qubits, a single measurement at the end.

Scope, per the 2026-07-28 decision to test only what was promising:
  baseline -- the architecture itself. Essential.
  reupload -- same wire count (16) so nearly free, and R3 returned a genuine
              null for it (+1.5 pts inside a 3.8-pt limit) rather than a clear
              rejection, so it is the only variant worth confirming.
  NOT ported: mixed_channel (-10.9 pts) and the spatial ancilla (-19.5 pts).
              Both were rejected decisively at node level and each ancilla
              doubles the statevector (20-21 wires => 16-32x cost). Paying that
              to re-confirm a rejection is poor value.

Cost: ~17.5 min per 30-epoch run on lightning.qubit with adjoint, against ~15s
for the hybrid, because the hybrid only ever simulated 4 qubits at a time. Run
one worker per (arm, seed-shard); checkpoints after every seed.

Pilot mode (--pilot) trains a few seeds and reports where the validation curve
plateaus, so the epoch budget can be cut before spending the full seed budget.

Run (pilot):  python -m qnlp.image_tower.classification.quantum.run_r7_coherent --pilot --seeds 0 1 2
Run (full):   python -m qnlp.image_tower.classification.quantum.run_r7_coherent --arm baseline --seeds 0 1 2 3 4
"""

import argparse

import numpy as np

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import CoherentQTTNClassifier

ARMS = ("baseline", "reupload")


def plateau_epoch(curve, tol=1.0):
    """First epoch after which the curve never improves by more than `tol`
    points over its running best. Used to decide whether 30 epochs are needed."""
    best, plateau = -1e9, len(curve)
    for i, v in enumerate(curve):
        if v > best + tol:
            best, plateau = v, i + 1
    return plateau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="baseline", choices=ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=pc.PROTOCOL["epochs"])
    ap.add_argument(
        "--shared-l1",
        action="store_true",
        help="Share level-1 weights across the four blocks (211 params). Default is per-block "
        "(287 params), which is what train_synthetic_shapes.py used for its 75%% result and what "
        "the R7 readout diagnostic used.",
    )
    ap.add_argument("--pilot", action="store_true", help="Report plateau analysis instead of a full comparison.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = args.arm
    out = args.out or f"r7_{mode}"
    print(
        f"R7 coherent tree | arm={mode} | seeds={args.seeds} | epochs={args.epochs}\n"
        f"architecture={pc.COHERENT_ARCH} | l1_weights={'shared' if args.shared_l1 else 'per-block'}"
        f" | ~17.5 min/run at 30 epochs",
        flush=True,
    )

    curves = []
    for s in args.seeds:
        curve, model = pc.train_run(
            lambda: CoherentQTTNClassifier(**pc.COHERENT_ARCH, mode=mode, share_level1_weights=args.shared_l1),
            seed=s,
            epochs=args.epochs,
        )
        curves.append(curve)
        score = sum(curve[-pc.SCORE_LAST_K :]) / pc.SCORE_LAST_K
        # Sanity check the tree is still coherent for this trained model, not
        # just at initialisation.
        import torch

        bloch = model.survivor_bloch_length(torch.rand(8, 3, 16, 16))
        print(
            f"  seed {s:>2}: score={score:.1f}%  final={curve[-1]:.1f}%  "
            f"plateau@epoch{plateau_epoch(curve):>3}  survivor|r|={bloch:.4f}",
            flush=True,
        )
        pc.save(
            {"arm": mode, "partial": True, "seeds_done": len(curves), "val_acc_curves_per_seed": curves},
            f"{out}_partial.json",
        )

    arm = pc.summarise(
        mode,
        curves,
        coherent=True,
        epochs=args.epochs,
        num_params=sum(
            p.numel()
            for p in CoherentQTTNClassifier(
                **pc.COHERENT_ARCH, mode=mode, share_level1_weights=args.shared_l1
            ).parameters()
        ),
    )
    # The chance-level guard is a real failure for a full run, but a 2-3 epoch
    # diagnostic legitimately sits at chance, so warn rather than abort there.
    if args.epochs >= 10:
        pc.assert_not_chance_level(mode, arm["score_mean"])
    elif pc.is_chance_level(arm["score_mean"]):
        print(f"\nNOTE: {arm['score_mean']:.1f}% is at chance, expected for a {args.epochs}-epoch diagnostic run.")

    pc.print_arms(f"R7 coherent tree: {mode}, {args.epochs} epochs, {len(args.seeds)} seeds", [arm])

    if args.pilot:
        plateaus = [plateau_epoch(c) for c in curves]
        curve_mean = np.array(curves).mean(axis=0)
        print(f"\nPlateau epochs per seed: {plateaus}  (median {int(np.median(plateaus))})")
        print("Mean validation curve by epoch:")
        for i in range(0, len(curve_mean), 5):
            print(
                f"  epoch {i+1:>2}-{min(i+5, len(curve_mean)):>2}: "
                + " ".join(f"{v:5.1f}" for v in curve_mean[i : i + 5])
            )
        rec = int(np.median(plateaus))
        print(
            f"\nRECOMMENDATION: the curve stops improving by ~epoch {rec}. "
            + (
                f"Cutting the budget from {args.epochs} to ~{min(args.epochs, rec + 3)} epochs would save "
                f"~{100 * (1 - min(args.epochs, rec + 3) / args.epochs):.0f}% of the remaining compute."
                if rec + 3 < args.epochs
                else "The full epoch budget is being used; do not cut it."
            )
        )
        arm["plateau_epochs"] = plateaus

    pc.save(
        {
            "architecture": pc.COHERENT_ARCH,
            "shared_l1": args.shared_l1,
            "coherent": True,
            "epochs": args.epochs,
            "arm": arm,
        },
        f"{out}_results.json",
    )


if __name__ == "__main__":
    main()
