"""Task C2: is the readout wide enough to carry four attributes at once?

PREDICTED BOTTLENECK, TESTED EXPLICITLY. The tower currently emits four numbers
(`top_layer_qubits`: <Z> on wires 0/4/8/12). CLEVR asks for four simultaneous
attributes spanning 8 x 3 x 2 x 2 = 96 combinations. Four real numbers feeding
four heads is very likely too narrow -- and R7 measured exactly this failure
mode, where widening the readout from 3 to 4 values was worth 43 points.

  top_layer_qubits       4 values   <Z> on each top-layer wire (current)
  top_layer_multi_pauli 12 values   <X>,<Y>,<Z> on each -- the full single-qubit
                                    information, at no extra wires and no extra
                                    gates, only more measurements

If 12 clearly beats 4, adopt it for C3/C4 and record that the BOND, not the
circuit, was again the binding constraint.

Cost: ~17.5 min per seed per arm on lightning.qubit. Run at most 4-5 concurrent
workers -- ten concurrent 16-qubit processes exhausted 18 GB in Phase 1 and six
were killed silently. Checkpoints after every seed.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.clevr.run_c2_readout --seeds 0 1 2 3 4
"""

import argparse

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import READOUT_DIM, CoherentQTTNClassifier

READOUTS = ("top_layer_qubits", "top_layer_multi_pauli")
PATCH_FOR = {16: 4, 32: 8, 64: 16}


def model_factory(readout, img_size, heads):
    arch = {**pc.COHERENT_ARCH, "readout": readout}
    return lambda: CoherentQTTNClassifier(
        **arch,
        img_size=img_size,
        patch_size=PATCH_FOR[img_size],
        n_classes=heads,
        share_level1_weights=False,
    )


def readout_verdict(narrow, wide, cmps, heads, params, n_seeds):
    """Phrase the C2 outcome. Shared with `combine_clevr` so the sharded and
    single-process paths cannot disagree about what the same numbers mean."""
    better = [h for h in heads if cmps[h]["resolved"] and cmps[h]["difference"] > 0]
    worse = [h for h in heads if cmps[h]["resolved"] and cmps[h]["difference"] < 0]
    if better and not worse:
        return (
            f"ADOPT {wide} for C3/C4. It resolves better on {better} at no extra wires and no "
            f"extra gates ({params[narrow]} -> {params[wide]} params, all of it in the classical "
            f"heads). The BOND, not the circuit, was again the binding constraint -- the same "
            f"finding as R7's +43 pts from widening the readout 3 -> 4."
        )
    if worse and not better:
        return (
            f"KEEP {narrow}. The wider readout is resolvably WORSE on {worse}. Check the parameter "
            f"counts before concluding anything about the bond: {wide} adds "
            f"{params[wide] - params[narrow]} head parameters, so this may be overfitting rather "
            f"than the readout width."
        )
    if better and worse:
        return (
            f"MIXED: {wide} is better on {better} and worse on {worse}. Do not pick a readout on "
            f"the mean across heads -- report per head and choose on the attributes the thesis "
            f"actually claims."
        )
    return (
        f"UNRESOLVED at {n_seeds} seeds: no head shows a resolved difference. Four values are NOT "
        f"demonstrably a bottleneck for this task, which is itself a reportable finding -- report "
        f"it with the resolution limits, not as evidence the readout is adequate in general. "
        f"KEEP {narrow} (fewer parameters, Phase-1-validated) unless seeds_needed shows a "
        f"plausible effect within reach."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=16, choices=sorted(PATCH_FOR), help="Use the C1-chosen resolution.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=cc.CLEVR_PROTOCOL["epochs"])
    ap.add_argument("--readouts", nargs="+", default=list(READOUTS), choices=list(READOUTS))
    ap.add_argument("--out", default="c2_readout")
    args = ap.parse_args()

    heads = cc.HEADS
    floors = cc.majority_baselines(task="objects", img_size=args.img_size, seed=args.seeds[0])
    print(
        f"C2 readout width | {args.img_size}x{args.img_size} | seeds={args.seeds} | epochs={args.epochs}\n"
        f"~62 min/seed/arm at 30 epochs (MEASURED on CLEVR 2026-07-30; Phase 1's ~17.5 min figure "
        f"was synthetic shapes and does not carry over). Run at most 4-5 workers concurrently.",
        flush=True,
    )

    arms_by_name, params_by_name = {}, {}
    for readout in args.readouts:
        factory = model_factory(readout, args.img_size, heads)
        n_params = sum(p.numel() for p in factory().parameters())
        params_by_name[readout] = n_params
        print(f"\n--- {readout} ({n_params} params, readout width {READOUT_DIM[readout]}) ---", flush=True)

        curves = []
        for s in args.seeds:
            c, _ = cc.train_run_multihead(factory, seed=s, task="objects", img_size=args.img_size, epochs=args.epochs)
            curves.append(c)
            print(f"  seed {s:>2}: " + "  ".join(f"{h}={cc.score_of(c[h]):.1f}%" for h in heads), flush=True)
            # Checkpoint per seed: a killed worker must cost one seed, not all of them.
            cc.save(
                {"readout": readout, "partial": True, "seeds_done": len(curves), "curves": curves},
                f"{args.out}_{readout}_partial.json",
            )
        arms_by_name[readout] = cc.summarise_heads(
            readout, curves, heads, num_params=n_params, img_size=args.img_size, epochs=args.epochs
        )

    cc.print_head_table(
        f"C2: readout width on single-object CLEVR, {args.img_size}x{args.img_size}, {len(args.seeds)} seeds",
        arms_by_name,
        heads,
        floors=floors,
    )

    result = {
        "task": "objects",
        "img_size": args.img_size,
        "protocol": {**cc.CLEVR_PROTOCOL, "epochs": args.epochs},
        "seeds": args.seeds,
        "majority_floors": floors,
        "num_params": params_by_name,
        "arms": arms_by_name,
    }

    if len(args.readouts) == 2:
        narrow, wide = args.readouts
        cmps = cc.compare_heads(arms_by_name[narrow], arms_by_name[wide], heads)
        cc.print_head_comparisons(f"{narrow} -> {wide}", cmps, heads)
        result["comparisons"] = cmps

        verdict = readout_verdict(narrow, wide, cmps, heads, params_by_name, len(args.seeds))
        print(f"\nVERDICT: {verdict}")
        result["verdict"] = verdict

    cc.save(result, f"{args.out}_results.json")


if __name__ == "__main__":
    main()
