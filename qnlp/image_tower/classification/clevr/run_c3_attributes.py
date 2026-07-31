"""Task C3: single-object attribute classification, quantum vs classical.

Four heads off ONE shared readout. Five arms on identical splits and seeds:

  quantum_coherent    COHERENT_ARCH + the C2-chosen readout, per-block level-1
  classical_bare      CP quad-tree, no residual, no dropout -- the matched-
                      constraint analogue that answers Question A.3
  classical_full      CP + residual + dropout, i.e. what classical practice does
  mlp_reference       task-difficulty ceiling
  mlp_param_matched   size-matched MLP, so no "beats the MLP" claim has a size
                      confound hiding in it

PASS CRITERION (the roadmap's revised one): per-attribute accuracy within a
stated margin of the MLP reference, AT FEWER PARAMETERS. The old ">80% on all 4
heads" is not defensible once C1 shows an attribute is data-limited, and
parameter efficiency is the claim Phase 1 actually supports.

Rank is SWEPT, never derived. Deriving CP rank from a parameter budget silently
forced rank=1 in Phase 1 and produced three different "measurements" of the same
baseline (33.9 -> 56.7 -> 79.6).

Run:
  # classical arms are seconds; shard the quantum arm across workers
  ... run_c3_attributes --skip-quantum
  ... run_c3_attributes --only-quantum --seeds 0 1 --out-suffix _w0
"""

import argparse

import numpy as np

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import CoherentQTTNClassifier
from qnlp.image_tower.classification.quantum.run_r4_classical_control import (
    ClassicalTTNClassifier,
    MLPReference,
)

PATCH_FOR = {16: 4, 32: 8, 64: 16}


def tune_classical(
    use_residual,
    dropout_p,
    q_params,
    img_size,
    heads,
    task="objects",
    seeds=(0, 1, 2),
    lrs=(0.003, 0.01, 0.03, 0.1),
    bond_dims=(2, 4, 8),
    ranks=(1, 2, 4, 8, 16),
    max_param_ratio=1.5,
    epochs=10,
):
    """Fair-shot hyperparameter search for a classical arm, rank free.

    Mirrors `run_r4_classical_control.tune_classical` but scores on the mean over
    heads. Parameter matching is an upper BOUND, never a way of deriving rank --
    that is the Phase-1 correction this signature encodes.
    """
    budget = q_params * max_param_ratio
    ps = PATCH_FOR[img_size]
    best, considered = None, 0
    for lr in lrs:
        for bd in bond_dims:
            for rank in ranks:
                make = lambda: ClassicalTTNClassifier(  # noqa: E731
                    rank=rank,
                    bond_dim=bd,
                    img_size=img_size,
                    patch_size=ps,
                    n_classes=heads,
                    use_residual=use_residual,
                    dropout_p=dropout_p,
                )
                n_params = sum(p.numel() for p in make().parameters())
                if n_params > budget:
                    continue
                considered += 1
                scores = []
                for s in seeds:
                    c, _ = cc.train_run_multihead(make, seed=s, task=task, img_size=img_size, lr=lr, epochs=epochs)
                    scores.append(float(np.mean([cc.score_of(c[h]) for h in heads])))
                mean = float(np.mean(scores))
                if best is None or mean > best["score"]:
                    best = {"lr": lr, "bond_dim": bd, "rank": rank, "params": n_params, "score": mean}
    if best is None:
        raise RuntimeError(f"No classical config fits within {budget:.0f} params. Widen ranks/bond_dims.")
    print(
        f"  best: lr={best['lr']} bond_dim={best['bond_dim']} rank={best['rank']} "
        f"params={best['params']} ({best['score']:.1f}% mean over heads)  "
        f"[{considered} configs within budget {budget:.0f}]",
        flush=True,
    )
    if best["rank"] == 1:
        print(
            "  WARNING: winning CP rank is 1 (a single outer product) -- degenerate. The parameter "
            "budget is too tight for a fair comparison; do NOT treat this arm as representative.",
            flush=True,
        )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=16, choices=sorted(PATCH_FOR))
    ap.add_argument(
        "--readout",
        default="top_layer_multi_pauli",
        help="C2 (2026-07-31) chose the 12-value readout on stability grounds.",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--epochs", type=int, default=cc.CLEVR_PROTOCOL["epochs"])
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Quantum-arm learning rate. Defaults to PROTOCOL's 0.03, inherited from Phase 1 where "
        "the loss had ONE head rather than four summed. Task C3b sweeps this: the classical arms got "
        "a 60-config search while the quantum arm got a single lr, and that asymmetry is what makes "
        "the C3 colour result uninterpretable.",
    )
    ap.add_argument("--skip-quantum", action="store_true", help="Classical arms only; combine later.")
    ap.add_argument("--only-quantum", action="store_true", help="Quantum arm only, for sharding.")
    ap.add_argument(
        "--out",
        default="c3",
        help="Checkpoint prefix. Shard with --out c3_s0, c3_s1, ... so `combine_clevr --prefix c3` "
        "can find them: it globs `{prefix}_s{seed}_{arm}_partial.json`.",
    )
    ap.add_argument("--out-suffix", default="", help="Extra suffix on the final results file.")
    args = ap.parse_args()

    heads, img_size, ps = cc.HEADS, args.img_size, PATCH_FOR[args.img_size]
    floors = cc.majority_baselines(task="objects", img_size=img_size, seed=args.seeds[0])

    q_arch = {**pc.COHERENT_ARCH, "readout": args.readout}
    q_factory = lambda: CoherentQTTNClassifier(  # noqa: E731
        **q_arch, img_size=img_size, patch_size=ps, n_classes=heads, share_level1_weights=False
    )
    q_params = sum(p.numel() for p in q_factory().parameters())
    print(
        f"C3 attributes | {img_size}x{img_size} | readout={args.readout} | seeds={args.seeds}\n"
        f"quantum_coherent: {q_params} params, arch={q_arch}",
        flush=True,
    )

    q_lr = args.lr if args.lr is not None else pc.PROTOCOL["lr"]
    specs = {}
    if not args.skip_quantum:
        specs["quantum_coherent"] = (q_factory, q_lr, None)

    if not args.only_quantum:
        print("\nTuning classical_bare (rank swept freely):", flush=True)
        tuned_bare = tune_classical(False, 0.0, q_params, img_size, heads)
        print("Tuning classical_full:", flush=True)
        tuned_full = tune_classical(True, 0.1, q_params, img_size, heads)
        specs["classical_bare"] = (
            lambda: ClassicalTTNClassifier(
                rank=tuned_bare["rank"],
                bond_dim=tuned_bare["bond_dim"],
                img_size=img_size,
                patch_size=ps,
                n_classes=heads,
                use_residual=False,
                dropout_p=0.0,
            ),
            tuned_bare["lr"],
            tuned_bare,
        )
        specs["classical_full"] = (
            lambda: ClassicalTTNClassifier(
                rank=tuned_full["rank"],
                bond_dim=tuned_full["bond_dim"],
                img_size=img_size,
                patch_size=ps,
                n_classes=heads,
                use_residual=True,
                dropout_p=0.1,
            ),
            tuned_full["lr"],
            tuned_full,
        )
        specs["mlp_reference"] = (
            lambda: MLPReference(hidden=16, img_size=img_size, patch_size=ps, n_classes=heads),
            0.01,
            None,
        )
        specs["mlp_param_matched"] = (
            lambda: MLPReference(hidden=2, img_size=img_size, patch_size=ps, n_classes=heads),
            0.01,
            None,
        )
    else:
        tuned_bare = tuned_full = None

    arms_by_name, tuned_by_name = {}, {}
    for name, (factory, lr, tuned) in specs.items():
        n_params = sum(p.numel() for p in factory().parameters())
        print(f"\n--- {name} ({n_params} params, lr={lr}, {len(args.seeds)} seeds) ---", flush=True)
        curves = []
        for s in args.seeds:
            c, _ = cc.train_run_multihead(factory, seed=s, task="objects", img_size=img_size, lr=lr, epochs=args.epochs)
            curves.append(c)
            print(f"  seed {s:>2}: " + "  ".join(f"{h}={cc.score_of(c[h]):.1f}%" for h in heads), flush=True)
            cc.save(
                {"arm": name, "partial": True, "seeds_done": len(curves), "curves": curves},
                f"{args.out}_{name}_partial.json",
            )
        arms_by_name[name] = cc.summarise_heads(
            name, curves, heads, num_params=n_params, img_size=img_size, epochs=args.epochs, lr=lr
        )
        tuned_by_name[name] = tuned

        # A head at its own chance floor is a broken run, not a result.
        if args.epochs >= 10:
            cc.assert_not_chance_level_per_head(name, {h: arms_by_name[name][h]["score_mean"] for h in heads}, heads)

    cc.print_head_table(
        f"C3: single-object CLEVR attributes, {img_size}x{img_size}, {len(args.seeds)} seeds",
        arms_by_name,
        heads,
        floors=floors,
    )

    result = {
        "task": "objects",
        "img_size": img_size,
        "readout": args.readout,
        "protocol": {**cc.CLEVR_PROTOCOL, "epochs": args.epochs},
        "seeds": args.seeds,
        "majority_floors": floors,
        "quantum_params": q_params,
        "tuned": {k: v for k, v in tuned_by_name.items() if v},
        "arms": arms_by_name,
    }

    if "quantum_coherent" in arms_by_name and len(arms_by_name) > 1:
        q = arms_by_name["quantum_coherent"]
        cmps = {}
        for other in ("classical_bare", "classical_full", "mlp_reference", "mlp_param_matched"):
            if other in arms_by_name:
                cmps[other] = cc.compare_heads(q, arms_by_name[other], heads)
                cc.print_head_comparisons(f"quantum_coherent -> {other}", cmps[other], heads)
        result["comparisons"] = cmps

        # Interpretability gate, straight from R4: if the MLP reference beats the
        # classical CP arm, the CP node -- not classical computation -- is what is
        # underperforming, and the quantum-vs-CP gap says nothing about unitarity.
        if "mlp_reference" in arms_by_name and "classical_bare" in arms_by_name:
            cp_vs_mlp = cc.compare_heads(arms_by_name["classical_bare"], arms_by_name["mlp_reference"], heads)
            broken = [h for h in heads if cp_vs_mlp[h]["resolved"] and cp_vs_mlp[h]["difference"] > 0]
            result["a3_answerable"] = not broken
            if broken:
                print(
                    f"\nQuestion A.3 NOT ANSWERABLE on heads {broken}: the MLP reference beats the "
                    f"classical CP arm there, so the CP quad-node is what is underperforming, not "
                    f"classical computation. DO NOT cite the raw quantum-vs-CP gap on those heads."
                )
            else:
                print(
                    "\nInterpretability gate PASSED: classical_bare is competitive with the MLP "
                    "reference, so the quantum-vs-CP comparison is meaningful on every head."
                )

        if "mlp_reference" in arms_by_name:
            mlp = arms_by_name["mlp_reference"]
            print("\nParameter efficiency (the claim Phase 1 supports):")
            print(f"  quantum_coherent {q_params} params vs mlp_reference {mlp[list(heads)[0]]['num_params']} params")
            for h in heads:
                d = mlp[h]["score_mean"] - q[h]["score_mean"]
                print(
                    f"    {h:<9} quantum trails the MLP by {d:>5.1f} pts "
                    f"(resolves {cmps['mlp_reference'][h]['min_detectable_effect']:.1f})"
                )

    cc.save(result, f"c3_attributes{args.out_suffix}_results.json")


if __name__ == "__main__":
    main()
