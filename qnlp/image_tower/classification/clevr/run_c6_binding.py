"""Task C6: the shape-binding compositional probe. THE CLOSING EXPERIMENT.

WHAT THIS TESTS THAT NOTHING ELSE IN THE PHASE COULD
----------------------------------------------------
`mlp_reference` beats every TTN arm on every head of C3, and beats them on C4's
relations. Read naively that contradicts the project's premise -- that tensor
networks capture COMPOSITIONAL structure better than the bag-of-features
behaviour which makes frozen CLIP fail on ARO/SugarCrepe.

It does not, because neither of those tasks is compositional. Attribute
classification is pure perception; C4's relation task is a single-referent
directional readout. `MLPReference` flattens POSITIONED patch embeddings, so it
is near-ideal for both. Neither task has the property that breaks a
bag-of-features model -- the same features arranged two ways giving two labels.

C6 does. One cube and one sphere per image, label = which is on the left, with
the class-conditional marginals IDENTICAL BY CONSTRUCTION.

THE HEADLINE IS AN INTERACTION, NOT A LEVEL -- AND BOTH DIRECTIONS ARE
PRE-REGISTERED
---------------------------------------------------------------------
On perception the MLP beats the TTNs. The question here is whether that
advantage NARROWS OR REVERSES on binding.

  * narrows/reverses -> the compositional claim is supported;
  * holds            -> THE COMPOSITIONAL ADVANTAGE DOES NOT HOLD IN THE VISION
                        TOWER, and that is what gets written up.

This experiment is designed to be able to fail. Deciding which reading applies
after seeing the numbers is how the phase produced three different
"measurements" of `classical_bare`, so the rule is fixed here, in code, before
the run.

QUESTION C.2(ii) GETS ITS SHARPEST TEST HERE
--------------------------------------------
Binding demands that position be CONJOINED WITH CONTENT ("cube AND left"), not
merely read out ("something is left"). C4's `+1.8` (limit 5.4) may reflect the
low positional demand of a directional readout rather than the encoding itself.
`quantum_none` vs `quantum_on_wire` is one architecture +/- an explicit encoding:
same qubits, topology, depth and gates, differing by a learned rotation pair per
patch (+32 params).

  ⚠️ THOSE 32 PARAMETERS ARE A CONFOUND. A win cannot be cleanly separated from
  the extra capacity -- which is exactly why `ancilla2` was gated behind a
  RESOLVED effect. Report it; do not quietly drop it.

Run:
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.run_c6_binding --skip-quantum
  conda run -n qnlp python -m qnlp.image_tower.classification.clevr.run_c6_binding --only-quantum --seeds 0
"""

import argparse
import json

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.clevr.run_c3_attributes import tune_classical
from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import CoherentQTTNClassifier
from qnlp.image_tower.classification.quantum.run_r4_classical_control import (
    ClassicalTTNClassifier,
    MLPReference,
)
from qnlp.utils.data import clevr_binding as cb

POSITIONAL_ARMS = ("none", "on_wire")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=cb.CANVAS, help="Composite canvas; 32 keeps the tree at 16 qubits.")
    ap.add_argument("--readout", default="top_layer_multi_pauli", help="C2's choice, on stability grounds.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    ap.add_argument("--epochs", type=int, default=90, help="C3c showed the tower needs ~3x the 30-epoch budget.")
    ap.add_argument("--lr", type=float, default=None, help="Quantum-arm lr; defaults to PROTOCOL's 0.03.")
    ap.add_argument("--positional", nargs="+", default=list(POSITIONAL_ARMS), choices=list(POSITIONAL_ARMS))
    ap.add_argument(
        "--no-cosine",
        action="store_true",
        help="Disable cosine lr decay. Decay is ON by default here: C4 collapsed 3 of 7 seeds AFTER "
        "they had already reached 35-50%%, which is a late-training lr signature.",
    )
    ap.add_argument(
        "--shuffled",
        action="store_true",
        help="MANIPULATION CHECK on the DATA, not a per-architecture score. Controlled marginals mean "
        "every architecture must land at exactly chance; any arm resolvably above it means a "
        "non-positional cue got baked into the composites and the task must be rebuilt.",
    )
    ap.add_argument("--skip-quantum", action="store_true", help="Cheap arms only (minutes).")
    ap.add_argument("--only-quantum", action="store_true", help="Quantum arms only, for sharding.")
    ap.add_argument(
        "--attribute",
        default=cb.DEFAULT_ATTRIBUTE,
        help="Which attribute is bound. `size` because EVERY arm perceives it (C3: 84.8-99.1%%), so a "
        "failure is a BINDING failure. NOT `shape`: classical_bare (36.5) and mlp_param_matched "
        "(36.9) sit at its 35.4 floor, so their scores there measured perception, not composition.",
    )
    ap.add_argument(
        "--tune-epochs",
        type=int,
        default=30,
        help="Budget for the classical hyperparameter search. 30, not the run's 90: the grid is ~36 "
        "configs x 3 seeds and scales linearly, so tuning at 90 cost 5 h of the first C6 run for a "
        "config RANKING that 30 epochs already gives. Not 10 either -- tuning at 10 while running at "
        "90 selects configs good at 10, which moves the budget asymmetry rather than removing it.",
    )
    ap.add_argument(
        "--reuse-tuning",
        default=None,
        help="Path to a previous results JSON whose tuned classical configs should be reused. Use it "
        "for the --shuffled manipulation check: re-running the grid there costs another full sweep "
        "to rank configs on data where every arm is at chance by construction.",
    )
    ap.add_argument("--out", default="c6")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    heads, img_size = cc.BINDING_HEAD, args.img_size
    # 32px canvas with 8px patches keeps the 4x4 patch grid, so the tree is still
    # 16 qubits (C5 route (a)). This is resolution scaling, NOT quantum scaling.
    ps = img_size // 4
    protocol = {
        "epochs": args.epochs,
        "cosine_decay": not args.no_cosine,
        "shuffled": args.shuffled,
        "attribute": args.attribute,
    }
    floors = cc.majority_baselines(task="binding", img_size=img_size, seed=args.seeds[0], **protocol)
    q_arch = {**pc.COHERENT_ARCH, "readout": args.readout}
    q_lr = args.lr if args.lr is not None else pc.PROTOCOL["lr"]

    print(
        f"C6 binding on `{args.attribute}` | {img_size}x{img_size} (patch {ps}, 16 qubits) | "
        f"readout={args.readout}\n"
        f"seeds={args.seeds} | epochs={args.epochs} | cosine={not args.no_cosine} | "
        f"SHUFFLED={args.shuffled} | chance=50.0% | majority floor={floors['binding']:.1f}%",
        flush=True,
    )
    if args.shuffled:
        print(
            "  MANIPULATION CHECK RUN. Every arm should land at chance. Anything resolvably above it "
            "invalidates the task -- rebuild the composites before reading any C6 result.",
            flush=True,
        )

    specs = {}
    if not args.skip_quantum:
        for positional in args.positional:
            specs[f"quantum_{positional}"] = (
                lambda p=positional: CoherentQTTNClassifier(
                    **q_arch,
                    img_size=img_size,
                    patch_size=ps,
                    n_classes=heads,
                    share_level1_weights=False,
                    positional=p,
                ),
                q_lr,
                None,
            )

    if not args.only_quantum:
        # Size the classical budget off the quantum model even when the quantum
        # arms are not being trained here, so a sharded run tunes identically.
        q_params = sum(
            p.numel()
            for p in CoherentQTTNClassifier(
                **q_arch, img_size=img_size, patch_size=ps, n_classes=heads, share_level1_weights=False
            ).parameters()
        )
        print(f"\nquantum reference: {q_params} params (classical budget = 1.5x)", flush=True)
        if args.reuse_tuning:
            with open(args.reuse_tuning) as f:
                prev = json.load(f)["arms"]
            tb = prev["classical_bare"]["binding"]["tuned"]
            tf = prev["classical_full"]["binding"]["tuned"]
            print(f"Reusing tuned configs from {args.reuse_tuning}: bare={tb} full={tf}", flush=True)
        else:
            print("Tuning classical_bare (rank swept freely):", flush=True)
            tb = tune_classical(False, 0.0, q_params, img_size, heads, task="binding", epochs=args.tune_epochs)
            print("Tuning classical_full:", flush=True)
            tf = tune_classical(True, 0.1, q_params, img_size, heads, task="binding", epochs=args.tune_epochs)
        for name, tuned, resid, drop in (("classical_bare", tb, False, 0.0), ("classical_full", tf, True, 0.1)):
            specs[name] = (
                lambda t=tuned, r=resid, d=drop: ClassicalTTNClassifier(
                    rank=t["rank"],
                    bond_dim=t["bond_dim"],
                    img_size=img_size,
                    patch_size=ps,
                    n_classes=heads,
                    use_residual=r,
                    dropout_p=d,
                ),
                tuned["lr"],
                tuned,
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

    if not specs:
        raise SystemExit("--skip-quantum with --only-quantum leaves nothing to run.")

    arms_by_name = {}
    for name, (factory, lr, tuned) in specs.items():
        n_params = sum(p.numel() for p in factory().parameters())
        print(f"\n--- {name} ({n_params} params, lr={lr}) ---", flush=True)
        curves = []
        for s in args.seeds:
            c, _ = cc.train_run_multihead(factory, seed=s, task="binding", img_size=img_size, lr=lr, **protocol)
            curves.append(c)
            print(f"  seed {s:>2}: binding={cc.score_of(c['binding']):.1f}%", flush=True)
            cc.save(
                {
                    "arm": name,
                    "partial": True,
                    "seeds_done": len(curves),
                    "curves": curves,
                    "num_params": n_params,
                },
                f"{args.out}_{name}_partial.json",
            )
        arms_by_name[name] = cc.summarise_heads(
            name, curves, heads, num_params=n_params, img_size=img_size, epochs=args.epochs, lr=lr, tuned=tuned
        )

    cc.print_head_table(
        f"C6: CLEVR {args.attribute} binding, {img_size}x{img_size}, {len(args.seeds)} seeds"
        + (" [PATCH-SHUFFLED MANIPULATION CHECK]" if args.shuffled else ""),
        arms_by_name,
        heads,
        floors=floors,
    )

    result = {
        "task": "binding",
        "img_size": img_size,
        "patch_size": ps,
        "readout": args.readout,
        "protocol": {**cc.CLEVR_PROTOCOL, **protocol, "quantum_lr": q_lr},
        "seeds": args.seeds,
        "majority_floors": floors,
        "shuffled": args.shuffled,
        "arms": arms_by_name,
        "attribute": args.attribute,
        "binding_manifest": cb.load_binding_manifest(args.attribute),
        "confound": (
            "quantum_on_wire carries 32 more parameters than quantum_none (+7%), so a win is not "
            "cleanly separable from the extra capacity. This is why ancilla2 was gated behind a "
            "RESOLVED effect."
        ),
    }

    if args.shuffled:
        # The gate. Above chance here means the composites leak a non-positional
        # cue -- lighting, an intensity gradient, a pasting artifact correlated
        # with the class -- and every unshuffled C6 number is uninterpretable.
        bad = []
        for name, arm in arms_by_name.items():
            a = arm["binding"]
            limit = cc.mde_unpaired(a["score_std"], a["score_std"], a["n_seeds"], a["n_seeds"])
            if a["score_mean"] - 50.0 > limit:
                bad.append(f"{name} {a['score_mean']:.1f}% (limit {limit:.1f})")
        result["manipulation_check_passed"] = not bad
        if bad:
            print(
                "\n❌ MANIPULATION CHECK FAILED. Above chance on shuffled input: "
                + "; ".join(bad)
                + "\n   The composites leak a non-positional cue. REBUILD the task -- every unshuffled "
                "C6 number is uninterpretable until this passes."
            )
        else:
            print("\n✅ Manipulation check passed: every arm at chance on shuffled input, as required.")

    for other in ("classical_bare", "classical_full", "mlp_reference", "mlp_param_matched"):
        base = "quantum_none" if "quantum_none" in arms_by_name else None
        if base and other in arms_by_name:
            cc.print_head_comparisons(
                f"{base} -> {other}",
                cc.compare_heads(arms_by_name[base], arms_by_name[other], heads),
                heads,
            )

    if "quantum_none" in arms_by_name and "quantum_on_wire" in arms_by_name:
        cmp = cc.compare_heads(arms_by_name["quantum_none"], arms_by_name["quantum_on_wire"], heads)
        cc.print_head_comparisons("quantum_none -> quantum_on_wire (Question C.2(ii))", cmp, heads)
        result["question_c2ii"] = cmp
        c = cmp["binding"]
        if c["resolved"] and c["difference"] > 0:
            verdict = (
                f"C.2(ii): explicit per-patch position HELPS where position must be BOUND to content "
                f"({c['difference']:+.1f} pts, limit {c['min_detectable_effect']:.1f}). That SCOPES R3's "
                f"-19.5 rather than contradicting it: R3 measured a per-quadrant ancilla on a "
                f"translation-invariant task where position is irrelevant. ⚠️ CONFOUNDED by on_wire's "
                f"+32 params -- escalate to ancilla2 to separate mechanism from capacity."
            )
        elif c["resolved"]:
            verdict = (
                f"C.2(ii): explicit position HURTS even where position must be bound to content "
                f"({c['difference']:+.1f} pts, limit {c['min_detectable_effect']:.1f}). Generalises R3 "
                f"rather than scoping it, and is the stronger result. Do not escalate."
            )
        else:
            verdict = (
                f"C.2(ii): BOUNDED NULL. Explicit position does not change binding accuracy by more "
                f"than {c['min_detectable_effect']:.1f} pts (observed {c['difference']:+.1f}). Combined "
                f"with C.2(i) -- the implicit topology reaching classical parity on relations -- this "
                f"supports omitting positional encoding. A legitimate finding; quote it AT ITS LIMIT, "
                f"never as 'no difference'. Do NOT escalate to ancilla2 on a null."
            )
        print(f"\nVERDICT: {verdict}")
        result["question_c2ii_verdict"] = verdict

    cc.save(result, f"c6_binding{args.out_suffix}_results.json")


if __name__ == "__main__":
    main()
