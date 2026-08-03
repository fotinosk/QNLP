"""Task C4: relational reasoning, with and without explicit position.

THIS IS QUESTION C.2 AND IT IS REQUIRED, NOT OPTIONAL.

Two-object crops; predict left/right/front/behind of the other object relative to
the CENTRED reference object. R3 measured the spatial ancilla at -19.5 pts, but on
a translation-invariant single-object task where position barely affects the label
-- a setting where a positional mechanism has nothing to contribute and a negative
result is close to structurally guaranteed. Here POSITION IS THE LABEL, so this is
the only place the question can be settled.

It is also the first genuine test of whether the implicit tree topology encodes
spatial relations at all, which is the assumption behind dropping positional
encoding in the first place.

WHAT IS ACTUALLY BEING TESTED -- state this in the thesis, do not blur it:
  * R3 tested a PER-QUADRANT ANCILLA (4 positions, extra wires).
  * The original design was a PER-PATCH ANCILLA (16 positions) -- that is 16 extra
    wires, i.e. 32 qubits, which statevector simulation cannot reach.
  * This tests a PER-PATCH POSITIONAL ENCODING on the existing wires (`on_wire`):
    a learned rotation pair on each patch's own wire. Zero extra wires.
  These are three different mechanisms. Report which one was measured.

ESCALATION: only if `on_wire` shows a RESOLVED effect is it worth building the
2-ancilla-wire variant (18 qubits, ~4x cost) to confirm the mechanism rather than
the encoding. On a null, stop -- that is the agreed budget rule.

Run: conda run -n qnlp python -m qnlp.image_tower.classification.clevr.run_c4_relational --seeds 0 1 2 3 4
"""

import argparse

from qnlp.image_tower.classification.clevr import clevr_common as cc
from qnlp.image_tower.classification.clevr.run_c3_attributes import PATCH_FOR, tune_classical
from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import CoherentQTTNClassifier
from qnlp.image_tower.classification.quantum.run_r4_classical_control import (
    ClassicalTTNClassifier,
    MLPReference,
)

ARMS = ("none", "on_wire")


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
        help="Quantum-arm learning rate; defaults to PROTOCOL's 0.03. SWEEP THIS BEFORE ADDING "
        "SEEDS. The 2026-08-02 re-run collapsed 12 of 20 seeds at lr=0.03, and 3 of those had "
        "already reached 35-50%% before falling back to chance -- a learning-rate signature, not a "
        "capability limit. Contaminated pooled std is 11.5 against ~3 on converged seeds, which is "
        "the difference between ~40 and ~9 seeds per arm for Question C.2.",
    )
    ap.add_argument("--positional", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument(
        "--tune-epochs",
        type=int,
        default=10,
        help="Budget for the classical hyperparameter search. Raise it to --epochs for C3d, or "
        "tuning at 10 while running at 90 moves the budget asymmetry onto the hyperparameter axis.",
    )
    ap.add_argument("--skip-classical", action="store_true", help="Quantum arms only, for sharding.")
    ap.add_argument(
        "--skip-quantum",
        action="store_true",
        help="Classical arms only. The quantum model is still built once (unrained) to size the "
        "tuning budget at 1.5x its parameter count.",
    )
    ap.add_argument("--out", default="c4", help="Checkpoint prefix; shard with --out c4_s0, c4_s1, ...")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    heads, img_size, ps = cc.RELATION_HEAD, args.img_size, PATCH_FOR[args.img_size]
    floors = cc.majority_baselines(task="relations", img_size=img_size, seed=args.seeds[0])
    q_arch = {**pc.COHERENT_ARCH, "readout": args.readout}

    print(
        f"C4 relational (Question C.2) | {img_size}x{img_size} | readout={args.readout}\n"
        f"positional arms={args.positional} | seeds={args.seeds} | chance=25.0% | "
        f"majority floor={floors['relation']:.1f}%",
        flush=True,
    )

    q_lr = args.lr if args.lr is not None else pc.PROTOCOL["lr"]
    specs = {}
    for positional in args.positional:
        specs[f"quantum_{positional}"] = (
            lambda p=positional: CoherentQTTNClassifier(
                **q_arch, img_size=img_size, patch_size=ps, n_classes=heads, share_level1_weights=False, positional=p
            ),
            q_lr,
            None,
        )

    if not args.skip_classical:
        q_params = sum(p.numel() for p in specs[f"quantum_{args.positional[0]}"][0]().parameters())
        print("\nTuning classical_bare (rank swept freely):", flush=True)
        tb = tune_classical(False, 0.0, q_params, img_size, heads, task="relations", epochs=args.tune_epochs)
        print("Tuning classical_full:", flush=True)
        tf = tune_classical(True, 0.1, q_params, img_size, heads, task="relations", epochs=args.tune_epochs)
        specs["classical_bare"] = (
            lambda: ClassicalTTNClassifier(
                rank=tb["rank"],
                bond_dim=tb["bond_dim"],
                img_size=img_size,
                patch_size=ps,
                n_classes=heads,
                use_residual=False,
                dropout_p=0.0,
            ),
            tb["lr"],
            tb,
        )
        specs["classical_full"] = (
            lambda: ClassicalTTNClassifier(
                rank=tf["rank"],
                bond_dim=tf["bond_dim"],
                img_size=img_size,
                patch_size=ps,
                n_classes=heads,
                use_residual=True,
                dropout_p=0.1,
            ),
            tf["lr"],
            tf,
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

    if args.skip_quantum:
        # Drop the quantum arms only AFTER they have sized the tuning budget.
        for positional in args.positional:
            specs.pop(f"quantum_{positional}", None)
        if not specs:
            raise SystemExit("--skip-quantum with --skip-classical leaves nothing to run.")
        print("\nSKIPPING the quantum arm(s) (--skip-quantum). Run them separately and combine.", flush=True)

    arms_by_name = {}
    for name, (factory, lr, tuned) in specs.items():
        n_params = sum(p.numel() for p in factory().parameters())
        print(f"\n--- {name} ({n_params} params, lr={lr}) ---", flush=True)
        curves = []
        for s in args.seeds:
            c, _ = cc.train_run_multihead(
                factory, seed=s, task="relations", img_size=img_size, lr=lr, epochs=args.epochs
            )
            curves.append(c)
            print(f"  seed {s:>2}: relation={cc.score_of(c['relation']):.1f}%", flush=True)
            cc.save(
                # num_params travels WITH the checkpoint. Without it a sharded run
                # merges to a table reading `params 0`, which is an accuracy with no
                # size beside it -- the one thing standing requirement 1 forbids.
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
        f"C4: CLEVR spatial relations, {img_size}x{img_size}, {len(args.seeds)} seeds",
        arms_by_name,
        heads,
        floors=floors,
    )

    result = {
        "task": "relations",
        "img_size": img_size,
        "readout": args.readout,
        "protocol": {**cc.CLEVR_PROTOCOL, "epochs": args.epochs, "quantum_lr": q_lr},
        "seeds": args.seeds,
        "majority_floors": floors,
        "arms": arms_by_name,
        "mechanism_tested": (
            "per-patch positional ENCODING on existing wires (on_wire). NOT the per-quadrant ancilla "
            "R3 rejected at -19.5 pts, and NOT the per-patch ancilla of the original design, which "
            "would need 32 qubits."
        ),
    }

    # Does the task work at all? Above chance is necessary before the ancilla
    # question means anything -- a model at 25% cannot inform Question C.2.
    #
    # VIABILITY IS A PROPERTY OF THE DATA, SO ONLY A CLASSICAL ARM MAY JUDGE IT.
    # This block used to fall back to `quantum_none`, and on 2026-08-01 that is
    # exactly what it did: the quantum arms scored 28.4/29.2% against a 29.3%
    # floor and the run concluded "the relational task is NOT learnable at this
    # resolution". It is learnable. Re-run with the classical arms present, the
    # same data gives classical_bare 49.6%, classical_full 51.3% and
    # mlp_reference 65.4% -- and the classical TTNs are the quantum tower's
    # structural counterparts at 428/434 params against its 462. So C4 measured a
    # MODEL failure and reported it as a DATA failure, which is the R4 error with
    # the roles reversed: a quantum arm was used as its own control.
    #
    # A quantum arm at chance is now exactly as consistent with "the model cannot
    # do this" as with "the task is impossible", and this script must not choose
    # between those for you.
    viability_arms = [a for a in ("mlp_reference", "classical_full", "classical_bare") if a in arms_by_name]
    baseline_name = viability_arms[0] if viability_arms else None
    if baseline_name is None:
        # `viable` stays None = UNKNOWN, which is not the same as False. The
        # quantum arms may still be compared to each other below -- that is a
        # model-vs-model question and does not need viability -- but no claim
        # about the TASK may be made from this run.
        base, viable = None, None
        print(
            "\nTask viability: NOT ASSESSED -- this run has no classical arm (--skip-classical). "
            "A quantum arm at the floor is equally consistent with 'the model failed' and 'the task "
            "is impossible'. Run submit_c4_classical.sh and read the two together."
        )
        result["above_chance"] = {"assessed": False, "reason": "no classical arm in this run"}
    else:
        base = arms_by_name[baseline_name]["relation"]
        above_chance = base["score_mean"] - floors["relation"]
        limit = cc.mde_unpaired(base["score_std"], base["score_std"], base["n_seeds"], base["n_seeds"])
        print(
            f"\nTask viability ({baseline_name}, classical): {base['score_mean']:.1f}% vs a "
            f"{floors['relation']:.1f}% majority floor ({above_chance:+.1f}, resolves {limit:.1f})."
        )
        viable = bool(above_chance > limit)
        result["above_chance"] = {
            "arm": baseline_name,
            "margin": float(above_chance),
            "limit": float(limit),
            "resolved": viable,
        }
        if not viable:
            print(
                f"  The relational task is NOT learnable by the classical {baseline_name} arm at this "
                "resolution, so it is the DATA that is the limit. Question C.2 cannot be answered "
                "from this run -- report it as a resolution limit, not as evidence about positional "
                "encoding."
            )
        else:
            print(
                "  The task is viable. A quantum arm at the floor here is a MODEL result, not a data "
                "limit, and must be reported as one."
            )

    if viable is False and "quantum_none" in arms_by_name and "quantum_on_wire" in arms_by_name:
        # Comparing two arms on data that no classical model can learn measures
        # nothing. Note the `is False`: an UNASSESSED task (no classical arm) must
        # not take this branch, or a sharded quantum-only run would silently
        # suppress its own verdict on a task that is perfectly fine.
        verdict = (
            f"Question C.2: NO VERDICT. The task itself is not learnable here (classical "
            f"{baseline_name} {base['score_mean']:.1f}% vs a {floors['relation']:.1f}% floor), so the "
            f"with-vs-without-position comparison is between two arms that are both at chance. Fix "
            f"viability first -- more seeds will not help. Check resolution, the epoch budget, and lr."
        )
        print(f"\nVERDICT: {verdict}")
        result["question_c2_verdict"] = verdict
    elif "quantum_none" in arms_by_name and "quantum_on_wire" in arms_by_name:
        cmp = cc.compare_heads(arms_by_name["quantum_none"], arms_by_name["quantum_on_wire"], heads)
        cc.print_head_comparisons("quantum_none -> quantum_on_wire (Question C.2)", cmp, heads)
        result["question_c2"] = cmp
        c = cmp["relation"]
        if c["resolved"] and c["difference"] > 0:
            verdict = (
                f"Question C.2: explicit per-patch position HELPS on a relational task "
                f"({c['difference']:+.1f} pts, limit {c['min_detectable_effect']:.1f}). This does NOT "
                f"contradict R3's -19.5: that measured a per-quadrant ancilla on a translation-"
                f"invariant task where position is irrelevant. The two results together say the value "
                f"of positional encoding depends on whether position is task-relevant, which is the "
                f"scoping R3's entry explicitly asked for. ESCALATE: build positional='ancilla2' to "
                f"confirm this is the mechanism and not just extra parameters."
            )
        elif c["resolved"]:
            verdict = (
                f"Question C.2: explicit position HURTS even when position is the label "
                f"({c['difference']:+.1f} pts, limit {c['min_detectable_effect']:.1f}). That "
                f"generalises R3's rejection rather than scoping it, and is the stronger result. "
                f"Do not escalate to ancilla2."
            )
        else:
            verdict = (
                f"Question C.2: UNRESOLVED at {len(args.seeds)} seeds "
                f"({c['difference']:+.1f} pts vs a {c['min_detectable_effect']:.1f}-pt limit). The "
                f"implicit tree topology is not measurably worse than explicit position even where "
                f"position is the label -- which is a real, reportable finding and supports the "
                f"decision to omit positional encoding. Use pc.seeds_needed() "
                f"({cc.seeds_needed(c['pooled_std'], max(abs(c['difference']), 0.5))} seeds/arm for "
                f"this effect) before spending more. Do NOT escalate to ancilla2 on a null."
            )
        print(f"\nVERDICT: {verdict}")
        result["question_c2_verdict"] = verdict

    # Standing requirement 1: never report a quantum accuracy without a size-matched
    # classical reference beside it. So the quantum arm is the LEFT side whenever it
    # is present -- `baseline_name` is now a classical arm and using it here printed
    # `classical_bare -> classical_bare`, a self-comparison, which is no reference
    # at all.
    compare_from = "quantum_none" if "quantum_none" in arms_by_name else baseline_name
    for other in ("classical_bare", "classical_full", "mlp_reference", "mlp_param_matched"):
        if other in arms_by_name and compare_from in arms_by_name and other != compare_from:
            cc.print_head_comparisons(
                f"{compare_from} -> {other}",
                cc.compare_heads(arms_by_name[compare_from], arms_by_name[other], heads),
                heads,
            )

    cc.save(result, f"c4_relational{args.out_suffix}_results.json")


if __name__ == "__main__":
    main()
