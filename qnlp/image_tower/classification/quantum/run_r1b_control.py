"""
Task R1b (roadmap Section 7): the control arm R1 is missing.

R1 passed the >=70% gate, but not via the mechanism the Code Audit predicted:

    readout:  scalar -> root_multi_pauli   55.0 +/- 3.5  ->  57.5 +/- 7.2   (inside noise)
    encoding: scalar_ry -> multi_axis      57.5 +/- 7.2  ->  59.1 +/- 6.4   (inside noise)
    protocol: 256/15 -> 1024/30            59.1          ->  78.75 +/- 7.2  (large)

All of the gate-clearing movement came from the training protocol; both
architecture changes are inside seed noise at n=5. And the capacity fallback
was only ever run on the *winning* config -- nobody ran the original scalar
readout at 1024/30. So the 2x2 has a hole in it, and the causal claim in
research_log.md ("the scalar-readout bottleneck invalidated the July-26/27
ablations") is currently unsupported by the experiment as run.

This script fills the hole and reports the complete 2x2.

Part A -- CONTROL CELL. Runs readout=scalar, encoding=scalar_ry at 1024/30
  using the *identical* code path as R1 (imports run_r1_remediation.train_config
  directly, rather than reimplementing it) so the new number is strictly
  comparable to R1's 78.75%. Decides the diagnosis:
    - scalar stays well below ~78%  -> audit's diagnosis confirmed; readout width
      was genuinely binding, it just needed enough training to express itself.
      Keep root_multi_pauli + multi_axis as the architecture of record.
    - scalar reaches ~78%           -> diagnosis was wrong. The July-26/27
      ablations were underpowered because of the *protocol*, not the readout.
      R3 still stands (an underpowered protocol invalidates them just as
      thoroughly), but the recorded reason changes and the architecture of
      record should revert to the simpler model (105 params vs 211).
    - ambiguous                     -> report as ambiguous, default to simpler.

Part B -- PAIRED-PROTOCOL VALIDATION (also a deliverable for R3's precondition).
  R1's per-seed spread was 20 points ([70.3, 76.6, 73.4, 90.6, 82.8]), which at
  n=10 unpaired resolves only ~+/-6 point differences -- while R3 hunts 2-3 point
  effects. This part re-runs both configs at 1024/30 with the data ordering
  pinned per seed (identical across variants), so per-seed deltas can be taken.
  It reports the minimum detectable effect for the paired vs unpaired design on
  the same data, which is the number R3 needs before it spends 10-seed runs.

  Note on R1's pairing: R1's train_config calls torch.manual_seed(seed) and then
  constructs the model *before* iterating the DataLoader (shuffle=True draws from
  the global RNG). Configs with different parameter counts consume different
  amounts of RNG at init, so the shuffle order diverges between variants even at
  the same seed. Part B fixes that by building the loaders before model init.

Run:  conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r1b_control
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.image_tower.classification.quantum.qttn_core import HierarchicalQTTNClassifier
from qnlp.image_tower.classification.quantum.run_r1_remediation import train_config
from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

RESULTS_DIR = "qnlp/image_tower/classification/quantum/results"
R1_RESULTS = os.path.join(RESULTS_DIR, "r1_remediation_results.json")
OUT_PATH = os.path.join(RESULTS_DIR, "r1b_control_results.json")

SEEDS = (0, 1, 2, 3, 4)
GATE_THRESHOLD = 70.0

# The two corners of the 2x2 that matter: the audited/regressed config and R1's winner.
OLD_CONFIG = ("scalar", "scalar_ry")
NEW_CONFIG = ("root_multi_pauli", "multi_axis")

PROTOCOL_OLD = {"train_samples": 256, "epochs": 15}
PROTOCOL_NEW = {"train_samples": 1024, "epochs": 30}

# If the control lands within this many points of R1's winner, the readout/encoding
# change is not what cleared the gate.
EQUIVALENCE_MARGIN = 5.0


# ---------------------------------------------------------------------------
# Part B: paired training (data order pinned independently of model init)
# ---------------------------------------------------------------------------
def train_config_paired(
    readout, encoding, seed, epochs=30, train_samples=1024, test_samples=64, batch_size=32, lr=0.03
):
    """Same optimisation setup as run_r1_remediation.train_config, but the data
    ordering is made independent of how much RNG the model constructor consumes,
    so seed k gives the *same* batch sequence for every variant.
    """
    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=batch_size,
        train_samples=train_samples,
        test_samples=test_samples,
        img_size=16,
        seed=seed,
    )
    # Pin the shuffle stream to the seed alone. Note this must be set on the
    # sampler, not the DataLoader: RandomSampler.__iter__ reads its own
    # .generator and falls back to the *global* RNG when it is None, which is
    # exactly the coupling to model-init RNG consumption we are removing.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    train_loader.sampler.generator = loader_gen

    # Model init gets its own stream, drawn after the loaders exist.
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HierarchicalQTTNClassifier(readout=readout, encoding=encoding)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    val_accs = []
    for epoch in range(epochs):
        model.train()
        for batch_imgs, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_imgs), batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_imgs, batch_labels in test_loader:
                preds = model(batch_imgs).argmax(dim=1)
                correct += preds.eq(batch_labels).sum().item()
                total += batch_labels.size(0)
        val_accs.append(100.0 * correct / total)
        print(
            f"  [paired {readout}/{encoding} seed={seed}] " f"epoch {epoch+1:02d}/{epochs} val_acc={val_accs[-1]:.1f}%"
        )
    return val_accs


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
# Two-sided t critical values, df = n-1, alpha = 0.05. Small lookup avoids a
# scipy dependency for what is a 5-row table.
_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
}


def _t_crit(df):
    return _T_CRIT_95.get(df, 1.96)


def min_detectable_effect(std_a, std_b, n, paired_std=None):
    """95% CI half-width on the difference of means -- i.e. the smallest effect
    the design could distinguish from zero. Unpaired uses the pooled between-seed
    std; paired uses the std of the per-seed deltas.
    """
    if paired_std is not None:
        return float(_t_crit(n - 1) * paired_std / np.sqrt(n))
    pooled = np.sqrt((std_a**2 + std_b**2) / 2.0)
    return float(_t_crit(2 * n - 2) * pooled * np.sqrt(2.0 / n))


def summarise(name, val_accs, **extra):
    val_accs = np.array(val_accs)
    final = val_accs[:, -1]
    last5 = val_accs[:, -5:].mean(axis=1)
    out = {
        "config": name,
        "final_val_acc_per_seed": final.tolist(),
        "final_val_acc_mean": float(final.mean()),
        "final_val_acc_std": float(final.std()),
        "peak_val_acc_mean": float(val_accs.max(axis=1).mean()),
        "val_acc_mean_curve": val_accs.mean(axis=0).tolist(),
        # Per-seed curves are kept so R3 can compare estimators (single final
        # epoch vs. last-5-epoch average) without re-running anything.
        "val_acc_curves_per_seed": val_accs.tolist(),
        "last5_val_acc_per_seed": last5.tolist(),
        "last5_val_acc_mean": float(last5.mean()),
        "last5_val_acc_std": float(last5.std()),
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
def load_r1_cells():
    """The three 2x2 cells R1 already measured."""
    with open(R1_RESULTS) as f:
        r1 = json.load(f)
    s1, s2 = r1["sweep1_readout"], r1["sweep2_encoding"]
    fb = r1["capacity_fallback"]
    return {
        "old_config@old_protocol": {
            "config": "scalar/scalar_ry",
            "protocol": "256/15",
            "final_val_acc_per_seed": s1["readout=scalar,encoding=scalar_ry"]["final_val_acc_per_seed"],
            "final_val_acc_mean": s1["readout=scalar,encoding=scalar_ry"]["final_val_acc_mean"],
            "final_val_acc_std": s1["readout=scalar,encoding=scalar_ry"]["final_val_acc_std"],
            "num_params": s1["readout=scalar,encoding=scalar_ry"]["num_params"],
            "source": "R1 sweep 1",
        },
        "new_config@old_protocol": {
            "config": "root_multi_pauli/multi_axis",
            "protocol": "256/15",
            "final_val_acc_per_seed": s2["readout=root_multi_pauli,encoding=multi_axis"]["final_val_acc_per_seed"],
            "final_val_acc_mean": s2["readout=root_multi_pauli,encoding=multi_axis"]["final_val_acc_mean"],
            "final_val_acc_std": s2["readout=root_multi_pauli,encoding=multi_axis"]["final_val_acc_std"],
            "num_params": s2["readout=root_multi_pauli,encoding=multi_axis"]["num_params"],
            "source": "R1 sweep 2",
        },
        "new_config@new_protocol": {
            "config": "root_multi_pauli/multi_axis",
            "protocol": "1024/30",
            "final_val_acc_per_seed": fb["final_val_acc_per_seed"],
            "final_val_acc_mean": fb["final_val_acc_mean"],
            "final_val_acc_std": fb["final_val_acc_std"],
            "source": "R1 capacity fallback",
        },
    }


def verdict(control_mean, winner_mean, control_std, winner_std, n):
    """Which of the three branches in the roadmap's R1b outcome handling."""
    gap = winner_mean - control_mean
    mde = min_detectable_effect(control_std, winner_std, n)
    control_clears_gate = control_mean >= GATE_THRESHOLD

    if not control_clears_gate and gap > mde:
        return (
            "AUDIT_CONFIRMED",
            "The scalar readout does NOT reach the gate at 1024/30, and the gap to the "
            "winner exceeds what this design can resolve. Readout width was genuinely "
            "binding; it needed adequate training to express itself. Keep "
            "root_multi_pauli+multi_axis as the architecture of record and strengthen "
            "the Code Audit entry's attribution.",
        )
    if control_clears_gate and abs(gap) <= EQUIVALENCE_MARGIN:
        return (
            "AUDIT_REFUTED",
            "The scalar readout reaches the gate at 1024/30 too. The bottleneck "
            "diagnosis was wrong: the July-26/27 ablations were underpowered because of "
            "the TRAINING PROTOCOL, not the readout. R3 still stands unchanged, but the "
            "Code Audit entry must be corrected and the architecture of record should "
            "revert to the simpler 105-param scalar model.",
        )
    return (
        "AMBIGUOUS",
        "Neither branch is clean (control clears gate: {}; gap {:+.1f} pts vs "
        "resolvable {:.1f} pts). Report as ambiguous and default to the simpler "
        "architecture; do not manufacture a narrative from a noisy result.".format(control_clears_gate, gap, mde),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-paired",
        action="store_true",
        help="Run Part A (control cell) only; skip the Part B paired-protocol validation.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help="Part A seeds. Defaults to R1's 5 seeds so the control cell is "
        "strictly comparable to R1's capacity-fallback number.",
    )
    ap.add_argument(
        "--paired-seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Part B seeds. Defaults to 10 to match R3's planned seed count, so "
        "the reported min-detectable-effect is the one R3 will actually face.",
    )
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    n = len(seeds)
    paired_seeds = tuple(args.paired_seeds)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {"seeds": list(seeds), "cells": load_r1_cells()}

    # ---------------- Part A: the missing control cell ----------------
    print("=" * 74)
    print("PART A -- CONTROL CELL: readout=scalar, encoding=scalar_ry @ 1024 train / 30 epochs")
    print("(identical code path as R1, imported from run_r1_remediation)")
    print("=" * 74)

    readout, encoding = OLD_CONFIG
    control_accs = [train_config(readout, encoding, s, **PROTOCOL_NEW, test_samples=64) for s in seeds]
    control = summarise(
        "scalar/scalar_ry",
        control_accs,
        protocol="1024/30",
        source="R1b control",
        num_params=sum(p.numel() for p in HierarchicalQTTNClassifier(readout=readout, encoding=encoding).parameters()),
    )
    results["cells"]["old_config@new_protocol"] = control

    winner = results["cells"]["new_config@new_protocol"]
    branch, explanation = verdict(
        control["final_val_acc_mean"],
        winner["final_val_acc_mean"],
        control["final_val_acc_std"],
        winner["final_val_acc_std"],
        n,
    )
    results["verdict"] = {
        "branch": branch,
        "explanation": explanation,
        "control_mean": control["final_val_acc_mean"],
        "winner_mean": winner["final_val_acc_mean"],
        "gap": winner["final_val_acc_mean"] - control["final_val_acc_mean"],
        "min_detectable_effect_unpaired": min_detectable_effect(
            control["final_val_acc_std"], winner["final_val_acc_std"], n
        ),
        "gate_threshold": GATE_THRESHOLD,
        "control_clears_gate": control["final_val_acc_mean"] >= GATE_THRESHOLD,
    }

    # ---------------- Part B: paired-protocol validation ----------------
    if not args.skip_paired:
        print("\n" + "=" * 74)
        print("PART B -- PAIRED PROTOCOL (data order pinned per seed across variants)")
        print("Delivers the per-seed-delta harness R3's precondition requires.")
        print("=" * 74)

        np_ = len(paired_seeds)
        paired = {}
        for label, (ro, en) in (("old", OLD_CONFIG), ("new", NEW_CONFIG)):
            accs = [train_config_paired(ro, en, s, **PROTOCOL_NEW, test_samples=64) for s in paired_seeds]
            paired[label] = summarise(f"{ro}/{en}", accs, protocol="1024/30 (paired)")

        old_f = np.array(paired["old"]["final_val_acc_per_seed"])
        new_f = np.array(paired["new"]["final_val_acc_per_seed"])
        deltas = new_f - old_f
        paired_std = float(deltas.std(ddof=1)) if np_ > 1 else 0.0
        mde_paired = min_detectable_effect(None, None, np_, paired_std=paired_std)
        mde_unpaired = min_detectable_effect(
            paired["old"]["final_val_acc_std"], paired["new"]["final_val_acc_std"], np_
        )

        results["paired"] = {
            "old": paired["old"],
            "new": paired["new"],
            "seeds": list(paired_seeds),
            "per_seed_delta": deltas.tolist(),
            "mean_delta": float(deltas.mean()),
            "delta_std": paired_std,
            "wins_for_new": int((deltas > 0).sum()),
            "n": np_,
            "min_detectable_effect_paired": mde_paired,
            "min_detectable_effect_unpaired": mde_unpaired,
            "variance_reduction_factor": (mde_unpaired / mde_paired) if mde_paired > 0 else None,
            "seed_correlation_between_variants": float(np.corrcoef(old_f, new_f)[0, 1]),
        }

        # Estimator comparison: is "final epoch" a noisy way to score a run?
        # If averaging the last 5 epochs cuts the std, R3 gets sharper
        # resolution for free -- no extra seeds, no extra compute.
        est = {}
        for key, suffix in (("final", "final_val_acc"), ("last5", "last5_val_acc")):
            o = np.array(paired["old"][f"{suffix}_per_seed"])
            w = np.array(paired["new"][f"{suffix}_per_seed"])
            d = w - o
            est[key] = {
                "old_std": float(o.std(ddof=1)),
                "new_std": float(w.std(ddof=1)),
                "mean_delta": float(d.mean()),
                "delta_std": float(d.std(ddof=1)),
                "mde_unpaired": min_detectable_effect(float(o.std(ddof=1)), float(w.std(ddof=1)), np_),
                "mde_paired": min_detectable_effect(None, None, np_, paired_std=float(d.std(ddof=1))),
            }
        results["estimator_comparison"] = est

        # How many seeds would R3 actually need? Solve MDE = t*s*sqrt(2/n) for n,
        # iterating because t depends on n. Uses the better (last5) estimator's
        # pooled std. Runs cost ~18s each at 1024/30, so large n is affordable
        # here -- the point is to state the requirement rather than guess it.
        s_pooled = float(np.sqrt((est["last5"]["old_std"] ** 2 + est["last5"]["new_std"] ** 2) / 2.0))
        power = {}
        for target in (2.0, 3.0, 5.0, 8.0):
            k = 4
            for _ in range(100):
                k_new = int(np.ceil(2.0 * (_t_crit(2 * k - 2) * s_pooled / target) ** 2))
                if k_new == k:
                    break
                k = k_new
            power[f"{target:g}pt_effect"] = {"seeds_per_arm": int(k), "est_minutes_per_arm": round(k * 18 / 60.0, 1)}
        results["power_analysis"] = {"pooled_std_last5": s_pooled, "requirements": power}

    # ---------------- Report ----------------
    print("\n" + "=" * 74)
    print("R1b RESULT -- THE COMPLETED 2x2")
    print("=" * 74)
    print(f"{'':<34}{'256 train / 15 ep':>20}{'1024 train / 30 ep':>20}")
    for cfg_label, old_key, new_key in (
        ("scalar / scalar_ry  (105 params)", "old_config@old_protocol", "old_config@new_protocol"),
        ("root_multi_pauli / multi_axis (211)", "new_config@old_protocol", "new_config@new_protocol"),
    ):
        o, w = results["cells"][old_key], results["cells"][new_key]
        print(
            f"{cfg_label:<34}"
            f"{o['final_val_acc_mean']:>13.1f} +/-{o['final_val_acc_std']:>4.1f}"
            f"{w['final_val_acc_mean']:>13.1f} +/-{w['final_val_acc_std']:>4.1f}"
        )

    v = results["verdict"]
    print(f"\nGate threshold: {GATE_THRESHOLD:.0f}%   " f"Control clears gate: {v['control_clears_gate']}")
    print(
        f"Architecture gap at 1024/30: {v['gap']:+.1f} pts "
        f"(this design resolves >= {v['min_detectable_effect_unpaired']:.1f} pts, unpaired n={n})"
    )
    print(f"\nVERDICT: {v['branch']}\n{v['explanation']}")

    if "paired" in results:
        p = results["paired"]
        print(
            f"\nPaired comparison (n={p['n']}): mean delta {p['mean_delta']:+.1f} "
            f"+/- {p['delta_std']:.1f}, new config won on {p['wins_for_new']}/{p['n']} seeds"
        )
        print(
            f"  min detectable effect -- paired: {p['min_detectable_effect_paired']:.1f} pts, "
            f"unpaired: {p['min_detectable_effect_unpaired']:.1f} pts"
        )
        print(
            f"  cross-variant seed correlation: {p['seed_correlation_between_variants']:+.2f} "
            f"(pairing only helps when this is strongly positive)"
        )
        print(
            f"  min detectable effect -- paired: {p['min_detectable_effect_paired']:.1f} pts, "
            f"unpaired: {p['min_detectable_effect_unpaired']:.1f} pts "
            "-> pairing "
            + ("HELPS" if p["min_detectable_effect_paired"] < p["min_detectable_effect_unpaired"] else "DOES NOT HELP")
        )

        e = results["estimator_comparison"]
        print("\n  Estimator comparison (how a run is scored):")
        for key, label in (("final", "final epoch     "), ("last5", "mean last 5 ep  ")):
            r = e[key]
            print(
                f"    {label} old std {r['old_std']:5.1f}  new std {r['new_std']:5.1f}  "
                f"delta {r['mean_delta']:+5.1f} +/- {r['delta_std']:4.1f}  "
                f"MDE(unpaired) {r['mde_unpaired']:.1f} pts"
            )
        better = "last5" if e["last5"]["mde_unpaired"] < e["final"]["mde_unpaired"] else "final"
        print(f"    -> R3 should score runs by: {better}")

        pa = results["power_analysis"]
        print(f"\n  Seeds R3 needs (pooled std {pa['pooled_std_last5']:.1f}, ~18s per run at 1024/30):")
        for k, v in pa["requirements"].items():
            print(
                f"    to resolve a {k.replace('pt_effect','')}-point effect: "
                f"{v['seeds_per_arm']:>3} seeds/arm  (~{v['est_minutes_per_arm']:.0f} min/arm)"
            )

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
