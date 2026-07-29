"""Shared training / statistics harness for Phase 1.5 tasks R2-R4.

Exists so R2, R3 and R4 do not each re-fork the training loop -- that drift is
what let the scalar-readout regression go unnoticed across four scripts (see
research_log.md 2026-07-27 "Code Audit").

Protocol and statistics defaults encode what R1/R1b measured (research_log.md
2026-07-27 / 2026-07-28):
  - PROTOCOL: 1024 train / 64 test / 30 epochs. The old 256/15 protocol is
    underpowered for this task regardless of architecture.
  - Runs are scored by the mean of the last 5 epochs, not the single final
    epoch (MDE 8.0 -> 7.2 pts, and it equalises arm variances).
  - Comparisons are UNPAIRED. R1b measured cross-variant seed correlation at
    -0.22, so pairing has nothing to cancel and measurably hurt (MDE 10.2
    paired vs 7.6 unpaired). Do not reintroduce pairing without re-measuring
    that correlation.
  - Seed counts come from R1b's power analysis (pooled std ~8.2): resolving a
    3-pt effect needs ~58 seeds/arm, 5-pt needs ~21, 8-pt needs ~9.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.utils.data.synthetic_shapes import get_synthetic_shapes_loaders

RESULTS_DIR = "qnlp/image_tower/classification/quantum/results"

# Architecture of record, established by R1/R1b (readout) and R2 (encoding, ansatz).
# Kept here rather than as qttn_core constructor defaults so that R1/R1b remain
# reproducible exactly as logged -- those runs predate R2 and used the module's
# original `strongly_entangling` default. R3 onward should use ARCH.
#
# R2 caveat worth carrying: the *encoding* is what is actually resolved
# (multi_axis beats scalar_ry by +12.6 pts vs a 3.6-pt limit). The *ansatz*
# choice is NOT resolved on accuracy (iqp vs strongly_entangling: +1.6 pts vs a
# 4.8-pt limit); iqp is adopted because it matched the documented choice and had
# visibly lower seed variance (4.9 vs 9.7), which buys resolution downstream --
# not because it is measurably more accurate.
ARCH = {"readout": "root_multi_pauli", "encoding": "multi_axis", "ansatz": "iqp"}

# The correct readout is TOPOLOGY-DEPENDENT, which the single ARCH above hid.
# In the hybrid, information reaches the root through classical scalar bonds, so
# reading the root qubit's Bloch vector is adequate. In the coherent tree the
# root is a single qubit's marginal of a 16-qubit state -- a severe bottleneck.
# Measured on the coherent tree, one seed, 15 epochs (2026-07-28):
#     root_multi_pauli (root qubit only) : 35.6%
#     top_layer_qubits (4 wires)         : 78.4%   <-- +43 points
# The gap is not "more numbers" (3 vs 4); it is that one qubit's marginal cannot
# carry 16 patches. See research_log.md 2026-07-28 "R7".
COHERENT_ARCH = {"readout": "top_layer_qubits", "encoding": "multi_axis", "ansatz": "iqp"}

PROTOCOL = {"train_samples": 1024, "test_samples": 64, "epochs": 30, "batch_size": 32, "lr": 0.03}
SCORE_LAST_K = 5

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
    15: 2.131,
    20: 2.086,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t_crit(df):
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    keys = sorted(_T_CRIT_95)
    for k in keys:
        if df < k:
            return _T_CRIT_95[k]
    return 1.96


def mde_unpaired(std_a, std_b, n_a, n_b=None):
    """95% CI half-width on a difference of means: the smallest effect this
    design could distinguish from zero.

    Uses Welch's unequal-variance, unequal-n form. The earlier version took
    n = min(n_a, n_b) and applied the equal-n pooled formula, which throws away
    every observation in the larger arm. That mattered in practice: the R7
    coherent baseline (n=4, std 3.9) against the hybrid (n=30, std 8.0) looked
    "unresolved" at a 10.8-pt limit, when the correct limit is ~5.8 and the
    9.9-pt gap is resolved. Small n on one arm is not by itself a reason for
    more seeds -- especially when that arm is the low-variance one.
    """
    if n_b is None:
        n_b = n_a
    va, vb = std_a**2 / n_a, std_b**2 / n_b
    se = np.sqrt(va + vb)
    if se == 0:
        return float("inf")
    # Welch-Satterthwaite degrees of freedom.
    denom = (va**2 / max(n_a - 1, 1)) + (vb**2 / max(n_b - 1, 1))
    df = int(max(1, round((va + vb) ** 2 / denom))) if denom > 0 else 1
    return float(t_crit(df) * se)


def seeds_needed(pooled_std, target_effect):
    n = 4
    for _ in range(100):
        n_new = int(np.ceil(2.0 * (t_crit(2 * n - 2) * pooled_std / target_effect) ** 2))
        if n_new == n:
            break
        n = n_new
    return n


def is_chance_level(score, n_classes=4, margin=2.5):
    """Is this score indistinguishable from random guessing?

    A run at chance is a broken run, not a result. R4's first classical baseline
    scored 25.8% on a 4-class task and would have produced a spectacular fake
    "quantum beats classical by 57.8 points" headline had it been taken at face
    value. Any arm landing here needs debugging before it is compared to anything.
    """
    return score <= (100.0 / n_classes) + margin


def assert_not_chance_level(name, score, n_classes=4, margin=2.5):
    assert not is_chance_level(score, n_classes, margin), (
        f"arm {name!r} scored {score:.1f}%, at chance level for {n_classes} classes "
        f"({100.0 / n_classes:.1f}%). This is a broken run, not a result -- debug it "
        f"before using it in any comparison."
    )


def train_run(model_factory, seed, p_noise=0.0, **protocol):
    """One training run. `model_factory` is a zero-arg callable returning a
    fresh nn.Module, so this harness stays agnostic to quantum vs classical.
    Returns the per-epoch val-accuracy curve.
    """
    cfg = {**PROTOCOL, **protocol}
    train_loader, test_loader = get_synthetic_shapes_loaders(
        batch_size=cfg["batch_size"],
        train_samples=cfg["train_samples"],
        test_samples=cfg["test_samples"],
        img_size=cfg.get("img_size", 16),
        seed=seed,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_factory()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def _fwd(imgs):
        return model(imgs, p_noise=p_noise) if p_noise > 0 else model(imgs)

    val_accs = []
    for _ in range(cfg["epochs"]):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(_fwd(imgs), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                correct += _fwd(imgs).argmax(dim=1).eq(labels).sum().item()
                total += labels.size(0)
        val_accs.append(100.0 * correct / total)
    return val_accs, model


def summarise(name, curves, **extra):
    """Score an arm. `score` (mean of last SCORE_LAST_K epochs) is the headline
    metric; `final` is kept for comparability with pre-R1b log entries."""
    curves = np.array(curves)
    final = curves[:, -1]
    score = curves[:, -SCORE_LAST_K:].mean(axis=1)
    out = {
        "config": name,
        "n_seeds": int(curves.shape[0]),
        "score_per_seed": score.tolist(),
        "score_mean": float(score.mean()),
        "score_std": float(score.std(ddof=1)) if curves.shape[0] > 1 else 0.0,
        "final_val_acc_per_seed": final.tolist(),
        "final_val_acc_mean": float(final.mean()),
        "final_val_acc_std": float(final.std(ddof=1)) if curves.shape[0] > 1 else 0.0,
        "peak_val_acc_mean": float(curves.max(axis=1).mean()),
        "val_acc_curves_per_seed": curves.tolist(),
    }
    out.update(extra)
    return out


def compare(arm_a, arm_b):
    """Unpaired comparison of two summarised arms, reporting the resolution
    limit alongside the difference -- a null is meaningless without it."""
    n_a, n_b = arm_a["n_seeds"], arm_b["n_seeds"]
    n = min(n_a, n_b)
    diff = arm_b["score_mean"] - arm_a["score_mean"]
    m = mde_unpaired(arm_a["score_std"], arm_b["score_std"], n_a, n_b)
    pooled = float(np.sqrt((arm_a["score_std"] ** 2 + arm_b["score_std"] ** 2) / 2.0))
    # Degenerate guard: zero observed variance (e.g. n=1, or every seed landing
    # identically) would otherwise report any difference as "resolved".
    if pooled == 0.0 or n < 2:
        m = float("inf")
    return {
        "baseline": arm_a["config"],
        "variant": arm_b["config"],
        "difference": float(diff),
        "min_detectable_effect": m,
        "resolved": bool(abs(diff) > m),
        "verdict": (
            "variant better"
            if diff > m
            else "variant worse"
            if -diff > m
            else f"unresolved (|{diff:+.1f}| <= {m:.1f} pts)"
        ),
        "n_seeds": n,
        "n_seeds_a": n_a,
        "n_seeds_b": n_b,
        "pooled_std": pooled,
        "seeds_needed_for_observed_diff": seeds_needed(pooled, max(abs(diff), 0.5)),
    }


def print_arms(title, arms):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"{'config':<44}{'score (last5)':>18}{'final':>16}")
    for a in arms:
        print(
            f"{a['config']:<44}{a['score_mean']:>11.1f} +/-{a['score_std']:>4.1f}"
            f"{a['final_val_acc_mean']:>10.1f} +/-{a['final_val_acc_std']:>4.1f}"
        )


def print_comparisons(cmps):
    print(f"\n{'comparison':<52}{'diff':>9}{'resolves':>10}  verdict")
    for c in cmps:
        print(
            f"{c['baseline']+' -> '+c['variant']:<52}{c['difference']:>+9.1f}"
            f"{c['min_detectable_effect']:>10.1f}  {c['verdict']}"
        )


def save(results, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {path}")
    return path
