"""Training harness for the CLEVR phase (roadmap Section 3, tasks C1-C4).

WHY THIS IS A SEPARATE MODULE
-----------------------------
`phase15_common` encodes the Phase-1 protocol exactly as R1-R7 measured it, and
every one of those results must stay reproducible. CLEVR needs two things Phase 1
never did -- a real dataset behind the loader, and several simultaneous
classification heads -- so the training loop is forked here rather than
generalised in place.

Forked training loops are also precisely what caused both Phase-1 audits, so the
fork is kept as thin as possible:

  * EVERY statistic comes from `phase15_common` (compare, summarise, mde_unpaired,
    seeds_needed, is_chance_level, t_crit, save, print_arms, print_comparisons).
    There is no second implementation of Welch's test in this repo.
  * EVERY model comes from `qttn_core` / `run_r4_classical_control`. No new model
    class is defined here. Per-script model drift is what let the scalar-readout
    regression survive a month of ablations.

What is genuinely new: the loader plumbing, the multi-head loss, and a
chance-level guard that knows each head's own number of classes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.utils.data.clevr_binding import get_binding_loaders
from qnlp.utils.data.clevr_objects import ATTRIBUTES, RELATIONS, get_clevr_loaders

# Re-exported so runners import statistics from one place and cannot accidentally
# grow a local copy.
compare = pc.compare
summarise = pc.summarise
mde_unpaired = pc.mde_unpaired
seeds_needed = pc.seeds_needed
is_chance_level = pc.is_chance_level
print_arms = pc.print_arms
print_comparisons = pc.print_comparisons
save = pc.save
SCORE_LAST_K = pc.SCORE_LAST_K

HEADS = dict(ATTRIBUTES)  # {"color": 8, "shape": 3, "material": 2, "size": 2}
RELATION_HEAD = {"relation": len(RELATIONS)}
BINDING_HEAD = {"binding": 2}  # Task C6: which shape is on the left.

# Phase-1 protocol, with ONE deliberate change: 512 test samples instead of 64.
# 64 samples across an 8-way `color` head is ~8 per class, which makes the
# evaluation noise larger than most effects being measured. Evaluation is cheap
# (no gradients), so this costs almost nothing. Everything else -- 1024 train,
# 30 epochs, batch 32, lr 0.03, last-5-epoch scoring, unpaired Welch -- is
# unchanged, so CLEVR numbers stay methodologically comparable to Phase 1.
CLEVR_PROTOCOL = {**pc.PROTOCOL, "test_samples": 512}

TASK_HEADS = {"objects": HEADS, "relations": RELATION_HEAD, "binding": BINDING_HEAD}


def get_loaders(task, img_size, cfg, seed, shuffled=False, patch_size=None, attribute=None):
    """One dispatch point for every task's loader.

    Binding lives in its own module because its images are COMPOSED rather than
    cropped, and because it carries the patch-shuffle manipulation check that no
    other task needs. Routing here keeps `train_run_multihead` and
    `majority_baselines` task-agnostic -- per-script data loading is the same
    class of drift as per-script model classes, which caused both Phase-1 audits.
    """
    if task == "binding":
        return get_binding_loaders(
            img_size=img_size,
            batch_size=cfg["batch_size"],
            train_samples=cfg["train_samples"],
            test_samples=cfg["test_samples"],
            seed=seed,
            shuffled=shuffled,
            patch_size=patch_size or img_size // 4,
            **({"attribute": attribute} if attribute else {}),
        )
    return get_clevr_loaders(
        task=task,
        img_size=img_size,
        batch_size=cfg["batch_size"],
        train_samples=cfg["train_samples"],
        test_samples=cfg["test_samples"],
        seed=seed,
    )


def chance_rate(n_classes):
    return 100.0 / n_classes


def assert_not_chance_level_per_head(name, scores_by_head, heads, margin=2.5):
    """A head sitting at ITS OWN chance floor is a broken run, not a result.

    The per-head part matters: chance is 12.5% for `color` but 50% for the binary
    `material` and `size` heads. Calling `pc.assert_not_chance_level` with its
    4-class default would wave through a completely dead binary head at 50%.
    """
    dead = [
        f"{h} {scores_by_head[h]:.1f}% (chance {chance_rate(k):.1f}%)"
        for h, k in heads.items()
        if pc.is_chance_level(scores_by_head[h], n_classes=k, margin=margin)
    ]
    assert not dead, (
        f"arm {name!r} has head(s) at chance level: {', '.join(dead)}. "
        f"That is a broken run, not a result -- debug it before using it in any comparison."
    )


def majority_baselines(task="objects", img_size=16, seed=0, **protocol):
    """Per-head majority-class rate on the actual val split.

    Chance assumes balanced classes; CLEVR's attributes are not guaranteed
    balanced, so this is the floor an accuracy must actually clear. Reported
    beside every result.
    """
    cfg = {**CLEVR_PROTOCOL, **protocol}
    heads = TASK_HEADS[task]
    _, test_loader = get_loaders(
        task, img_size, cfg, seed, shuffled=cfg.get("shuffled", False), attribute=cfg.get("attribute")
    )
    counts = {h: np.zeros(k, dtype=np.int64) for h, k in heads.items()}
    for _, labels in test_loader:
        for h in heads:
            counts[h] += np.bincount(labels[h].numpy(), minlength=heads[h])
    return {h: float(100.0 * c.max() / max(c.sum(), 1)) for h, c in counts.items()}


def train_run_multihead(model_factory, seed, task="objects", img_size=16, **protocol):
    """One training run with several heads off a shared readout.

    Differences from `pc.train_run`, and only these:
      * the loader is CLEVR rather than synthetic shapes;
      * the loss is the SUM of per-head cross-entropies (equal weight -- there is
        no basis for weighting one attribute over another, and an unequal
        weighting would need its own justification);
      * it returns a dict of per-head validation curves rather than one curve.

    Returns (curves_by_head, model).
    """
    cfg = {**CLEVR_PROTOCOL, **protocol}
    heads = TASK_HEADS[task]
    train_loader, test_loader = get_loaders(
        task, img_size, cfg, seed, shuffled=cfg.get("shuffled", False), attribute=cfg.get("attribute")
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_factory()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Cosine decay, OFF by default so every pre-2026-08-03 result stays exactly
    # reproducible. It targets a specific measured failure mode rather than being
    # general tidying: in C4, 3 of quantum_none's 7 collapsed seeds had ALREADY
    # LEARNED the task -- one reached 50.2% at epoch 12, above the classical
    # baseline's mean -- and then fell back to chance for the last ten epochs.
    # That is a too-high-late-in-training signature. The other 4 never left
    # chance, which decay cannot help; expect it to fix at most half the
    # collapses.
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if cfg.get("cosine_decay") else None
    )

    curves = {h: [] for h in heads}
    for _ in range(cfg["epochs"]):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = torch.stack([criterion(out[h], labels[h]) for h in heads]).sum()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        correct = {h: 0 for h in heads}
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                out = model(imgs)
                for h in heads:
                    correct[h] += out[h].argmax(dim=1).eq(labels[h]).sum().item()
                total += imgs.shape[0]
        for h in heads:
            curves[h].append(100.0 * correct[h] / total)
    return curves, model


def summarise_heads(name, curves_per_seed, heads, **extra):
    """`curves_per_seed` is a list (one per seed) of {head: curve}.

    Returns {head: pc.summarise(...)} so every downstream comparison is
    `pc.compare` on a standard arm dict.
    """
    return {h: pc.summarise(name, [c[h] for c in curves_per_seed], head=h, **extra) for h in heads}


def score_of(curves):
    return float(np.mean(curves[-SCORE_LAST_K:]))


def print_head_table(title, arms_by_name, heads, floors=None):
    """One row per arm, one column per head, with the parameter count.

    The parameter column is not decoration: Phase 1's result is a
    parameter-efficiency claim, and an accuracy reported without a size beside it
    is what produced the fake 57.8-pt headline (R4).
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"{'arm':<24}{'params':>8}  " + "".join(f"{h:>16}" for h in heads))
    for name, arms in arms_by_name.items():
        params = arms[list(heads)[0]].get("num_params", 0)
        cells = "".join(f"{arms[h]['score_mean']:>10.1f} +/-{arms[h]['score_std']:>4.1f}" for h in heads)
        print(f"{name:<24}{params:>8}  {cells}")
    if floors:
        print(f"{'(majority-class floor)':<24}{'':>8}  " + "".join(f"{floors[h]:>16.1f}" for h in heads))
        print(f"{'(uniform chance)':<24}{'':>8}  " + "".join(f"{chance_rate(k):>16.1f}" for k in heads.values()))


def compare_heads(arm_a, arm_b, heads):
    """`pc.compare` per head. Every comparison carries its resolution limit."""
    return {h: pc.compare(arm_a[h], arm_b[h]) for h in heads}


def print_head_comparisons(title, cmps_by_head, heads):
    print(f"\n{title}")
    print(f"{'head':<12}{'diff':>9}{'resolves':>10}  verdict")
    for h in heads:
        c = cmps_by_head[h]
        print(f"{h:<12}{c['difference']:>+9.1f}{c['min_detectable_effect']:>10.1f}  {c['verdict']}")
