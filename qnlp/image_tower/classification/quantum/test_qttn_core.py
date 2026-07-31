"""Regression tests for the quantum image tower (roadmap Section 8, Phase C).

Every test here targets a failure this project actually shipped. None are
hypothetical:

  test_noisy_matches_clean_at_zero_noise
      A PennyLane 0.43.2 `default.mixed` parameter-broadcasting bug made batched
      noisy evaluation return *wrong values with the right shape* for multi-axis
      encoding. Silent corruption of every noise sweep. The check exploits the
      fact that DepolarizingChannel(0) is exactly the identity, so a noisy device
      must reproduce the clean one to machine precision -- a framework-agnostic
      invariant that would catch this bug class anywhere.

  test_readout_width_matches_config
      The whole image was being compressed to one scalar through `nn.Linear(1, 4)`
      in four separate experiment scripts, invalidating a month of ablations
      (research_log.md 2026-07-27 "Code Audit").

  test_batched_and_rowwise_agree
      The first attempted fix for the broadcasting bug (power-of-4 chunking)
      produced correct shapes and wrong numbers. Only caught because chunk
      invariance was checked explicitly.

  test_chance_level_guard
      R4's classical baseline sat at 25.8% -- exactly chance for 4 classes --
      and would have produced a spectacular fake "quantum beats classical by
      57.8 points" headline had it not been questioned.

  test_gradients_reach_every_parameter
      The mixed-channel dead-gradient trap: a learnable mixing angle that drifts
      to zero starves the ansatz weights of gradient.

Run: conda run -n qnlp python -m pytest qnlp/image_tower/classification/quantum/test_qttn_core.py -q
"""

import numpy as np
import pytest
import torch

from qnlp.image_tower.classification.quantum import phase15_common as pc
from qnlp.image_tower.classification.quantum.qttn_core import (
    ANSATZE,
    ENCODINGS,
    MODES,
    READOUT_DIM,
    READOUTS,
    CoherentQTTNClassifier,
    HierarchicalQTTNClassifier,
    QuantumNode,
)

CLEVR_HEADS = {"color": 8, "shape": 3, "material": 2, "size": 2}

CONFIGS = [(e, a) for e in ENCODINGS for a in ANSATZE]


def _img(batch=6):
    torch.manual_seed(0)
    return torch.rand(batch, 3, 16, 16)


@pytest.mark.parametrize("encoding,ansatz", CONFIGS)
@pytest.mark.parametrize("mode", MODES)
def test_noisy_matches_clean_at_zero_noise(encoding, ansatz, mode):
    """DepolarizingChannel(p=0) is the identity, so the noisy device must equal
    the clean one. Any deviation means the mixed-state device is mis-evaluating
    the circuit -- the exact silent failure that corrupted multi-axis noise runs.
    """
    torch.manual_seed(0)
    model = HierarchicalQTTNClassifier(readout="root_multi_pauli", encoding=encoding, ansatz=ansatz, mode=mode)
    x = _img()
    with torch.no_grad():
        clean = model(x)
        noisy_at_zero = model(x, p_noise=1e-12)
    assert torch.allclose(clean, noisy_at_zero, atol=1e-5), (
        f"{encoding}+{ansatz}+{mode}: noisy device disagrees with clean device at p=0 "
        f"by {float((clean - noisy_at_zero).abs().max()):.3e}. Do not trust any noise "
        f"sweep from this configuration."
    )


@pytest.mark.parametrize("encoding,ansatz", CONFIGS)
def test_noisy_matches_clean_at_zero_noise_with_ancilla(encoding, ansatz):
    torch.manual_seed(0)
    model = HierarchicalQTTNClassifier(readout="root_multi_pauli", encoding=encoding, ansatz=ansatz, use_ancilla=True)
    x = _img()
    with torch.no_grad():
        assert torch.allclose(model(x), model(x, p_noise=1e-12), atol=1e-5)


@pytest.mark.parametrize("readout", READOUTS)
def test_readout_width_matches_config(readout):
    """Guards against the `nn.Linear(1, 4)` bottleneck reappearing silently."""
    model = HierarchicalQTTNClassifier(readout=readout, encoding="multi_axis")
    assert model.head.in_features == READOUT_DIM[readout], (
        f"readout={readout!r} should feed {READOUT_DIM[readout]} values into the head, " f"got {model.head.in_features}"
    )
    with torch.no_grad():
        n2 = model.level2(torch.rand(2, 1, 4, 1))
    assert n2.shape[-1] == READOUT_DIM[readout]


def test_scalar_readout_is_a_real_bottleneck():
    """Documents *why* the scalar readout was fatal, so the reason survives the
    code: it collapses the whole image to one number, and a Linear(1, n) head
    can only order classes along a single axis."""
    model = HierarchicalQTTNClassifier(readout="scalar", encoding="multi_axis")
    assert model.head.in_features == 1
    assert READOUT_DIM["root_multi_pauli"] > READOUT_DIM["scalar"]


@pytest.mark.parametrize("encoding,ansatz", CONFIGS)
def test_batched_and_rowwise_agree(encoding, ansatz):
    """Whichever evaluation path the probe selects, the other must agree where it
    is available. Catches a fix that returns plausible shapes with wrong values.
    """
    torch.manual_seed(0)
    model = HierarchicalQTTNClassifier(readout="root_multi_pauli", encoding=encoding, ansatz=ansatz)
    x = _img(batch=4)
    with torch.no_grad():
        whole = model(x, p_noise=0.05)
        per_row = torch.cat([model(x[i : i + 1], p_noise=0.05) for i in range(x.shape[0])])
    assert torch.allclose(whole, per_row, atol=1e-5), (
        f"{encoding}+{ansatz}: batch-size-dependent noisy results "
        f"(maxdiff {float((whole - per_row).abs().max()):.3e})"
    )


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_every_parameter(mode):
    torch.manual_seed(0)
    model = HierarchicalQTTNClassifier(readout="root_multi_pauli", encoding="multi_axis", mode=mode)
    model(_img()).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"mode={mode}: no gradient reaches {dead}"


def test_probe_selects_a_working_path():
    """The broadcasting probe must flag exactly the configurations that need the
    unbatched path -- multi_axis on this PennyLane version, scalar_ry not."""
    assert QuantumNode(encoding="scalar_ry", ansatz="iqp")._noisy_rowwise is False
    assert QuantumNode(encoding="multi_axis", ansatz="iqp")._noisy_rowwise is True


def test_chance_level_guard():
    """A result at chance is a broken run, not a finding. R4's first classical
    baseline sat at 25.8% (chance for 4 classes) and nearly produced a fake
    quantum-advantage headline."""
    assert pc.is_chance_level(25.8, n_classes=4)
    assert pc.is_chance_level(27.0, n_classes=4)
    assert not pc.is_chance_level(79.0, n_classes=4)
    with pytest.raises(AssertionError, match="chance level"):
        pc.assert_not_chance_level("classical_bare", 25.8, n_classes=4)
    pc.assert_not_chance_level("quantum", 79.0, n_classes=4)


def test_compare_reports_resolution_limit():
    """A null result is meaningless without the effect size the design could have
    detected. `compare` must always surface it, and must not call a difference
    resolved when it is inside the limit."""
    rng = np.random.default_rng(0)
    a = pc.summarise("a", [[50.0] * 25 + list(50 + rng.normal(0, 8, 5))] * 10)
    b = pc.summarise("b", [[52.0] * 25 + list(52 + rng.normal(0, 8, 5))] * 10)
    c = pc.compare(a, b)
    assert "min_detectable_effect" in c
    assert c["min_detectable_effect"] > 0
    if not c["resolved"]:
        assert abs(c["difference"]) <= c["min_detectable_effect"]


def test_seeds_needed_converges_and_satisfies_its_own_criterion():
    """`seeds_needed` decides whether to spend more compute, so a wrong answer is
    expensive in both directions.

    The original fixed-point iteration did not converge when the answer was
    small: it oscillated (1 <-> 10 for std 1.49 / effect 8.9) because n_new = 1
    made t_crit(2n-2) = t_crit(0) fall through the table to 12.706. It returned
    whichever value the loop stopped on, and reported "~10 seeds/arm needed" for
    effects already resolved at 3 -- pure over-spend.
    """
    for std, effect in [(1.49, 8.9), (5.0, 20.0), (2.0, 15.0), (8.2, 3.0), (8.2, 5.0), (4.94, 2.0)]:
        n = pc.seeds_needed(std, effect)
        assert n >= 2, f"std={std} effect={effect}: {n} seeds/arm is not a runnable design"
        # It must actually satisfy the power criterion it claims to solve...
        assert n >= 2.0 * (pc.t_crit(2 * n - 2) * std / effect) ** 2, f"std={std} effect={effect}: n={n} too small"
        # ...and be the SMALLEST such n, not merely a sufficient one.
        if n > 2:
            m = n - 1
            assert m < 2.0 * (pc.t_crit(2 * m - 2) * std / effect) ** 2, f"std={std} effect={effect}: n={n} not minimal"

    # Bigger effects never need more seeds than smaller ones at the same variance.
    needs = [pc.seeds_needed(8.2, e) for e in (2, 3, 5, 8, 15)]
    assert needs == sorted(needs, reverse=True), f"not monotone in effect size: {needs}"

    # The R1b power analysis logged on 2026-07-28 must still reproduce.
    assert pc.seeds_needed(8.2, 2) == 130
    assert pc.seeds_needed(4.94, 3) == 22


def test_architecture_of_record_is_pinned():
    """R2 selected multi_axis+iqp; R1/R1b selected root_multi_pauli. If ARCH
    drifts from the logged decision, results stop being comparable."""
    assert pc.ARCH == {"readout": "root_multi_pauli", "encoding": "multi_axis", "ansatz": "iqp"}
    assert pc.PROTOCOL["train_samples"] == 1024 and pc.PROTOCOL["epochs"] == 30


# ---------------------------------------------------------------------
# Coherent tree (Task R7). These guard the second regression of the same
# class as the scalar readout: a coherent design silently replaced by a
# cheaper approximation. See research_log.md 2026-07-28 "Code Audit #2".
# ---------------------------------------------------------------------


def test_coherent_tree_is_actually_coherent():
    """The defining property: a survivor qubit must be entangled with the rest
    of the tree when it reaches level 2.

    A single qubit is mixed exactly when its Bloch vector is shorter than 1
    (purity = (1 + |r|^2) / 2). In the measure-and-re-encode hybrid the level-2
    inputs are freshly encoded from classical scalars, so they are pure product
    states with |r| = 1 exactly and carry no entanglement upward.
    """
    torch.manual_seed(0)
    model = CoherentQTTNClassifier()
    x = torch.rand(8, 3, 16, 16)
    r = model.survivor_bloch_length(x)
    assert r < 1.0 - 1e-3, (
        f"survivor Bloch length {r:.6f} is indistinguishable from 1, so the qubit reaching "
        f"level 2 is pure and carries no entanglement. The tree is not coherent."
    )


def test_coherent_tree_uses_one_device_for_the_whole_tower():
    """One circuit over all patch wires, not a circuit per node."""
    model = CoherentQTTNClassifier()
    assert model.n_wires == model.n_patches == 16
    assert hasattr(model, "_circuit"), "expected a single QNode spanning the tower"
    assert not hasattr(model, "level1"), "per-node sub-circuits indicate the hybrid, not the coherent tree"


def test_coherent_tree_matches_hybrid_parameter_count():
    """Parameter counts must match so coherent-vs-hybrid comparisons are not
    confounded by model size."""
    coherent = sum(p.numel() for p in CoherentQTTNClassifier(**pc.ARCH).parameters())
    hybrid = sum(p.numel() for p in HierarchicalQTTNClassifier(**pc.ARCH).parameters())
    assert coherent == hybrid, f"coherent {coherent} vs hybrid {hybrid} params"


@pytest.mark.parametrize("mode", ["baseline", "reupload"])
def test_coherent_gradients_reach_every_parameter(mode):
    torch.manual_seed(0)
    model = CoherentQTTNClassifier(mode=mode)
    model(torch.rand(4, 3, 16, 16)).sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"mode={mode}: no gradient reaches {dead}"


@pytest.mark.parametrize("mode", ["mixed_channel", "with_ancilla_placeholder"])
def test_rejected_variants_are_not_silently_available(mode):
    """R3 rejected mixed_channel (-10.9 pts) and the spatial ancilla (-19.5 pts)
    decisively, and each ancilla doubles the statevector. They are deliberately
    not ported; the class must say so rather than appear to support them."""
    with pytest.raises((NotImplementedError, AssertionError, TypeError)):
        CoherentQTTNClassifier(mode=mode)


def test_coherent_tree_refuses_noise():
    """Noiseless-only per the 2026-07-28 scope decision; 16 qubits on a
    density-matrix simulator needs ~68GB."""
    model = CoherentQTTNClassifier()
    with pytest.raises(NotImplementedError, match="noiseless-only"):
        model(torch.rand(2, 3, 16, 16), p_noise=0.05)


# ---------------------------------------------------------------------
# CLEVR (Phase 2) additions: the 12-value readout, multi-head output,
# explicit position, and non-4x4 patch sizes.
# ---------------------------------------------------------------------


def test_wide_readout_is_wider_than_the_one_it_replaces():
    """C2's premise. `top_layer_multi_pauli` must actually carry more numbers
    than `top_layer_qubits`, and must do it WITHOUT extra wires -- the point is
    that widening the bond is free here, unlike raising the bond dimension."""
    narrow = CoherentQTTNClassifier(readout="top_layer_qubits", encoding="multi_axis")
    wide = CoherentQTTNClassifier(readout="top_layer_multi_pauli", encoding="multi_axis")
    assert READOUT_DIM["top_layer_multi_pauli"] == 3 * READOUT_DIM["top_layer_qubits"] == 12
    assert wide.head.in_features == 12
    assert wide.n_wires == narrow.n_wires, "the wide readout must not cost extra wires"
    out = wide(torch.rand(4, 3, 16, 16))
    assert out.shape == (4, 4)


@pytest.mark.parametrize("cls", [HierarchicalQTTNClassifier, CoherentQTTNClassifier])
def test_multi_head_gives_one_head_per_attribute_off_a_shared_readout(cls):
    """CLEVR needs four simultaneous attributes. They must come off the SAME
    readout vector -- whether four numbers can carry all four is exactly what
    task C2 measures, so the heads must not each get their own circuit."""
    torch.manual_seed(0)
    model = cls(readout="top_layer_qubits", encoding="multi_axis", n_classes=CLEVR_HEADS)
    out = model(torch.rand(4, 3, 16, 16))
    assert set(out) == set(CLEVR_HEADS)
    for name, k in CLEVR_HEADS.items():
        assert out[name].shape == (4, k), f"head {name!r}: {out[name].shape}"
        assert model.head[name].in_features == READOUT_DIM["top_layer_qubits"]


def test_multi_head_gradients_reach_every_parameter():
    """A summed multi-head loss must not starve the shared circuit weights."""
    torch.manual_seed(0)
    model = CoherentQTTNClassifier(readout="top_layer_qubits", encoding="multi_axis", n_classes=CLEVR_HEADS)
    sum(v.sum() for v in model(torch.rand(4, 3, 16, 16)).values()).backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient reaches {dead}"


def test_single_head_path_is_unchanged_by_the_multi_head_addition():
    """int n_classes must keep returning a plain tensor off a plain nn.Linear,
    so every R1-R7 script and the readout-width test keep working."""
    model = CoherentQTTNClassifier(**pc.COHERENT_ARCH)
    assert isinstance(model.head, torch.nn.Linear)
    assert model(torch.rand(3, 3, 16, 16)).shape == (3, 4)


def test_positional_encoding_changes_the_circuit_and_trains():
    """C4's 'with position' arm. It must (a) add parameters, (b) add no wires,
    (c) actually alter the output, and (d) receive gradient. A positional
    mechanism that silently no-ops would make Question C.2 unanswerable, which
    is the failure mode this whole test file exists to prevent."""
    torch.manual_seed(0)
    plain = CoherentQTTNClassifier(**pc.COHERENT_ARCH)
    torch.manual_seed(0)
    posed = CoherentQTTNClassifier(**pc.COHERENT_ARCH, positional="on_wire")

    assert posed.n_wires == plain.n_wires, "on_wire position must not cost extra wires"
    n_plain = sum(p.numel() for p in plain.parameters())
    n_posed = sum(p.numel() for p in posed.parameters())
    assert n_posed == n_plain + 2 * posed.n_patches, f"{n_plain} -> {n_posed}"

    x = torch.rand(4, 3, 16, 16)
    with torch.no_grad():
        # Same seed, so every shared parameter matches; only the position imprint
        # differs. Its weights are small but nonzero, so the outputs must differ.
        assert not torch.allclose(plain(x), posed(x), atol=1e-6), "positional encoding is a no-op"

    posed(x).sum().backward()
    assert posed.pos_weights.grad is not None and posed.pos_weights.grad.abs().sum() > 0


def test_ancilla2_refuses_rather_than_silently_falling_back():
    """The C4 escalation arm is unbuilt by decision. It must say so loudly --
    silently degrading to 'no position' would fabricate a null result for
    Question C.2."""
    with pytest.raises(NotImplementedError, match="on_wire"):
        CoherentQTTNClassifier(positional="ancilla2")
    with pytest.raises(ValueError, match="positional"):
        CoherentQTTNClassifier(positional="nonsense")


@pytest.mark.parametrize("img_size,patch_size", [(16, 4), (32, 8), (64, 16)])
def test_bigger_patches_keep_the_tree_at_16_qubits(img_size, patch_size):
    """C5 route (a): higher resolution at a constant qubit count. This is the
    'works today' claim in the roadmap -- assert it rather than trusting it.

    It also pins down what the route does and does not buy: the circuit width is
    identical at all three resolutions, so the extra pixels are absorbed by the
    classical patch encoder. That is resolution scaling, not quantum scaling,
    and the thesis has to say so.
    """
    torch.manual_seed(0)
    model = CoherentQTTNClassifier(**pc.COHERENT_ARCH, img_size=img_size, patch_size=patch_size)
    assert model.n_wires == 16 and model.n_patches == 16
    assert model.patch_embed.in_features == patch_size * patch_size * 3
    assert model(torch.rand(3, 3, img_size, img_size)).shape == (3, 4)
