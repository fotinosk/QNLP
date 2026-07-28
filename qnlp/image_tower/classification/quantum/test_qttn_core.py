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
