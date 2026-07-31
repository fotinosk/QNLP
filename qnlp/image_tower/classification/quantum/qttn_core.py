"""
Shared hierarchical QTTN classifier for the Phase 1.5 remediation (roadmap
Section 7, Task R1 onward). Consolidates the near-duplicate model classes
that had drifted across investigate_quantum_residuals.py,
investigate_spatial_ancilla.py, investigate_ancilla_residuals.py, and
compare_topologies.py -- see research_log.md 2026-07-27 "Code Audit" for why
that drift mattered (it let the scalar-readout regression go unnoticed).

Tree structure (16x16 images, patch_size=4): 16 patches -> 4 level-1 nodes
(4 patches each, weight-shared) -> 1 level-2 node (4 level-1 outputs) -> head.

Configurable axes for the R1 ablation:
  - readout:  what the level-2 (root) node measures for classification.
      "scalar"           -- expval(PauliZ(0)) only. [B, 1] -> Linear(1, n_classes).
                             The regressed behaviour audited on 2026-07-27; kept
                             only so R1's A/B comparison is exact.
      "root_multi_pauli" -- expval(X(0)), expval(Y(0)), expval(Z(0)). [B, 3] ->
                             Linear(3, n_classes). No extra qubits, no circuit change.
      "level1_survivors" -- expval(PauliZ(w)) for all 4 level-2 wires. [B, 4] ->
                             Linear(4, n_classes). Matches the 2026-07-17 75%-val-acc
                             run's effective readout width.
  - encoding: how each patch's classical features become rotation angles at level 1.
      "scalar_ry"  -- Linear(48, 1) -> tanh*pi -> RY per wire (current/regressed default).
      "multi_axis" -- Linear(48, 3) -> tanh*pi -> RX, RY, RZ per wire.
    Level-2 encoding is always scalar RY: level-1 messages are scalar expectation
    values by construction (the tree's intra-level messages are not part of this
    ablation -- only the final root readout width is).
  - ansatz: {"strongly_entangling", "iqp"} per-node entangling block (Task R2).

`mode` is a reserved hook for the residual variants R3 will re-test
(data re-uploading, mixed-unitary channel) -- only "baseline" is implemented
here; anything else raises so R3 fails loudly instead of silently no-op'ing.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

NUM_QUBITS = 4
NUM_LAYERS = 2

# "level1_survivors" is a misleading name kept for backwards compatibility with
# R1's logged sweep: in the coherent tree it measures the four wires that ENTER
# the top node, read AFTER that node has acted -- i.e. the root plus the three
# qubits the tree nominally discards at the top. "top_layer_qubits" is the
# accurate name and the one to use going forward.
READOUTS = (
    "scalar",
    "root_multi_pauli",
    "level1_survivors",
    "top_layer_qubits",
    "top_layer_multi_pauli",
)
READOUT_ALIASES = {"top_layer_qubits": "level1_survivors"}
ENCODINGS = ("scalar_ry", "multi_axis")
ANSATZE = ("strongly_entangling", "iqp")

# Node residual variants re-tested by Task R3b. Only the two mechanisms that
# showed any signal in the (bottlenecked) July-26 runs are implemented:
# near_identity had no effect in either regime and lcu/lcu-lite inherit
# mixed_channel's instability while remaining untestable under noise
# (default.mixed postselection limitation, PennyLane 0.43.2). See roadmap
# Section 7 R3b for why those are not re-run.
MODES = ("baseline", "reupload", "mixed_channel")

# Explicit-position mechanisms for the coherent tree (CLEVR task C4 / Question C.2).
# "ancilla2" is declared here but deliberately unbuilt -- see CoherentQTTNClassifier.
POSITIONALS = ("none", "on_wire", "ancilla2")

READOUT_DIM = {
    "scalar": 1,  # <Z> on the root qubit only
    "root_multi_pauli": 3,  # full Bloch vector of the root qubit -- still ONE qubit's marginal
    "level1_survivors": NUM_QUBITS,  # deprecated alias of top_layer_qubits
    "top_layer_qubits": NUM_QUBITS,  # <Z> on all four wires entering the top node
    # Full single-qubit information of every top-layer wire. Costs no extra wires
    # and no extra gates -- only more measurements -- so it is the cheapest way to
    # widen the bond, which R7 measured to be the binding constraint (+43 pts from
    # 3 -> 4 values). Added for CLEVR task C2, where four heads spanning
    # 8 x 3 x 2 x 2 = 96 attribute combinations must come off one readout.
    "top_layer_multi_pauli": 3 * NUM_QUBITS,
}


def build_head(readout_dim, n_classes):
    """Classification head(s) off the shared quantum readout.

    `n_classes` is either an int -- one head, `forward` returns a [B, n_classes]
    tensor, which is every Phase-1 run -- or a dict {name: k}, giving one linear
    head per attribute off the SAME readout vector and a dict of logits.

    The multi-head form is what CLEVR needs (colour x shape x material x size),
    and sharing the readout is deliberate: whether four real numbers can carry
    four simultaneous attributes is the question task C2 asks. Keeping the int
    path returning a bare nn.Linear means `model.head.in_features` still works,
    which the readout-width regression test relies on.
    """
    if isinstance(n_classes, int):
        return nn.Linear(readout_dim, n_classes)
    if not n_classes:
        raise ValueError("n_classes as a dict must be non-empty")
    return nn.ModuleDict({name: nn.Linear(readout_dim, k) for name, k in n_classes.items()})


def apply_head(head, features):
    return {name: h(features) for name, h in head.items()} if isinstance(head, nn.ModuleDict) else head(features)


def _iqp_block(wires, weights):
    """IQP-inspired ansatz, adapted from train_synthetic_shapes.py's iqp_node
    to the [NUM_LAYERS, NUM_QUBITS, 3] weight shape shared with
    strongly_entangling_layers (only the first two rotation columns are used).
    """
    for layer in weights:
        for w in wires:
            qml.Hadamard(wires=w)
        for i, w in enumerate(wires):
            qml.RZ(layer[i, 0], wires=w)
        for i in range(len(wires)):
            w1, w2 = wires[i], wires[(i + 1) % len(wires)]
            qml.IsingZZ(layer[i, 1], wires=[w1, w2])
        for w in wires:
            qml.Hadamard(wires=w)


def _run_noisy(circuit, single_circuit, inputs, *args, rowwise=False, **kwargs):
    """Evaluate a `default.mixed` QNode over a batch of rows.

    WORKAROUND for a PennyLane 0.43.2 `default.mixed` parameter-broadcasting
    bug, characterised 2026-07-28. With multi-axis encoding (three rotations on
    the same wire) the device mis-broadcasts:

      * at most batch sizes it returns a *square* number of results (32 -> 64,
        128 -> 256, 12 -> 36) or raises an internal reshape error (5, 7, 9, ...);
      * at batch sizes that ARE powers of 4 it returns the correct *shape* but
        numerically *wrong values* -- confirmed by the fact that at p=0, where
        DepolarizingChannel is exactly the identity, the noisy device disagrees
        with the clean device by up to 4.2e-01 (it should agree to ~1e-16).

    The second case is the dangerous one: right shape, wrong numbers, no error.
    Batched evaluation is therefore not trustworthy for affected circuits and
    the only correct mode is row-by-row.

    `scalar_ry` encoding is NOT affected -- batched noisy matches clean to
    8.9e-16 at p=0 -- so the pre-R1 noisy results in research_log.md, which all
    used scalar_ry, are not impacted.

    Rather than hard-coding which encodings are affected, QuantumNode probes
    each circuit once at construction (see _probe_noisy_broadcasting) and sets
    `rowwise` only when that circuit actually fails the p=0 identity check. If a
    future PennyLane release fixes the bug, the fast batched path is taken
    automatically with no code change.
    """
    n_rows = inputs.shape[0]

    def _row(v, i):
        # Per-row tensors (e.g. pos_angles) index alongside `inputs`; shared
        # parameters (weights, mix_angle, p_noise) pass through untouched.
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n_rows:
            return v[i]
        return v

    def _flat(res, expected_rows):
        if isinstance(res, (list, tuple)):
            res = torch.stack([torch.as_tensor(r) for r in res], dim=-1)
        else:
            res = torch.as_tensor(res).unsqueeze(-1)
        if res.dim() == 1:  # unbatched circuit returns [n_observables]
            res = res.unsqueeze(0)
        if res.shape[0] != expected_rows:
            raise RuntimeError(
                f"default.mixed returned {res.shape[0]} rows, expected {expected_rows}. "
                "Re-characterise the device before trusting any noisy result "
                "(see _run_noisy docstring)."
            )
        return res

    if not rowwise:
        return _flat(circuit(inputs, *args, **kwargs), n_rows)

    outs = [
        _flat(single_circuit(inputs[i], *[_row(a, i) for a in args], **{k: _row(v, i) for k, v in kwargs.items()}), 1)
        for i in range(n_rows)
    ]
    return torch.cat(outs, dim=0)


def _apply_ansatz(ansatz, wires, weights):
    if ansatz == "strongly_entangling":
        qml.StronglyEntanglingLayers(weights, wires=wires)
    elif ansatz == "iqp":
        _iqp_block(wires, weights)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz}")


def _encode(encoding, inputs, wires, p_noise, noisy, batched=True):
    """Encode patch features as rotation angles.

    inputs: [B, len(wires), enc_dim] when `batched`, else [len(wires), enc_dim].
    enc_dim is 1 (scalar_ry) or 3 (multi_axis). The unbatched form exists to
    dodge a PennyLane `default.mixed` broadcasting bug -- see _run_noisy.
    """

    def ang(idx, j):
        return inputs[:, idx, j] if batched else inputs[idx, j]

    for idx, w in enumerate(wires):
        if encoding == "scalar_ry":
            qml.RY(ang(idx, 0), wires=w)
        elif encoding == "multi_axis":
            qml.RX(ang(idx, 0), wires=w)
            qml.RY(ang(idx, 1), wires=w)
            qml.RZ(ang(idx, 2), wires=w)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        if noisy:
            qml.DepolarizingChannel(p_noise, wires=w)


class QuantumNode(nn.Module):
    """One tree level: NUM_QUBITS children -> node output(s).

    `readout` only matters when `is_root=True` (level-2 / final node); every
    non-root node always emits a single scalar expectation value on wire 0,
    since that scalar is the message passed up the tree, not the ablated
    quantity.

    Wire layout: [0 .. NUM_QUBITS-1] system, then (if enabled) the spatial
    ancilla, then (if enabled) the mixing ancilla. The spatial ancilla joins
    the ansatz wires; the mixing ancilla never does (it controls the ansatz).
    """

    def __init__(
        self,
        encoding="scalar_ry",
        ansatz="strongly_entangling",
        readout="scalar",
        is_root=False,
        mode="baseline",
        use_ancilla=False,
    ):
        super().__init__()
        assert encoding in ENCODINGS
        assert ansatz in ANSATZE
        assert readout in READOUTS
        readout = READOUT_ALIASES.get(readout, readout)
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        if mode == "reupload" and NUM_LAYERS < 2:
            raise ValueError("reupload splits the ansatz into two halves; needs NUM_LAYERS >= 2")
        self.encoding = encoding
        self.ansatz = ansatz
        self.readout = readout if is_root else "scalar"
        self.is_root = is_root
        self.mode = mode
        self.use_ancilla = use_ancilla

        sys_wires = list(range(NUM_QUBITS))
        next_wire = NUM_QUBITS
        spatial_wire = None
        if use_ancilla:
            spatial_wire = next_wire
            next_wire += 1
        mix_wire = None
        if mode == "mixed_channel":
            mix_wire = next_wire
            next_wire += 1
        ansatz_wires = sys_wires + ([spatial_wire] if use_ancilla else [])
        n_wires = next_wire

        self.weights = nn.Parameter(torch.randn(NUM_LAYERS, len(ansatz_wires), 3) * 0.1)
        if use_ancilla:
            # Learned embedding of "which spatial quadrant" (4 positions), matching
            # quantum_image_embedding_model_hea.py's spatial_ancilla_encoding.
            self.pos_weights = nn.Parameter(torch.randn(4, 2) * 0.1)
        if mode == "mixed_channel":
            # Near-identity start (small mixing angle -> mostly pass-through),
            # consistent with the residual-connection motivation.
            self.mix_angle = nn.Parameter(torch.randn(()) * 0.1)

        dev_clean = qml.device("default.qubit", wires=n_wires)
        dev_noisy = qml.device("default.mixed", wires=n_wires)
        encoding_, ansatz_, readout_, mode_ = self.encoding, self.ansatz, self.readout, self.mode

        def _noise_all(p_noise, wires_):
            for w in wires_:
                qml.DepolarizingChannel(p_noise, wires=w)

        def _body(inputs, weights, p_noise, pos_angles=None, mix_angle=None, batched=True):
            noisy = p_noise > 0
            _encode(encoding_, inputs, sys_wires, p_noise, noisy, batched)
            if use_ancilla:
                qml.RX(pos_angles[:, 0] if batched else pos_angles[0], wires=spatial_wire)
                qml.RY(pos_angles[:, 1] if batched else pos_angles[1], wires=spatial_wire)
                if noisy:
                    qml.DepolarizingChannel(p_noise, wires=spatial_wire)

            if mode_ == "reupload":
                # Encode -> half the ansatz -> RE-ENCODE the same node input ->
                # the other half. Same total layer count and weight shape as
                # baseline, so the comparison isolates re-injection, not depth.
                half = NUM_LAYERS // 2
                _apply_ansatz(ansatz_, ansatz_wires, weights[:half])
                _encode(encoding_, inputs, sys_wires, p_noise, noisy, batched)
                _apply_ansatz(ansatz_, ansatz_wires, weights[half:])
            elif mode_ == "mixed_channel":
                # rho -> (1-lambda) rho + lambda U rho U^dag, realised as a genuine
                # channel: RY on the ancilla, controlled-U on the system, ancilla
                # traced out (never measured or returned).
                qml.RY(mix_angle, wires=mix_wire)
                qml.ctrl(_apply_ansatz, control=mix_wire)(ansatz_, ansatz_wires, weights)
            else:
                _apply_ansatz(ansatz_, ansatz_wires, weights)

            if noisy:
                _noise_all(p_noise, ansatz_wires)

            if readout_ == "scalar":
                return qml.expval(qml.PauliZ(0))
            elif readout_ == "root_multi_pauli":
                return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            elif readout_ == "level1_survivors":
                return [qml.expval(qml.PauliZ(w)) for w in sys_wires]
            elif readout_ == "top_layer_multi_pauli":
                return [qml.expval(P(w)) for w in sys_wires for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]
            raise ValueError(f"Unknown readout: {readout_}")

        @qml.qnode(dev_clean, interface="torch")
        def circuit_clean(inputs, weights, pos_angles=None, mix_angle=None):
            return _body(inputs, weights, 0.0, pos_angles, mix_angle)

        @qml.qnode(dev_noisy, interface="torch")
        def circuit_noisy(inputs, weights, p_noise, pos_angles=None, mix_angle=None):
            return _body(inputs, weights, p_noise, pos_angles, mix_angle, batched=True)

        @qml.qnode(dev_noisy, interface="torch")
        def circuit_noisy_single(inputs, weights, p_noise, pos_angles=None, mix_angle=None):
            """One row, no broadcast dimension -- the only formulation that is
            correct on default.mixed for every encoding/ansatz combination here."""
            return _body(inputs, weights, p_noise, pos_angles, mix_angle, batched=False)

        self._circuit_clean = circuit_clean
        self._circuit_noisy = circuit_noisy
        self._circuit_noisy_single = circuit_noisy_single
        self._noisy_rowwise = self._probe_noisy_broadcasting(enc_dim=3 if encoding == "multi_axis" else 1)

    def _probe_noisy_broadcasting(self, enc_dim, rows=4, tol=1e-8):
        """Is batched evaluation on `default.mixed` trustworthy for this circuit?

        At p=0 the DepolarizingChannel is exactly the identity, so the noisy
        device must reproduce the clean device to machine precision. If it does
        not (or returns the wrong shape), batching is broken for this circuit
        and every noisy evaluation must go row-by-row. See _run_noisy.
        """
        with torch.no_grad():
            probe = torch.rand(rows, NUM_QUBITS, enc_dim)
            kwargs = {}
            if self.use_ancilla:
                kwargs["pos_angles"] = torch.rand(rows, 2)
            if self.mode == "mixed_channel":
                kwargs["mix_angle"] = self.mix_angle

            def _flat(res):
                if isinstance(res, (list, tuple)):
                    return torch.stack([torch.as_tensor(r) for r in res], dim=-1)
                return torch.as_tensor(res).unsqueeze(-1)

            try:
                clean = _flat(self._circuit_clean(probe, self.weights, **kwargs))
                noisy = _flat(self._circuit_noisy(probe, self.weights, 0.0, **kwargs))
            except Exception:
                return True
            if noisy.shape != clean.shape:
                return True
            if bool((noisy - clean).abs().max() > tol):
                return True

            # Batched looks fine; confirm the unbatched fallback agrees with it,
            # so the two paths are known to be interchangeable rather than merely
            # each self-consistent.
            try:
                single = torch.stack(
                    [
                        _flat(
                            self._circuit_noisy_single(
                                probe[i],
                                self.weights,
                                0.0,
                                **{
                                    k: (v[i] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == rows else v)
                                    for k, v in kwargs.items()
                                },
                            )
                        ).reshape(-1)
                        for i in range(rows)
                    ]
                )
            except Exception:
                return False  # batched path verified good; fallback unavailable but unneeded
            return bool((single - clean.reshape(rows, -1)).abs().max() > tol)

    def forward(self, x, p_noise=0.0):
        # x: [Batch, Nodes, NUM_QUBITS, enc_dim]
        batch, nodes, k, enc_dim = x.shape
        x_flat = x.reshape(batch * nodes, k, enc_dim)

        kwargs = {}
        if self.use_ancilla:
            # Quadrant index 0..3 repeats per batch item, matching node order.
            quadrant_idx = torch.arange(nodes).repeat(batch)
            kwargs["pos_angles"] = self.pos_weights[quadrant_idx]
        if self.mode == "mixed_channel":
            kwargs["mix_angle"] = self.mix_angle

        if p_noise > 0:
            # Chunked: works around a default.mixed broadcasting bug that would
            # otherwise silently return the wrong number of rows. See _run_noisy.
            out = _run_noisy(
                self._circuit_noisy,
                self._circuit_noisy_single,
                x_flat,
                self.weights,
                p_noise,
                rowwise=self._noisy_rowwise,
                **kwargs,
            ).float()
        else:
            q_out = self._circuit_clean(x_flat, self.weights, **kwargs)
            if isinstance(q_out, (list, tuple)):
                out = torch.stack(list(q_out), dim=-1).float()
            else:
                out = q_out.unsqueeze(-1).float()

        assert out.shape[0] == batch * nodes, f"expected {batch*nodes} rows, got {out.shape[0]}"
        return out.view(batch, nodes, out.shape[-1])


class HierarchicalQTTNClassifier(nn.Module):
    def __init__(
        self,
        readout="scalar",
        encoding="scalar_ry",
        ansatz="strongly_entangling",
        img_size=16,
        patch_size=4,
        n_classes=4,
        mode="baseline",
        use_ancilla=False,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid_dim = img_size // patch_size
        assert self.grid_dim == 4, "level-1 grouping assumes a 4x4 patch grid (16 patches -> 4 nodes of 4)"

        self.encoding = encoding
        enc_dim = 3 if encoding == "multi_axis" else 1
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, enc_dim)

        # The spatial ancilla is level-1 only: "which quadrant" is meaningful for
        # the 4 patch-pooling nodes, but the root is a single node with no
        # position to encode (matching investigate_spatial_ancilla.py's design).
        self.level1 = QuantumNode(encoding=encoding, ansatz=ansatz, is_root=False, mode=mode, use_ancilla=use_ancilla)
        self.level2 = QuantumNode(
            encoding="scalar_ry", ansatz=ansatz, readout=readout, is_root=True, mode=mode, use_ancilla=False
        )
        self.head = build_head(READOUT_DIM[readout], n_classes)

    @staticmethod
    def _group_2x2(grid):
        # grid: [B, H, W, enc_dim] -> [B, (H/2)*(W/2), 4, enc_dim]
        b, h, w, d = grid.shape
        grid = grid.view(b, h // 2, 2, w // 2, 2, d)
        grid = grid.permute(0, 1, 3, 2, 4, 5).contiguous()
        return grid.view(b, (h // 2) * (w // 2), 4, d)

    def forward(self, x, p_noise=0.0):
        b = x.shape[0]
        x = x.unfold(2, 4, 4).unfold(3, 4, 4)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(b, 16, 48)

        angles = torch.tanh(self.patch_embed(x)) * np.pi  # [B, 16, enc_dim]
        enc_dim = angles.shape[-1]
        angles = angles.view(b, self.grid_dim, self.grid_dim, enc_dim)

        x1 = self._group_2x2(angles)  # [B, 4, 4, enc_dim]
        n1 = self.level1(x1, p_noise=p_noise)  # [B, 4, 1] -- level-1 message is always scalar

        x2 = n1.view(b, 1, 4, 1)  # 4 level-1 scalars become the 4 wire inputs of the root node
        n2 = self.level2(x2, p_noise=p_noise)  # [B, 1, READOUT_DIM[readout]]

        return apply_head(self.head, n2.squeeze(1))


# =====================================================================
# Coherent QTTN (Task R7) -- the architecture as designed
# =====================================================================
# `HierarchicalQTTNClassifier` above is NOT a coherent quantum tree: each
# QuantumNode owns its own 4-qubit device, level-1 nodes return a single <Z>,
# and those *classical scalars* are re-encoded as rotation angles into a
# separate root circuit. That gives zero entanglement across tree levels and an
# inter-level bond of one real number -- narrower than the chi=2 qubit the
# design specifies. See research_log.md 2026-07-28 "Code Audit #2".
#
# The class below is the real architecture: ONE device holding every patch
# qubit, level-1 block unitaries, then the level-2 unitary applied directly to
# the surviving qubits, which stay quantum until a single measurement at the
# end. Reference implementation: train_synthetic_shapes.py:50-70 (2026-07-17),
# which this restores.
#
# Cost: ~17.5 min per 30-epoch run on lightning.qubit with adjoint
# differentiation, vs ~15s for the hybrid, because the hybrid only ever
# simulated 4 qubits at a time. lightning+adjoint is ~4x faster than
# default.qubit+backprop here and is what makes this affordable at all.
#
# Noiseless only, per the 2026-07-28 scope decision: a 16-qubit density-matrix
# simulation needs ~68GB. Noise work is Task R8, at single-node scale.

PATCH_WIRE_IS_PATCH_INDEX = True  # wire w always carries patch w; keeps indexing trivial


def _encode_on(encoding, inputs, wires):
    """Encode patch `w` onto wire `w` for every w in `wires`.

    inputs: [B, n_patches, enc_dim]. Unlike `_encode`, which indexes by position
    within a node, this indexes by absolute patch/wire number -- needed because
    the coherent circuit holds all patches on one device simultaneously.
    """
    for w in wires:
        if encoding == "scalar_ry":
            qml.RY(inputs[:, w, 0], wires=w)
        elif encoding == "multi_axis":
            qml.RX(inputs[:, w, 0], wires=w)
            qml.RY(inputs[:, w, 1], wires=w)
            qml.RZ(inputs[:, w, 2], wires=w)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")


class CoherentQTTNClassifier(nn.Module):
    """16 patches -> 4 level-1 blocks -> 1 root, as a single coherent circuit.

    Wire w carries patch w. Level-1 unitaries act on [0-3], [4-7], [8-11],
    [12-15] with weights SHARED across the four blocks (matching the hybrid, so
    parameter counts stay comparable). The level-2 unitary then acts directly on
    the survivors [0, 4, 8, 12] -- these are still entangled with their
    subtrees, which is the property the hybrid destroyed.

    Supported modes:
      baseline -- encode, ansatz per node.
      reupload -- each node's ansatz is split in half and that node's patch
                  inputs are re-encoded in between. At level 2 the node's
                  "inputs" are the patches sitting on the survivor wires
                  (0, 4, 8, 12), so the same rule applies uniformly at both
                  levels rather than being special-cased.

    `mixed_channel` and `use_ancilla` are deliberately NOT ported: R3 rejected
    both decisively at node level (-10.9 and -19.5 pts), and each ancilla
    doubles the state vector (20-21 wires => 16-32x cost). Re-confirming a
    rejection at that price is not worth it. Their node-level verdicts stand as
    node-level results.

    `positional` is a SEPARATE axis from `mode`, added for CLEVR task C4
    (Question C.2 -- does the implicit tree topology encode spatial relations, or
    is an explicit position mechanism needed?). It is deliberately not a `mode`
    value, so the guard that `mode` only ever accepts baseline/reupload keeps
    holding.
      "none"    -- current behaviour: position is implicit in the wire layout
                   (wire w carries patch w) and in per-block level-1 weights.
      "on_wire" -- a learned per-patch rotation pair on the patch's OWN wire.
                   Zero extra wires, so cost is unchanged. NOTE: this is a
                   per-patch positional *encoding*, NOT an ancilla -- a per-patch
                   ancilla would be 16 extra wires (32 qubits), which statevector
                   simulation cannot reach. Report it under that name; R3's
                   -19.5 pts measured a per-quadrant ancilla and is a different
                   mechanism again.
    """

    def __init__(
        self,
        readout="root_multi_pauli",
        encoding="multi_axis",
        ansatz="iqp",
        img_size=16,
        patch_size=4,
        n_classes=4,
        mode="baseline",
        share_level1_weights=True,
        device_name="lightning.qubit",
        diff_method="adjoint",
        positional="none",
    ):
        super().__init__()
        assert readout in READOUTS
        readout = READOUT_ALIASES.get(readout, readout)
        assert encoding in ENCODINGS
        assert ansatz in ANSATZE
        if positional == "ancilla2":
            raise NotImplementedError(
                "positional='ancilla2' is not built yet, by decision: it is the escalation arm for "
                "CLEVR task C4 and is only worth its ~4x cost (18 wires) if the free 'on_wire' arm "
                "shows a RESOLVED effect. Run C4 with positional='on_wire' first."
            )
        if positional not in POSITIONALS:
            raise ValueError(f"positional must be one of {POSITIONALS}, got {positional!r}")
        if mode not in ("baseline", "reupload"):
            raise NotImplementedError(
                f"mode={mode!r} is not ported to the coherent tree. R3 rejected mixed_channel "
                f"(-10.9 pts) and the spatial ancilla (-19.5 pts) decisively, and each ancilla "
                f"doubles the statevector. See the class docstring."
            )
        self.grid_dim = img_size // patch_size
        self.n_patches = self.grid_dim**2
        assert self.n_patches == 16, "coherent tree currently assumes a 4x4 patch grid (16 patches)"
        self.n_wires = self.n_patches
        self.encoding, self.ansatz, self.readout, self.mode = encoding, ansatz, readout, mode

        enc_dim = 3 if encoding == "multi_axis" else 1
        self.patch_embed = nn.Linear(patch_size * patch_size * 3, enc_dim)
        # Level-1 weights either shared across the four blocks (parameter parity
        # with the hybrid) or per-block. train_synthetic_shapes.py (2026-07-17),
        # which reached 75%, used PER-BLOCK weights, so this is an axis worth
        # controlling rather than inheriting.
        self.share_level1_weights = share_level1_weights
        l1_shape = (NUM_LAYERS, NUM_QUBITS, 3) if share_level1_weights else (4, NUM_LAYERS, NUM_QUBITS, 3)
        self.weights_l1 = nn.Parameter(torch.randn(*l1_shape) * 0.1)
        self.weights_l2 = nn.Parameter(torch.randn(NUM_LAYERS, NUM_QUBITS, 3) * 0.1)
        self.head = build_head(READOUT_DIM[readout], n_classes)

        # Per-patch position angles, shared across the batch (position is a
        # property of the wire, not of the sample). 32 parameters at 16 patches.
        self.positional = positional
        self.pos_weights = nn.Parameter(torch.randn(self.n_patches, 2) * 0.1) if positional == "on_wire" else None

        self.l1_blocks = [[4 * b + i for i in range(NUM_QUBITS)] for b in range(4)]
        self.l2_wires = [4 * b for b in range(4)]
        blocks, l2w = self.l1_blocks, self.l2_wires
        enc_, ans_, ro_, mode_, pos_ = encoding, ansatz, readout, mode, positional
        half = NUM_LAYERS // 2

        def _node(wires, weights, inputs):
            if mode_ == "reupload":
                _apply_ansatz(ans_, wires, weights[:half])
                _encode_on(enc_, inputs, wires)
                _apply_ansatz(ans_, wires, weights[half:])
            else:
                _apply_ansatz(ans_, wires, weights)

        def _prepare(inputs, pos):
            """Data encoding, then (optionally) the explicit position imprint.

            Position goes AFTER the data so it acts on an already-encoded qubit
            rather than being overwritten by it -- with multi_axis the encoding
            starts from |0> and fully determines the state, so a positional
            rotation applied first would be erased.
            """
            _encode_on(enc_, inputs, range(self.n_wires))
            if pos_ == "on_wire":
                for w in range(self.n_wires):
                    qml.RX(pos[w, 0], wires=w)
                    qml.RY(pos[w, 1], wires=w)

        def _body(inputs, w1, w2, pos=None):
            _prepare(inputs, pos)
            for bi, blk in enumerate(blocks):
                _node(blk, w1 if share_level1_weights else w1[bi], inputs)
            _node(l2w, w2, inputs)
            if ro_ == "scalar":
                return [qml.expval(qml.PauliZ(0))]
            if ro_ == "root_multi_pauli":
                return [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))]
            if ro_ == "level1_survivors":
                return [qml.expval(qml.PauliZ(w)) for w in l2w]
            if ro_ == "top_layer_multi_pauli":
                return [qml.expval(P(w)) for w in l2w for P in (qml.PauliX, qml.PauliY, qml.PauliZ)]
            raise ValueError(f"Unknown readout: {ro_}")

        dev = qml.device(device_name, wires=self.n_wires)
        self._circuit = qml.QNode(_body, dev, interface="torch", diff_method=diff_method)

        # Coherence diagnostic. A single qubit is mixed -- i.e. entangled with
        # the rest of the register -- exactly when its Bloch vector is shorter
        # than 1, since purity = (1 + |r|^2) / 2. Measuring three expectation
        # values is far cheaper than a reduced density matrix (vn_entropy at 16
        # qubits OOMs) and is equally decisive: in the hybrid the level-2 input
        # wires are freshly encoded from classical scalars, hence pure, hence
        # |r| = 1 exactly.
        def _bloch_body(inputs, w1, w2, wire, pos=None):
            _prepare(inputs, pos)
            for bi, blk in enumerate(blocks):
                _node(blk, w1 if share_level1_weights else w1[bi], inputs)
            return [qml.expval(qml.PauliX(wire)), qml.expval(qml.PauliY(wire)), qml.expval(qml.PauliZ(wire))]

        self._bloch = qml.QNode(_bloch_body, dev, interface="torch", diff_method=None)

    def _patches(self, x):
        b = x.shape[0]
        ps = x.shape[-1] // self.grid_dim
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, self.n_patches, -1)
        return torch.tanh(self.patch_embed(x)) * np.pi  # [B, 16, enc_dim]

    def forward(self, x, p_noise=0.0):
        if p_noise > 0:
            raise NotImplementedError(
                "The coherent tree is noiseless-only (2026-07-28 scope decision): a 16-qubit "
                "density-matrix simulation needs ~68GB. Noise work is Task R8, at single-node scale."
            )
        angles = self._patches(x)
        out = self._circuit(angles, self.weights_l1, self.weights_l2, self.pos_weights)
        out = torch.stack([torch.as_tensor(o) for o in out], dim=-1).float()
        return apply_head(self.head, out)

    def survivor_bloch_length(self, x, wire=0):
        """Bloch-vector length of a survivor qubit after level 1, averaged over
        the batch. Strictly below 1 iff that qubit is entangled with the rest of
        the tree, which is the property the measure-and-re-encode hybrid lacks
        (there it is exactly 1). Used by the coherence regression test.
        """
        with torch.no_grad():
            r = self._bloch(self._patches(x), self.weights_l1, self.weights_l2, wire, self.pos_weights)
            r = torch.stack([torch.as_tensor(v) for v in r], dim=-1)
            return float(r.norm(dim=-1).mean())
