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

READOUTS = ("scalar", "root_multi_pauli", "level1_survivors")
ENCODINGS = ("scalar_ry", "multi_axis")
ANSATZE = ("strongly_entangling", "iqp")

# Node residual variants re-tested by Task R3b. Only the two mechanisms that
# showed any signal in the (bottlenecked) July-26 runs are implemented:
# near_identity had no effect in either regime and lcu/lcu-lite inherit
# mixed_channel's instability while remaining untestable under noise
# (default.mixed postselection limitation, PennyLane 0.43.2). See roadmap
# Section 7 R3b for why those are not re-run.
MODES = ("baseline", "reupload", "mixed_channel")

READOUT_DIM = {"scalar": 1, "root_multi_pauli": 3, "level1_survivors": NUM_QUBITS}


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
        self.head = nn.Linear(READOUT_DIM[readout], n_classes)

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

        return self.head(n2.squeeze(1))
