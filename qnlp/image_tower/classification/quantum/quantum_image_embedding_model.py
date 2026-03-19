import pennylane as qml
import torch
from torch import nn

# --- HYPERPARAMETERS ---
NUM_PATCHES = 16
# 4 qubits for the 2x2 pixel grid (RGB dense encoded) + 1 Ancilla for Position
QUBITS_PER_PATCH = 5
TOTAL_WIRES = NUM_PATCHES * QUBITS_PER_PATCH  # 80 Qubits total

dev = qml.device("default.qubit", wires=TOTAL_WIRES)


def dense_rgb_encoding(patch_data, wires):
    """Encodes the 4 pixels of RGB data into the first 4 wires of the patch."""
    for i, w in enumerate(wires):
        qml.RX(patch_data[i, 0], wires=w)
        qml.RY(patch_data[i, 1], wires=w)
        qml.RZ(patch_data[i, 2], wires=w)


def spatial_ancilla_encoding(pos_weights, wire):
    """
    Encodes the learned spatial position onto the dedicated 5th ancilla qubit.
    Uses RX and RY to map the 2D positional embedding onto the Bloch sphere.
    """
    qml.RX(pos_weights[0], wires=wire)
    qml.RY(pos_weights[1], wires=wire)


def quad_node_unitary(wires, weights):
    """
    The parameterized entangling ansatz.
    Applies local rotations then a ring of CNOTs to mix Color and Position.
    """
    for i, w in enumerate(wires):
        qml.RX(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)

    for i in range(len(wires)):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % len(wires)]])


@qml.qnode(dev, interface="torch")
def qttn_forward(inputs, pos_weights, layer_1_weights, layer_2_weights):
    # 1. State Prep: Load Color & Position separately per patch
    for i in range(NUM_PATCHES):
        start_wire = i * QUBITS_PER_PATCH
        color_wires = range(start_wire, start_wire + 4)
        ancilla_wire = start_wire + 4

        # Encode What (Color)
        dense_rgb_encoding(inputs[i], color_wires)
        # Encode Where (Learned Position)
        spatial_ancilla_encoding(pos_weights[i], ancilla_wire)

    # 2. Tree Contraction - LAYER 1 (16 patches -> 4 nodes)
    # We take blocks of 20 qubits (4 patches * 5 qubits).
    surviving_layer_1 = []
    for i in range(4):
        block_wires = list(range(i * 20, (i + 1) * 20))
        # This ansatz physically entangles the spatial ancilla with the color data
        quad_node_unitary(block_wires, layer_1_weights)
        # We trace out 16 qubits, keeping 4 general feature qubits for the next layer
        surviving_layer_1.extend(block_wires[:4])

    # 3. Tree Contraction - LAYER 2 (4 nodes -> 1 root node)
    # 16 surviving qubits total.
    quad_node_unitary(surviving_layer_1, layer_2_weights)
    root_wires = surviving_layer_1[:4]

    # 4. Extract continuous vector
    return [qml.expval(qml.PauliZ(w)) for w in root_wires]


class QuantumImageEmbedder(nn.Module):
    """
    Quantum Tree Tensor Network (QTTN) Image Embedder

    This module implements a Quantum adaptation of the classical CPQuadRankLayer
    vision model. It translates standard deep learning tensor operations into
    strictly unitary quantum circuits.

    ### Key Architectural Differences:

    1. Data Representation (Features -> Qubits)
       - Classical: A node is a floating-point vector of size `bond_dim`.
       - Quantum: A node is a quantum register. Data is embedded via Dense Angle
         Encoding, mapping 3 color channels to the R_x, R_y, and R_z axes of a single
         qubit's Bloch sphere.

    2. The "Weights" & Contraction (einsum -> Unitary Ansatz)
       - Classical: 4 child nodes are merged using `torch.einsum` with parameterized
         tensors (factors) of a specific `rank`.
       - Quantum: 4 child registers are entangled using a Parameterized Quantum Circuit
         (PQC). The classical weights become the learned rotation angles of the quantum
         gates. The tensor "rank" is naturally dictated by the depth of the CNOT
         entanglement ring.

    3. Dimensionality Reduction (Summation -> Partial Trace)
       - Classical: `einsum` mathematically sums over axes to reduce the shape.
       - Quantum: Reversible unitary matrices cannot change size. We reduce dimensions
         by performing a "partial trace"—physically ignoring/discarding 3 out of every
         4 qubits after entanglement, passing only the "surviving" qubits forward.

    4. Normalization (RMS Norm -> Unitary Evolution)
       - Classical: Requires `_rms_norm` and `LayerNorm` to prevent continuous matrix
         multiplications from exploding or vanishing.
       - Quantum: Operations are strictly unitary ($U^\\dagger U = I$). The $L_2$ norm
         of the quantum state vector is physically guaranteed to remain 1.0. No
         artificial normalization is needed inside the tree.

    5. The Output (Linear Head -> Observable Measurement)
       - Classical: A final linear layer projects the vector to the target embedding size.
       - Quantum: The continuous embedding is extracted by measuring the Pauli-Z
         expectation values of the surviving root qubits, which can then be classically
         scaled if needed.

    ### NOTE ON POSITIONAL EMBEDDINGS (Explicit vs. Implicit):
    In this implementation, we explicitly allocate a 5th 'Ancilla' qubit to each
    image patch strictly to hold learned spatial coordinates. This prevents the
    positional rotations from overwriting the color rotations on the Bloch sphere.

    However, mathematically, this explicit ancilla is not strictly necessary.
    A Quantum Tree Tensor Network naturally acts like a classical Convolutional
    Neural Network (CNN): the spatial geometry is implicitly encoded in the wiring
    diagram itself. Because Patch 1 is physically hardwired to interact only with
    Patches 2, 3, and 4 in the first layer, the parameterized ansatz naturally
    learns their spatial relationships without needing explicit coordinate data.
    The ancilla is included here as an educational demonstration of pure quantum
    tensor product state expansion.

    ### Dimensions:
    - Input: [Batch, 16 Patches, 4 Pixels, 3 Channels]
    - Quantum Qubits: 80 (16 patches * (4 color + 1 space))
    - Output: [Batch, final_embedding_dim]
    """

    def __init__(self, final_embedding_dim=512):
        super().__init__()
        # Positional weights: 16 patches, 2 angles (RX, RY) per patch
        self.pos_weights = nn.Parameter(torch.randn(NUM_PATCHES, 2) * 0.01)

        # Layer 1 acts on 20-qubit blocks (4 patches * 5 qubits)
        self.layer_1_weights = nn.Parameter(torch.randn(20, 3) * 0.1)

        # Layer 2 acts on the 16 surviving qubits
        self.layer_2_weights = nn.Parameter(torch.randn(16, 3) * 0.1)

        self.classical_head = nn.Linear(4, final_embedding_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        embeddings = []

        for i in range(batch_size):
            q_out = qttn_forward(x[i], self.pos_weights, self.layer_1_weights, self.layer_2_weights)
            q_out_stacked = torch.stack(q_out)
            embeddings.append(q_out_stacked)

        quantum_batch_out = torch.stack(embeddings)
        final_out = self.classical_head(quantum_batch_out)

        return nn.functional.normalize(final_out, p=2, dim=-1)
