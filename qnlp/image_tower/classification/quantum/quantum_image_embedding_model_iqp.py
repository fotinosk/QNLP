import pennylane as qml
import torch
from pennylane import numpy as pnp
from torch import nn

# --- HYPERPARAMETERS ---
NUM_PATCHES = 16
QUBITS_PER_PATCH = 5  # 4 for RGB + 1 for Spatial Ancilla
TOTAL_WIRES = NUM_PATCHES * QUBITS_PER_PATCH  # 80 Qubits total

dev = qml.device("default.qubit", wires=TOTAL_WIRES)


def dense_rgb_encoding(patch_data, wires):
    for i, w in enumerate(wires):
        qml.RX(patch_data[i, 0], wires=w)
        qml.RY(patch_data[i, 1], wires=w)
        qml.RZ(patch_data[i, 2], wires=w)


def spatial_ancilla_encoding(pos_weights, wire):
    qml.RX(pos_weights[0], wires=wire)
    qml.RY(pos_weights[1], wires=wire)


def iqp_node_unitary(wires, weights):
    """
    Trainable IQP-inspired Ansatz.
    weights shape: (len(wires), 2)
    - Column 0: Local RZ angles (Z-fields)
    - Column 1: Ising ZZ coupling angles (Interactions)
    """
    # 1. Enter Superposition (The X-basis)
    for w in wires:
        qml.Hadamard(wires=w)

    # 2. Local Phase Shifts (Parameterized Z-rotations)
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)

    # 3. The Entanglement Layer (Parameterized ZZ Couplings)
    # We connect adjacent wires in a ring topology
    for i in range(len(wires)):
        w1 = wires[i]
        w2 = wires[(i + 1) % len(wires)]
        qml.IsingZZ(weights[i, 1], wires=[w1, w2])

    # 4. Return to the Computational Basis (The Z-basis)
    for w in wires:
        qml.Hadamard(wires=w)


@qml.qnode(dev, interface="torch")
def qttn_forward(inputs, pos_weights, layer_1_weights, layer_2_weights):
    # 1. State Prep
    for i in range(NUM_PATCHES):
        start_wire = i * QUBITS_PER_PATCH
        color_wires = range(start_wire, start_wire + 4)
        ancilla_wire = start_wire + 4

        dense_rgb_encoding(inputs[i], color_wires)
        spatial_ancilla_encoding(pos_weights[i], ancilla_wire)

    # 2. Tree Contraction - LAYER 1 (IQP blocks on 20-qubit groups)
    surviving_layer_1 = []
    for i in range(4):
        block_wires = list(range(i * 20, (i + 1) * 20))
        iqp_node_unitary(block_wires, layer_1_weights)
        surviving_layer_1.extend(block_wires[:4])

    # 3. Tree Contraction - LAYER 2 (IQP block on 16 surviving qubits)
    iqp_node_unitary(surviving_layer_1, layer_2_weights)
    root_wires = surviving_layer_1[:4]

    # 4. Extract continuous vector
    return [qml.expval(qml.PauliZ(w)) for w in root_wires]


class QuantumIQPImageEmbedder(nn.Module):
    def __init__(self, final_embedding_dim=512):
        super().__init__()
        self.pos_weights = nn.Parameter(torch.randn(NUM_PATCHES, 2) * 0.01)

        # IQP weights now require 2 angles per wire (RZ and ZZ)
        # Layer 1 acts on 20-qubit blocks. Shape: [20, 2]
        self.layer_1_weights = nn.Parameter(torch.randn(20, 2) * 0.1)

        # Layer 2 acts on the 16 surviving qubits. Shape: [16, 2]
        self.layer_2_weights = nn.Parameter(torch.randn(16, 2) * 0.1)

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


def export_full_circuit(qnode, *args, filename="full_circuit_diagram.txt"):
    """
    Generates a complete text-based diagram of any QNode and saves it to a file.
    This bypasses visual rendering limitations, allowing you to view massive
    circuits (like 80+ qubits) in any standard text editor.
    """
    num_wires = len(qnode.device.wires)
    print(f"Tracing quantum circuit across {num_wires} wires. This may take a moment...")

    # Generate the text representation
    # We use qml.draw() instead of qml.draw_mpl() for pure text output
    circuit_text = qml.draw(qnode)(*args)

    # Save to a text file with UTF-8 to preserve quantum gate symbols
    with open(filename, "w", encoding="utf-8") as f:
        f.write(circuit_text)

    print(f"Success! Full circuit successfully exported to '{filename}'.")
    print("Tip: Open this file in a text editor with 'word-wrap' disabled to explore the full architecture.")

    return circuit_text


if __name__ == "__main__":
    mock_inputs = pnp.random.random((16, 4, 3))
    mock_pos = pnp.random.random((16, 2))  # 16 patches, 2 spatial angles
    mock_layer_1 = pnp.random.random((20, 2))  # Layer 1 weights
    mock_layer_2 = pnp.random.random((16, 2))  # Layer 2 weights

    # 2. Pass the QNode and the arguments to the export function
    circuit_string = export_full_circuit(
        qttn_forward, mock_inputs, mock_pos, mock_layer_1, mock_layer_2, filename="iqp_80_qubit_model.txt"
    )
