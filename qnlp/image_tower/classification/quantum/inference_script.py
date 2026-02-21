import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

# Mapping: BOND_DIM 64 -> 6 Qubits.
# For 16 patches, that would be 96 qubits (too large for simulation).
# We use 1 qubit per patch for this visualization of the tree structure.
NUM_PATCHES = 16
QUBITS_PER_BOND = 1
TOTAL_WIRES = NUM_PATCHES * QUBITS_PER_BOND

dev = qml.device("default.qubit", wires=TOTAL_WIRES)


def quad_node_unitary(wires, weights):
    """The Quantum version of CPQuadRankLayer."""
    # 1. Local Rotations (The 'Factors')
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)

    # 2. Quad-Entanglement (The 'Merged Interaction')
    # Connects TL, TR, BL, BR in a loop
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[3], wires[0]])


@qml.qnode(dev)
def ttn_inference_circuit(inputs, weights):
    # State Preparation: Load patch data
    qml.AngleEmbedding(inputs, wires=range(TOTAL_WIRES))

    # --- LAYER 1: 16 Patches -> 4 Nodes ---
    # We process 4-qubit blocks. Qubits [0,1,2,3] merge, [4,5,6,7] merge, etc.
    # We assume 'surviving' info stays on the first qubit of each block.
    for i in range(0, TOTAL_WIRES, 4):
        quad_node_unitary(wires=[i, i + 1, i + 2, i + 3], weights=weights[0])

    # --- LAYER 2: 4 Nodes -> 1 Root ---
    # The 'surviving' qubits from Layer 1 are 0, 4, 8, 12
    survivors = [0, 4, 8, 12]
    quad_node_unitary(wires=survivors, weights=weights[1])

    return qml.expval(qml.PauliZ(0))


# Mock weights for visualization
# 2 layers, 4 inputs per node, 3 angles per input
weights = np.random.random((2, 4, 3))
inputs = np.random.random(TOTAL_WIRES)

# Generate Schematic
qml.drawer.use_style("black_white")
fig, ax = qml.draw_mpl(ttn_inference_circuit)(inputs, weights)
plt.title("Quad-Tree Variational Quantum Circuit")
plt.show()
