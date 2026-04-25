import pennylane as qml
from pennylane import numpy as np

# Let's assume 1 qubit per bond for simplicity in the code,
# but the logic scales to 6 qubits for your Bond Dim 64.
n_qubits = 16  # Represents 16 patches
dev = qml.device("default.qubit", wires=n_qubits)


def quantum_quad_node(wires, weights):
    """
    Acts as the CPQuadRankLayer.
    It entangles 4 incoming 'wires' (or bundles) and
    concentrates information.
    """
    # 1. Analog to 'factor_tl, tr, bl, br' (Local Rotations)
    for i, wire in enumerate(wires):
        qml.Rot(*weights[i], wires=wire)

    # 2. Analog to 'Merged interaction' (Entanglement)
    # We use a chain of CNOTs to mix the 4 branches
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[3], wires[0]])

    # 3. Analog to 'factor_out'
    qml.Rot(*weights[4], wires=wires[0])


@qml.qnode(dev)
def quad_tree_vqc(inputs, weights):
    """
    The full Quad-Tree structure.
    """
    # Step 1: State Preparation (Patch Embedding)
    # Encodes your 'patch_embed' output into quantum amplitudes
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Step 2: Layer 1 (16 patches -> 4 nodes)
    # Each node takes 4 qubits (TL, TR, BL, BR)
    weight_idx = 0
    for i in range(0, 16, 4):
        quantum_quad_node(wires=[i, i + 1, i + 2, i + 3], weights=weights[weight_idx])
        weight_idx += 1

    # Step 3: Layer 2 (4 nodes -> 1 final node)
    # We use the 'surviving' qubits from the previous layer: 0, 4, 8, 12
    quantum_quad_node(wires=[0, 4, 8, 12], weights=weights[weight_idx])

    # Measurement (Final classification head)
    return qml.expval(qml.PauliZ(0))


# Example shapes
# 5 nodes total in this tree, each node needs 5 sets of 3 Euler angles
weights = np.random.random((5, 5, 3), requires_grad=True)
inputs = np.random.random(16)
print(quad_tree_vqc(inputs, weights))
