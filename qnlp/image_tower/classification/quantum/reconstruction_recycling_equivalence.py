import time

import numpy as np
import pennylane as qml


# =====================================================================
# Circuit Node Definition (3 CNOT optimal ansatz)
# =====================================================================
def quad_node_unitary(wires, weights):
    """
    Standard block unitary mapping 4 children to 1 parent.
    Uses 1-qubit rotations followed by 3 CNOTs to entangle
    all children (wires[1, 2, 3]) directly to the parent (wires[0]).
    """
    # 1. Local Rotations
    for i, w in enumerate(wires):
        qml.RZ(weights[i, 0], wires=w)
        qml.RY(weights[i, 1], wires=w)
        qml.RZ(weights[i, 2], wires=w)

    # 2. Entanglement (All children to parent)
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.CNOT(wires=[wires[2], wires[0]])
    qml.CNOT(wires=[wires[3], wires[0]])


# =====================================================================
# 1. Standard 16-Qubit QTTN
# =====================================================================
dev_std = qml.device("default.qubit", wires=16)


@qml.qnode(dev_std)
def standard_qttn(inputs, weights):
    # State Preparation: Load 16 patch inputs
    for i in range(16):
        qml.RY(inputs[i], wires=i)

    # Layer 1: 16 patches -> 4 parents (located on wires 0, 4, 8, 12)
    for b in range(4):
        block_wires = [4 * b, 4 * b + 1, 4 * b + 2, 4 * b + 3]
        quad_node_unitary(block_wires, weights[b])

    # Layer 2: 4 parents -> 1 root (located on wire 0)
    quad_node_unitary([0, 4, 8, 12], weights[4])

    return qml.expval(qml.PauliZ(0))


# =====================================================================
# 2. Recycled 7-Qubit QTTN
# =====================================================================
# Wires:
# - [0, 1, 2, 3]: Parent storage qubits (P0, P1, P2, P3)
# - [4, 5, 6]: Scratchpad qubits (S1, S2, S3)
dev_rec = qml.device("default.qubit", wires=25)


def recycle_wires(wires):
    """
    Resets a list of wires back to state |0> using mid-circuit measurements.
    """
    for w in wires:
        qml.measure(w, reset=True)


@qml.qnode(dev_rec)
def recycled_qttn(inputs, weights):
    # --- Block 0 ---
    # Load inputs [0, 1, 2, 3] on [P0, S1, S2, S3]
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=4)
    qml.RY(inputs[2], wires=5)
    qml.RY(inputs[3], wires=6)

    # Process Block 0. Parent is stored in wire 0 (P0)
    quad_node_unitary([0, 4, 5, 6], weights[0])

    # Recycle scratchpad
    recycle_wires([4, 5, 6])

    # --- Block 1 ---
    # Load inputs [4, 5, 6, 7] on [P1, S1, S2, S3]
    qml.RY(inputs[4], wires=1)
    qml.RY(inputs[5], wires=4)
    qml.RY(inputs[6], wires=5)
    qml.RY(inputs[7], wires=6)

    # Process Block 1. Parent is stored in wire 1 (P1)
    quad_node_unitary([1, 4, 5, 6], weights[1])

    # Recycle scratchpad
    recycle_wires([4, 5, 6])

    # --- Block 2 ---
    # Load inputs [8, 9, 10, 11] on [P2, S1, S2, S3]
    qml.RY(inputs[8], wires=2)
    qml.RY(inputs[9], wires=4)
    qml.RY(inputs[10], wires=5)
    qml.RY(inputs[11], wires=6)

    # Process Block 2. Parent is stored in wire 2 (P2)
    quad_node_unitary([2, 4, 5, 6], weights[2])

    # Recycle scratchpad
    recycle_wires([4, 5, 6])

    # --- Block 3 ---
    # Load inputs [12, 13, 14, 15] on [P3, S1, S2, S3]
    qml.RY(inputs[12], wires=3)
    qml.RY(inputs[13], wires=4)
    qml.RY(inputs[14], wires=5)
    qml.RY(inputs[15], wires=6)

    # Process Block 3. Parent is stored in wire 3 (P3)
    quad_node_unitary([3, 4, 5, 6], weights[3])

    # Recycle scratchpad
    recycle_wires([4, 5, 6])

    # --- Layer 2 ---
    # Process the 4 stored parents [P0, P1, P2, P3] (wires [0, 1, 2, 3])
    quad_node_unitary([0, 1, 2, 3], weights[4])

    return qml.expval(qml.PauliZ(0))


# =====================================================================
# Main Experiment Execution
# =====================================================================
if __name__ == "__main__":
    print("====================================================")
    print("Running QTTN Equivalence & Scaling Validation...")
    print("====================================================")

    # Generate random input angles for 16 patches
    np.random.seed(1234)
    mock_inputs = np.random.uniform(0, 2 * np.pi, size=16)

    # Generate weights: 5 nodes in tree, each needs 4 wires * 3 rotations
    mock_weights = np.random.uniform(0, 2 * np.pi, size=(5, 4, 3))

    # 1. Run Standard QTTN (16 Qubits)
    t0 = time.time()
    std_out = standard_qttn(mock_inputs, mock_weights)
    std_time = time.time() - t0

    # 2. Run Recycled QTTN (7 Qubits)
    t0 = time.time()
    rec_out = recycled_qttn(mock_inputs, mock_weights)
    rec_time = time.time() - t0

    # 3. Validation and Analysis
    difference = np.abs(std_out - rec_out)

    print("\nRESULTS:")
    print("-" * 55)
    print(f"Standard QTTN Output (16 Qubits): {std_out:.8f}")
    print(f"Recycled QTTN Output (7 Qubits):  {rec_out:.8f}")
    print(f"Numerical Difference:             {difference:.8e}")
    print("-" * 55)
    print(f"Standard execution time:          {std_time:.4f} seconds")
    print(f"Recycled execution time:          {rec_time:.4f} seconds")
    print("-" * 55)

    # Assert equivalence to verify correctness
    tolerance = 1e-7
    if difference < tolerance:
        print("\nSUCCESS: Mathematical equivalence verified!")
        print(f"The outputs are identical up to {tolerance:.1e} tolerance.")
        print(f"Qubit width reduced from 16 to 7 (a {((16-7)/16)*100:.1f}% reduction).")
    else:
        print("\nFAILURE: Outputs do not match!")

    print("\n====================================================")
