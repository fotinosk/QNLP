# Quad-Tree VQC: From Tensors to Quanta

This project translates a classical **Tree Tensor Network (TTN)** classification 
model into a **Variational Quantum Circuit (VQC)** using PennyLane and PyTorch.

### 1. Conceptual Mapping
In the classical version, you used `einsum` to merge four vectors via CP-Decomposition. 
In the quantum version:
- **Input Data:** Encoded as qubit rotation angles.
- **CP-Rank Interaction:** Replaced by **Entangling Layers** (CNOTs).
- **Pooling:** We merge 4 qubits through a unitary gate but only use the 
  expectation value of 1 qubit as the "parent" feature.

### 2. Why Hybrid?
Pure quantum simulation is slow. We use a **Hybrid approach**:
1. **Classical Embedding:** A simple Linear layer reduces patch dimensions 
   to a manageable number of qubits (4).
2. **Quantum Core:** The hierarchical tree structure is handled by `qml.qnn.TorchLayer`.
3. **Classical Head:** Converts the final quantum measurement into class probabilities.

### 3. Pointers for Scaling
- **Bond Dimension:** If you want `BOND_DIM=64`, you must use 6 qubits per wire. 
  Note: This will make the 4-node interaction a 24-qubit simulation, which is 
  heavy for `default.qubit`.
- **Gradients:** Use `qml.qnode(dev, interface="torch", diff_method="backprop")` 
  for faster simulation training, or `"parameter-shift"` if deploying to real hardware.
- **Normalization:** Unlike your PyTorch code, quantum gates are **automatically unitary**. 
  You no longer need to manually normalize your factor weights!
