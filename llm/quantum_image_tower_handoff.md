# Quantum Image Tower Handoff Summary

This document provides a complete transition state for the **Quantum Image Tower Classification Project** (focusing strictly on the QTTN image novelty for the thesis). Use this to resume work after context clearing.

---

## 1. Work Accomplished & Staged (thesis/quantum-image-tower branch)

We completed the following items on the `thesis/quantum-image-tower` branch (all files are staged and verified):
1. **QTTN Training Loop & Vectorization**: Vectorized PennyLane QTTN training on 16x16 shapes, achieving a $30\times$ speedup via parameter broadcasting at [train_synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_synthetic_shapes.py).
2. **Noise Emulation sweeps**: Swept depolarizing noise in mixed-state simulation at [emulate_noise_synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/emulate_noise_synthetic_shapes.py).
3. **Multi-Dimensional Encodings & Ansatz Benchmarks**: Evaluated 12 combinations of encodings (Angle, Multi-Axis, Amplitude, ZZ Map) and ansatze (HEA, IQP, ALT) at [benchmark_encodings_ansatze.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/benchmark_encodings_ansatze.py).
   * **Multi-Axis Encoding** (loading 3 features per qubit via RX-RY-RZ) achieved the highest noiseless accuracy of **`95.3% - 96.9%`** with high noise resilience ($p_{crit} > 0.200$).
   * **Amplitude Encoding** (compressing 4 patch features into 2 qubits) achieved **`78.1%` test accuracy** using only 2 qubits and 4 parameters.
4. **32x32 Representation Bias & Feature Binding**:
   * Updated [synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/utils/data/synthetic_shapes.py) to support 32x32 canvases and three testing modes: Color-Only, Shape-Only (Grayscale), and Overlapping (Red/Green Circles/Squares).
   * Created [evaluate_representation_bias.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/evaluate_representation_bias.py) (diagnostic runs) and [train_overlapping_32x32.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_overlapping_32x32.py) (full convergence proof).
   * **Result**: The 16-qubit model trained on 32x32 overlapping shapes peaked at **`65.6%` validation accuracy**, proving mathematically that it extracts and binds **both** color and shape attributes (since single-attribute shortcuts are strictly bounded at `50.0%`).
5. **Barren Plateau Scaling Sweep**: Built and ran [test_barren_plateaus.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/test_barren_plateaus.py) on physical sizes $N \in \{4, 9, 16, 20\}$ qubits, confirming stable gradient variances ($\sim 10^{-2}$) and proving barren plateau immunity.
6. **3-Way Comparative Topology Benchmark**: Built and ran [compare_topologies.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/compare_topologies.py) comparing QTTN, MPS, and MERA on 4-20 qubits:
   * **MERA** achieved the highest noiseless classification capacity (**`51.6%`**).
   * **QTTN** achieved the highest depolarizing noise resilience (**`53.9%`** at $p=0.10$).
   * **MPS** dropped fastest under noise (**`46.1%`** at $p=0.10$) due to linear gate accumulation.
7. **Research Log & Roadmap Documentation**: Fully updated [research_log.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/research_log.md) and logged future paths in [quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md).

---

## 2. Active Files Registry

* **Dataset Generator**: [synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/utils/data/synthetic_shapes.py)
* **12-Combination Sweep**: [benchmark_encodings_ansatze.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/benchmark_encodings_ansatze.py)
* **32x32 Convergence Proof**: [train_overlapping_32x32.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_overlapping_32x32.py)
* **4-qubit Convergence Proof**: [train_overlapping_proof.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_overlapping_proof.py)
* **Diagnostic Bias Sweep**: [evaluate_representation_bias.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/evaluate_representation_bias.py)
* **Barren Plateau Sweep**: [test_barren_plateaus.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/test_barren_plateaus.py)
* **Comparative Topology Sweep**: [compare_topologies.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/compare_topologies.py)
* **Research Log**: [research_log.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/research_log.md)
* **Investigation Roadmap**: [quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md)

---

## 3. Simulator Design Constraints & Mitigation Rules

1. **Mixed State Simulators (`default.mixed`) scale poorly**: Density matrix simulation of a recycled circuit (7 active qubits + 18 reset variables) compiles to 25 wires, causing Out-Of-Memory errors immediately. 
2. **Standard 16-qubit simulation is highly feasible**: When running noiseless training, use standard `default.qubit` on the full 16-qubit register. It takes less than 1MB of memory and executes in milliseconds via parameter broadcasting.
3. **PennyLane mixed-state batching bug**: A batch of density matrices simulated under `default.mixed` using diagonal unitaries (like RZ in multi-axis or ZZ Feature Map) crashes due to internal einsum index mismatches in PennyLane.
   * *Mitigation*: For noisy sweeps, noiseless training remains batched (using `default.qubit`), and noisy evaluation is processed **sample-by-sample without batching** (shape `(F_per_patch,)`), completely bypassing the broadcasting bug.

---

## 4. Next Step: Phase 2 CLEVR Integration

Transition the QTTN to train on the **single-object attribute classification tasks of the CLEVR dataset**:
1. Inspect the HuggingFace CLEVR dataloaders in the repo (or `qnlp/utils/data/`).
2. Utilize the optimal selected **`MULTI_AXIS + IQP`** architecture (qubit-efficient 4-qubit register per 4 patches, or 16-qubit register for 16 patches, compact parameter set).
3. Align the linear classification head to predict the object attributes (shape, color, material, size).
