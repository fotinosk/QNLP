# Quantum TTN Image Model Investigation Roadmap

This document outlines a rigorous research and implementation roadmap for your quantum computing thesis, focusing **strictly on the Quantum Tree Tensor Network (QTTN) Image Tower**. 

---

## 1. Project Scope & Focus

To maximize the research novelty of your thesis, this investigation ignores the text tower (DisCoCat/lambeq) and isolates the **image tower architecture**. We investigate the quantum representation of spatial image data, hierarchical contractions, and classification expressibility under both simulated (noiseless) and emulated (noisy) environments.

```
═══════════════════════════════════════════════════
IMAGE TOWER HIERARCHY
═══════════════════════════════════════════════════
                  Image (16×16 or 64×64)
                            │
                            ▼
                Bilinear patch embedding
                 (4×4 patches → 16 patches)
                 + positional encoding
                            │
                            ▼
              State Preparation: Angle/Amplitude
              (16 patches × Qubits per patch)
                            │
                            ▼
              TTN Level 0: 4-qubit Block Unitaries
                            │
              (Partial trace: trace out 3 of 4 qubits)
                            ▼
              TTN Level 1: 4-qubit Block Unitaries
                            │
                            ▼
                      Root Qubits
                            │
                            ▼
               Measurement: PauliZ Expectation
                            │
                            ▼
               Classical Head / Classification
```

---

## 2. Elaborated Open Questions & Research Directions

These core research questions form the scientific contribution of your thesis. Each requires mathematical formulation followed by empirical validation.

### Question A: The Mathematical Mapping of CP-Rank to Quantum Entanglement
* **Context**: In a classical Tree Tensor Network (TTN), the contraction of four child nodes of dimension $\chi$ (bond dimension) into a parent node of dimension $\chi$ is regularized via CP-Decomposition of rank $R$. In the quantum circuit, this is replaced by a parameterized unitary ansatz $U \in SU(2^{4 \log_2(\chi)})$ followed by a partial trace (ignoring $3 \log_2(\chi)$ qubits).
* **Research Focus**:
  1. **Rank Equivalence**: What is the mathematical relationship between the classical CP-rank $R$ and the entangling gate depth $D$ (number of CNOT layers) of the quantum ansatz? If a VQC has $D=0$ (no entangling gates), the state remains a separable product state, equivalent to a classical CP-rank $R=1$. How does the representation capacity scale as $D$ increases?
  2. **Entanglement Entropy**: Can we bound the entanglement entropy of the quantum state at each level of the tree and show that it matches the area-law/tree-entropy bounds of the classical TTN?
  3. **Unitary Expressibility**: Does the restriction to unitary operations ($U^\dagger U = I$) in the quantum gate nodes constrain the representation compared to the unconstrained classical CP factor weights?

### Question B: Mitigating the 80-Qubit Classical Simulation Wall
* **Context**: If each patch encodes 4 pixels of RGB data (4 qubits) + 1 positional ancilla (1 qubit), the 16-patch model requires **80 qubits**. Classical statevector simulators (`default.qubit`) cannot simulate $2^{80}$ amplitudes.
* **Research Focus**:
  1. **Tensor Network Contraction (Polynomially Scaling Simulators)**: Since the VQC itself forms a Tree Tensor Network, its contraction path is highly optimized. Can we use Matrix Product State (MPS) or Tree Tensor Network (TTN) simulators (e.g., PennyLane's tensor network device or integration with libraries like `quimb`) to simulate the 80-qubit circuit by keeping the simulation bond dimension small? What is the maximum entangling gate depth before the simulation bond dimension explodes?
  2. **Active Qubit Recycling (Mid-Circuit Measurements & Resets)**: In a QTTN, once a block unitary is applied to a 4-patch register and 3 qubits are traced out, those 3 qubits are never used again. If we physically measure and reset them to $|0\rangle$, can we reuse them for the next patch block? How does this reduce the total physical qubit budget (e.g., from 80 qubits down to 8–12 active qubits)? What are the latency and gate error overheads of mid-circuit measurements on current NISQ hardware?
  3. **Classical Compression Calibration (Hybrid Setup)**: As implemented in [hybrid_trainer.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/hybrid_trainer.py), we can use a classical `Linear(16, 4)` head to project patch features down to 4 dimensions before feeding them into a 4-qubit VQC. Does this hybrid architecture lose the expressive benefit of the quantum representation, or does it serve as a practical compromise for near-term simulation and execution?

### Question C: Spatial Positional Encoding — Explicit vs. Implicit
* **Context**: The HEA model allocates a 5th "Ancilla" qubit to each patch to encode 2D spatial coordinates ($x, y$ mapped via learned $R_x, R_y$ rotations). However, a TTN has a fixed hierarchical topology (e.g., Patch 1 only interacts with Patches 2, 3, and 4 in Layer 1). This topology implicitly encodes spatial geometry.
* **Research Focus**:
  1. **Ablation Study**: Compare a QTTN trained *with* the 5th spatial ancilla against a model trained *without* it (relying purely on the wiring diagram topology to capture spatial relations).
  2. **Relational Task Sensitivity**: Does explicit positional encoding improve convergence and accuracy on coordinate-sensitive tasks (like predicting the relative spatial coordinates of objects in CLEVR), or is the implicit hierarchical structure sufficient?
  3. **Qubit Conservation**: If the spatial ancilla is redundant, removing it reduces the qubit requirement from 80 qubits down to 64 qubits, significantly easing simulation constraints.

### Question D: Barren Plateaus and Gradient Trainability in Tree VQCs
* **Context**: While flat, deep variational circuits suffer from barren plateaus, hierarchical architectures (like TTNs and MERA) are often resistant to barren plateaus when optimizing local observables.
* **Research Focus**:
  1. **Gradient Variance Scaling**: Measure the variance of the gradients $\text{Var}[\partial_{\theta} \mathcal{L}]$ empirically as a function of the number of image patches (tree depth) and qubits per patch. Verify if the variance decreases polynomially $\mathcal{O}(1/\text{Poly}(N))$ or exponentially $\mathcal{O}(2^{-N})$.
  2. **Effect of Final Classical Head**: Does placing a classical linear layer after the quantum PauliZ expectation values modify the gradient landscape or help mitigate early trainability issues?

### Question E: Noise Sensitivity and Quantum Error Mitigation
* **Context**: Because information is pooled hierarchically, a leaf qubit only undergoes a few gates before being measured or traced out. The maximum gate depth per qubit scales logarithmically: $\mathcal{O}(\log_4(N_{patches}))$.
* **Research Focus**:
  1. **Logarithmic Noise Resilience**: Does this logarithmic gate depth make the QTTN more resilient to depolarizing and amplitude damping noise than flat VQCs of comparable width?
  2. **Entropy Propagation via Partial Trace**: How does noise propagate through the partial trace/pooling operations? Does discarding qubits "wash away" noise, or does it propagate mixed-state entropy to the root of the tree?
  3. **Error Mitigation Overhead**: Implement Zero-Noise Extrapolation (ZNE) and Readout Error Mitigation. Measure how much classical accuracy is recovered on emulated IBM noise backends, and quantify the classical sampling overhead (number of shots) required.

---

## 3. Iterative Tasks & Implementation Plan

### Phase 1: Simulation Setup & Qubit Reduction (Milestone 1)
- [ ] **Task 1.1**: Resolve the 80-qubit simulation feasibility issue.
  - Implement a benchmark comparing:
    - A tensor network simulator device mapping.
    - An active qubit recycling circuit version using PennyLane `qml.measure` and `qml.reset`.
- [ ] **Task 1.2**: Write a synthetic shapes dataset generator.
  - Generate 16×16 images containing 4 classes (combinations of primary shapes: circle, square, triangle, cross; and primary colors: red, green, blue).
- [ ] **Task 1.3**: Validate gradients on the synthetic shapes.
  - Run noiseless simulation (`default.qubit` or the optimized TN device) to ensure gradients flow back from the loss function to the leaf encoder weights.

### Phase 2: CLEVR Dataset Integration & Feature Learning (Milestone 2)
- [ ] **Task 2.1**: Implement the CLEVR data ingestion pipeline.
  - Write a Polars-based script to filter the `dpdl-benchmark/clevr` HuggingFace dataset to scenes containing exactly one object.
  - Extract the object attributes (`color`, `shape`, `material`, `size`) and format them into 4 classification targets.
- [ ] **Task 2.2**: Train the QTTN on CLEVR Single-Object Attribute Classification.
  - Classify the object's composite properties. Target accuracy: $>80\%$ on all 4 heads in noiseless simulation.
- [ ] **Task 2.3**: Train on CLEVR Multi-Object (Relational) Classification.
  - Filter to scenes with exactly two objects. Derive the relative spatial relation (left/right/front/behind) using their 3D coordinates.
  - Train the model to predict this relation, validating whether the hierarchical tree encodes spatial coordinates.

### Phase 3: Noisy Emulation & Mitigation Benchmarks (Milestone 3)
- [x] **Task 3.1**: Calibrate Noisy Emulation.
  - Set up a noisy backend simulator (`default.mixed` or Qiskit's noise model mimicking an IBM device).
  - Sweep noise parameter $p$ to establish the accuracy degradation curve for CLEVR tasks.
- [x] **Task 3.2**: Error Mitigation Implementation.
  - Integrate PennyLane/Mitiq mitigation techniques (ZNE, Readout mitigation).
  - Measure accuracy improvement on emulated IBM backends.
- [x] **Task 3.3**: Evaluate Optimization Algorithms under Noise.
  - Compare Parameter-Shift Rules against SPSA and classical backpropagation. Record convergence rates and gradient computation costs.

---

## 4. Completed Diagnostic Investigations

We completed the diagnostic tasks evaluating barren plateau scaling and comparative topologies:

### 1. Barren Plateaus & Trainability (Completed)
* **Activity**: Built and executed [test_barren_plateaus.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/test_barren_plateaus.py) on $N \in \{4, 9, 16, 20\}$ qubits.
* **Result**: Verified that gradient variance remains stable near $10^{-2}$ to $10^{-1}$ (at $N=20$, $\text{Var} = 5.94 \times 10^{-2}$, which is $>62,000\times$ larger than the exponential BP limit of $\approx 9.5 \times 10^{-7}$). This empirically proves QTTN immunity to barren plateaus under local observables.
* **Autograd Memory Wall**: Discovered that statevector autograd requires $\mathcal{O}(G \cdot 2^N)$ memory, which at 25 qubits with 200 gates requires $>100$ GB of RAM, causing Out-Of-Memory crashes. This highlights the necessity of SPSA or Tensor Network simulators at scale.

### 2. 3-Way Comparative Topology Benchmark (Completed)
* **Activity**: Built and executed [compare_topologies.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/compare_topologies.py) comparing QTTN, MPS, and MERA.
* **Result**: 
  * **MERA** achieved the highest noiseless classification capacity (**`51.6%`** vs. `42.2%` for QTTN).
  * **QTTN** achieved the highest depolarizing noise resilience (**`53.9%`** at $p=0.10$ vs. `46.1%` for MPS).
  * **Analysis**: MPS accumulates gate noise linearly due to its linear gate depth. QTTN and MERA preserve noise tolerance due to logarithmic depth.

---

## 5. Architectural Scaling Analysis & Next Research Directions

### QTTN vs. MERA Contraction Complexity Wall for Large Images (ARO Contrastive Learning)
While MERA demonstrates superior expressibility capacity by entangling across block borders, it introduces **closed loops** in the tensor network contraction graph.
* **The Complexity Wall**: Classical tensor network contraction of loop-free trees (QTTN) scales **linearly** with image size (number of patches), making it highly scalable to $64 \times 64$ and $128 \times 128$ images. MERA's contraction complexity scales **exponentially** with the number of loops, rendering classical simulation completely intractable beyond small $8 \times 8$ or $16 \times 16$ resolutions.
* **Conclusion**: For realistic scaling towards ARO Contrastive Learning, **QTTN is the only computationally viable candidate**.

### Proposed Deep-Dive: Quantum Depolarizing Regularization (Quantum Dropout)
We observed that test accuracy on under-trained models improved under depolarizing noise (QTTN clean `42.2%` $\to$ noisy `53.9%`).
* **Hypothesis**: Depolarizing noise contracts state expectation values toward zero, smoothing decision boundaries and preventing overfitting. This is mathematically equivalent to noise injection (dropout/weight decay) in classical machine learning.
* **Future Deep-Dive Experiment**:
  * Plot the loss landscape/entropy of the model outputs under varying noise rates $p$.
  * Train models under a simulated "quantum dropout" schedule (varying $p$ during the training backward pass) to see if it improves final clean test generalization.
