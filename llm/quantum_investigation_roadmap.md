# Quantum TTN Image Model Investigation Roadmap

This document outlines a rigorous research and implementation roadmap for your quantum computing thesis, focusing **strictly on the Quantum Tree Tensor Network (QTTN) Image Tower**. 

---

## 1. Project Scope & Focus

To maximize the research novelty of your thesis, this investigation ignores the text tower (DisCoCat/lambeq) and isolates the **image tower architecture**. We investigate the quantum representation of spatial image data, hierarchical contractions, and classification expressibility under both simulated (noiseless) and emulated (noisy) environments.

```
═══════════════════════════════════════════════════
IMAGE TOWER HIERARCHY (updated 2026-07-27, see status notes below)
═══════════════════════════════════════════════════
                  Image (16×16 validated; 32×32/64×64
                  forward-pass only, see status notes)
                            │
                            ▼
                Bilinear patch embedding
                 (4×4 patches → N patches)
                 [NO positional encoding --
                  confirmed unnecessary, Question C]
                            │
                            ▼
              State Preparation: Angle/Amplitude
              (N patches × Qubits per patch)
                            │
                            ▼
              TTN Level 0: 4-qubit Block Unitaries
                            │
              (Partial trace: trace out 3 of 4 qubits)
                            ▼
              TTN Level 1: 4-qubit Block Unitaries
                            │
                     (additional levels for
                      depth-3/depth-4 trees)
                            ▼
                      Root Qubit
                            │
                            ▼
               Measurement: PauliZ Expectation
                            │
                            ▼
               Classical Head / Classification
               [minimal only -- no classical
                capacity substituting for
                quantum work, see NOTE in
                quantum_implementation_plan.md]
```

**Status notes (2026-07-27)**:
- **Image size**: 16×16 (depth-2 tree, 16 leaf patches) is the only size with validated end-to-end *training* (every experiment in this investigation used this size). 32×32 (depth-3) and 64×64 (depth-4) have validated *forward-pass* simulation only, via `default.tensor(method='tn')` — training at these sizes is blocked on gradient cost (parameter-shift is impractically slow; SPSA, Task 1.5, is the proposed fix, not yet implemented).
- **No spatial ancilla**: Question C (closed 2026-07-26) found the explicit positional-encoding qubit is not beneficial (mildly worse) and safely removable — the diagram above and any CLEVR implementation should omit it, reducing qubit count per patch from 5 to 4.
- **No residual connections**: five quantum-native mechanisms tested and rejected (Section 5) — do not add skip connections to tree nodes.
- **No classical capacity inside the pipeline**: only the minimal patch-embedding and classification-head `Linear` layers are permitted (unavoidable I/O boundary); see `quantum_implementation_plan.md`'s NOTE — Classical Hybrid Shortcuts.

---

## 2. Elaborated Open Questions & Research Directions

These core research questions form the scientific contribution of your thesis. Each requires mathematical formulation followed by empirical validation.

### Question A: The Mathematical Mapping of CP-Rank to Quantum Entanglement
* **Context**: In a classical Tree Tensor Network (TTN), the contraction of four child nodes of dimension $\chi$ (bond dimension) into a parent node of dimension $\chi$ is regularized via CP-Decomposition of rank $R$. In the quantum circuit, this is replaced by a parameterized unitary ansatz $U \in SU(2^{4 \log_2(\chi)})$ followed by a partial trace (ignoring $3 \log_2(\chi)$ qubits).
* **Research Focus**:
  1. **Rank Equivalence**: What is the mathematical relationship between the classical CP-rank $R$ and the entangling gate depth $D$ (number of CNOT layers) of the quantum ansatz? If a VQC has $D=0$ (no entangling gates), the state remains a separable product state, equivalent to a classical CP-rank $R=1$. How does the representation capacity scale as $D$ increases?
  2. ~~**Entanglement Entropy**: Can we bound the entanglement entropy of the quantum state at each level of the tree and show that it matches the area-law/tree-entropy bounds of the classical TTN?~~ — **Done 2026-07-26**, [entropy_vs_tree_depth.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/entropy_vs_tree_depth.py), full writeup in `research_log.md`. Correction to the framing: the bound match ($\log_2(\chi)$, $\chi=2$ here) is **architecturally guaranteed** by the single-qubit-per-node bottleneck, not an empirical result to verify — a single qubit's entropy mathematically cannot exceed $\ln(2)$ regardless of tree depth. The real finding is that root-qubit entropy climbs from `72.7%` (depth 1, 4 leaves) to `89.7%` (depth 2, 16 leaves) of that shared ceiling as depth increases — deeper trees generate entanglement that saturates the fixed bottleneck more fully, a genuine (non-guaranteed) trend about entanglement-generation efficiency vs. depth. Only 2 depths tested. **Correction (2026-07-26)**: the assumption that qubit recycling would make depth-3 entropy tractable turned out to be wrong — see Question B's correction note below and `research_log.md` "Depth-3 Recycling Feasibility." Recycling hits a hard simulation wall at depth 3 for both deferred-measurement (ancilla-wire blowup) and `tree-traversal` (exponential branch-count blowup) MCM strategies. A depth-3 entropy measurement is not currently blocking for the thesis narrative, but if pursued, it would need the `default.tensor` device (Task 1.4) instead — untested for entropy-type measurements specifically (only tested for `expval`-based classification forward passes so far).
  3. **Unitary Expressibility**: Does the restriction to unitary operations ($U^\dagger U = I$) in the quantum gate nodes constrain the representation compared to the unconstrained classical CP factor weights? — Still open; needs a different methodology (function-fitting capacity comparison against an unconstrained classical CP node), not an entropy measurement.

### Question B: Mitigating the 80-Qubit Classical Simulation Wall
* **Context**: If each patch encodes 4 pixels of RGB data (4 qubits) + 1 positional ancilla (1 qubit), the 16-patch model requires **80 qubits**. Classical statevector simulators (`default.qubit`) cannot simulate $2^{80}$ amplitudes. (The positional ancilla itself is now known to be unnecessary — Question C, closed 2026-07-26 — so the realistic qubit count for a 16-patch/depth-2 model is 64, or 16 without the ancilla; the wall below is stated in terms of leaf-patch/qubit count generally, independent of the ancilla question.)
* **Correction (2026-07-26)**: sub-question 2 (active qubit recycling) was only ever validated at **depth 2** (16 qubits → 7, 2026-07-17). Attempting to extend it to depth 3 (64 leaves, the actual next scaling step) revealed a hard simulation wall — see Task 1.4 in Section 3 and `research_log.md` for the full writeup. Recycling does **not** provide a path past 16×16 images; the "8–12 active qubits" framing in sub-question 2 below describes a *real-hardware* qubit budget, not something classically simulable beyond depth 2 with current tools.
* **Research Focus**:
  1. ~~**Tensor Network Contraction (Polynomially Scaling Simulators)**: Since the VQC itself forms a Tree Tensor Network, its contraction path is highly optimized. Can we use Matrix Product State (MPS) or Tree Tensor Network (TTN) simulators (e.g., PennyLane's tensor network device or integration with libraries like `quimb`) to simulate the 80-qubit circuit by keeping the simulation bond dimension small? What is the maximum entangling gate depth before the simulation bond dimension explodes?~~ — **Resolved 2026-07-26, see Task 1.4.** `default.tensor(method='tn')` works for forward-pass classification circuits at depth-3 (32×32, 0.13s) and depth-4 (64×64, 1.2s). Training-cost question (parameter-shift is exact but slow, 56.7s/step at depth-3) is tracked separately as Task 1.5 (SPSA).
  2. **Active Qubit Recycling (Mid-Circuit Measurements & Resets)**: In a QTTN, once a block unitary is applied to a 4-patch register and 3 qubits are traced out, those 3 qubits are never used again. If we physically measure and reset them to $|0\rangle$, can we reuse them for the next patch block? How does this reduce the total physical qubit budget (e.g., from 80 qubits down to 8–12 active qubits)? What are the latency and gate error overheads of mid-circuit measurements on current NISQ hardware? — **Validated at depth 2 only (2026-07-17); confirmed NOT to extend to depth 3 for classical simulation purposes (2026-07-26). Still valid as a real-hardware qubit-budget technique.**
  3. ~~**Classical Compression Calibration (Hybrid Setup)**~~ — **CLOSED 2026-07-26, rejected on principle (no experiment run).** [hybrid_trainer.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/hybrid_trainer.py)'s classical `Linear(16, 4)` compression before the 4-qubit VQC substitutes classical capacity for quantum circuit width — this is a hybrid-architecture shortcut, not a purely-quantum solution to the qubit wall. The project's stated direction is a purely quantum implementation; qubit-budget problems should be solved via genuinely quantum means (active qubit recycling — sub-question 2, done — or tensor-network simulators — sub-question 1) rather than classical compression. `hybrid_trainer.py` is deprecated as an architecture direction; see `llm/quantum_implementation_plan.md`'s NOTE — Classical Hybrid Shortcuts.

### Question C: Spatial Positional Encoding — Explicit vs. Implicit — CLOSED for sub-questions 1 & 3 (2026-07-26)
* **Context**: The HEA model allocates a 5th "Ancilla" qubit to each patch to encode 2D spatial coordinates ($x, y$ mapped via learned $R_x, R_y$ rotations). However, a TTN has a fixed hierarchical topology (e.g., Patch 1 only interacts with Patches 2, 3, and 4 in Layer 1). This topology implicitly encodes spatial geometry.
* **Verdict (sub-questions 1 & 3)**: Ablation tested on 5 seeds ([investigate_spatial_ancilla.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_spatial_ancilla.py), full writeup in `research_log.md` 2026-07-26). The explicit ancilla is **not better, and mildly worse**: noiseless final val acc `52.8% ± 5.7` vs. `55.0% ± 3.5` without it, higher variance, one seed failed to converge, and it costs more (119 vs. 105 params, +1 qubit per level-1 node). Noise-swept (averaged across all 5 seeds) the two variants are statistically indistinguishable, with the ancilla trending slightly worse. **Recommendation: do not use the explicit spatial ancilla for CLEVR** — the fixed tree topology already provides sufficient implicit positional signal via slot ordering.
* **Research Focus**:
  1. ~~**Ablation Study**: Compare a QTTN trained *with* the 5th spatial ancilla against a model trained *without* it~~ — ✅ Done, see verdict above.
  2. **Relational Task Sensitivity**: Does explicit positional encoding improve convergence and accuracy on coordinate-sensitive tasks (like predicting the relative spatial coordinates of objects in CLEVR), or is the implicit hierarchical structure sufficient? — **Still open**, needs CLEVR's multi-object relative-position labels (Phase 2); this single-object classification ablation can't test relational reasoning.
  3. ~~**Qubit Conservation**: If the spatial ancilla is redundant, removing it reduces the qubit requirement from 80 qubits down to 64 qubits~~ — ✅ Confirmed redundant, safe to drop to 64 qubits.

### Question D: Barren Plateaus and Gradient Trainability in Tree VQCs
* **Context**: While flat, deep variational circuits suffer from barren plateaus, hierarchical architectures (like TTNs and MERA) are often resistant to barren plateaus when optimizing local observables.
* **Research Focus**:
  1. **Gradient Variance Scaling**: Measure the variance of the gradients $\text{Var}[\partial_{\theta} \mathcal{L}]$ empirically as a function of the number of image patches (tree depth) and qubits per patch. Verify if the variance decreases polynomially $\mathcal{O}(1/\text{Poly}(N))$ or exponentially $\mathcal{O}(2^{-N})$.
  2. ~~**Effect of Final Classical Head**~~ — **CLOSED 2026-07-26, rejected on principle (no experiment run).** This asks whether *tuning* the classical readout layer improves quantum trainability — a hybrid-architecture optimization question, not a purely-quantum one. The minimal classical head (measurement → class logits) stays as an unavoidable I/O boundary since some classical decision layer is required for any classifier, but it is not a research target for further investigation.

### Question E: Noise Sensitivity and Quantum Error Mitigation
* **Context**: Because information is pooled hierarchically, a leaf qubit only undergoes a few gates before being measured or traced out. The maximum gate depth per qubit scales logarithmically: $\mathcal{O}(\log_4(N_{patches}))$.
* **Research Focus**:
  1. **Logarithmic Noise Resilience**: Does this logarithmic gate depth make the QTTN more resilient to depolarizing and amplitude damping noise than flat VQCs of comparable width?
  2. ~~**Entropy Propagation via Partial Trace**: How does noise propagate through the partial trace/pooling operations? Does discarding qubits "wash away" noise, or does it propagate mixed-state entropy to the root of the tree?~~ — **Done 2026-07-27**, [investigate_noise_regularization.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_noise_regularization.py), full writeup in `research_log.md`. Partial answer only: the entropy gap between level-1 and root narrows with noise (consistent with washing away), but this is confounded by root already sitting close to the $\ln(2)$ ceiling (less room to grow regardless of propagation). Not a clean result — would need a design keeping both levels similarly far from saturation to answer definitively.
  3. **Error Mitigation Overhead**: Implement Zero-Noise Extrapolation (ZNE) and Readout Error Mitigation. Measure how much classical accuracy is recovered on emulated IBM noise backends, and quantify the classical sampling overhead (number of shots) required.

### Question F: Depolarizing Noise as Implicit Regularization — CLOSED (scheduling sub-question) 2026-07-27
* **Context (added 2026-07-26, promoted from an informal Section 5 note)**: Test accuracy on under-trained QTTN models has twice been observed to *improve* under depolarizing noise rather than degrade — clean `42.2%` $\to$ noisy `53.9%` at $p=0.10$ in the topology benchmark, and clean `85.9%` $\to$ `91.4%` at $p=0.005$ in the synthetic-shapes noise sweep (see Experiment & Metrics Record). This is distinct from Question E's noise-*tolerance* framing — E asks how much accuracy survives noise; F asks why noise sometimes makes the model *better*.
* **Verdict (sub-question 2, scheduling)**: **No — training under noise does not act as a trainable regularizer.** Tested directly (2026-07-27, [investigate_noise_regularization.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_noise_regularization.py), 5 seeds, always evaluated clean): final clean accuracy was `55.0% ± 3.5` (noiseless-trained) vs. `54.1% ± 5.0` for both `p=0.02` and `p=0.05` trained variants — noisy training is mildly *worse*, not better, monotonically across both noise levels tested. **The eval-time noise-improves-accuracy effect and "noise as a trainable regularizer" are different phenomena that only superficially resemble each other** — the project has solid repeated evidence for the former, but the classical-dropout analogy the latter implied does not hold up.
* **Research Focus**:
  1. **Mechanism**: Does depolarizing noise contract expectation values toward zero in a way that smooths decision boundaries and suppresses overfitting, analogous to classical dropout/weight decay? Plot the loss landscape and output entropy as a function of noise rate $p$. — **Not attempted**; given sub-question 2 came back negative, this is now lower priority (the eval-time effect's mechanism remains an open curiosity, not an actionable lead).
  2. ~~**Scheduling**: Does training under a scheduled noise rate (varying $p$ during the backward pass, evaluating clean at test time) improve final generalization versus training and evaluating noiseless throughout?~~ — **Done, negative result, see verdict above.**
  3. ~~**Relationship to E.2**~~ — confirmed, same infrastructure used for both (see `research_log.md` 2026-07-27 for the combined writeup); E.2 answered separately (partial answer only, see Question E above).
* **Status**: The actionable sub-question (scheduling) is closed with a clear negative result. Sub-question 1 (mechanism) remains open but deprioritized — not worth pursuing unless a future need for the eval-time effect specifically (not training-time) arises.

---

## 3. Iterative Tasks & Implementation Plan

### Phase 1: Simulation Setup & Qubit Reduction (Milestone 1)
- [x] **Task 1.1a**: Active qubit recycling — validated at depth 2 (16 qubits → 7, 2026-07-17), confirmed a hard simulation wall at depth 3 (2026-07-26, see `research_log.md` and Task 1.4 below). Not a viable path to images larger than 16×16.
- [x] **Task 1.1b**: Tensor network simulator device mapping (quimb / PennyLane TN device). Done 2026-07-26 — `default.tensor(method='tn')` (quimb backend, installed via `pip install quimb`) correctly runs forward passes at depth-2 (matches `default.qubit` ground truth exactly), depth-3/32×32 (0.13s), and depth-4/64×64 (1.2s, reproducible). Gradients are exact via `parameter-shift` (matches `backprop` to 1e-11) but expensive — 56.7s per gradient step at depth-3 (252 params), since this device doesn't support `backprop`. Training-cost fix (SPSA) is Task 1.5.
- [x] **Task 1.2**: Write a synthetic shapes dataset generator.
  - Generate 16×16 images containing 4 classes (combinations of primary shapes: circle, square, triangle, cross; and primary colors: red, green, blue). Done 2026-07-17, `qnlp/utils/data/synthetic_shapes.py`.
- [x] **Task 1.3**: Validate gradients on the synthetic shapes.
  - Run noiseless simulation (`default.qubit`) to ensure gradients flow back from the loss function to the leaf encoder weights. Done 2026-07-17 and reconfirmed across every 2026-07-26 experiment (residual, ancilla, entropy investigations) — all trained successfully on 16×16.
- [x] **Task 1.4: Enable 32×32 image support (depth-3 QTTN, 64 leaf patches).** **Forward-pass goal met 2026-07-26**, training cost still open (→ Task 1.5).
  - **Motivation**: All training to date was capped at 16×16 images (depth-2 tree, 16 leaf patches) — the actual current ceiling of the architecture, not an arbitrary dataset choice. Scaling to CLEVR-realistic resolutions, or even just 32×32, requires a depth-3 tree (64 leaves), which was blocked as of the recycling-wall finding (see `research_log.md` 2026-07-26, "Depth-3 Recycling Feasibility").
  - **What didn't work**: direct 64-qubit statevector simulation ($2^{64}$, infeasible); active qubit recycling via deferred measurement (~65-70 total simulated wires needed, infeasible) or `tree-traversal` MCM (exponential in reset count, blows up between 12 and 24 resets, far short of depth-3's ~50).
  - **What worked**: `default.tensor(method='tn')` (quimb backend) — contracts the circuit's tensor network directly rather than materializing a statevector or branching over outcomes, exploiting the fact (established in Question A.2) that every internal bond in this QTTN has $\chi=2$. Forward pass: depth-3 (32×32) in 0.13s, depth-4/64×64 in 1.2s, both correct and reproducible. `method='mps'` also works at depth-3 (1.73s) but OOM-kills at depth-4 — not needed since `tn` handles both.
  - **Still open**: gradients are exact (`parameter-shift` matches `backprop` to 1e-11) but expensive — 56.7s/gradient-step at depth-3 for one sample, since `default.tensor` doesn't support `backprop` and parameter-shift costs ~2 evaluations per parameter (252 params at depth-3, 1020 at depth-4). This makes actual *training* impractical at these sizes even though *inference* is fast. See Task 1.5.

- [ ] **Task 1.5 (new, 2026-07-26): Make depth-3/depth-4 training practical via SPSA.**
  - **Motivation**: Task 1.4 solved forward-pass tractability at 32×32/64×64 but not training — parameter-shift's per-parameter cost (56.7s/step at depth-3) makes real training runs (many samples × many epochs) impractical.
  - **Plan**: implement SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimation, which needs a constant ~2 circuit evaluations per step *regardless of parameter count*, instead of parameter-shift's 2×(num params). At depth-3's per-eval cost this would turn ~56.7s/step into an estimated ~0.2s/step (~250x speedup) — plausibly making depth-3 training practical and depth-4 worth a serious attempt.
  - **Est. time**: 1 day (SPSA gradient estimator + a short training run at depth-3 to confirm convergence quality, since SPSA gradients are noisier/approximate compared to exact parameter-shift).

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
> **⚠️ CHECKBOX DISCREPANCY (flagged 2026-07-26, not yet resolved)**: These three tasks were previously marked `[x]` complete with no corresponding log entries — an audit found no ZNE, Mitiq, or SPSA-vs-parameter-shift-vs-backprop comparison anywhere in `research_log.md`. Corrected to `[ ]` below to reflect actual state. See Section 6 backlog item 1. **CLEVR-relevant partial coverage exists elsewhere**: Task 3.1's noise-sweep goal is substantially covered by the many depolarizing noise sweeps run throughout this investigation (topology benchmark, residual/ancilla investigations, entropy-vs-noise) — just not on CLEVR data specifically, which doesn't exist yet. Task 3.2 (ZNE/Mitiq) and Task 3.3 (SPSA vs. parameter-shift vs. backprop convergence comparison, distinct from Task 1.5's SPSA-for-speed motivation) remain genuinely undone.
- [ ] **Task 3.1**: Calibrate Noisy Emulation.
  - Set up a noisy backend simulator (`default.mixed` or Qiskit's noise model mimicking an IBM device).
  - Sweep noise parameter $p$ to establish the accuracy degradation curve for CLEVR tasks. (Noise-sweep methodology is validated on synthetic data; needs re-running on actual CLEVR once Task 2.1/2.2 exist.)
- [ ] **Task 3.2**: Error Mitigation Implementation.
  - Integrate PennyLane/Mitiq mitigation techniques (ZNE, Readout mitigation).
  - Measure accuracy improvement on emulated IBM backends.
- [ ] **Task 3.3**: Evaluate Optimization Algorithms under Noise.
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

### Resolved: bespoke Quantum Dropout mechanism — deprioritized
We observed that test accuracy on under-trained models improved under depolarizing noise (QTTN clean `42.2%` $\to$ noisy `53.9%`), and again on the synthetic-shapes noise sweep (`85.9%` clean $\to$ `91.4%` at $p=0.005$).
* **Correction (2026-07-26)**: *building a bespoke quantum-dropout mechanism* is deprioritized — the naturally-occurring depolarizing noise already provides a dropout-equivalent regularization effect, so hand-rolling gate/qubit/parameter dropout on top is not worth the engineering cost right now.
* **This does NOT resolve the underlying question of *why* it happens** — see Question F below, which remains open and is promoted out of this informal note into the main Open Questions list.
* **Superseded candidate list** (kept for reference, not scheduled): classical head dropout, noise-as-dropout training schedule, qubit/patch dropout, gate/parameter dropout, entangling-layer dropout, finite-shot dropout, data re-uploading + re-upload dropout.

### CLOSED (2026-07-26): Quantum-Native Residual Connections — Conclusion: Do Not Use
**Verdict**: Five quantum-native residual mechanisms were tested against a no-residual baseline. None show a robustly-confirmed benefit. **Quantum tree nodes in this project should not use residual/skip connections going forward** — this is now a binding design rule, recorded in `llm/quantum_implementation_plan.md`'s NOTE block. This investigation is closed; do not re-open it without a genuinely new hypothesis distinct from the five below.

| Method | Noiseless result | Noise-robustness result | Verdict |
|---|---|---|---|
| Data re-uploading | Highest peak accuracy (59.9%), but 2.6x higher seed variance | **Worse than baseline** — collapses to 28.1% at $p\ge0.10$ vs. baseline's 50.0% floor | Rejected — trades expressivity for fragility |
| Near-identity init | No measurable difference from baseline | No measurable difference from baseline | Rejected — no effect either way |
| Mixed-unitary channel | Small edge on converged runs (55.6% vs. 54.1%), but 10% training-failure rate (dead-gradient trap) | **No benefit** — an initial single-seed result suggesting a benefit did not replicate on 10 seeds; baseline is equal-or-better at high noise | Rejected — apparent benefit was a sampling artifact |
| LCU / LCU-lite | Same instability as mixed-unitary channel, no accuracy edge | Untestable (`default.mixed` postselection limitation) | Rejected — inherits instability, adds real shot overhead, unproven benefit |
| Classical-bypass residual (prototyped, out of scope) | Modest edge, noise-robust | Noise-robust (mechanistically trivial — bypasses the noisy channel entirely) | Excluded from scope — not a quantum-architecture finding |

* **Audit finding (2026-07-26)**: The classical CP-quadtree layer (`qnlp/discoviz/models/cp_node.py:28,68-69`) has an explicit residual connection — `out = merged_output + res_proj(x.mean(dim=2))`. The quantum tower has **no residual/skip-connection mechanism anywhere** — and per the conclusion above, it should stay that way.
* **Scope restriction (2026-07-26)**: Investigation is restricted to mechanisms where the shortcut is realized *inside the quantum circuit* — via re-encoding, near-identity unitary structure, or ancilla-mediated mixing — not a classical value computed from pre-quantum inputs and added back in after measurement. A classical-bypass residual was prototyped and discarded from this line of investigation for exactly that reason: it answers "does adding a classical shortcut around a noisy component help," which is an engineering mitigation, not a quantum-architecture finding.
* **Hypothesis**: A shortcut path lets each tree level learn a small perturbation on top of its children's information rather than reconstructing it from scratch through a unitary — classical residual networks show this speeds convergence and improves gradient flow in deep stacks; the open question is whether an equivalent, genuinely-quantum mechanism gives the same benefit without leaving the Hilbert space.

#### Candidate Quantum-Native Residual Mechanisms
| # | Method | Mechanism | Notes | Est. time | Status |
|---|---|---|---|---|---|
| 1 | **Data re-uploading (re-injection residual)** | Re-encode the original node input $S(x)$ again partway through (or after) each node's ansatz — $U(\theta) S(x) U(\theta') S(x)$ — instead of encoding once at the leaf and letting later levels only see it through compounded unitaries | Cheapest genuinely in-circuit option; standard technique (Pérez-Salinas et al. 2020) repurposed here as a residual — repeated re-injection keeps the original signal "alive" deep in the tree the way a skip connection keeps early-layer activations alive in a ResNet | 1–1.5 days | ✅ **Done 2026-07-26** — see results below |
| 2 | **Near-identity ansatz initialization (warm-start residual)** | Initialize each node's rotation parameters near $0$ so the node unitary starts close to $U \approx I$ (pure pass-through), then let training gradually deviate | Not a structural shortcut, but achieves the same "start as identity, learn a small perturbation" property that residual initialization gives classical ResNets — nearly free, just changes `weights = torch.randn(...) * eps` | 2–3 hrs | ✅ **Done 2026-07-26** — see results below |
| 3 | **Mixed-unitary channel residual** | Realize $\rho \to (1-\lambda) \rho + \lambda\, U\rho U^\dagger$ as a genuine quantum channel with Kraus operators $\{\sqrt{1-\lambda}\, I,\ \sqrt{\lambda}\, U\}$ — a proper probabilistic mix of "apply $U$" vs. "do nothing," implementable on `default.mixed` | Physically valid channel-level residual (not classical arithmetic on measured values); $\lambda$ can be learnable | 1–1.5 days | ✅ **Done 2026-07-26** — see results below |
| 4 | **Ancilla-controlled soft mixing (LCU-lite)** | One ancilla qubit per node, $R_y(\theta)$-controlled branch between applying $U_{node}$ vs. Identity to the child registers — a learnable, fully unitary interpolation between "skip" and "transform" | Closest in spirit to a true quantum shortcut; requires +1 ancilla wire per node and gradient-flow validation | 1.5–2 days | ✅ **Done 2026-07-26** (noiseless only — see limitation below) |
| 5 | **Full LCU (Linear Combination of Unitaries) residual** | Coherent superposition $\alpha I + \beta U_{node}$ via standard LCU block-encoding with post-selection on the ancilla | Heaviest lift — post-selection success-probability handling and renormalization needed; likely not worth it near-term | 3+ days | ✅ **Done 2026-07-26** (overhead diagnostic only — see results below) |

#### Results (Methods 1 & 2, see `research_log.md` 2026-07-26 for full writeup)
Built [investigate_quantum_residuals.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_quantum_residuals.py), same hierarchical QTTN, 3 seeds × 15 epochs × 3 variants (baseline / reupload / near_identity) on 16×16 synthetic shapes, plus a depolarizing noise sweep on the seed=0 trained weights of each.
* **Near-identity init (Method 2)**: No measurable effect. Final/peak accuracy and noise-sweep curve are statistically indistinguishable from baseline — the near-identity start only shapes early training dynamics and washes out by the time training converges (15 epochs, 2-layer ansatz). Free to include or drop; not worth pursuing further at this depth.
* **Data re-uploading (Method 1)**: Reached the highest peak accuracy of all three variants (`59.9%` vs. `57.3%` baseline), clearly fitting training data harder — but with 2.6x higher seed-to-seed variance and one seed degrading late in training. Under the noise sweep it **collapsed to `28.1%`** at $p \ge 0.10$, worse than baseline's `50.0%` floor. Mechanism: re-uploading doubles the encode→ansatz cycle, which doubles the number of `DepolarizingChannel` injection points the circuit is exposed to — the same repeated signal-injection that helps expressivity noiselessly is exposed to noise twice, so re-uploading trades noiseless expressivity for noise fragility. This is the **opposite** of the robustness benefit skip connections give classical ResNets, and is a direct consequence of re-uploading not being a true bypass (unlike a classical residual, the re-injected signal still passes through the noisy channel on its way to the measurement).

#### Results (Methods 3, 4 & 5, see `research_log.md` 2026-07-26 for full writeup)
Built [investigate_ancilla_residuals.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_ancilla_residuals.py). Each node gets a 5th ancilla wire with a learnable mixing angle (init small, near-identity). `mixed_channel` traces out the ancilla (incoherent mixture); `lcu` postselects it on $|0\rangle$ (coherent LCU block-encoding).
* **Known limitation**: `default.mixed` (PennyLane 0.43.2) doesn't support `qml.measure(postselect=...)` via its deferred-measurement transform — confirmed by direct test (`WireError`/`ValueError: Postselection is not allowed on the device...`). So `lcu` was evaluated **noiseless only**.
* **Training instability**: Both ancilla methods showed severe instability — `mixed_channel` final `45.8% ± 20.3` (vs. baseline `56.8% ± 2.9`), `lcu` final `42.7% ± 18.6`. One seed for **both** variants got permanently stuck at loss ≈$\ln(4)$ (exactly uniform-random cross-entropy) for all 15 epochs — a dead-gradient trap, not slow convergence. Likely cause: the learnable `mix_angle` gates how much the controlled-`U` branch contributes; if it drifts toward 0 early, the ansatz weights stop receiving gradient signal through the controlled operation — structurally similar to classical gated units (LSTM-style) getting stuck closed early in training.
* **Noise robustness (seed=0 models that converged, 3-seed run)**: `mixed_channel` clean `51.6%` → improves and flattens at `54.7%` for all $p \ge 0.02$, consistently beating baseline's flat `50.0%` floor. **⚠️ Retracted 2026-07-26 — see below.**
* **LCU overhead diagnostic (Method 5)**: postselection success probability is level-dependent — Level 1 (patch-pooling) `P(ancilla=0)=0.986`, Level 2 (root) `P(ancilla=0)=0.515` — the root node discards roughly half of all trials, a concrete ~2x shot-overhead cost unique to LCU. (This diagnostic is unaffected by the retraction below — it's a property of the trained circuit, not a noise-sweep claim.)

#### Correction (2026-07-26): Extended 10-Seed Re-test Retracts the Noise-Robustness Claim
The 3-seed noise-sweep comparison above used a single trained model per variant (seed=0). Re-run on 10 seeds with the noise sweep averaged across **every** trained model (not just one) via [investigate_mixed_channel_seeds.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_mixed_channel_seeds.py):
* **Failure rate revised down**: 1/10 seeds (10%) failed, not 1/3 (33%) — the earlier sample was unlucky, not representative.
* **Noiseless edge confirmed**: `mixed_channel` converged-only mean `55.6% ± 5.2` vs. baseline `54.1% ± 3.6` — small, real, holds up with more seeds.
* **Noise robustness does NOT replicate**: averaged over converged seeds, baseline and mixed_channel are statistically tied at low noise, and **baseline is equal-or-better at high noise** ($p=0.20$: `51.6%` vs. `47.2%`). Per-seed noise-response std (4–11 points) is often larger than the between-method gap — the original single-seed comparison sampled an unrepresentative pair of models.
* **Decision**: `mixed_channel`'s noise-robustness benefit is retracted. The only surviving finding is a modest noiseless accuracy edge on converged runs, with a lower-than-feared (10%) but nonzero failure rate. `lcu` inherits the same instability without any demonstrated noise benefit (untestable under noise due to the `default.mixed` postselection limitation) and adds real shot overhead (Method 5 diagnostic above). None of the five quantum-native residual methods tested (Methods 1–5) show a clean, robustly-confirmed win for CLEVR. **Methodological lesson**: any future noise-robustness claim in this investigation needs multi-seed averaging (~10 seeds), not a single-seed spot check — noise-response variance at this system size is comparable to or larger than the effects being measured.

---

## 6. Consolidated Backlog — Proposed but Unexecuted Components (Audited 2026-07-26, updated through 2026-07-27)

Cross-checking this roadmap and `quantum_implementation_plan.md` against `research_log.md` surfaced several items that were proposed (or marked complete) but have no corresponding experiment in the log:

| # | Item | Source | Status | Est. time |
|---|---|---|---|---|
| 1 | **Phase 3 checkbox discrepancy**: ZNE, readout error mitigation (Mitiq), SPSA vs. parameter-shift vs. backprop comparison | Roadmap Phase 3 (marked `[x]` but unlogged) | Not actually done — checkboxes need real experiments or to be unchecked | 2–3 days (new Mitiq dependency + calibration) |
| 2 | ~~Question A.2: Entanglement entropy vs. classical TTN area-law~~ | Question A | **Done 2026-07-26** — bound match is architecturally guaranteed (single-qubit bottleneck); real finding is entropy saturating the bound more with depth (72.7%→89.7%, depth 1→2). See results above. | — |
| 3 | **Question A.3**: Does the unitary constraint limit expressibility vs. unconstrained classical CP factors? | Question A | Not started | 1–2 days (needs classical CP-decomposition baseline for comparison) |
| 4 | ~~Question B.1 / Task 1.4: Tensor-network simulator device benchmark~~ | Question B / Task 1.1b, 1.4 | **Done 2026-07-26** — forward pass works at 32×32 (0.13s) and 64×64 (1.2s) via `default.tensor(method='tn')`; gradients correct but slow (56.7s/step at depth-3). Training-cost fix is item 14 (SPSA, Task 1.5). | — |
| 5 | ~~Question B.3: Classical compression head ablation~~ | Question B | **CLOSED 2026-07-26, rejected on principle** — project direction is a purely quantum implementation; classical compression substituting for quantum circuit width is out of scope. `hybrid_trainer.py` deprecated as an architecture direction. | — |
| 6 | ~~Question C.1/C.3: Spatial ancilla ablation~~ | Question C | **Done 2026-07-26** — ancilla is not better (and mildly worse); confirmed safe to drop, reducing 80→64 qubits. Sub-question 2 (relational sensitivity) remains open, deferred to CLEVR. See results above. | — |
| 7 | ~~Question D.2: Effect of classical head on gradients~~ | Question D | **CLOSED 2026-07-26, rejected on principle** — tuning/investigating the classical readout layer is a hybrid-architecture question, not a purely-quantum one; the minimal readout stays as an unavoidable I/O boundary but isn't a research target. | — |
| 8 | ~~Question E.2: Entropy propagation through partial trace~~ | Question E | **Done 2026-07-27** — partial answer only (ceiling-effect confound); entropy gap narrows with noise, consistent with washing away but not conclusive. See results above. | — |
| 9 | ~~Bespoke Quantum Dropout mechanism~~ | Section 5 | **Deprioritized 2026-07-26** — noise already provides the regularization effect; see item 11 for the still-open "why" question | — |
| 10 | ~~Quantum-native residuals, Methods 1–2 (data re-uploading, near-identity init)~~ | Section 5 | **Done 2026-07-26** — near-identity: no effect; re-uploading: higher peak accuracy but worse noise robustness (collapses to 28.1% at p≥0.10 vs. baseline's 50.0% floor). See results above. | — |
| 11 | ~~Quantum-native residuals, Methods 3–5 (mixed-unitary channel, LCU, LCU overhead diagnostic)~~ | Section 5 | **Done 2026-07-26**, corrected same day after 10-seed re-test — mixed-channel's noise-robustness claim retracted (didn't replicate); only a modest noiseless accuracy edge survives, with a 10% training-failure rate. LCU inherits the instability, noiseless-only, pays ~2x shot overhead at the root node. See results above. | — |
| 12 | ~~Question F: Depolarizing noise as implicit regularization (scheduling)~~ | Question F (new) | **Done 2026-07-27** — negative result: training under noise does not improve clean accuracy (mildly worse, 5 seeds). Mechanism sub-question (F.1) deprioritized given this. See results above. | — |
| 13 | ~~Stabilize `mixed_channel` training~~ | Section 5 (follow-up) | **Deprioritized 2026-07-26** — originally motivated by the noise-robustness benefit, which was retracted after the 10-seed re-test. Remaining motivation (modest noiseless edge, 10% failure rate) doesn't justify prioritizing a training fix over other backlog items unless a future experiment finds a different genuine advantage. | — |
| 14 | **Task 1.5: SPSA gradient estimation for depth-3/4 training** | Section 3, Phase 1 (new) | Not started — needed to make 32×32/64×64 *training* practical now that forward passes work (item 4); parameter-shift costs 56.7s/step at depth-3, SPSA should reduce this ~250x | 1 day |

**Total backlog**: roughly 2–5 days of remaining work (items 2, 4, 5, 6, 7, 8, 9, 10, 11, 12 are done or closed; item 13 deprioritized). Remaining open items: item 1 (Phase 3 checkbox discrepancy — ZNE/Mitiq/optimizer comparison, 2-3 days, discretionary), item 3 (Question A.3, expressibility, 1-2 days), and item 14 (SPSA for depth-3/4 training, 1 day — only needed if CLEVR will use images larger than 16×16; otherwise park it as documented-but-not-executed infrastructure and proceed to CLEVR at the validated 16×16 size). **Any future backlog item that proposes adding or tuning a classical component inside the core quantum pipeline (compression heads, hybrid readouts, classical-bypass shortcuts) should be closed on principle without running an experiment** — the project direction is a purely quantum implementation; only the unavoidable classical I/O boundary (pixel-to-angle encoding in, class logits out) is exempt. **None of the five quantum-native residual methods tested show a robustly-confirmed win**, and the spatial ancilla is confirmed unnecessary — if either line is revisited, it should start from a fresh hypothesis rather than re-attempting to rescue a mechanism whose apparent benefit didn't survive a larger sample.
