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

> **⚠️ READ FIRST (2026-07-27)**: the three architecture rules in the status notes below (no spatial ancilla, no residuals, image-size ceiling framing) were decided by experiments that ran through a **scalar-readout bottleneck** — the whole image compressed to one number before a `nn.Linear(1, 4)` head. They are **PROVISIONAL** pending re-test. See `research_log.md` 2026-07-27 "Code Audit" and **Section 7 — Phase 1.5 Remediation Plan** below. Do not start Phase 2 (CLEVR) until Section 7's decision gate is passed.

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

### Question C: Spatial Positional Encoding — Explicit vs. Implicit — PROVISIONAL for sub-questions 1 & 3 (was CLOSED 2026-07-26; downgraded 2026-07-27)
> **✅ RE-TESTED 2026-07-28 (Task R3), verdict UPHELD and now resolved.** At the architecture of record, 30 seeds: `with_ancilla` `59.9 ± 6.9` vs baseline `79.4 ± 8.0` — **`−19.5` pts against a `3.8`-pt resolution limit.** The July verdict (`52.8 ± 5.7` vs `55.0 ± 3.5` at n=5, inside noise) reached the same conclusion without evidence; this one has it.
>
> **⚠️ SCOPE — this must travel with the result and is enforced in `combine_r3.py`.** Do **not** write this up as "positional encoding is harmful". Two limits: (a) the task is translation-invariant single-object classification, where position barely affects the label, so an explicit position mechanism has nothing to contribute and a negative result is close to structurally guaranteed; (b) it tests the **per-quadrant** ancilla (4 positions), not the original **per-patch** design (16). What it licenses: do not use it for this class of task.
>
> **Sub-question 2 (relational sensitivity) is now REQUIRED in CLEVR, not optional.** CLEVR's left/right/front/behind task is the only place where position *is* the label and the question can actually be settled. Run it with and without the ancilla.

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

### Question E: Noise Sensitivity and Quantum Error Mitigation — ❌ CLOSED 2026-07-28 (scope decision)
> Closed with the evidence already collected. E.1 (logarithmic noise resilience) and E.3 (error-mitigation overhead) are not pursued; E.2 remains a partial answer with its ceiling-effect confound. Task R8 provides one close-out sweep. All noise results characterise a single 4–5 qubit node, not the tree — a limitation to state, not fix.
* **Context**: Because information is pooled hierarchically, a leaf qubit only undergoes a few gates before being measured or traced out. The maximum gate depth per qubit scales logarithmically: $\mathcal{O}(\log_4(N_{patches}))$.
* **Research Focus**:
  1. **Logarithmic Noise Resilience**: Does this logarithmic gate depth make the QTTN more resilient to depolarizing and amplitude damping noise than flat VQCs of comparable width?
  2. ~~**Entropy Propagation via Partial Trace**: How does noise propagate through the partial trace/pooling operations? Does discarding qubits "wash away" noise, or does it propagate mixed-state entropy to the root of the tree?~~ — **Done 2026-07-27**, [investigate_noise_regularization.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_noise_regularization.py), full writeup in `research_log.md`. Partial answer only: the entropy gap between level-1 and root narrows with noise (consistent with washing away), but this is confounded by root already sitting close to the $\ln(2)$ ceiling (less room to grow regardless of propagation). Not a clean result — would need a design keeping both levels similarly far from saturation to answer definitively.
  3. **Error Mitigation Overhead**: Implement Zero-Noise Extrapolation (ZNE) and Readout Error Mitigation. Measure how much classical accuracy is recovered on emulated IBM noise backends, and quantify the classical sampling overhead (number of shots) required.

### Question F: Depolarizing Noise as Implicit Regularization — ❌ FULLY CLOSED 2026-07-28
> The scheduling sub-question already had a clean negative result (below). Sub-question 1 (mechanism) is now closed too, under the noise scope decision: the eval-time effect is recorded as an observation with a probable mechanism (expectation values contracting toward zero, which can move samples across a decision boundary) and is folded into Task R8's close-out sweep rather than investigated separately.
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

### Phase 3: Noisy Emulation & Mitigation Benchmarks (Milestone 3) — ❌ CLOSED OUT OF SCOPE 2026-07-28

> **All three tasks below are closed without further experiments.** Per the scope decision of 2026-07-28 (`research_log.md`), the project proceeds on **noiseless simulation only**. Noise is not the thesis contribution; the existing depolarizing sweeps are sufficient to characterise tolerance, and Task **R8** provides a single close-out sweep on the coherent base model. Tasks 3.1 (CLEVR noise calibration), 3.2 (ZNE/Mitiq) and 3.3 (optimizer-under-noise comparison) are **not** to be executed. Section 6 backlog item 1 is closed by this decision, and item 18 (Qiskit cross-check) is dropped.
>
> **Standing limitation for the thesis**: noisy simulation was capped at 4–5 qubits throughout by the $O(4^N)$ density-matrix cost, so all noise results characterise a single quantum node rather than the full tree. State this; do not fix it.
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

### CLOSED — CONFIRMED 2026-07-28: Quantum-Native Residual Connections — Conclusion: Do Not Use
> **✅ RE-TESTED 2026-07-28 (Task R3) at restored capacity, 30 seeds. Rejection UPHELD — and for the first time on positive evidence.** `mixed_channel` `68.5 ± 10.1` vs baseline `79.4 ± 8.0` = **`−10.9` pts against a `4.7`-pt limit**. `reupload` `80.9 ± 6.6` = `+1.5`, inside a `3.8`-pt limit: **a true null**.
>
> **Two July narratives in the table below did not survive and should not be repeated:**
> * *"mixed-channel has a ~10% dead-gradient training-failure rate"* — **0/30 failures** at restored capacity. It is not unstable; it is consistently worse. This also closes backlog item 13 (stabilise mixed_channel) outright: there is nothing left to stabilise.
> * *"re-uploading trades expressivity for noise fragility, collapsing to 28.1% at p≥0.10"* — does not reproduce. It degrades comparably to baseline (`69.2` vs `77.5` at p=0.10). The collapse was an artifact of the bottlenecked model, not a property of re-uploading.
>
> The correct statement for `reupload` is **"no demonstrated benefit at effects ≥ 3.8 pts"** — not that it is harmful.

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
| 1 | ~~Phase 3: ZNE, readout error mitigation (Mitiq), optimizer-under-noise comparison~~ | Roadmap Phase 3 | **CLOSED OUT OF SCOPE 2026-07-28** — project proceeds noiseless only. No experiments required; the checkboxes are resolved by the scope decision rather than by execution. | — |
| 2 | ~~Question A.2: Entanglement entropy vs. classical TTN area-law~~ | Question A | **Done 2026-07-26** — bound match is architecturally guaranteed (single-qubit bottleneck); real finding is entropy saturating the bound more with depth (72.7%→89.7%, depth 1→2). See results above. | — |
| 3 | **Question A.3**: Does the unitary constraint limit expressibility vs. unconstrained classical CP factors? | Question A | **ATTEMPTED 2026-07-28 (R4), NOT ANSWERED.** The classical CP baseline scored `33.9%` against an MLP's `96.0%` — the CP node is broken, so the quantum-vs-CP gap is an artifact and must not be cited. Needs a classical TTN competitive with the MLP reference before the question can be posed. | 1–2 days (fix or replace the CP baseline first) |
| 4 | ~~Question B.1 / Task 1.4: Tensor-network simulator device benchmark~~ | Question B / Task 1.1b, 1.4 | **Done 2026-07-26** — forward pass works at 32×32 (0.13s) and 64×64 (1.2s) via `default.tensor(method='tn')`; gradients correct but slow (56.7s/step at depth-3). Training-cost fix is item 14 (SPSA, Task 1.5). | — |
| 5 | ~~Question B.3: Classical compression head ablation~~ | Question B | **CLOSED 2026-07-26, rejected on principle** — project direction is a purely quantum implementation; classical compression substituting for quantum circuit width is out of scope. `hybrid_trainer.py` deprecated as an architecture direction. | — |
| 6 | ~~Question C.1/C.3: Spatial ancilla ablation~~ | Question C | **Done 2026-07-26** — ancilla is not better (and mildly worse); confirmed safe to drop, reducing 80→64 qubits. Sub-question 2 (relational sensitivity) remains open, deferred to CLEVR. See results above. | — |
| 7 | ~~Question D.2: Effect of classical head on gradients~~ | Question D | **CLOSED 2026-07-26, rejected on principle** — tuning/investigating the classical readout layer is a hybrid-architecture question, not a purely-quantum one; the minimal readout stays as an unavoidable I/O boundary but isn't a research target. | — |
| 8 | ~~Question E.2: Entropy propagation through partial trace~~ | Question E | **Done 2026-07-27** — partial answer only (ceiling-effect confound); entropy gap narrows with noise, consistent with washing away but not conclusive. See results above. | — |
| 9 | ~~Bespoke Quantum Dropout mechanism~~ | Section 5 | **Deprioritized 2026-07-26** — noise already provides the regularization effect; see item 11 for the still-open "why" question | — |
| 10 | ~~Quantum-native residuals, Methods 1–2 (data re-uploading, near-identity init)~~ | Section 5 | **Done 2026-07-26** — near-identity: no effect; re-uploading: higher peak accuracy but worse noise robustness (collapses to 28.1% at p≥0.10 vs. baseline's 50.0% floor). See results above. | — |
| 11 | ~~Quantum-native residuals, Methods 3–5 (mixed-unitary channel, LCU, LCU overhead diagnostic)~~ | Section 5 | **Done 2026-07-26**, corrected same day after 10-seed re-test — mixed-channel's noise-robustness claim retracted (didn't replicate); only a modest noiseless accuracy edge survives, with a 10% training-failure rate. LCU inherits the instability, noiseless-only, pays ~2x shot overhead at the root node. See results above. | — |
| 12 | ~~Question F: Depolarizing noise as implicit regularization (scheduling)~~ | Question F (new) | **Done 2026-07-27** — negative result: training under noise does not improve clean accuracy (mildly worse, 5 seeds). Mechanism sub-question (F.1) deprioritized given this. See results above. | — |
| 13 | ~~Stabilize `mixed_channel` training~~ | Section 5 (follow-up) | **CLOSED 2026-07-28 (R3)** — there is nothing to stabilise. At restored capacity `mixed_channel` had **0/30** failures (the ~10% dead-gradient rate was a bottleneck artifact) and is simply `−10.9` pts worse than baseline. No remaining motivation. | — |
| 14 | **Task 1.5: SPSA gradient estimation for depth-3/4 training** | Section 3, Phase 1 (new) | Not started — needed to make 32×32/64×64 *training* practical now that forward passes work (item 4); parameter-shift costs 56.7s/step at depth-3, SPSA should reduce this ~250x | 1 day |

> **⚠️ SUPERSEDED IN PART (2026-07-27)**: the "done/closed" statuses below for items 6, 10, 11 (and the framing of item 12) rest on experiments run through a scalar-readout bottleneck — see `research_log.md` 2026-07-27 "Code Audit" and **Section 7** below. They are downgraded to PROVISIONAL pending task R3.

| 15 | **Question C.2**: does explicit positional encoding help *relational* tasks? | Question C | **PROMOTED to REQUIRED 2026-07-28.** R3's `−19.5` ancilla result comes from a translation-invariant task where position barely affects the label, so it says nothing about relational reasoning. CLEVR's left/right/front/behind task is the only discriminating test. Run with and without the ancilla. | folded into Phase 2 |
| 16 | **Classical controls for every CLEVR accuracy claim** | R4 (new) | **REQUIRED 2026-07-28.** R4 showed a quantum-vs-classical claim is uninterpretable without a matched-parameter classical reference, and that a broken baseline can manufacture a 57.8-pt fake result. Build controls in from day one. | folded into Phase 2 |
| 19 | **Task R7: port to the coherent quantum tree** | Code Audit #2 (new) | **BLOCKING, added 2026-07-28.** The trained model measures and re-encodes between levels, so it has zero inter-level entanglement and is a quantum-node/classical-wiring hybrid, not a quantum TTN. Must be fixed before figures are regenerated. | 1 day |
| 20 | **Task R8: noise close-out** | Scope decision (new) | One sweep on the coherent base model, then the noise track closes. | 0.5 day |
| 17 | **Position on the purely-quantum scope rule** | §8.5 item 1 (new) | **OPEN, and sharpened by Code Audit #2** — the hybrid tree was a larger violation of the same rule than anything Question B.3 rejected. R2 measured that widening the *classical* encoder bought `+12.6` pts — more than any quantum architectural effect measured in this project — while Question B.3 was closed on principle for being "classical capacity substituting for quantum work". Needs a stated, defended position. | 0.5 day (writing, not experiment) |
| 18 | ~~Qiskit Aer cross-check of the noise chapter~~ | PennyLane bug | **DROPPED 2026-07-28** — proposed to validate the noise chapter after the silent-corruption bug; with the noise track closed the motivation goes with it. The `default.mixed` bug is downgraded to a documented curiosity: the workaround and its regression test stay, but nothing on the forward path uses that device. | — |

**Total backlog**: roughly 2–5 days of remaining work (items 2, 4, 5, 6, 7, 8, 9, 10, 11, 12 are done or closed; item 13 deprioritized). Remaining open items: item 1 (Phase 3 checkbox discrepancy — ZNE/Mitiq/optimizer comparison, 2-3 days, discretionary), item 3 (Question A.3, expressibility, 1-2 days), and item 14 (SPSA for depth-3/4 training, 1 day — only needed if CLEVR will use images larger than 16×16; otherwise park it as documented-but-not-executed infrastructure and proceed to CLEVR at the validated 16×16 size). **Any future backlog item that proposes adding or tuning a classical component inside the core quantum pipeline (compression heads, hybrid readouts, classical-bypass shortcuts) should be closed on principle without running an experiment** — the project direction is a purely quantum implementation; only the unavoidable classical I/O boundary (pixel-to-angle encoding in, class logits out) is exempt. **None of the five quantum-native residual methods tested show a robustly-confirmed win**, and the spatial ancilla is confirmed unnecessary — if either line is revisited, it should start from a fresh hypothesis rather than re-attempting to rescue a mechanism whose apparent benefit didn't survive a larger sample.

---

## 7. Phase 1.5 — Remediation Plan (added 2026-07-27): Close the Theory Phase Correctly Before CLEVR

**Why this section exists.** A code audit on 2026-07-27 (full writeup: `research_log.md`, entry "Code Audit: Scalar-Readout Bottleneck Invalidates the July-26/27 Decision Layer") found that every experiment behind the July-26/27 architecture decisions compresses the entire image to a **single scalar** before classification — `self.head = nn.Linear(1, 4)` — so 4-class classification is `argmax` over four affine functions of one number in $[-1, 1]$. This is a 1-dimensional representation of the whole image, and it explains why every result in that period sits in a `50–57%` band with repeated "flat 50.0% noise floors" (`50.0%` is exactly the single-attribute shortcut ceiling of the `overlapping` dataset mode). The earlier 2026-07-17 run that reached `75.0%` at the same image size used a 4-dimensional readout (`train_synthetic_shapes.py:91`, `nn.Linear(4, 4)`); the scalar readout is a regression introduced after it.

**What this means**: null results measured in that regime cannot distinguish "the mechanism doesn't help" from "the bottleneck dominates, so nothing would show up." The affected verdicts are **provisional**, not wrong — they may well survive re-testing, but they cannot currently be defended in a thesis.

**Instructions for whoever executes this section.** Tasks R1–R6 are self-contained and are written to be runnable without prior context on this project. Run them in order; R1 gates everything else. Environment is the `qnlp` conda env — prefix every Python command with `conda run -n qnlp` (the base env lacks torch/pennylane/polars). All scripts live in `qnlp/image_tower/classification/quantum/`; all outputs go to that directory's `results/` subfolder. Per the Log Maintenance Protocol at the top of `research_log.md`: write a log entry with objective/motivation/expected outcome **before** each task, and update it with actual metrics, plots, and exact reproduction commands **after**.

---

### Task R1 (BLOCKING — do first): Remove the scalar-readout bottleneck — DONE 2026-07-27

**Completed.** `qttn_core.py` built as specified below; readout×encoding swept per the method. Full writeup: `research_log.md` 2026-07-27 "Completed Task R1". **Result: the original 256-train/15-epoch protocol did not clear the 70% gate for any configuration** (best: `root_multi_pauli`+`multi_axis`, 59.1 ± 6.4%) — readout width alone moved the mean from 55.0% to 59.1%, a real but insufficient improvement. The capacity-fallback escalation (1024 train / 30 epochs, same best config) reached **78.75 ± 7.2%**, passing the gate. **Architecture and protocol of record for R2 onward: `readout=root_multi_pauli`, `encoding=multi_axis`, 1024 train / 64 test / 30 epochs** — R2/R3 must use this protocol, not the original 256/15 one, since 256/15 is now known to be underpowered independent of readout choice.

<details>
<summary>Original task specification (for reference)</summary>

**Problem.** Confirmed at these exact locations:

| File | Line | Code |
|---|---|---|
| `investigate_quantum_residuals.py` | 134 | `self.patch_embed = nn.Linear(patch_size*patch_size*3, 1)` |
| `investigate_quantum_residuals.py` | 137 | `self.head = nn.Linear(1, 4)` |
| `investigate_quantum_residuals.py` | 162 | `return self.head(root)` — `root` is `[B, 1]` |
| `investigate_spatial_ancilla.py` | 199, 202 | same `Linear(48,1)` / `Linear(1,4)` pair |
| `investigate_ancilla_residuals.py` | 194, 197 | same |
| `compare_topologies.py` | 119, 143 | `Linear(48,3)` in, but still `Linear(1,4)` out |

**Deliverable.** A single shared module, `qnlp/image_tower/classification/quantum/qttn_core.py`, containing one configurable `HierarchicalQTTNClassifier` that all subsequent investigations import — rather than each investigation re-declaring its own slightly-divergent copy (the divergence is what let the regression go unnoticed, and is also the cause of the ansatz drift in R2). It must expose, at minimum:

- `readout: {"scalar", "root_multi_pauli", "level1_survivors"}`
  - `"scalar"` — the current behaviour, kept **only** so R1's A/B comparison is exact.
  - `"root_multi_pauli"` — measure $\langle X \rangle, \langle Y \rangle, \langle Z \rangle$ on the root qubit → `nn.Linear(3, n_classes)`. Cheapest fix; no extra qubits, no circuit change.
  - `"level1_survivors"` — return all 4 level-1 survivor expectation values *in addition to* the root → `nn.Linear(4, n_classes)` (or `Linear(5, ...)` including root). This is what the 07-17 `75.0%` run effectively did.
- `encoding: {"scalar_ry", "multi_axis"}`
  - `"scalar_ry"` — current: `Linear(48,1)` → `tanh` → `RY`.
  - `"multi_axis"` — `Linear(48,3)` → `tanh` → consecutive `RX, RY, RZ` on the same qubit. This is the encoding the 2026-07-17 benchmark actually selected (see R2).
- `ansatz: {"strongly_entangling", "iqp"}` (see R2), `img_size`, `n_classes`, `p_noise`, and a `mode` hook for the residual variants so R3 can reuse it.

**Method — change one thing at a time.** Confounding readout width with encoding width is how you end up unable to attribute the improvement.

1. Fix readout only: `encoding="scalar_ry"`, sweep `readout ∈ {scalar, root_multi_pauli, level1_survivors}`. 5 seeds × 15 epochs, 16×16 `overlapping` shapes, 256 train / 64 test (matching the existing protocol so results are comparable to the log).
2. Then fix encoding: take the winning readout, sweep `encoding ∈ {scalar_ry, multi_axis}`, 5 seeds.

**Acceptance criterion.** At least one configuration reaches **≥ 70% mean final val accuracy over 5 seeds** on 16×16 `overlapping` shapes. Rationale: `50.0%` is the single-attribute shortcut ceiling, the 07-17 4-dim-readout run reached `75.0%` at this exact size and dataset, and the 8×8 benchmarks reached `95%+` — so ≥70% is a demonstrated, not aspirational, target. Below 70%, the model is still too close to the shortcut ceiling for ablations to be informative, and R3 must not proceed.

**If the criterion is not met**: do not proceed to R3, and do not paper over it. Escalate with the numbers. Likely next things to check, in order: (a) the patch encoder is still `48 → 1` and throwing away nearly everything — try `multi_axis` immediately rather than after the readout sweep; (b) train set of 256 samples / 15 epochs is too small — try 1024 samples / 30 epochs as a capacity check; (c) `lr=0.03` with AdamW may be mistuned for the wider head.

**Est. time**: 0.5–1 day (the module consolidation is most of it; the runs themselves are minutes at 16×16 on `default.qubit`).

</details>

---

### ✅ Task R1b — DONE 2026-07-28: The missing control arm — run the OLD config at the NEW protocol

> **RESULT: AUDIT_CONFIRMED.** The `scalar`/`scalar_ry` control reaches only **`62.5 ± 6.9`** at 1024/30 — it does *not* clear the 70% gate, versus the winner's `78.8 ± 7.2`. The +16.2-pt architecture gap exceeds the 10.3 pts this design resolves. **The Code Audit's diagnosis stands and `root_multi_pauli` + `multi_axis` remains the architecture of record.**
>
> **The sharpened claim** (better than either the original audit wording or R1's write-up): protocol alone buys `+7.5` pts, architecture alone buys `+4.1` pts, but architecture *at adequate training* buys `+16.2` — **readout width is only expressible once there is enough data to train it.** That interaction is why R1's one-factor-at-a-time sweeps at 256/15 each looked like noise.
>
> Part B additionally refuted this roadmap's own pairing advice and produced the power analysis that rewrote R3's design — see the R3 precondition below and `research_log.md` 2026-07-28. Full details there; the rest of this task description is retained as the specification that was executed.


**Why this exists.** R1 passed the gate, but not in the way the Code Audit predicted, and the experiment as run cannot attribute the gain:

| change | effect (5 seeds) | resolvable at n=5? |
|---|---|---|
| readout: `scalar` → `root_multi_pauli` | `55.0 ± 3.5` → `57.5 ± 7.2` | no — well inside noise |
| encoding: `scalar_ry` → `multi_axis` | `57.5 ± 7.2` → `59.1 ± 6.4` | no — well inside noise |
| protocol: 256/15 → **1024/30** | `59.1` → **`78.75 ± 7.2`** | yes — large |

All of the gate-clearing movement came from the training protocol. Both architecture changes are inside seed noise. **And the capacity fallback was only run on the winning config** — nobody ran the original `scalar` readout at 1024/30.

**The question this decides.** Does `readout=scalar, encoding=scalar_ry` also reach ~78% at 1024/30? If it does, the Code Audit's central diagnosis — that the `nn.Linear(1, 4)` bottleneck is what invalidated the July-26/27 ablations — is **wrong**, and the real cause is simply that 256 samples / 15 epochs is undertrained for this task. That conclusion is currently written into `research_log.md` as established fact and would need correcting before it reaches the thesis.

**Deliverable.** One run: `readout=scalar`, `encoding=scalar_ry`, 1024 train / 64 test / 30 epochs, the same 5 seeds as R1's fallback. Minutes of compute on `default.qubit`. Report the full 2×2 (`{scalar, root_multi_pauli+multi_axis} × {256/15, 1024/30}`) so the attribution is complete.

**Outcome handling — all three branches are fine, they just tell different stories:**
- **Scalar stays ≪ 78% at 1024/30** → the audit's diagnosis is confirmed, readout width was genuinely the binding constraint (it just needed enough training to express itself). Keep `root_multi_pauli` + `multi_axis` as the architecture of record and strengthen the log entry's attribution.
- **Scalar reaches ≈ 78% at 1024/30** → the diagnosis was wrong. Correct the Code Audit entry: the July-26/27 ablations were underpowered because of the *training protocol*, not the readout. The ablations still need re-running (R3 stands unchanged — an underpowered protocol invalidates them just as thoroughly), but the stated reason changes, and **the architecture of record should revert to the simpler/cheaper option** (105 params vs. 211 — simpler is also a better story for the CLEVR scaling argument).
- **Ambiguous / partial gap** → report it as ambiguous and pick the simpler architecture by default; do not manufacture a clean narrative from a noisy result.

**Also fix while here (R1 reporting gaps):**
1. The R1 results never reached the **Experiment & Metrics Record** table — they exist only inside the log entry, contrary to the Log Maintenance Protocol. Add rows for both sweeps and the capacity fallback.
2. Soften the R1 entry's line *"readout width matters as of Sweep 1 (58 vs 55, direction is consistent, if noisy)"* — a 2.5-point gap with stds of 3.5 and 7.2 at n=5 is not evidence of a direction. State it as unresolved at this sample size and let R1b settle it.

**Est. time**: 1–2 hours.

---

### ✅ Task R2 — DONE 2026-07-28: Reconcile the ansatz-of-record with the code

> **RESULT: CONFIRM_DOCUMENTED.** `multi_axis+iqp` is the best arm at `80.9 ± 4.9` (21 seeds). The documented choice survives.
>
> **The decomposition is the finding**: the *encoding* is resolved and large (`+12.6` pts vs a `3.6`-pt limit); the *ansatz* is **not** resolved (`+1.6` vs `4.8`). **The July code's consequential error was `scalar_ry`, not `strongly_entangling`.** IQP is adopted for its lower seed variance (`4.9` vs `9.7`), which buys resolution downstream — not because it is measurably more accurate. Architecture of record pinned in `phase15_common.ARCH`. Full writeup: `research_log.md` 2026-07-28.


**Problem.** `research_log.md` Current Status and the 2026-07-17 encoding/ansatz benchmark both state the selected architecture is **Multi-Axis Encoding + IQP ansatz**. The July-26/27 scripts actually use `qml.RY(inputs[:, i], wires=i)` + `qml.StronglyEntanglingLayers` (`investigate_quantum_residuals.py:77-96`) — neither multi-axis nor IQP. So every recent architecture decision was made on a different circuit than the project's stated selection.

**Also note**: the original selection came from a 12-configuration sweep at **8×8** with apparently one seed per configuration, and Multi-Axis+IQP (`95.3%`) was chosen over Multi-Axis+HEA (`96.9%`) on parameter-efficiency grounds. A single-seed selection among 12 candidates at a smaller image size is a weak basis for a decision the whole thesis rests on.

**Deliverable.** Using the R1 module (`qttn_core.py`), re-run the encoding × ansatz comparison at **16×16** with the fixed readout (`root_multi_pauli`, per R1's result — R2 was written assuming R1 would pick a fixed readout, and it did): `{scalar_ry, multi_axis} × {strongly_entangling, iqp}`, 5 seeds each, **using R1's protocol of record (1024 train / 30 epochs), not the original 256/15** — the 256/15 protocol is now known to be underpowered for this task regardless of ansatz. Then either (a) confirm Multi-Axis+IQP and port it into `qttn_core.py` as the default, or (b) if `StronglyEntanglingLayers` wins at 16×16, **update the stated architecture of record** in `research_log.md` Current Status and in this roadmap — do not leave the documentation and the code disagreeing.

**Est. time**: 0.5 day (reuses R1's harness entirely).

---

### ✅ Task R3 — DONE 2026-07-28: Re-run the decisive ablations at restored capacity

> **RESULT: all rejections upheld, now on positive evidence rather than absence of it.** 30 seeds/arm, 0/30 failures in every arm.
>
> | arm | score | vs. baseline | limit | verdict |
> |---|---|---|---|---|
> | `reupload` | `80.9 ± 6.6` | `+1.5` | 3.8 | unresolved |
> | `baseline` | `79.4 ± 8.0` | — | — | — |
> | `mixed_channel` | `68.5 ± 10.1` | `−10.9` | 4.7 | **worse, resolved** |
> | `with_ancilla` | `59.9 ± 6.9` | `−19.5` | 3.8 | **worse, resolved** |
>
> Two things genuinely changed. `mixed_channel`'s dead-gradient failures are **gone** (0/30 vs 1/10 in July) — it is not unstable, just worse, which closes backlog item 13 outright. And `reupload` is a **true null**, not a trade-off: its July noise-collapse (`28.1%` at p≥0.10) does not reproduce; it degrades similarly to baseline. That narrative was an artifact of the bottlenecked model.
>
> **Ancilla scope caveat — do not write this up as "positional encoding is harmful".** See Question C above and §8.5. Full writeup and noise sweeps: `research_log.md` 2026-07-28.


> **⚠️ PRECONDITION — statistical power (added 2026-07-27, REVISED 2026-07-28 after R1b measured it).**
>
> The concern stands: at n=10 this system resolves only ~**7.6-point** differences, while the residual/ancilla effects R3 is hunting were **2–3 points**. **As originally specified, R3 would produce another inconclusive null** — the 2026-07-26 mixed-channel failure mode at higher accuracy. But R1b measured the variance structure directly (`research_log.md` 2026-07-28, `results/r1b_control_results.json`), and the fix is not the one first proposed here:
>
> 1. ~~**Run paired, not unpaired.**~~ — **RETRACTED, measured and refuted.** R1b built the paired harness (data ordering pinned per seed, independent of model-init RNG consumption — verified working) and found the cross-variant seed correlation is **−0.22**. Seed identity is *not* a shared difficulty factor between variants, so pairing has nothing to cancel: min detectable effect was **10.2 pts paired vs. 7.6 pts unpaired** — pairing made it *worse*. Run-to-run variance here is dominated by within-run optimization stochasticity, not by the data draw. **Do not pair.** (The harness in `run_r1b_control.py:train_config_paired` is still useful if a future comparison *does* show positive cross-variant correlation — check before assuming either way.)
> 2. **Score runs by the mean of the last 5 epochs, not the single final epoch.** Free improvement: MDE `8.0 → 7.2` pts, and it equalizes the arms' stds (`8.1/10.0 → 8.2/8.2`). Per-seed val-acc curves are already saved in `r1b_control_results.json` if you want to re-check this.
> 3. **Run ~60 seeds per arm, not 10.** This is the actual fix. At pooled std `8.2` and ~18s per run at 1024/30:
>
>    | effect size to resolve | seeds per arm | ≈ time per arm |
>    |---|---|---|
>    | 2 pts | 130 | 39 min |
>    | **3 pts** | **58** | **17 min** |
>    | 5 pts | 21 | 6 min |
>    | 8 pts | 9 | 3 min |
>
>    The runs are cheap enough that this is simply affordable — 10 seeds resolves only ~8-point effects, which is not the regime R3 operates in. Budget **~60 seeds/arm**.
> 4. **Optionally reduce the variance at source.** The per-seed spread is large (`46.9 → 78.1` for the baseline arm at n=10). An lr decay or longer schedule may cut the std; if it does, the seed requirement drops quadratically. Worth 30 minutes of trying before committing to 60-seed runs.
>
> **Report for every R3 comparison**: mean ± std, and the minimum effect size the design could have resolved. "No significant difference" without that second number is not a finding.

Only two mechanisms are worth re-testing. Do **not** re-run all five residual methods.

**R3a — Spatial ancilla (Question C.1).** Current verdict rests on `52.8% ± 5.7` vs. `55.0% ± 3.5` at n=5, which is well inside noise in either direction. Re-run `investigate_spatial_ancilla.py` on the R1/R2 architecture, **10 seeds**, noise sweep averaged over all trained seeds (already the practice in that script — preserve it).

**R3b — The two residual mechanisms that showed any signal.** `reupload` (Method 1: real noiseless expressivity gain, collapsed under noise) and `mixed_channel` (Method 3: small noiseless edge, 10% dead-gradient failure rate). 10 seeds each vs. baseline, both stages. Skip `near_identity` (no effect in either regime, and mechanistically it only perturbs initialization — a capacity fix does not change that argument), `lcu`/`lcu-lite` (inherits `mixed_channel`'s instability, adds ~2× shot overhead at the root, and remains untestable under noise due to the `default.mixed` postselection limitation in PennyLane 0.43.2), and the classical-bypass residual (out of scope by the purity rule).

**R3c — Topology benchmark (optional, do only if time permits).** `compare_topologies.py` also used `Linear(1,4)`, and its QTTN arm scored `42.2%` — *below* the 50% single-attribute ceiling, i.e. that arm had not reliably learned even one attribute. The "MERA is more expressive than QTTN" conclusion (`51.6%` vs `42.2%`) is therefore weakly grounded. It is not load-bearing for CLEVR (the QTTN-vs-MERA decision is settled on contraction-complexity grounds, Section 5, which is a scaling argument independent of these accuracies), so re-run it only if the accuracy claim will be cited in the thesis.

**Reporting rule.** For each mechanism, report the mean ± std over 10 seeds *and* state explicitly whether the difference exceeds seed-to-seed variance. The 2026-07-26 mixed-channel retraction happened precisely because a single-seed spot check produced a conclusion that reversed on a larger sample — do not repeat it.

**Outcome handling.**
- If the verdicts hold at restored capacity: promote them from PROVISIONAL back to binding rules, now with defensible evidence, and note in the log that they were re-validated post-remediation.
- If any verdict flips: update the corresponding `NOTE` block in `quantum_implementation_plan.md`, Section 5 of this roadmap, and the Current Status block in `research_log.md`. A flip is a *good* outcome for the thesis (it means the ablation is measuring something), not a setback.

**Est. time**: 1–1.5 days.

---

### ✅ Task R4 — DONE 2026-07-28: Classical control (does NOT close Question A.3)

> **RESULT: the first quantum-vs-classical measurement in the project's history — and A.3 is still open.**
>
> | arm | score | params |
> |---|---|---|
> | MLP reference | `96.0 ± 1.5` | 999 |
> | **quantum** | **`79.0 ± 8.7`** | **211** |
> | MLP, parameter-matched | `70.5 ± 18.1` | 257 |
> | classical CP + residual + dropout | `59.3 ± 2.9` | 332 |
> | classical CP, bare | `33.9 ± 12.4` | 308 |
>
> At **matched parameter count** the quantum tower is statistically tied with a classical MLP (`+8.5` vs an `8.9`-pt limit); it loses by `17.0` pts only when the MLP gets ~5x the parameters. **The defensible thesis claim is parameter efficiency, not raw accuracy.**
>
> **Question A.3 is NOT answerable from this run and the script refuses to emit a verdict**: `CPQuadRankLayer` scores `33.9%` where a comparable MLP reaches `96.0%`, so the CP node — not classical computation — is what underperforms. The quantum-vs-CP gap is a **baseline artifact and must not be cited as quantum advantage**. A.3 needs a classical TTN competitive with the MLP reference first.
>
> Also measured: the repo's classical node depends heavily on residual+dropout (`+25.4` pts, limit `5.6`) — load-bearing classically, while the quantum tower's rules against them were upheld by R3. That contrast belongs in the thesis.


**Problem.** There is no classical control anywhere in this investigation — not one run of a matched classical tensor-network model on the same data at the same sizes. Every result to date is quantum-vs-quantum, which cannot support a claim about quantum models *relative to their classical analogue*. This is the largest structural gap for the thesis, and it is cheap to close because the classical CP-tree implementation already exists in this repo (`qnlp/discoviz/models/cp_node.py`, and the classical TTN image tower under `qnlp/image_tower/`).

This is simultaneously **Question A.3** (does the unitary constraint $U^\dagger U = I$ limit representation capacity versus unconstrained classical CP factor weights?), which is Section 6 backlog item 3 and currently the only substantive un-run item in Question A.

**Deliverable.** A classical CP-TTN with matched structure (same 4×4 patch grid, same quad-tree topology, same tree depth) and **matched parameter count** to the R1/R2 quantum model, trained on identical synthetic-shapes data and seeds. Report side by side. Two variants are informative: (a) faithful analogue — no residual, no dropout, matching the quantum model's constraints; (b) the repo's actual classical node with residual + dropout, as the "what classical practice does" reference point.

**Why variant (b) matters.** The existing classical tower uses residuals (`cp_node.py:28,68-69`) and dropout and got good results, while this investigation has issued binding rules *against* both for the quantum tower. That divergence needs an explanation in the thesis, and right now the only bridge is "quantum-native versions behave differently" — an argument resting on the provisional evidence R3 is re-testing. Running (a) and (b) makes the comparison explicit instead of implicit.

**Est. time**: 1–2 days.

---

### Task R7 (BLOCKING, added 2026-07-28): Port to the coherent quantum tree

**Problem.** The trained model is not a coherent quantum tree. Every `QuantumNode` builds **its own 4-qubit device** (`qttn_core.py:229`); level-1 nodes return a single `⟨Z⟩` each, and those four **classical scalars** are re-encoded as `RY` angles into a *separate* root circuit (`qttn_core.py:214-218`). The tree measures at every level, so there is **zero entanglement across tree levels**, and the inter-level bond is one real scalar — narrower than the $\chi=2$ single qubit the design specifies. Full writeup: `research_log.md` 2026-07-28 "Code Audit #2".

**The coherent version already exists**: `train_synthetic_shapes.py` (2026-07-17) — one 16-qubit device, level-1 unitaries on wires `[0-3] [4-7] [8-11] [12-15]`, level-2 applied **directly to the survivors** `[0, 4, 8, 12]`, measured only at the end. It reached `75.0%` on the *old* 256/15 protocol. The July-26 scripts replaced it, undocumented — the same regression pattern as the scalar readout.

**Why it happened, and why the reason no longer applies**: `default.mixed` is $O(4^N)$, so a 16-qubit *noisy* circuit needs ~68GB (OOM-confirmed) and every noisy circuit was capped at 4–5 qubits. The hybrid is what permits a "tree" under that cap. But it only ever applied to **noisy** simulation — noiseless 16-qubit statevector is $2^{16}$ amplitudes, trivial, and `default.tensor` runs the coherent tree at 64 and 256 qubits already.

**Deliverable.**
1. Add a coherent tree to `qttn_core` (single device, all patches encoded together, survivors passed as qubits, measured once at the end), keeping the existing `readout` / `encoding` / `ansatz` / `mode` / `use_ancilla` axes. Use `train_synthetic_shapes.py:50-70` as the reference implementation.
2. Re-baseline on the protocol of record. **Do not expect the hybrid's numbers to reproduce** — these are different models, not two implementations of one.
3. Re-run **R3** (4 arms × 30 seeds; ~25 min with the parallel workers) and **R4** (classical control, 21 seeds) on the coherent model. R2's encoding/ansatz choice should carry over but is cheap to confirm.
4. Add a regression test asserting the tree is coherent — one device for the whole tower, no intermediate `expval` — so this specific regression cannot recur silently. It is the second instance of a coherent design being replaced by a cheaper approximation without a record.

**What carries over and what does not.** The node-level ablations (residuals, spatial ancilla) are mechanisms *within* a node and each node is a genuine 4-qubit VQC, so they stand as **node-level** results — but they are not tree-level results until re-checked here. Question A.2 (entropy saturation) and E.2 (noise propagation) were measured on coherent circuits and describe the coherent model, not the hybrid that was trained.

**Est. time**: 1 day.

---

### Task R8 (added 2026-07-28): Noise close-out

Per the scope decision of 2026-07-28, the project proceeds **noiseless only**. This task is the single close-out experiment, not the start of a noise programme.

**Deliverable.** One depolarizing sweep on the coherent **base** model at the largest coherently-simulable size, covering the standing observation that a small amount of noise sometimes *improves* accuracy (evidence already exists: `42.2% → 53.9%` at p=0.10 in the topology benchmark, `85.9% → 91.4%` at p=0.005 in the synthetic-shapes sweep). Report the degradation curve, answer the small-noise question, close the track.

**Explicitly NOT in scope** — none of these require an experiment to close: quantum trajectories, a Qiskit Aer cross-check, ZNE/Mitiq (Task 3.2), the optimizer-under-noise comparison (Task 3.3), CLEVR noise calibration (Task 3.1), and per-variant noise sweeps.

**Standing limitation to state in the thesis rather than fix**: noisy simulation was capped at 4–5 qubits throughout by the $O(4^N)$ density-matrix cost, so every noise result characterises a **single quantum node**, not the full tree. Whether node-level noise tolerance extends to a deep tree is untested and will remain so.

**Est. time**: 0.5 day.

---

### Task R5: Documentation and consistency fixes (no experiments)

1. **Resolve the $p_{crit}$ contradiction.** The 2026-07-17 noise sweep reports $p_{crit} \approx 0.05$ for a 4-qubit QTTN on 8×8 shapes; the same-day encoding/ansatz benchmark reports $p_{crit} > 0.200$ for essentially every configuration including same-family ones. Both are in the Experiment & Metrics Record and, as written, contradict each other. Read `emulate_noise_synthetic_shapes.py` and `benchmark_encodings_ansatze.py`, determine each script's actual threshold definition, and annotate both rows in the metrics table with the definition used.
2. **Rescope the barren-plateau claim.** Four system sizes ($N \le 20$), non-monotonic variance (`1.51e-1, 6.08e-2, 8.61e-2, 5.94e-2`), currently reported as an "Empirical BP Immunity Proof" against an asymptotic $2^{-N}$ strawman. Hierarchical/TTN barren-plateau resistance under local observables is an established theoretical result — reframe the data as *consistent with* that theory and cite it, rather than presenting it as an independent proof. Fix in both `research_log.md` (2026-07-18 entry) and Section 4 of this roadmap.
3. **Re-frame Question F's premise.** Its founding observation (`42.2% → 53.9%` under noise) came from a `Linear(1,4)` model; depolarizing noise contracts the scalar toward 0, sliding samples across a 1-D interval partition. Record this as the leading candidate mechanistic explanation. F.2's negative result (training under noise does not improve clean accuracy) **stands regardless** — but the phenomenon it was chasing may be an artifact. If R1's fixed-capacity runs no longer show eval-time noise improving accuracy, that confirms the artifact hypothesis and Question F can be closed outright rather than left "open but deprioritized."
4. **Give the purity rule a quantitative boundary.** Question B.3 was closed *on principle* because `hybrid_trainer.py` had a classical `Linear(16, 4)` before a 4-qubit VQC — yet `Linear(48, 1)` per patch, used in every current experiment, is a strictly more aggressive classical compression. Either state what encoder width counts as "minimal I/O boundary" (e.g. output dimension ≤ qubits-per-patch × rotations-per-qubit, which `multi_axis`'s `Linear(48,3)` satisfies and `Linear(16,4)`-into-4-qubits arguably also does), or reopen B.3. Update the `NOTE — Classical Hybrid Shortcuts` block in `quantum_implementation_plan.md` accordingly.

5. **Annotate the superseded training protocol (added 2026-07-27).** R1 established 1024 train / 30 epochs as the protocol of record, replacing 256/15, which is now known to be underpowered for this task *regardless of architecture*. Every pre-R1 row in the Experiment & Metrics Record was measured at 256/15 (or the older 512/128 at 8×8). Annotate those rows with their protocol so no thesis reader compares a 256/15 number against a 1024/30 number and reads the difference as an architecture effect. If R1b shows the protocol was the dominant factor all along, this annotation becomes the single most important caveat in the table.

**Est. time**: 0.5 day.

---

### Task R6: SPSA for depth-3/4 training (this is Task 1.5, restated with a realistic budget)

Unchanged in substance from Section 3, Task 1.5, but **its priority is now established rather than discretionary**: CLEVR images are 480×320. Downsampled to 16×16 an object occupies a handful of pixels, at which point `material` (rubber vs. metal — essentially specular-highlight detection) and `size` are close to unlearnable. The Step 2 pass criterion in `quantum_implementation_plan.md` (>80% on all 4 attribute heads) is **not reachable at 16×16**. So either R6 lands and CLEVR runs at 32×32, or the CLEVR task definition must be revised down (see the decision gate below).

**Correction to the cost estimate.** The roadmap currently projects "~56.7s/step → ~0.2s/step (~250x speedup)". That compares a full parameter-shift gradient against 2 SPSA circuit evaluations for **one sample**, and is optimistic on two counts: (a) a real training step uses a batch — at depth-3's 0.13s forward pass, a batch of 32 costs ~8s/step, not 0.2s; (b) SPSA gradients are much noisier than exact ones and need substantially more steps to converge. Budget the actual wall-clock cost of a full training run before committing, and report convergence quality (not just per-step speed) versus a parameter-shift reference run at depth-2 where both are affordable.

**Est. time**: 1 day for the estimator + a short depth-3 convergence check; add 0.5–1 day if a full depth-3 training run is needed to establish the CLEVR resolution.

---

### Decision Gate: when is the theory phase closed and CLEVR ready to start?

Proceed to Phase 2 when **all** of the following hold:

1. ✅ **R1 acceptance criterion met (2026-07-27)** — 78.75 ± 7.2% mean val accuracy over 5 seeds at 16×16 (`readout=root_multi_pauli`, `encoding=multi_axis`, 1024 train / 30 epochs), i.e. the model demonstrably binds both attributes rather than sitting near the 50% shortcut ceiling. Note: this required the capacity-fallback protocol, not the original 256/15 one — carry 1024/30 forward into R2/R3. ✅ **R1b completed 2026-07-28 — gate item 1 fully closed.** The control (`scalar`/`scalar_ry` at 1024/30) reached only `62.5 ± 6.9`, confirming the readout was genuinely binding; architecture of record unchanged.
2. ✅ **R2 done (2026-07-28)** — architecture of record pinned in `phase15_common.ARCH`, code and documentation agree, and a regression test fails if it drifts.
3. ✅ **R3 done (2026-07-28) at 30 seeds** — every verdict re-validated on positive evidence, with the resolution limit reported alongside each. Nothing flipped; two narratives (mixed-channel instability, reupload noise-collapse) were shown to be bottleneck artifacts.
4. ✅ **R4 done (2026-07-28)** — classical control exists. Outcome: quantum is parameter-competitive, not accuracy-dominant. **Question A.3 remains open** (CP baseline broken). CLEVR must carry matched-parameter classical controls from day one.
5. **R7 — STILL OPEN, BLOCKING.** The tower is a coherent quantum tree, re-baselined, with R3 and R4 re-run on it. Until this lands, every result describes a quantum-node/classical-wiring hybrid rather than a quantum TTN.
6. **R8 — STILL OPEN.** Noise close-out done and the noise track formally closed.
7. **R5 — STILL OPEN.** No known internal contradictions left in the log.
8. **R6 — STILL OPEN.** Resolved one way or the other — either 32×32 training is practical, or the CLEVR task is explicitly re-scoped (e.g. drop `material`, keep `color`/`shape`/`size`, and state in the thesis that resolution, not architecture, is the limiting factor). **Do not silently run CLEVR at 16×16 against an >80%-on-4-heads pass criterion that resolution cannot support.**

**Guidance for CLEVR execution once the gate is passed** (recorded here so it is not re-litigated):
- **Noiseless simulation is the primary experiment** — this is where the architectural claim lives. `default.tensor(method='tn')` + SPSA at 32×32.
- **A matched classical CP-TTN on identical CLEVR data is a mandatory control, not a substitute.** 80% on 4 heads means nothing in isolation: unimpressive if a small classical TTN gets 98%, genuinely interesting if it gets 82%.
- **No noisy emulation on CLEVR at all** (revised 2026-07-28). The noise track closes with Task R8 on synthetic shapes; CLEVR is run noiseless. This supersedes the earlier plan for a reduced-qubit CLEVR noise proxy.
- **Re-open Question C.2 as part of the relational task.** Whether explicit positional encoding helps *relational* reasoning is the one place the spatial ancilla could earn its keep, and the current "do not use" rule was decided on single-object classification, which structurally cannot test it.

**Total Phase 1.5 estimate**: 4–6 days (R1 0.5–1, R2 0.5, R3 1–1.5, R4 1–2, R5 0.5, R6 1–2).

---

## 8. Phase 1.6 — Purge, Regenerate & Re-examine (added 2026-07-28)

**Why this section exists.** R1b, R2 and R4 established the architecture of record and, in doing so, confirmed that a large fraction of this project's figures and stated conclusions were produced by a model that is not the one going forward. Rather than patch case by case, this phase does a single controlled pass: freeze the current state, triage every artifact, consolidate the code, regenerate what still matters, and re-examine the assumptions that survived unexamined.

### 8.0 Governing principle: supersede and archive, do NOT erase

This is a deliberate departure from a literal "purge", and it is not negotiable for thesis integrity:

* **The research log is a lab notebook.** Rewriting history in it destroys the property that makes it credible. If an examiner asks "how do you know the current numbers are right?", the strongest answer is "here is the audit that found the error, here is exactly what it invalidated, here is the re-run." Delete that and you are left with unexplained numbers that silently changed.
* **Some superseded results are themselves findings.** The mixed-channel retraction (2026-07-26) is a real methodological lesson about single-seed noise sweeps, and it only exists because the flawed run is on record.
* **Superseded scripts are the reproduction record** for logged results. They move to `deprecated/` with a status README; they are not deleted.

What *is* purged: the **authority** of bad results. Every superseded artifact gets an explicit status marker so it cannot be silently reused. Nothing is quietly removed and nothing is quietly kept.

### 8.1 Gate

~~**Do not start Phase B until R3 completes.**~~ R3 completed 2026-07-28.

**REVISED GATE (2026-07-28): do not start Phase B until Task R7 completes.** Code Audit #2 found that R1–R4 all ran on a measure-and-re-encode hybrid rather than a coherent quantum tree. Regenerating figures from a model that is about to be replaced would be wasted work, so the figure numbers must come from the coherent model (R7), not from R3-on-the-hybrid. The **triage in Phase B is unaffected** — which figures to keep, regenerate or retire does not change; only the source of the regenerated numbers does. Phase A is complete (tag `phase16-pre-purge`).

---

### Phase A — Freeze (30 min)

Commit the current working tree and tag it **`phase16-pre-purge`**. This makes every subsequent move reversible and gives the audit a fixed reference point to diff against. Do this **before** anything is moved or deleted.

(Tag named for the purge, not the remediation: the R1–R4 remediation work is already on disk and part of what gets frozen. `git diff phase16-pre-purge` then shows exactly what Phase C/D changed.)

**Done 2026-07-28** — commit `chore(phase1.6): freeze pre-purge state`, tag `phase16-pre-purge`.

---

### Phase B — Artifact triage (half day)

Every figure gets exactly one disposition. Rationale for each is in `research_log.md` 2026-07-28 (figure audit).

**Note (verified 2026-07-28): the PennyLane `default.mixed` broadcasting bug does NOT affect any existing figure.** Both scripts that could have hit it already evaluate noise sample-by-sample with a comment naming the bug (`benchmark_encodings_ansatze.py:253`, `compare_topologies.py:164`). That knowledge existed on 2026-07-17 but never reached the log and never propagated to the July-26 scripts — which is itself the strongest argument for Phase C's consolidation.

#### KEEP — no classifier head involved, data valid as-is
| Fig | File | Note |
|---|---|---|
| 1 | `fidelity_distributions.png` | 4-qubit node, random rotations. |
| 6 | `barren_plateau_scaling.png` | **Caption fix only** — "empirical BP immunity proof" overstates 4 points at N≤20 against an asymptotic strawman. Reframe as consistent with known TTN theory. |
| 7 | `topology_barren_plateaus.png` | Separate `dev_bp` path (`compare_topologies.py:190`), no head. Same caption fix. |
| 17 | `entropy_vs_tree_depth.png` | Random weights, structural entropy. |
| 18 | `entropy_propagation_vs_noise.png` | Ceiling-effect confound already documented; keep that caveat. |

#### KEEP + CAVEAT — usable once annotated
| Fig | File | Required annotation |
|---|---|---|
| 2 | `training_metrics.png` | 4-dim readout, so not bottlenecked. Valid as "gradients flow, it converges"; **not** valid for accuracy claims (256/15 protocol superseded). |
| 3 | `noise_tolerance_curve.png` | Reconcile $p_{crit}$ definition against Fig 4 (R5.1) **before** citing either. |
| 4 | `encoding_ansatz_sweep.png` | One seed per config across 12 configs. R2 supersedes the conclusion: encoding resolved, ansatz choice not. |
| 5 | `representation_bias_results.png` | Data fine; the "proves the QTTN learns spatial representations" claim needs the R4 classical control alongside it. |

#### REGENERATE
| Fig | File | Produced by |
|---|---|---|
| 9–12, 15, 16 | `quantum_residual_*`, `ancilla_residual_*`, `spatial_ancilla_*` | R3 (already running) |
| 13, 14 | `mixed_channel_extended_seeds_*` | Superseded by R3's 58-seed arms; the *methodological* lesson survives in text and needs no figure. |
| 19 | `noise_scheduling_training.png` | Re-run on the architecture of record — Question F's premise is suspect (see 8.5). |

#### RETIRE — do not regenerate
| Fig | File | Why |
|---|---|---|
| 8 | `topology_noise_resilience.png` | The QTTN-vs-MERA decision rests on **contraction complexity** (Section 5), a scaling argument independent of accuracy. Regenerating a comparison whose QTTN arm scored 42.2% — below the 50% single-attribute ceiling — to re-derive a conclusion already held on other grounds is wasted effort. Retire the accuracy claim; keep the complexity argument. |

---

### Phase C — Code consolidation (1 day)

1. **Single model path.** Everything routes through `qttn_core.HierarchicalQTTNClassifier` and `phase15_common`. No experiment script may define its own model class — that drift is precisely what let the scalar-readout regression and the un-propagated PennyLane workaround both go unnoticed across four files.
2. **Move to `deprecated/`** (with a README naming what each produced and what supersedes it): `investigate_quantum_residuals.py`, `investigate_ancilla_residuals.py`, `investigate_mixed_channel_seeds.py`, `investigate_spatial_ancilla.py`, `hybrid_trainer.py`. Retain — do not delete — since they reproduce logged results.
3. **Regression tests** (`test_qttn_core.py`), each targeting a failure this project actually hit:
   * **p→0 identity**: depolarizing at p=0 is exactly the identity, so noisy must equal clean to ~1e-15. Catches the `default.mixed` broadcasting bug class in any framework.
   * **Readout width**: assert the classifier's readout dimension matches the configured `readout`, so a `Linear(1,4)` bottleneck cannot reappear silently.
   * **Chance-level guard**: fail loudly when any arm lands within a few points of 25%, rather than reporting it as a result (this is what would have caught R4's dead classical baseline immediately).
   * **Batched-vs-rowwise agreement** on the noisy path.

---

### ✅ Phase D — Regenerate figures — DONE 2026-07-29

> `make_figures.py` produces all regenerated figures from current results, with consistent styling and provenance stated in every caption. Three figures written to `results/figures/`:
> * **`model_comparison.png`** — the headline. Accuracy and accuracy-per-parameter for all six models, with chance, the single-attribute ceiling, and the MLP reference marked.
> * **`ablations_and_noise.png`** — R3's node-level ablations and noise sweeps, captioned explicitly as **hybrid** data, since `mixed_channel` and the ancilla were deliberately not ported to the coherent tree.
> * **`readout_bottleneck.png`** — the R7 readout finding (34.1 / 35.6 / 78.4), with the explanation that the gap is not "more numbers" but a one-qubit marginal.
>
> The MLP reference line is enforced in code, not left to discipline. Figures 6/7's caption rescoping and the $p_{crit}$ reconciliation remain part of R5 (they are edits to retained figures, not regenerations).

A single `make_figures.py` producing every thesis figure from the current architecture with consistent seeds, protocol and styling.

**Non-negotiable: every accuracy figure carries the MLP reference line.** That control is what converts "our model scores X%" into an interpretable statement, and its absence is exactly what let R4's classical baseline sit at chance level unnoticed. Any figure comparing architectures without a classical reference is not publishable in this project.

---

### Phase E — Document reconciliation (half day)

Status markers on every log entry and figure reference; the figure-status table; and these framing corrections:
* Barren-plateau claim rescoped (Figs 6, 7).
* $p_{crit}$ definitions reconciled (Figs 3, 4).
* Question F's premise re-examined (8.5).
* R4's result stated correctly: **at matched parameter count the quantum tower is competitive with a classical MLP** (79.0 ± 8.7 vs 70.5 ± 18.1, difference inside the 8.9-pt limit); it loses only when the MLP is given ~5× the parameters (96.0 ± 1.5, +17.0 pts, resolved). The defensible thesis claim is **parameter efficiency**, not raw accuracy.
* **Question A.3 remains open and is NOT answered by R4** — `CPQuadRankLayer` scored 33.9% against the MLP's 96.0% (+62.1 pts, limit 5.5), so the CP node, not classical computation, is what underperforms. The quantum-vs-CP gap is a baseline artifact and must not be cited as quantum advantage.

---

### 8.5 Assumptions to re-examine

Carried unexamined until now; each needs an explicit position in the thesis.

1. **The purely-quantum scope rule is in tension with the project's own data.** R2 measured that widening the classical patch encoder from `Linear(48,1)` to `Linear(48,3)` bought **+12.6 pts** — a larger effect than any quantum architectural change measured anywhere in this investigation. Meanwhile Question B.3 was closed *on principle* because a `Linear(16,4)` constituted "classical capacity substituting for quantum work". The classical encoder is demonstrably doing more measurable work than the quantum topology. This needs a stated, defended position — either a quantitative boundary for what counts as minimal I/O (R5.4), or an acknowledgement that encoder width is a genuine architectural variable — not a rule inherited without scrutiny.
2. **Question F's founding observation** (eval-time noise improving accuracy) most likely came from a `Linear(1,4)` readout sliding samples across a 1-D decision partition as expectation values contract toward zero. Re-test on the architecture of record; if the effect vanishes, close Question F outright rather than leaving it "open but deprioritized".
3. **Noise emulation at 4–8 qubits is assumed to generalize to the 16-qubit model.** Never tested, and untestable directly — `default.mixed` is $O(4^N)$ and a genuine 16-qubit noisy circuit needs ~68GB (confirmed OOM, 2026-07-27). State it as a standing assumption in the noise chapter rather than leaving it implicit.
4. **16×16 as the architectural ceiling** — resolved or not by R6/SPSA; gates the CLEVR resolution decision.
5. **Benchmark discriminability is adequate** (revised 2026-07-28): an earlier concern that synthetic shapes might be saturated does **not** hold up — the MLP reaches 96.0% while the quantum tower sits at 79.0%, leaving ample headroom. The real constraint is per-run variance (±8.7), which the 58-seed protocol already accommodates. Synthetic shapes remains usable; no benchmark change needed.

**Total Phase 1.6 estimate**: ~3–4 days after R3 lands.
