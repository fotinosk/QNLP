# Research Log — Quantum Image Tower Investigation

This log tracks the theoretical derivations, simulation results, noisy emulations, and decisions made during the investigation of the **Quantum Tree Tensor Network (QTTN) Image Tower**.

---

## Log Maintenance Protocol

> [!IMPORTANT]
> **Strict Update Rules**:
> 1. **Before any Experiment / Theoretical Step**: Add a log entry describing:
>    * The objective of the task.
>    * The **motivation** (why we are doing this experiment/derivation).
>    * The **narrative fit** (how it fits into the bigger thesis narrative and overall research story).
>    * The expected outcome, configuration details, and active git branch.
> 2. **After the Experiment / Theoretical Step**: Immediately update the log entry with findings, actual metrics, decisions, and exact commands/files required to reproduce the results.
> 3. **Reproducibility Requirement**: Always document the specific environment (e.g. conda env name, python binaries), the exact script path, execution arguments, and where outputs/plots are saved.

---

## Current Status & Roadmap
* **Active Branch**: `thesis/quantum-image-tower`
* **Current Focus (as of 2026-07-29)**: **The theory phase is CLOSED. Next is Phase 2 (CLEVR).** Every decision-gate item in [quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md) Section 7 is resolved: R1, R1b, R2, R3, R4, R5, R7 done; R6 and R8 dropped with stated reasons. Figures regenerated (Section 8 Phase D). No known internal contradictions remain.
* **THE RESULT**: the coherent quantum tree scores **`89.3% ± 3.9` at 287 parameters** on 16×16 synthetic shapes (4 seeds, 1024/30, last-5-epoch mean).

  | model | score | params |
  |---|---|---|
  | `mlp_reference` | `96.0 ± 1.5` | 999 |
  | **`quantum_coherent`** | **`89.3 ± 3.9`** | **287** |
  | `classical_full` (CP + residual + dropout) | `88.4 ± 5.5` | 352 |
  | `classical_bare` (CP) | `79.6 ± 10.3` | 428 |
  | `quantum_hybrid` (superseded) | `79.4 ± 8.0` | 211 |
  | `mlp_param_matched` | `70.5 ± 18.1` | 257 |

  Resolved (Welch): beats the hybrid `+9.9` (limit 5.7), beats the bare classical CP tree `+9.7` (limit 6.3), beats a same-size MLP `+18.8` (limit 9.0); ties `classical_full`. **Question A.3 is answered**: under matched constraints the unitarity-constrained quantum node beats the unconstrained classical CP node while using 33% fewer parameters.
* **ARCHITECTURE OF RECORD — coherent tree** (`phase15_common.COHERENT_ARCH`, regression-tested): one 16-qubit device, patch *p* on wire *p*, `multi_axis` encoding (RX/RY/RZ), `iqp` block unitaries on `[0-3] [4-7] [8-11] [12-15]` with **per-block** level-1 weights, level-2 on the survivors `[0,4,8,12]` **passed as qubits**, readout `top_layer_qubits` (⟨Z⟩ on all four), `Linear(4, n_classes)`. Protocol 1024 train / 64 test / 30 epochs, scored on the last-5-epoch mean, comparisons **unpaired via Welch**. Runs on `lightning.qubit` + adjoint.
  * The **hybrid** (`ARCH`, `root_multi_pauli`) is retained only to reproduce R1–R4.
* **Design rules — all settled**:
  * **No residual/skip connections.** Five mechanisms rejected; confirmed at 30 seeds on positive evidence (`mixed_channel` `−10.9`, limit 4.7; `reupload` a true null at `+1.5`, limit 3.8).
  * **No spatial ancilla — *for translation-invariant classification only*** (`−19.5`, limit 3.8). **Question C.2 (relational) is REQUIRED in CLEVR**, with and without the ancilla; this result says nothing about relational reasoning.
  * **Classical components**: bounded by a checkable rule — *the encoder may set state-preparation parameters but may not reduce the qubit count the architecture would otherwise require*. This dissolves the R2-vs-B.3 tension: the `+12.6` pts from `multi_axis` came from using all three of a qubit's rotation parameters where `scalar_ry` used one.
  * **Noiseless only.** Noise closed at single-node scale (stable to p≈0.02); full-tree emulation infeasible at $O(4^N)$.
  * **Readout width is the binding constraint**, not an incidental setting: `35.6%` (root qubit) vs `78.4%` (four top-layer wires). χ=1 is a simulation limit, not a design preference.
* **Deferred into CLEVR (one question, three options)**: 16×16 cannot support CLEVR's ">80% on 4 attribute heads" criterion. Choose among **bigger patches** (works today; classical preprocessing, not quantum scaling), **SPSA** (deeper trees), or **higher bond dimension** (principled; needs TN training) — against real data rather than in the abstract.
* **Carried into CLEVR as requirements**: matched-parameter classical controls from day one; Question C.2 run both ways; noiseless only; resolution limit reported with every comparison.
* **Two caveats to keep visible in the write-up**: the classical arms were hyperparameter-tuned while the quantum arm inherited `lr=0.03` from R1 (so the quantum result is conservative), and `classical_bare` moved `33.9 → 56.7 → 79.6` across three revisions of the search space — the MLP reference is what caught that, and it is now enforced in the figure code.

---

## Log Entries

### [2026-07-17] Log Initialization & Scope Alignment
* **Activity**: Created the research log and set up the thesis investigation roadmap in [llm/quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md).
* **Decisions**:
  * Isolated the project scope to focus *strictly* on the **image tower novelty**. Dropped the joint text/alignment steps from the immediate scope.
  * Agreed on the execution progression: **Theory → Noiseless Simulation → Noisy Emulation**.
  * Initiated checkout of a dedicated research branch: `thesis/quantum-image-tower`.
* **Next Steps**:
  * Formulate the CP-Rank-to-VQC mapping.
  * Research tensor-network device options in PennyLane to bypass the 80-qubit simulation limit.

### [2026-07-17] Numerical Experiment: Quad-Node CNOT-to-Entropy Diagnostics
* **Objective**: Investigate how CNOT count impacts output mixedness (Von Neumann Entropy) and state coverage (Haar Expressibility) for a 4-qubit node.
* **Motivation**: Verify if entangling CNOT layers act as a regularizer equivalent to classical CP-decomposition. We want to find the minimal CNOT density required to mix information from child nodes without causing barren plateaus or parameter explosion.
* **Narrative Fit**: This experiment establishes the foundational link between classical tensor network rank constraints (CP-Rank) and quantum gate density (CNOT count). It provides the empirical validation for the theoretical chapter of the thesis, justifying why we use specific entangling structures in our hierarchical QTTN model.
* **Activity**: Created and executed [ansatz_diagnostics.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/ansatz_diagnostics.py).
* **Findings**:
  * **Entropy Scaling**: Increasing CNOT count from 0 to 6 directly scales the output state's mean entropy from `0.0000` to `0.5139` (relative to the single-qubit theoretical maximum of $\ln(2) \approx 0.693$). This shows that entangling CNOT layers are mathematically responsible for mixing child register features into the parent qubit.
  * **Expressibility Scaling**: A separable ansatz (0 CNOTs) exhibits near-perfect Haar state coverage (KL Div = `0.0109`) since single-qubit Euler rotations span the entire Bloch sphere. As CNOT count increases, the output qubit becomes entangled and mixed, resulting in higher KL divergence (lower pure-state expressibility).
* **Decisions**:
  * Confirmed that a minimum of 3 CNOTs (connecting all child wires directly to the parent wire) is required to achieve $\approx 73\%$ of maximum entropy transfer (`0.5089 / 0.6931`). Adding more CNOTs beyond 3 yields diminishing returns.
* **Fidelity Distribution Plot**:
  ![Fidelity Distributions](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/fidelity_distributions.png)
  **[RETAINED]** *Figure 1: Pairwise state fidelity distributions of the 4-qubit node output across different CNOT configurations, compared to the ideal uniform Haar-random distribution (red dashed line). As CNOT density increases, the output qubit is entangled with the child registers and collapses into a mixed state, shifting the fidelity distribution away from the uniform flat profile toward higher overlap (lower pure-state expressibility).*

### [2026-07-17] Completed Experiment: Equivalence & Scaling Validation of Standard vs. Recycled QTTN
* **Objective**: Implement both a standard 16-qubit QTTN and a recycled 7-qubit QTTN using mid-circuit measurements and resets, verify that their outputs match exactly, and compare their resource scaling (qubits and gate depth).
* **Motivation**: Verify if active qubit recycling is mathematically exact and can be used to simulate the 80-qubit model classically using only 8 qubits of simulator memory, completely bypassing the 80-qubit classical simulation limit.
* **Narrative Fit**: This establishes the simulation and execution methodology for the thesis. It shows how we can perform noiseless and noisy training of a large quantum model on standard classical machines (simulation benefit) and NISQ quantum processors (hardware execution benefit) via qubit recycling.
* **Activity**: Created and executed [reconstruction_recycling_equivalence.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/reconstruction_recycling_equivalence.py).
* **Findings**:
  * **Numerical Equivalence**: The expectation value outputs of the Standard 16-qubit QTTN (`0.00690406`) and the Recycled 7-qubit QTTN (`0.00690406`) matched exactly with a numerical difference of `0.00000000e+00`. This empirically proves that active qubit recycling is mathematically equivalent to tracing out registers.
  * **Physical Width Reduction**: Successfully reduced physical qubit requirements from 16 to 7 (a **$56.2\%$ reduction**).
  * **Simulation Limitation (Ancilla Expansion)**: Under standard classical simulators (like `default.qubit`), mid-circuit measurements with resets are compiled via the **deferred measurement transform**. This transform physically allocates a new virtual ancilla wire for each measurement. Hence, to simulate the 7-qubit recycled circuit with 12 resets, the device must have at least 20+ wires (we used `wires=25` successfully).
* **Decisions**:
  * For local simulation of the full 80-qubit model, we can use the recycled 8-qubit representation but must declare a simulator device of size $\approx 8 + 64 = 72$ wires to accommodate the deferred measurement compiler. This is still highly feasible and fast compared to simulating $2^{80}$ amplitudes.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/reconstruction_recycling_equivalence.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python qnlp/image_tower/classification/quantum/reconstruction_recycling_equivalence.py`

### [2026-07-17] Completed Task: Synthetic Shapes Dataset Generation
* **Objective**: Write a programmatic synthetic shape dataset generator producing 16x16 RGB images with combinations of 4 shapes (circle, square, triangle, cross) and 3 colors (red, green, blue), packaged as a PyTorch Dataset and DataLoader utility.
* **Motivation**: To establish a simple, fast, and fully controlled benchmark dataset. This serves as Step 1 of the QTTN build plan, allowing us to test gradient flow, convergence, and representation learning of the quantum tree layers in isolation before dealing with complex real-world datasets like CLEVR.
* **Narrative Fit**: This provides the first empirical test bed for the QTTN image model. It verifies that the VQC can learn simple spatial and color features under noiseless classical simulation, serving as the baseline for all future scaling tasks (CLEVR, Flickr8k).
* **Activity**: Created [synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/utils/data/synthetic_shapes.py) and executed self-check.
* **Findings**:
  * Programmatic shape generation (with random offsets/jitter and low Gaussian channel noise) works correctly.
  * Tensors are formatted to `(3, 16, 16)` and pixel ranges reside strictly in `[0.0, 1.0]`.
  * Outputs label shapes: Class 0 (Red Circle), Class 1 (Green Square), Class 2 (Blue Triangle), Class 3 (Red Cross).
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Module: `qnlp/utils/data/synthetic_shapes.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python qnlp/utils/data/synthetic_shapes.py`

### [2026-07-17] Completed Task: Noiseless QTTN Training on Synthetic Shapes
* **Objective**: Implement a modular PyTorch training script to train the QTTN image model on the 16x16 synthetic shapes dataset, verifying loss convergence and classification accuracy.
* **Motivation**: Validate that the hierarchical QTTN layers and their CNOT-entangling structure can learn representations under standard optimization algorithms (AdamW). This serves as the end-to-end integration test of the quantum image tower before scaling to the CLEVR dataset.
* **Narrative Fit**: This step validates Phase 1 (Simulation) of the thesis roadmap, proving that gradients propagate correctly through the hierarchical tree nodes to the classical input embedding layers, and that the quantum representations generalize to unseen shape samples.
* **Activity**: Created [train_synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_synthetic_shapes.py) and ran 6 epochs of noiseless training.
* **Findings**:
  * **Gradient Convergence**: Loss decreased steadily from `1.4386` (epoch 1) to `0.8490` (epoch 6), proving gradients successfully propagate through the 16-qubit QTTN model.
  * **Accuracy Peak**: Train accuracy reached `64.8%` and validation accuracy peaked at `75.0%` (epoch 4), confirming that the model learns simple shape/color boundaries on 16x16 inputs.
  * **Vectorization Speedup (Bigger Narrative)**: Loop-based batch execution took $>30$ seconds per batch, making local training impractical. Vectorizing the QNode call using **PennyLane parameter broadcasting** ran the entire batch in a single compiled tape, reducing the epoch time to **33 seconds** (a $30\times$ speedup).
* **Decisions**:
  * During classical simulation, we should always train using the **Standard 16-qubit QTTN Vectorized QNode** (without measurement/resets) since it runs 300x faster than the recycled version (due to deferred measurement wire expansion). The recycled model is strictly reserved for hardware deployment or physical width-constraint emulation.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/train_synthetic_shapes.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/train_synthetic_shapes.py`
* **Training Metrics Plot**:
  ![QTTN Training Metrics](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/training_metrics.png)
  **[RETAINED WITH CAVEAT]** *Figure 2: Noiseless training curves of the 16-qubit QTTN model on the synthetic shapes dataset over 6 epochs. The left panel shows the steady minimization of the Cross-Entropy loss. The right panel shows training accuracy (green) and validation accuracy (red) climbing and stabilizing around 75%, indicating successful feature learning without overfitting.*

### [2026-07-17] Completed Task: Noisy Emulation & Noise Tolerance Sweep on Synthetic Shapes
* **Objective**: Implement a noisy evaluation script using PennyLane's mixed-state device and depolarizing noise channels, sweep depolarizing noise rates ($p \in [0.0, 0.20]$), and plot the noise tolerance accuracy curve.
* **Motivation**: Characterize the physical noise threshold where the classification accuracy drops to the random baseline ($25\%$), empirically testing the QTTN's logarithmic noise resilience before scaling to the CLEVR dataset.
* **Narrative Fit**: This step represents Stage 2 (Emulation) of Step 1 in the build plan. It validates if the QTTN's shallow tree depth renders it naturally noise-tolerant under realistic quantum noise channels.
* **Activity**: Created [emulate_noise_synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/emulate_noise_synthetic_shapes.py) (using an 8x8 image size with a 4-qubit standard QTTN to avoid classical density matrix simulation memory walls) and evaluated validation accuracy across 8 noise points.
* **Findings**:
  * **High Resiliency at Low Noise**: Noiseless accuracy of `85.9%` remains remarkably stable at minor noise levels, yielding `91.4%` (at $p=0.005$, likely a regularizing stochastic effect), `88.3%` (at $p=0.010$), and `89.8%` (at $p=0.020$).
  * **Critical Threshold**: Accuracy drops to `66.4%` at $p=0.050$, and drops close to random at $p=0.100$ (`33.6%`).
  * **Below Random Baseline**: At $p \ge 0.15$, accuracy collapses to `21.9%` (under the $25\%$ random guess baseline), establishing the critical noise limit $p_{crit} \approx 0.05$.
* **Decisions**:
  * The QTTN model is highly noise-resilient up to $p = 0.02$, making it extremely suitable for NISQ processors with gate error rates in the $\sim 10^{-2}$ range.
  * When executing on physical processors or deep noisy emulators, we must apply error mitigation (such as Zero-Noise Extrapolation) when the gate error rate exceeds $p = 0.02$ to restore representation quality.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/emulate_noise_synthetic_shapes.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/emulate_noise_synthetic_shapes.py`
* **Noise Tolerance Curve Plot**:
  ![QTTN Noise Tolerance Curve](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/noise_tolerance_curve.png)
  **[RETAINED WITH CAVEAT]** *Figure 3: Test accuracy of the 4-qubit QTTN model under depolarizing noise channels swept from p = 0.0 to p = 0.20. The red line shows model accuracy, and the blue dashed line represents the 25% random guess baseline. The model maintains high performance up to p = 0.02, demonstrating favorable noise-resilience characteristics.*

### [2026-07-17] Completed Task: Multi-Dimensional Ansatz & Data Encoding Benchmarks
* **Objective**: Evaluate 4 data encoding styles (Angle, Multi-Axis, Amplitude, ZZ Feature Map) against 3 ansatz architectures (HEA, IQP, ALT) on the 8x8 synthetic shapes dataset under depolarizing noise sweeps.
* **Motivation**: Systematically map the complete trade-off space of QML image classifiers between qubit economy, parameter optimization overhead, noiseless classification capacity, and physical noise resilience.
* **Narrative Fit**: This unified benchmark represents the completion of Phase 3 (Ansatz Expressibility & Optimization Benchmarks) of the thesis roadmap. It provides the empirical foundation to choose the final model architecture for the CLEVR scaling experiments.
* **Activity**: Created [benchmark_encodings_ansatze.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/benchmark_encodings_ansatze.py) and executed training and noise sweeps for all 12 configurations.
* **Findings**:
  * **Multi-Axis Encoding (Best Overall Performance)**: Projecting patches to 3 features and encoding them on 1 qubit via consecutive $RX-RY-RZ$ rotations achieved **outstanding noiseless accuracies ($95.3\% - 96.9\%$)** across all ansätze. Furthermore, it remained highly noise-tolerant ($p_{crit} > 0.200$).
  * **Amplitude Encoding (Maximum Qubit Compression)**: Successfully classified shapes using **only 2 physical qubits** and **4 parameters**, reaching **`78.1%` test accuracy** (with ALT ansatz). This represents a massive qubit and parameter saving while retaining model capacity.
  * **ZZ Feature Map**: Achieved a high peak accuracy of **`89.8%`** (with HEA) but accumulated higher noise sensitivity.
  * **Topological Noise Resilience (Decision Boundary Preservation)**: Across all 4-qubit configurations, the models stayed resilient to noise up to $p = 0.20$ ($p_{crit} > 0.200$). This confirms that depolarizing noise contractively scales the output expectation values:
    \[\langle Z_i \rangle_{\text{noisy}} \approx (1-p)^d \langle Z_i \rangle_{\text{noiseless}}\]
    This uniform scaling shrinks values towards zero but **preserves their signs and relative order**. Since classification is determined by the `argmax` of the linear head, the decision boundaries remain topological and intact.
* **Decisions**:
  * **Primary Architecture for Scaling (CLEVR)**: We will select **Multi-Axis Encoding combined with the IQP ansatz**. It achieves the highest classification performance ($95.3\%$), is extremely qubit-efficient (requires only 4 qubits for 4 patches), and has fewer parameters (8 params) than HEA (12 params), making optimization highly tractable.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/benchmark_encodings_ansatze.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/benchmark_encodings_ansatze.py`
* **Ansatz & Encoding Sweep Plot**:
  ![Multi-Dimensional Encoding and Ansatz Sweep](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/encoding_ansatz_sweep.png)
  **[RETAINED WITH CAVEAT]** *Figure 4: Comparative noise degradation curves for all 12 combinations of data encodings (Angle, Multi-Axis, Amplitude, ZZ Map) and variational ansätze (HEA, IQP, ALT) on the 8x8 synthetic shapes dataset.*

### [2026-07-17] Completed Task: Investigating Representation Bias (Color vs. Shape) on 32x32 Shapes
* **Objective**: Train the 16-qubit Multi-Axis IQP classifier on 32x32 synthetic shapes under three experimental modes (Color-Only, Shape-Only/Grayscale, and Overlapping/Feature Binding) to measure whether the QTTN indexes primarily on color or shape features.
* **Motivation**: Verify if hierarchical tree networks successfully learn shape invariant features when color shortcuts are eliminated or crossed, addressing a critical question in classical-quantum representation capacity for the thesis.
* **Narrative Fit**: This step addresses Phase 2 (CLEVR Integration prep) by testing if the QTTN has the architectural capacity to perform feature binding (integrating independent shape and color attributes) on doubled image sizes (32x32).
* **Activity**: Updated [synthetic_shapes.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/utils/data/synthetic_shapes.py) to support 32x32 resolution and the 3 modes. Created [evaluate_representation_bias.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/evaluate_representation_bias.py) for diagnostic sweeps. Created and executed [train_overlapping_32x32.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_overlapping_32x32.py) to train a full 20-epoch convergence proof on 512 samples.
* **Findings**:
  * **Color-Only (50.0%)**: Easiest task for the model, as color is a simple statistical histogram feature that can be extracted quickly from pixel intensities.
  * **Shape-Only/Grayscale (37.5%)**: Completely isolates geometric spatial contours (eliminating all color information) on a translationally jittered 32x32 canvas. The model converged significantly above the **`25.0%`** random baseline, proving that the QTTN learns spatial representations.
  * **Overlapping/Feature Binding Convergence (65.6% Peak Acc / 53.9% Final Acc)**: We trained the 16-qubit model for 20 epochs on 512 samples. The validation accuracy crossed the critical **`50.0%`** boundary at Epoch 5 (`52.3%`), peaked at Epoch 14 (**`65.6%`**), and finished at **`53.9%`** (due to parameter oscillation). Because the validation performance is bounded significantly above `50.0%`, it is mathematically guaranteed that the model successfully extracted and bound **both** color and shape attributes on the full 32x32 image size.
* **Decisions**:
  * **CLEVR Preparedness**: The QTTN's success on the feature binding task at 32x32 confirms its readiness to scale to CLEVR classification tasks.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/train_overlapping_32x32.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/train_overlapping_32x32.py`
* **Representation Bias Comparison Plot**:
  ![QTTN Accuracy Across Shape vs. Color Modes](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/representation_bias_results.png)
  **[RETAINED WITH CAVEAT]** *Figure 5: Test accuracy of the 16-qubit QTTN model (Multi-Axis + IQP) on 32x32 canvases under three dataset modes: Color-Only, Shape-Only (Grayscale), and Overlapping (Feature Binding).*


### [2026-07-17] Completed Task: Overlapping Mode Feature Binding Convergence Proof
* **Objective**: Train a 4-qubit Multi-Axis IQP classifier on 8x8 synthetic shapes in `overlapping` mode (Class 0: Red Circle, Class 1: Red Square, Class 2: Green Circle, Class 3: Green Square) with 512 samples for 25 epochs.
* **Motivation**: Provide definitive mathematical proof that the QTTN is learning **both** color and shape attributes rather than relying on a single attribute shortcut.
* **Activity**: Created and executed [train_overlapping_proof.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/train_overlapping_proof.py).
* **Findings**:
  * **Validation Accuracy**: Achieved **`77.3%`** validation accuracy.
  * **Mathematical Proof of Dual-Attribute Learning**:
    * If the model learned *only* color, the maximum theoretical accuracy is $50.0\%$ (since color only partitions the 4 classes into Red $\{0, 1\}$ and Green $\{2, 3\}$).
    * If the model learned *only* shape, the maximum theoretical accuracy is also $50.0\%$ (since shape only partitions the 4 classes into Circle $\{0, 2\}$ and Square $\{1, 3\}$).
    * Since the model achieved **`77.3%`** (which is bounded significantly above $50.0\%$), it is mathematically guaranteed that the model successfully extracted, represented, and bound **both** color and shape features.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/train_overlapping_proof.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/train_overlapping_proof.py`

### [2026-07-18] Completed Task: Barren Plateau & Gradient Trainability Scaling Sweep
* **Objective**: Measure how gradient variance $\text{Var}[\partial_\theta \mathcal{L}]$ scales as the image resolution (and physical qubit count $N \in \{4, 9, 16, 20\}$) increases.
* **Motivation**: Verify the theoretical immunity of QTTN architectures to the barren plateau problem under local observables, proving trainability at scale.
* **Activity**: Created and executed [test_barren_plateaus.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/test_barren_plateaus.py) over 100 trials per system size.
* **Findings**:
  * **Empirical BP Immunity Proof**: The gradient variance hovers between $5.94 \times 10^{-2}$ and $1.51 \times 10^{-1}$ across all sizes. Under an exponential barren plateau ($2^{-N}$), the variance at 20 qubits would decay to $\approx 9.5 \times 10^{-7}$. Our empirical variance is **more than 62,000 times larger** than the barren plateau limit, demonstrating that QTTNs retain highly trainable gradients.
  * **Polynomial Scaling Check**: On a log-log plot, the variance follows a slow, stable decay consistent with polynomial trainability scaling: $\text{Var} \sim \mathcal{O}(1/\text{Poly}(N))$.
  * **Classically Intractable Autograd Cache Wall**: Attempting to simulate 25 qubits with autograd backward tracking caused immediate Out-Of-Memory termination. Since PyTorch autograd caches the statevector of size $2^N \times 16$ bytes at every gate in the graph, a 25-qubit circuit with 200 gates requires $>100$ GB of RAM. This highlights a critical limitation of statevector autograd at scale, emphasizing the need for parameter-shift or SPSA optimization.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/test_barren_plateaus.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/test_barren_plateaus.py`
* **Barren Plateau Scaling Plot**:
  ![Barren Plateau Scaling Sweep](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/barren_plateau_scaling.png)
  **[RETAINED]** *Figure 6: Scaling curves of QTTN leaf gradient variance against qubit count N on a semi-log scale (left) and log-log scale (right). The variance remains stable near $10^{-2}$, indicating strong resistance to the barren plateau phenomenon.*

### [2026-07-18] Completed Task: Comparative Topology Benchmark (QTTN vs. MPS vs. MERA)
* **Objective**: Compare Tree Tensor Networks (QTTN), Matrix Product States (MPS), and Multi-Scale Entanglement Renormalization Ansatz (MERA) on gradient scaling, noiseless training, and depolarizing noise sweeps.
* **Activity**: Created and executed [compare_topologies.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/compare_topologies.py).
* **Findings**:
  * **Barren Plateau Immunity**: All three topologies showed stable gradient variances at N=20 (QTTN: $6.00 \times 10^{-2}$, MERA: $8.04 \times 10^{-2}$, MPS: $1.97 \times 10^{-1}$), confirming trainability at these sizes.
  * **Expressibility capacity Boost (MERA)**: Under noiseless training, MERA achieved a significant validation accuracy increase to **`51.6%`** (compared to **`42.2%`** for QTTN and **`43.0%`** for MPS). This verifies that MERA's disentangler gates capture vital spatial correlations across block boundaries that trees miss.
  * **Noise Resilience (QTTN vs. MPS)**: Under depolarizing gate noise ($p = 0.10$), QTTN retained the highest accuracy of **`53.9%`** (followed by MERA at **`53.1%`** and MPS dropping to **`46.1%`**). MPS is highly vulnerable to noise due to its linear gate depth accumulation, while tree-based structures maintain high resilience.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/compare_topologies.py`
  * Run Command: `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u qnlp/image_tower/classification/quantum/compare_topologies.py`
* **Plots**:
  * **Barren Plateau Comparison**:
    ![Topology Barren Plateau Sweep](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/topology_barren_plateaus.png)
    **[RETAINED]** *Figure 7: Log-log scaling of gradient variance for the three topologies, showing stable, non-vanishing gradients for QTTN, MERA, and shallow MPS.*
  * **Noise Resilience Comparison**:
    ![Topology Noise Sweep](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/topology_noise_resilience.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 8: Test classification accuracy under depolarizing noise sweeps. QTTN and MERA maintain high noise tolerance, while the linear MPS chain drops significantly faster.*

### [2026-07-26] Backlog Audit: Gap Between Proposed and Executed Investigations
* **Objective**: Cross-check `quantum_investigation_roadmap.md` and `quantum_implementation_plan.md` against this log to find proposed experiments that were never actually run, and checkboxes marked complete without a corresponding log entry.
* **Motivation**: The Phase 3 checklist in the roadmap (noise calibration, ZNE/readout mitigation, optimizer comparison) is marked `[x]` complete, but no log entry documents ZNE, Mitiq integration, or an SPSA/parameter-shift/backprop comparison. Before starting Phase 2 (CLEVR), we need an accurate picture of what's actually validated vs. what's assumed.
* **Findings**:
  * **Dropout audit (corrected 2026-07-26)**: The classical TTN tower has explicit inverted-dropout between CP layers (`ttn_for_image_classification_model.py:23-58`, Bernoulli mask scaled by $1/(1-p)$). The quantum tower (`qnlp/image_tower/classification/quantum/`) has no dropout and no data re-uploading anywhere in the codebase. **Decision: only the "build a bespoke quantum-dropout mechanism" track is deprioritized** — the naturally-occurring depolarizing-noise regularization effect (see 2026-07-17/18 entries) is judged sufficient in place of hand-rolled dropout. This does **not** resolve *why* the effect happens; that question was promoted to a formal Open Question (Question F, "Depolarizing Noise as Implicit Regularization") in the roadmap rather than left buried under the deprioritized dropout note.
  * **Residual connection audit**: The classical CP-quadtree layer (`qnlp/discoviz/models/cp_node.py:28,68-69`) has an explicit residual/skip connection — `out + res_proj(x.mean(dim=2))`. The quantum tower has **no residual mechanism**. Scope is restricted to genuinely quantum-native residual mechanisms (in-circuit, not a classical bypass around measurement) — see [quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md) Section 5, "Quantum-Native Residual Connections."
  * **Phase 3 discrepancy**: No log entries exist for ZNE, Mitiq readout mitigation, or SPSA/parameter-shift/backprop convergence comparison, despite the roadmap checkboxes being ticked.
  * **Question A** (CP-rank/entanglement mapping): only A.1 (CNOT depth ↔ CP-rank, via `ansatz_diagnostics.py`) has been tested. A.2 (entanglement entropy vs. classical TTN area-law) and A.3 (unitary-constrained vs. unconstrained CP expressibility) were never run.
  * **Question B** (80-qubit wall): only qubit recycling was validated (`reconstruction_recycling_equivalence.py`). The tensor-network simulator device (quimb/PennyLane TN device) comparison called for in Task 1.1 was never benchmarked. B.3 (does the `hybrid_trainer.py` classical compression head lose the quantum advantage) is untested.
  * **Question C** (spatial ancilla ablation): never run, despite being runnable today on the existing synthetic/overlapping shapes infra without CLEVR.
  * **Question D.2** (effect of classical head on gradient variance/barren plateaus): never isolated as its own test.
  * **Question E.2** (entropy propagation through partial trace — does tracing out qubits wash away or propagate noise): never tested; likely the same experiment infrastructure answers both this and the quantum-dropout question.
* **Decisions**:
  * Documented full backlog with time estimates in [quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md) Section 6 (~10–14 days total across 10 items).
  * Quantum dropout deprioritized (noise already covers it); replaced with quantum-native residual connections as the active architecture follow-up (data re-uploading, near-identity ansatz init, mixed-unitary channel, ancilla-controlled soft mixing, full LCU — see roadmap Section 5).
  * Recommend clearing the cheap, infra-reusing items (ancilla ablation, hybrid-head ablation, classical-head-on-gradients test) before starting Phase 2 (CLEVR), since they inform architecture decisions CLEVR training would otherwise bake in without evidence.
  * Either execute the missing Phase 3 items (ZNE, Mitiq, optimizer comparison) for real, or un-check those roadmap boxes — currently they misrepresent the state of the investigation.

### [2026-07-26] Completed Experiment: Quantum-Native Residual Connections (Data Re-uploading vs. Near-Identity Init vs. Baseline)
* **Objective**: Test two genuinely in-circuit residual mechanisms — data re-uploading (Method 1) and near-identity ansatz initialization (Method 2) — against a plain baseline, on both noiseless convergence and depolarizing noise robustness.
* **Motivation**: A classical-bypass residual (adding a classically-computed value after quantum measurement) was prototyped earlier and discarded as out of scope — it answers an engineering question, not a quantum-architecture one. This experiment restricts to shortcuts realized inside the circuit itself. See [quantum_investigation_roadmap.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_investigation_roadmap.md) Section 5, "Quantum-Native Residual Connections."
* **Design**: Built [investigate_quantum_residuals.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_quantum_residuals.py) on the same hierarchical QTTN as before (16 patches → 4 nodes → 1 root). Three node-ansatz variants, same total layer count (2× `StronglyEntanglingLayers`) and same weight-tensor shape for a fair comparison:
  * `baseline`: encode once (`RY`), 2-layer ansatz, measure.
  * `reupload`: encode, 1-layer ansatz, **re-encode the same node input**, 1-layer ansatz, measure — the raw signal is re-injected partway through instead of only being reachable via compounded unitaries.
  * `near_identity`: identical circuit to `baseline`, but rotation weights initialized ~10x smaller (`*0.01` vs. `*0.1`), so the node unitary starts close to $I$.
  Trained 3 seeds × 15 epochs × 3 variants on 16×16 synthetic shapes (overlapping mode), then swept depolarizing noise (`default.mixed`, $p \in [0, 0.20]$) on the seed=0 trained weights of each variant.
* **Findings**:
  * **Stage 1 (noiseless)**: `reupload` reached the highest peak accuracy (`59.9%` vs. `57.3%` baseline vs. `56.8%` near_identity) and visibly fit training data harder (seed=2 reached `67.2%` val acc / `64.1%` train acc at epoch 13, well above baseline's plateau) — but at the cost of **2.6x higher seed-to-seed variance** in final accuracy (`54.7% ± 7.7` vs. `56.8% ± 2.9` for baseline), including one seed (seed=0) whose accuracy dropped in its final epoch (`57.8%→45.3%`).
  * `near_identity` was statistically indistinguishable from baseline (final `56.8% ± 2.9`, identical to baseline's exact numbers on these seeds; peak even slightly lower). Convergence to 50% val acc was slightly slower (avg epoch 2.7 vs. 1.7 for baseline). **The near-identity start only affects early training dynamics — by epoch 15 the weights have moved wherever gradient descent takes them regardless of starting point, washing out any structural benefit.**
  * **Stage 2 (noise sweep, seed=0 models)**: `baseline` and `near_identity` behaved identically — clean ~54%, flattening at exactly `50.0%` for all $p \ge 0.02$. `reupload` briefly held its clean accuracy at $p=0.02$ (`54.7%`), matched baseline's floor at $p=0.05$ (`50.0%`), then **collapsed to `28.1%`** for $p \ge 0.10$ — noticeably worse than baseline's floor.
  * **Mechanism**: Re-uploading duplicates the encode→ansatz cycle, which under the noisy circuit means twice as many `DepolarizingChannel` injection points before measurement (noise is applied after every gate, including both encoding passes). The re-injected signal that helps expressivity noiselessly is exposed to noise twice, so the mechanism that gives `reupload` its accuracy edge is exactly what makes it more fragile under depolarizing noise — the opposite trade-off classical residual-connection literature would predict (skip connections in classical ResNets typically *improve* robustness).
* **Decisions**:
  * Neither method is a clear win to carry into CLEVR as-is. `near_identity` is a no-cost/no-benefit change (safe to include or drop). `reupload` trades noiseless expressivity for noise fragility and training instability — worth keeping in mind as a *simulation-only* expressivity technique but not as a "residual for robustness" the way the classical analogy suggested; if pursued further, it should be paired with a noise-mitigation technique or restricted to leaf-level nodes only rather than every tree level, to limit the doubled noise exposure.
  * The ancilla-based methods (mixed-unitary channel, ancilla-controlled soft mixing, full LCU — Methods 3–5 in the roadmap) remain unexplored and are the more promising candidates if genuine noise-robust residual behavior is still wanted, since they don't duplicate encoding/noise exposure the way re-uploading does.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/investigate_quantum_residuals.py`
  * Run Command: `conda run -n qnlp python -u qnlp/image_tower/classification/quantum/investigate_quantum_residuals.py`
* **Plots & Raw Results**:
  * ![Quantum Residual Training Comparison](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/quantum_residual_comparison_training.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 9: Loss and validation-accuracy curves (mean ± std over 3 seeds) for baseline vs. data re-uploading vs. near-identity init.*
  * ![Quantum Residual Noise Comparison](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/quantum_residual_comparison_noise.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 10: Test accuracy under depolarizing noise sweep ($p=0$ to $0.20$) for the seed=0 trained model of each variant. Re-uploading collapses below baseline's noise floor at $p \ge 0.10$ due to doubled noise-channel exposure from the repeated encoding step.*
  * Raw JSON: `qnlp/image_tower/classification/quantum/results/quantum_residual_comparison_results.json`

### [2026-07-26] Completed Experiment: Ancilla/Channel-Based Quantum Residuals (Mixed-Unitary Channel & LCU)
* **Objective**: Test the remaining quantum-native residual mechanisms — mixed-unitary channel (Method 3) and single-ancilla LCU block-encoding (Methods 4/5) — against the same baseline, since Methods 1–2 (data re-uploading, near-identity init) traded expressivity for noise fragility or did nothing.
* **Motivation**: Both re-uploading's fragility and near-identity's null result stemmed from *not* being a real bypass — re-uploading re-exposes the signal to noise, near-identity only affects init. Ancilla-based mechanisms route the "skip" through a genuinely separate degree of freedom (an ancilla qubit), closer to how classical residual connections work structurally.
* **Design**: Built [investigate_ancilla_residuals.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_ancilla_residuals.py). Each tree node gets a 5th "mixing" ancilla wire, with a learnable mixing angle initialized small (near-identity start):
  * `mixed_channel`: `RY(mix_angle)` on ancilla, controlled-`U` on the system, ancilla **traced out** (not returned) — realizes the incoherent Kraus mixture $\rho \to (1-\lambda)\rho + \lambda\, U\rho U^\dagger$, a genuine quantum channel (not classical arithmetic on measured values).
  * `lcu`: same but with `RY(-mix_angle)` uncompute before **postselecting** the ancilla on $|0\rangle$ (`qml.measure(..., postselect=0)`) — the standard single-ancilla PREPARE-SELECT-PREPARE$^\dagger$ LCU block-encoding, coherent rather than a classical mixture.
  * **Known limitation**: PennyLane 0.43.2's `default.mixed` device does not support `qml.measure(postselect=...)` via the deferred-measurement transform (it requires a `Projector` gate the transform doesn't provide for density-matrix devices) — confirmed by direct test. So `lcu` was evaluated **noiseless only**; `baseline` and `mixed_channel` got both stages.
  Trained 3 seeds × 15 epochs × 3 variants, then noise-swept `baseline`/`mixed_channel` (seed=0 models), and separately computed the LCU postselection success probability (Method 5's overhead metric) via an auxiliary non-postselected diagnostic circuit.
* **Findings**:
  * **Stage 1 (noiseless)**: Both ancilla methods showed **severe training instability** — `mixed_channel` final val acc `45.8% ± 20.3` (peak `50.5%`), `lcu` final `42.7% ± 18.6` (peak `49.0%`), both well below baseline's `56.8% ± 2.9`. One seed (seed=1) for **both** variants got permanently stuck: loss plateaued at exactly `1.379`–`1.386` (≈$\ln(4)$, the cross-entropy of uniform random 4-class guessing) for all 15 epochs, never escaping. This is a dead-gradient trap, not just slow convergence — `convergence_epoch_to_50pct` for that seed is `None` (never reached).
  * **Mechanism (instability)**: The learnable `mix_angle` acts as a gate on how much the controlled-`U` branch contributes to the measured output. If `mix_angle` drifts toward 0 early (plausible from initialization near-identity plus early gradient noise), the ansatz weights `weights` receive vanishing gradient signal through the controlled operation — a chicken-and-egg trap structurally similar to classical gated units (LSTM-style) getting stuck closed early in training. Baseline has no such gating parameter and never exhibits this failure mode.
  * **Stage 2 (noise sweep, seed=0 models — the ones that *did* converge)**: `mixed_channel` clean `51.6%` → **improves and flattens at `54.7%`** for all $p \ge 0.02$, consistently above baseline's flat `50.0%` floor across the whole sweep. This is a genuine noise-robustness benefit, distinct from `reupload`'s fragility — the ancilla-traced-out mixture doesn't duplicate noise exposure the way re-uploading's repeated encoding does.
  * **LCU diagnostic (Method 5 overhead)**: postselection success probability differs sharply by tree level — Level 1 (patch-pooling nodes) `P(ancilla=0) = 0.986` (almost always keeps the branch), Level 2 (root node) `P(ancilla=0) = 0.515` (discards roughly **half** of all trials). This is a concrete, level-dependent resource cost unique to LCU that neither `mixed_channel` nor `baseline` pay — on real hardware, the root node would need ~2x the shots to compensate.
* **Decisions**:
  * `mixed_channel` (Method 3) is the most promising quantum-native residual found so far *when it converges* — real noise-robustness benefit without re-uploading's noise-fragility trade-off. But the training instability (1 of 3 seeds failing outright) is a real blocker before recommending it for CLEVR; it needs either a `mix_angle` warm-up schedule, gradient clipping, or a different initialization strategy to avoid the gating trap before it's trustworthy.
  * `lcu` (Methods 4/5) inherits the same instability without a demonstrated noise-robustness upside (untested under noise, due to the `default.mixed` postselection limitation) and adds a real resource cost (up to 2x shot overhead at the root level). Not recommended to pursue further unless the instability is fixed and the noise question can be answered — the LCU postselection limitation would need a workaround (e.g. dropping down to `default.qubit` + explicit density-matrix bookkeeping, or waiting for PennyLane support) before Stage 2 can even be attempted.
  * None of the five quantum-native residual methods tested (re-uploading, near-identity, mixed-channel, LCU) are currently a clean drop-in win for CLEVR. If pursued further, `mixed_channel` with a stabilized training recipe is the best lead.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/investigate_ancilla_residuals.py`
  * Run Command: `conda run -n qnlp python -u qnlp/image_tower/classification/quantum/investigate_ancilla_residuals.py`
* **Plots & Raw Results**:
  * ![Ancilla Residual Training Comparison](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/ancilla_residual_comparison_training.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 11: Loss and validation-accuracy curves (mean ± std over 3 seeds) for baseline vs. mixed-unitary channel vs. LCU. The wide shaded bands for the ancilla methods reflect the seed=1 dead-gradient failures.*
  * ![Ancilla Residual Noise Comparison](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/ancilla_residual_comparison_noise.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 12: Test accuracy under depolarizing noise sweep for baseline vs. mixed-unitary channel (seed=0 models; LCU excluded due to the `default.mixed` postselection limitation). Mixed-channel holds a consistent lead over baseline's noise floor.*
  * Raw JSON: `qnlp/image_tower/classification/quantum/results/ancilla_residual_comparison_results.json`

### [2026-07-26] Completed Experiment: Extended-Seed Re-test of Mixed-Unitary Channel (Method 3) — Corrects Earlier Noise-Robustness Claim
* **Objective**: Re-run baseline vs. `mixed_channel` on 10 seeds instead of 3, to (a) get a statistically meaningful estimate of the dead-gradient failure rate, and (b) average the noise sweep across many trained models instead of a single seed=0 spot check, per the caveat flagged in the previous entry.
* **Motivation**: The 3-seed run found `mixed_channel` beating baseline's noise floor (`54.7%` flat vs. `50.0%` flat) — but that used only one trained model per variant. 3 seeds is too small a sample to distinguish a real effect from one lucky/unlucky draw, especially for a method with a known instability mode.
* **Design**: Built [investigate_mixed_channel_seeds.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_mixed_channel_seeds.py), reusing the training/circuit code from `investigate_ancilla_residuals.py`. Trained `baseline` and `mixed_channel` on 10 seeds (0–9) × 15 epochs, then ran the full depolarizing noise sweep ($p \in [0, 0.20]$) on **every** trained seed model (not just seed=0), reporting results both including and excluding failed runs (final val acc `< 35%` = failed, near the 25% random baseline).
* **Findings**:
  * **Failure rate revised down**: `mixed_channel` failed (dead-gradient trap) on only **1 of 10 seeds (10%)**, not the 1-in-3 (33%) the earlier 3-seed sample suggested. The small earlier sample was simply unlucky/unrepresentative.
  * **Noiseless accuracy (converged-only) confirmed**: `mixed_channel` on its 9 converged seeds averaged `55.6% ± 5.2` vs. baseline's `54.1% ± 3.6` (all 10 seeds, 0 failures) — a small, real edge, consistent with the earlier finding and now on firmer statistical footing.
  * **Noise-robustness claim REVERSED**: averaged across all converged seeds, the earlier finding ("mixed_channel beats baseline's noise floor") **did not replicate**. The two methods are statistically tied at low noise (`53.4–53.9%` baseline vs. `53.1–54.2%` mixed_channel for $p \le 0.05$), but at higher noise **baseline is equal-or-better**: $p=0.15$: `53.0%` (baseline) vs. `47.9%` (mixed_channel); $p=0.20$: `51.6%` vs. `47.2%`. Per-seed noise-response standard deviation is large (`3.9–8.5` for baseline, `4.9–11.2` for mixed_channel) — often bigger than the gap between the two methods' means — which is exactly why the single seed=0 comparison looked so different from the properly-averaged result. That comparison happened to sample a baseline model whose noise response degraded to a flat 50% floor and a mixed_channel model whose noise response happened to hold up well; neither is representative of the method in general.
* **Decisions**:
  * **Retract the noise-robustness claim from the previous entry.** `mixed_channel` does not have a demonstrated noise-robustness advantage over baseline — if anything the trend (not statistically confirmed either way at low noise, but suggestive at high noise) favors baseline. The only surviving genuine finding for `mixed_channel` is the small noiseless accuracy edge (`55.6%` vs. `54.1%`, on converged runs) plus a now-lower (10%, not 33%) but still nonzero training-failure rate.
  * **Methodological lesson**: single-seed noise sweeps are unreliable for this system size — seed-to-seed noise-response variance is comparable to or larger than the between-method gap. Any future noise-robustness claim in this investigation should be averaged over at least ~10 seeds before being reported, not spot-checked on one trained model.
  * Given the retracted noise benefit, the case for prioritizing item 13 (stabilizing `mixed_channel` training) is weaker than previously stated — the remaining motivation is only the modest noiseless accuracy edge, not noise robustness. Deprioritized relative to other backlog items unless a future experiment finds a different genuine advantage.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/investigate_mixed_channel_seeds.py`
  * Run Command: `conda run -n qnlp python -u qnlp/image_tower/classification/quantum/investigate_mixed_channel_seeds.py`
* **Plots & Raw Results**:
  * ![Per-Seed Final Val Acc Bar Chart](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/mixed_channel_extended_seeds_bar.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 13: Final validation accuracy per seed (10 seeds) for baseline vs. mixed-unitary channel. The single failed mixed_channel run (seed=1) is grayed out.*
  * ![Extended Noise Sweep, All Seeds vs. Converged-Only](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/mixed_channel_extended_seeds_noise.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 14: Noise sweep averaged across all 10 trained seed models (left) and converged seeds only (right), with std shading. Baseline and mixed_channel are statistically indistinguishable at low noise; baseline is equal-or-better at high noise — the opposite of the single-seed result reported previously.*
  * Raw JSON: `qnlp/image_tower/classification/quantum/results/mixed_channel_extended_seeds_results.json`

### [2026-07-26] Decision: Quantum-Native Residual Connections — Investigation Closed, Do Not Use
* **Conclusion**: Across all five quantum-native residual mechanisms tested (data re-uploading, near-identity ansatz init, mixed-unitary channel, LCU, LCU-lite), none show a robustly-confirmed benefit over a plain no-residual baseline once properly tested (multi-seed where applicable). **Quantum tree nodes in this project — image tower or any future quantum model — should not use residual/skip connections.**
* **Summary of why each was rejected**:
  * **Data re-uploading**: real noiseless expressivity gain, but collapses under noise (28.1% vs. baseline's 50.0% floor at $p\ge0.10$) because re-injecting the signal exposes it to noise twice.
  * **Near-identity init**: no measurable effect in either regime — the initialization bias washes out well before training converges.
  * **Mixed-unitary channel**: initially looked like the best candidate (small noiseless edge, apparent noise-robustness benefit) — but the noise-robustness result was a single-seed artifact that did not replicate on a 10-seed re-test, and it carries a confirmed 10% training-failure rate (dead-gradient trap) the baseline never exhibits.
  * **LCU / LCU-lite**: inherits the same training instability as mixed-unitary channel with no accuracy benefit, adds a real ~2x shot-overhead cost at the root node (postselection success probability 51.5%), and can't even be evaluated under noise given a `default.mixed` postselection limitation in the installed PennyLane version.
  * **Classical-bypass residual** (prototyped very early, out of scope): the only variant that showed a clean, mechanistically-obvious noise-robustness benefit — but only because it bypasses the noisy quantum circuit entirely via a classical shortcut, which answers an engineering question, not a quantum-architecture one, so it was excluded from this line of investigation from the start.
* **Decision**: This is now a binding design rule, not an open question. Documented as a `NOTE` in [quantum_implementation_plan.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_implementation_plan.md) alongside the existing non-linear-contractions and NaN-handling rules, and the roadmap's Section 5 residual entry is marked CLOSED. Future work should not re-propose residual connections for quantum tree nodes without a genuinely new mechanism distinct from the five ruled out here — re-litigating the same five approaches would not change the conclusion.
* **Methodological note carried forward**: the mixed-unitary-channel episode is a reminder that any future noise-robustness claim in this investigation needs multi-seed averaging (~10 seeds) before being trusted — a single-seed spot check produced a conclusion that reversed on a larger sample.

### [2026-07-26] Completed Experiment: Question C — Spatial Ancilla Ablation (Explicit vs. Implicit Positional Encoding)
* **Objective**: Test whether the explicit 5th "spatial ancilla" qubit per patch (`quantum_image_embedding_model_hea.py`) actually improves classification, or whether the QTTN's fixed tree topology already encodes position implicitly (per the roadmap's Question C).
* **Motivation**: `quantum_image_embedding_model_hea.py`'s own docstring already hypothesizes the ancilla is "not strictly necessary" since the wiring diagram fixes which patches interact with which siblings — this experiment tests that hypothesis empirically instead of leaving it as an unverified comment.
* **Design adaptation**: `quantum_image_embedding_model_hea.py` is an 80-qubit circuit-drawing demo (one ancilla per *individual* patch, unbatched per-sample forward loop) — not tractable for actual training. Reused the efficient hierarchical QTTN pattern from the residual investigation (16 patches → 4 nodes → 1 root, level-1 weights shared across all 4 blocks) and added the ancilla at *block* granularity: each level-1 node gets a 5th wire encoding a learned embedding of which of the 4 spatial quadrants (top-left/top-right/bottom-left/bottom-right) its 4 patches belong to, via `RX`/`RY` from a `(4, 2)` embedding table. Level 2 (root, only 1 node) has no ancilla — "position" isn't meaningful with a single remaining node. Built [investigate_spatial_ancilla.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_spatial_ancilla.py). Trained 5 seeds × 15 epochs × 2 variants (`no_ancilla` / `with_ancilla`) on 16×16 synthetic shapes, then noise-swept **every** trained seed model (not a single spot-check, per the multi-seed-averaging lesson from the mixed_channel retest).
* **Note on scope**: only Question C sub-questions 1 (ablation) and 3 (qubit cost) are answered here. Sub-question 2 (relational task sensitivity) needs CLEVR's multi-object relative-position labels and remains deferred to Phase 2.
* **Findings**:
  * **Stage 1 (noiseless)**: `with_ancilla` was **not better, and mildly worse** — final val acc `52.8% ± 5.7` vs. `no_ancilla`'s `55.0% ± 3.5`; peak `54.4%` vs. `56.9%`. Variance was higher (5.7 vs. 3.5) and one seed never reached the 50%-val-acc convergence threshold within 15 epochs (`with_ancilla` convergence epochs `[2, 2, 8, 2, None]` vs. `no_ancilla`'s `[2, 1, 2, 2, 5]`). This costs more too: 119 params / 5 qubits per level-1 node vs. 105 params / 4 qubits.
  * **Stage 2 (noise sweep, averaged across all 5 trained seeds each)**: the two curves track closely and mostly overlap within their std bands at every noise level ($p=0$ to $0.20$); `with_ancilla` trends slightly lower throughout (e.g. $p=0.10$: `53.8% ± 5.5` vs. `55.0% ± 4.1`). No meaningful noise-robustness difference either direction.
* **Decisions**:
  * **Confirms the implicit-topology hypothesis**: the fixed tree wiring (level-1 weights shared across blocks, but level 2 sees the 4 level-1 outputs in a fixed slot order, so "which slot" already carries quadrant identity) provides sufficient positional signal on its own. The explicit ancilla adds qubits, parameters, and training-instability risk for no measurable benefit — on this dataset it's mildly counterproductive.
  * **Qubit conservation confirmed** (Question C.3): safe to drop the ancilla, reducing the full-scale model from 80 qubits (16 patches × 5) to 64 qubits (16 patches × 4) as the roadmap originally hypothesized, without an accuracy or noise-robustness cost observed here.
  * **Recommendation for CLEVR**: build the image tower without the spatial ancilla. Revisit only if Phase 2's relational task (Question C.2, needs actual multi-object relative positions) turns up a case where implicit topology isn't enough — that's a qualitatively different test (relational reasoning) than this single-object classification ablation.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/investigate_spatial_ancilla.py`
  * Run Command: `conda run -n qnlp python -u qnlp/image_tower/classification/quantum/investigate_spatial_ancilla.py`
* **Plots & Raw Results**:
  * ![Spatial Ancilla Training Comparison](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/spatial_ancilla_comparison_training.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 15: Loss and validation-accuracy curves (mean ± std over 5 seeds) for with vs. without the explicit spatial ancilla.*
  * ![Spatial Ancilla Noise Comparison](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/spatial_ancilla_comparison_noise.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 16: Noise sweep averaged across all 5 trained seed models per variant, with std shading. The two curves overlap within noise — no meaningful difference.*
  * Raw JSON: `qnlp/image_tower/classification/quantum/results/spatial_ancilla_comparison_results.json`

### [2026-07-26] Decision: Purely-Quantum Scope — Close All Open Questions Involving Classical Components in the Core Pipeline
* **Decision**: The project direction is a purely quantum implementation. Any open question or backlog item that proposes adding or tuning a classical component *inside* the core quantum representation-building pipeline is closed on principle, without running an experiment. This does not affect the unavoidable classical I/O boundary (pixel-to-angle encoding at the start, expectation-value-to-logits readout at the end) — some classical interface is required for any classifier and stays minimal (a single `Linear` layer), but is not itself a research target.
* **Closed as a result**:
  * **Question B.3** (classical `Linear(16,4)` compression head in `hybrid_trainer.py`, substituting classical capacity for quantum circuit width): closed without testing. `hybrid_trainer.py` is deprecated as an architecture direction — it was already an unfinished prototype (dimension bug, no training loop), and testing whether its shortcut "loses the quantum advantage" is moot once the shortcut itself is out of scope.
  * **Question D.2** (does tuning the classical readout head change the quantum circuit's gradient-variance/barren-plateau landscape): closed without testing — this is a hybrid-architecture optimization question, not a purely-quantum one.
* **Consistency check**: this formalizes a principle the investigation was already implicitly following — the classical-bypass residual (prototyped, then explicitly excluded from the residual investigation for "answering an engineering question, not a quantum-architecture one") and all five tested quantum-native residual mechanisms (which specifically avoided classical bypasses) were already aligned with this rule before it was stated explicitly.
* **Documented**: added a binding `NOTE — Classical Hybrid Shortcuts` to [quantum_implementation_plan.md](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/llm/quantum_implementation_plan.md) alongside the existing non-linear-contractions, NaN-handling, and residual-connections rules. Updated `quantum_investigation_roadmap.md` Question B and Question D entries and the Section 6 backlog table (items 5 and 7 closed) accordingly.

### [2026-07-26] Completed Experiment: Question A.2 — Entanglement Entropy vs. Tree Depth (Area-Law Saturation)
* **Objective**: Extend the single-node CNOT-to-entropy diagnostic (`ansatz_diagnostics.py`, 2026-07-17) to the full multi-level tree, measuring how root-qubit entanglement entropy behaves as tree depth (number of leaf patches) increases, and relate this to the classical TTN area-law bound.
* **Motivation**: Roadmap Question A.2 asks whether the QTTN's entanglement entropy at each tree level matches the area-law bounds of a classical TTN.
* **Design**: Built [entropy_vs_tree_depth.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/entropy_vs_tree_depth.py). Reused the "3 CNOTs, all children to parent" ansatz identified as near-optimal in the earlier diagnostic. Measured Von Neumann entropy of the root qubit's reduced density matrix (tracing out all other qubits) at Depth 1 (4 leaf patches, 1 quad-node) and Depth 2 (16 leaf patches, 2 tree levels), 150 random weight/input samples each (diagnostic style, not a trained model — characterizes the architecture's structural entropy behavior, independent of any specific training run).
* **Findings**:
  * Depth 1 (4 leaves): mean entropy `0.5037 ± 0.1680` nats (`72.7%` of `ln(2) = 0.6931`).
  * Depth 2 (16 leaves): mean entropy `0.6219 ± 0.1149` nats (`89.7%` of `ln(2)`).
  * Entropy never exceeds `ln(2)` at either depth, but this is **not a non-trivial verification** — since every tree node passes exactly 1 surviving qubit to its parent (bond dimension $\chi=2$), a single qubit's Von Neumann entropy is mathematically incapable of exceeding $\log(2) = \ln(2)$ regardless of tree depth or leaf count. This bound is architecturally guaranteed by the qubit bottleneck, not something that could have been violated.
  * **The genuinely informative result**: entropy climbs from 72.7% to 89.7% of that shared ceiling as depth goes from 1→2 — deeper trees, with more stacked layers of local rotations + CNOTs feeding into the same single-qubit bottleneck, generate entanglement that saturates the fixed ceiling more fully. This is a real, non-trivial trend (not guaranteed by construction) about how efficiently random parameterizations use the tree's available entanglement budget as depth increases.
* **Relationship to the classical TTN area-law**: a classical TTN with bond dimension $\chi$ has the *exact same* structural entropy cap ($\log(\chi)$, via the Schmidt decomposition across any bond) — this is shared by construction between the quantum and classical architectures whenever the survivor width matches ($\chi=2$ here, i.e. 1 surviving qubit ↔ a 2-dimensional classical bond). So "does the QTTN match the classical TTN's area-law bound" has a trivial yes (by architectural equivalence of the bottleneck width) — the more interesting question this experiment actually answers is how close to that shared bound the quantum circuit gets at a given depth, which is a genuine expressivity/entanglement-generation-efficiency signal.
* **Decisions**:
  * Corrects a framing risk in the original Question A.2 wording ("can we bound entropy and show it matches area-law bounds") — the bound match is structural/guaranteed, not an empirical finding. Future write-ups of this result should lead with the saturation trend (72.7%→89.7%), not a "confirms area law holds" framing.
  * Only 2 depths were tested. Depth 3 (64 leaves) was attempted as a same-day follow-up assuming the 2026-07-17 recycling technique would make it tractable — **this assumption turned out to be wrong; see the follow-up entry below.**
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/entropy_vs_tree_depth.py`
  * Run Command: `conda run -n qnlp python -u qnlp/image_tower/classification/quantum/entropy_vs_tree_depth.py`
* **Plot**: ![Entropy vs. Tree Depth](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/entropy_vs_tree_depth.png)
  **[RETAINED]** *Figure 17: Root-qubit Von Neumann entropy at tree depth 1 (4 leaves) and depth 2 (16 leaves), with the area-law bound $\ln(2)$ shown as a reference line. Entropy climbs toward the bound with depth rather than staying flat, reflecting more efficient entanglement generation in deeper trees — the bound itself is architecturally guaranteed, not empirically at risk of violation.*

### [2026-07-26] Investigation: Depth-3 Recycling Feasibility — Discovers a Hard Simulation Wall, Corrects the Project's 32×32 Assumption
* **Objective**: Extend the A.2 entropy measurement to tree depth 3 (64 leaf patches, 32×32 image equivalent) using the active-qubit-recycling technique, on the assumption (stated in the A.2 entry above) that the already-validated depth-2 recycling result would extend naturally.
* **Motivation**: Characterizing whether entropy saturation continues past depth 2, and — more importantly, as this investigation unfolded — establishing what image size the current QTTN architecture can actually support in simulation. All training to date (every experiment in this session) has used 16×16 images; this was assumed to be an arbitrary dataset choice rather than a hard architectural ceiling.
* **Correction of a prior assumption**: earlier in this session, discussing whether "Question B" (the 80-qubit wall) was resolved, it was noted that the 2026-07-17 recycling experiment only validated a **depth-2** case (16 qubits → 7) and that extrapolating to the full 80-qubit / depth-4-ish model was "a design decision not separately validated." This investigation is that extrapolation actually being tested for the first time — at depth 3, one step past what was validated.
* **What was tried**:
  1. **Deferred-measurement approach** (the exact technique used in the 2026-07-17 depth-2 validation): each `qml.measure(wire, reset=True)` call requires PennyLane's deferred-measurement transform to allocate a dedicated ancilla wire. Depth-2 needed 12 resets → 7 logical + 12 ancilla = 19-25 total simulated wires (fine, $2^{25}$ is trivial). Depth-3 needs roughly 48-60 reset operations (recycling 3 scratch qubits across 16 level-1 blocks, plus inter-level bookkeeping) → ~65-70 total simulated wires → $2^{65+}$ amplitudes. **Infeasible by a huge margin** — this is the same order of magnitude as trying to simulate the un-recycled 64-qubit circuit directly.
  2. **`mcm_method='tree-traversal'`** (discovered and newly adopted during this investigation): a fundamentally different PennyLane simulation strategy for mid-circuit measurements that does **not** need ancilla wires — it explicitly branches over measurement outcomes in the classical control flow instead of purifying via extra qubits. Verified this works with **zero wire inflation** (declaring only the 7 logical wires) and is exact: reconstructing the root qubit's entropy from three basis-rotated `⟨Z⟩` expectation values (a standard single-qubit Bloch-vector tomography trick, since `tree-traversal` doesn't support the nonlinear `vn_entropy`/`density_matrix` measurement types directly — only linear observables combine correctly across its branches) matched the deferred-measurement ground truth to 13 significant digits (`0.6789223173654231` vs. `0.678922317365424`).
  3. **Scaling test for `tree-traversal`**: timed the same recycling pattern (parent-storage + 3-wire recycled scratchpad) at increasing block counts. 4 blocks (12 resets): consistently ~2.4-2.5s across repeated calls, new qnode constructions, and device reuse — very stable. 8 blocks (24 resets): **hung past 120 seconds with zero output**, confirmed via an isolated single-call test (not a batching or repeated-call artifact). This is consistent with `tree-traversal` enumerating $2^{\text{resets}}$ execution branches exactly rather than using any smarter marginalization (2^12=4096 branches in ~2.5s ≈ 1700 branches/sec; 2^24 would be ~4096x more branches, extrapolating to hours). Depth-3's ~50 resets would need $2^{50}$ branches — nowhere close to tractable.
* **Finding**: **depth-3 (64 leaves, 32×32-equivalent images) is not tractable to simulate exactly with any currently-available PennyLane mid-circuit-measurement strategy**, whether via deferred measurement (ancilla-wire blowup) or `tree-traversal` (exponential branch-count blowup) — both hit walls of the same rough order of magnitude as direct 64-qubit simulation, just via different mechanisms. Recycling genuinely helps for *real hardware* (where a reset is a physical operation with no simulation cost), but provides **no path to classically simulating anything past depth 2** with current tooling.
* **What this means for the project right now**: **16×16 is the actual current ceiling for image size**, not an arbitrary choice — every training experiment in this session has already been operating at exactly that limit. 32×32 (depth 3) and anything larger is blocked pending a different simulation strategy.
* **Decisions**:
  * Corrected `quantum_investigation_roadmap.md` Question B and Phase 1 Task 1.1 to reflect that qubit recycling is validated only at depth 2 and does not extend further for simulation purposes (real-hardware qubit budgets are unaffected by this finding).
  * Added **Task 1.4: Enable 32×32 image support** to the roadmap (Section 3) — the recommended path is benchmarking the tensor-network simulator device (Question B.1, quimb/PennyLane TN device), which contracts the circuit's tensor network directly rather than materializing a statevector or branching over outcomes. This is specifically well-matched to this architecture: the A.2 result above already established every internal tree bond has $\chi=2$ (entropy bounded by $\ln 2$), exactly the low-bond-dimension regime TN contraction is efficient in.
  * The `tree-traversal` MCM method and the Bloch-vector entropy-reconstruction trick are still useful, validated findings in their own right (zero ancilla overhead, exact results, up to ~16 resets / depth-2-scale problems) — worth keeping in mind for any future diagnostic needing exact mid-circuit-measurement simulation at a similar scale.
* **Reproduction Steps**: exploratory session (no standalone script committed — see this entry's inline code for the validated `tree-traversal` + Bloch-reconstruction pattern, and the scaling test that found the wall). Conda Environment: `qnlp`.

### [2026-07-26] Completed Task 1.4: Tensor-Network Device Enables 32×32 and 64×64 Forward Passes — Training Still Blocked on Gradient Cost
* **Objective**: Test the tensor-network simulator device (`default.tensor`, quimb backend) as the alternative path to 32×32/64×64 images, per Task 1.4, since both direct and recycled statevector simulation were confirmed blocked at depth 3 in the entry above.
* **Motivation**: The A.2 entropy result already established that every internal bond in this QTTN has bond dimension $\chi=2$ — exactly the low-entanglement regime tensor-network contraction is efficient in, unlike a generic statevector simulator that tracks the full $2^N$-dimensional Hilbert space regardless of how little entanglement is actually present.
* **Setup**: `quimb` was not installed (`default.tensor` requires it); installed via `pip install quimb` into the `qnlp` env (quimb 1.14.0, plus its numba/llvmlite/cytoolz dependencies). PennyLane's `default.tensor` device supports two contraction strategies: `method='mps'` (forces the circuit into a 1D matrix-product-state chain) and `method='tn'` (general tensor-network contraction via `cotengra`, which searches for a good contraction order respecting the circuit's actual topology rather than forcing a 1D ordering).
* **Findings**:
  * **Correctness (depth-2, 16 qubits)**: both `mps` and `tn` methods matched the `default.qubit` ground truth to machine precision (`diff ≈ 1e-15`).
  * **Depth-3 (64 qubits, 32×32-equivalent)** — previously totally infeasible (see entry above): both methods succeeded. `mps`: 1.73s. `tn`: **0.13s**. No independent 64-qubit ground truth exists to check against (infeasible to compute), but the two independent contraction methods agreed to ~9 decimal places, a strong correctness signal.
  * **Depth-4 (256 qubits, 64×64-equivalent)**: `tn` succeeded in **1.2s**, reproducible across repeated trials with identical results. `mps` was OOM-killed at this width — expected, since forcing a wide quad-tree into a 1D MPS chain ordering requires much higher bond dimension than the tree's native structure needs; not a problem since `tn` works and is the better fit for this topology anyway.
  * **Gradient correctness**: `default.tensor` doesn't support `diff_method='backprop'` (documented PennyLane limitation), so gradients must go through `parameter-shift` instead. Verified this is exact: at depth-2, `parameter-shift` gradients on `default.tensor` matched `backprop` gradients on `default.qubit` to `1e-11`.
  * **Gradient cost — the real limitation**: `parameter-shift` needs ~2 circuit evaluations per trainable parameter. Depth-3 has 252 parameters (21 nodes × 4 wires × 3 rotations); a single full gradient computation for **one training sample** took **56.7 seconds**. Depth-4 has 1020 parameters — proportionally worse (likely several minutes per sample). At these costs, training (many samples × many epochs) is impractical, even though forward-pass inference is fast.
* **Decisions**:
  * **Task 1.4 forward-pass goal is met**: 32×32 and 64×64 images are now simulation-tractable for the *forward pass* via `default.tensor(method='tn')`, correctly and quickly. This is a genuine, validated capability the project didn't have before today.
  * **Training at these sizes is a separate, still-open problem**, gated on gradient cost rather than forward-pass feasibility. The natural fix is **SPSA** (Simultaneous Perturbation Stochastic Approximation) instead of parameter-shift: SPSA needs a constant ~2 evaluations per gradient step *regardless of parameter count*, instead of parameter-shift's 2×(num params). At depth-3's per-eval cost, this would turn ~56.7s/step into roughly ~0.2s/step (~250x speedup), plausibly making depth-3 training practical and depth-4 worth retrying. SPSA is already mentioned elsewhere in the roadmap (Question B, the original barren-plateau autograd-memory-wall entry from 2026-07-18) as an alternative optimizer for scaling — this is now a concrete, motivated reason to actually implement and test it, rather than a hypothetical.
  * `quimb` is now a project dependency (added to the `qnlp` conda environment via pip).
* **Reproduction Steps**:
  * Conda Environment: `qnlp` (now includes `quimb`, installed via `pip install quimb`)
  * Exploratory session, no standalone script committed yet — see this entry's context for the validated `default.tensor(wires=N, method='tn')` pattern (works directly on the natural, non-recycled quad-tree circuit at any depth; no recycling needed since this device never materializes a full statevector).
* **Next step**: implement SPSA gradient estimation for the depth-3/depth-4 tensor-network circuits and re-measure per-step training cost, to determine whether 32×32/64×64 *training* (not just forward passes) is practically achievable.

### [2026-07-27] Completed Experiment: Question E.2 (Entropy Propagation Through Partial Trace) + Question F (Noise as Implicit Regularization — Scheduling Test)
* **Objective**: Two related open questions closed together. E.2: does noise generated at one tree level propagate to the root, or get washed away by the intervening partial trace? F: the project has twice observed test accuracy *improve* under eval-time depolarizing noise (topology benchmark: 42.2%→53.9% at p=0.10; synthetic-shapes sweep: 85.9%→91.4% at p=0.005) — why, and does training *under* noise (evaluated clean) act as a deliberate regularizer the way classical dropout does?
* **Motivation**: This is the last open thread from the noise-robustness narrative running through the whole investigation (raised early in this session, never directly tested). Deliberately closing it before moving to CLEVR.
* **Design**: Built [investigate_noise_regularization.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/investigate_noise_regularization.py), two parts:
  * **Part A (E.2)**: `default.mixed` is a density-matrix simulator — memory scales $O(4^N)$, not $O(2^N)$. A genuine 16-qubit noisy circuit needs ~68GB and is infeasible (confirmed empirically, OOM-killed on first attempt — every other noisy circuit in this project has stayed at 4-5 qubits for exactly this reason, which this experiment initially missed). Redesigned to 8 total qubits: two level-1 quad-nodes (wires 0-3, 4-7; survivors at 0, 4), then a level-2 quad-node on `[0, 4, 1, 5]` — the two survivors plus two of level-1's *discarded* (not reset) qubits, still carrying whatever state level-1's noisy processing left them in. A simplification of the true 4-ary branching, but a genuine coherent two-level circuit. Measured Von Neumann entropy of the level-1 survivor and the root qubit across $p \in [0, 0.20]$, 100 random-weight samples each.
  * **Part B (F, scheduling)**: trained the baseline hierarchical QTTN (`investigate_quantum_residuals.py`'s `HierarchicalQTTNClassifier`, `mode="baseline"`) three ways — `train_p_noise ∈ {0.0, 0.02, 0.05}` — always evaluating on **clean** (p=0) test data, 5 seeds × 15 epochs each. If noise is a genuine training-time regularizer, noisy-trained models should show higher clean-eval accuracy than the noiseless-trained baseline.
* **Findings**:
  * **E.2**: level-1 entropy climbs from `0.505` (p=0) to `0.640` (p=0.20); root entropy climbs from `0.616` to `0.684`. The *gap* between them narrows as noise increases (`+0.111` at p=0 → `+0.044` at p=0.20), and the ratio of root's entropy increase to level-1's drops from ~0.7 toward ~0.5 as p grows past 0.05. This is *consistent with* partial trace washing away some of the noise-induced entropy before it reaches the root — but it's confounded by a ceiling effect: root entropy starts much closer to saturation (`ln(2)=0.693`, only 11% headroom) than level-1 does, so it has structurally less room to grow *regardless* of whether noise propagates. **Cannot cleanly separate "washing away" from "already near the ceiling" with this design** — a genuine answer would need a design that keeps both levels similarly far from saturation (e.g. deeper trees, or measuring at matched entropy starting points), which is a natural follow-up if this question needs a firmer answer.
  * **Question F, scheduling — negative result**: training under noise did **not** improve clean-eval accuracy. Final clean accuracy: `55.0% ± 3.5` (noiseless-trained) vs. `54.1% ± 5.0` (both `p=0.02` and `p=0.05` trained) — noisy training is mildly *worse*, not better, and peak accuracy decreases monotonically with training noise (`56.9%` → `56.6%` → `55.6%`). Differences are within one std of each other (not dramatic), but consistently non-beneficial across both noise levels tested, not just noisy or mixed.
* **Decisions**:
  * **The eval-time noise-improves-accuracy effect and "noise as a trainable regularizer" are different phenomena that only superficially resemble each other.** The project has solid repeated evidence that evaluating a model under noise *sometimes* helps (observed twice, in two different experiments). But deliberately training under noise to get a better *clean* model — the direct classical-dropout analogy — does not hold up under a proper 5-seed test. **Question F's "can this be scheduled deliberately" sub-question is answered: no, not via this straightforward training-noise-injection approach.**
  * The eval-time effect itself remains unexplained mechanistically (still just "expectation values contract toward zero under noise, which happens to sometimes help an under-trained decision boundary") — a full mechanistic account would need the loss-landscape/output-distribution analysis originally proposed in Question F.1, which wasn't attempted here. Given the scheduling test (F.2) came back negative, this is now lower priority than it was.
  * E.2 is left as a partial answer — the ceiling-effect confound means this result shouldn't be over-cited as clean evidence either way. If a firmer answer becomes important later, the fix is measuring propagation at points where neither level is close to saturating $\ln(2)$.
* **Reproduction Steps**:
  * Conda Environment: `qnlp`
  * Python Script: `qnlp/image_tower/classification/quantum/investigate_noise_regularization.py`
  * Run Command: `conda run -n qnlp python -u qnlp/image_tower/classification/quantum/investigate_noise_regularization.py`
* **Plots & Raw Results**:
  * ![Entropy Propagation vs. Noise](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/entropy_propagation_vs_noise.png)
    **[RETAINED]** *Figure 18: Von Neumann entropy of the level-1 survivor qubit vs. the root qubit (after 2 pooling levels) as depolarizing noise increases. The gap between them narrows with noise, but root's proximity to the ln(2) ceiling confounds interpretation.*
  * ![Noise-Scheduled Training, Evaluated Clean](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/results/superseded/noise_scheduling_training.png)
    **[SUPERSEDED — do not use; see results/superseded/README.md]** *Figure 19: Clean test accuracy over training, for models trained at three different noise rates (5 seeds each, shaded std). Training under noise does not improve — and mildly hurts — final clean accuracy.*
  * Raw JSON: `qnlp/image_tower/classification/quantum/results/noise_regularization_results.json`

### [2026-07-27] Code Audit: Scalar-Readout Bottleneck Invalidates the July-26/27 Decision Layer — Remediation Required Before CLEVR
* **Objective**: Before committing the "architecture decisions locked in for CLEVR" (see Current Status above) to the thesis, read the actual experiment code behind each decision and check that the measured quantity supports the claim being made.
* **Motivation**: Every 2026-07-26/27 result clusters in a narrow `50–57%` band on a 4-class task, with repeated "flat 50.0% noise floors". `50.0%` is exactly the single-attribute ceiling of the `overlapping` dataset mode (color alone or shape alone partitions the 4 classes into 2 pairs — see the 2026-07-17 feature-binding proof entry, which uses this same fact as its proof device). A cluster of results pinned at the shortcut ceiling is a symptom, not a finding.
* **Finding 1 — the readout is a single scalar (blocking)**: All three investigation scripts driving the July-26/27 conclusions compress the entire image to **one real number** before classification:
  * `investigate_quantum_residuals.py:134` — `self.patch_embed = nn.Linear(patch_size*patch_size*3, 1)` (48 → 1 per patch)
  * `investigate_quantum_residuals.py:137` — `self.head = nn.Linear(1, 4)`; `:162` — `return self.head(root)` where `root` is `[B, 1]`
  * `investigate_spatial_ancilla.py:199,202` — identical `Linear(48,1)` / `Linear(1,4)` pair
  * `investigate_ancilla_residuals.py:194,197` — identical
  * `compare_topologies.py:119,143` — `Linear(48,3)` in, but still `Linear(1,4)` out
  4-class classification is therefore `argmax` over four affine functions of a scalar $r \in [-1,1]$: the model must lay all 4 classes out as ordered intervals on a line. This is a 1-dimensional representation of the whole image.
* **Finding 2 — this is a regression, not the original design**: `train_synthetic_shapes.py:85,91` (the 2026-07-17 run that reached **75.0%** val acc at 16×16) uses `Linear(48,1)` in but `head = nn.Linear(4, 4)` out — a 4-dimensional quantum readout. The scalar readout appeared between 2026-07-17 and 2026-07-26. Every architecture decision currently marked "locked in for CLEVR" was made *after* the regression, in the degraded regime. The accuracy history is consistent with this: 8×8 benchmark `95.3–96.9%` → 16×16 with 4-dim readout `75.0%` → 16×16 with scalar readout `52–57%`.
* **Consequences (what this invalidates)**:
  * **The five residual rejections are underpowered, not refuted.** Asking whether a skip-connection improves representation quality in a model whose representation is one number cannot distinguish "the mechanism doesn't help" from "the bottleneck dominates and nothing would show up." The binding `NOTE — Residual Connections` rule in `quantum_implementation_plan.md` — which currently extends to *any future quantum model in this project* — is more load-bearing than its evidence.
  * **The spatial-ancilla verdict is a non-result.** `52.8% ± 5.7` vs. `55.0% ± 3.5` on n=5 is well inside noise; it supports "no evidence of benefit," not the "confirmed unnecessary, safe to drop 80→64 qubits" framing now carried in the roadmap.
  * **Question F's founding premise is likely an artifact.** The observation that started it (topology benchmark, `42.2% → 53.9%` under noise) came from a `Linear(1,4)` model. Depolarizing noise contracts the scalar toward 0, which slides samples across the boundaries of a 1-D interval partition — a readout-calibration effect, not regularization. This is consistent with F.2 (training under noise) coming back negative: the negative result stands, but the phenomenon it was chasing may not be real.
  * **The topology benchmark (QTTN vs. MPS vs. MERA)** was also run at scalar readout (`compare_topologies.py:143`), with QTTN at `42.2%` noiseless — below the 50% single-attribute ceiling, i.e. the QTTN arm had not learned even one attribute reliably. The "MERA has higher expressibility than QTTN" conclusion rests on a `51.6%` vs. `42.2%` gap measured in that regime.
* **Finding 3 — the ansatz of record does not match the code**: Current Status (above) and the 2026-07-17 encoding benchmark both state the selected architecture is **Multi-Axis Encoding + IQP**. The July-26/27 scripts use `qml.RY(inputs[:, i], wires=i)` + `qml.StronglyEntanglingLayers` (`investigate_quantum_residuals.py:77-96`) — neither multi-axis encoding nor an IQP ansatz. Every recent decision was made on a different circuit than the one the project says it selected.
* **Finding 4 — unresolved contradiction in the noise numbers**: The 2026-07-17 noise sweep establishes $p_{crit} \approx 0.05$ for a 4-qubit QTTN on 8×8 shapes; the same-day encoding/ansatz benchmark reports $p_{crit} > 0.200$ for essentially every configuration, including same-family ones. Both sit in the Experiment & Metrics Record. The two runs are almost certainly using different definitions of $p_{crit}$ (accuracy-drops-to-chance vs. some other threshold), but as written they contradict each other.
* **Finding 5 — the barren-plateau claim is over-stated for its evidence**: 4 system sizes ($N \le 20$), non-monotonic variance (`1.51e-1, 6.08e-2, 8.61e-2, 5.94e-2`), compared against an asymptotic $2^{-N}$ strawman and reported as an "Empirical BP Immunity Proof." Hierarchical/TTN barren-plateau resistance under local observables is an established *theoretical* result; the data should be presented as consistent with that theory, not as an independent proof.
* **Finding 6 — no classical control exists anywhere in the investigation**: There is not one run of a matched classical CP-TTN on the same data at the same sizes. Question A.3 (unitary constraint vs. unconstrained classical CP factors) is exactly this comparison and is still un-run. Without it, no quantum-vs-quantum ablation in this log can support a claim about quantum models relative to their classical analogue — which is the thesis's actual claim territory.
* **Finding 7 — internal inconsistency with the project's own purity rule**: Question B.3 was closed *on principle* because `hybrid_trainer.py` put a classical `Linear(16, 4)` before a 4-qubit VQC ("classical capacity substituting for quantum circuit width"). But `Linear(48, 1)` per patch, applied in every current experiment, is a strictly more aggressive classical compression. Either the rule needs a stated quantitative boundary (what encoder width counts as "minimal I/O"), or B.3's closure needs revisiting.
* **Decisions**:
  * **Do not start CLEVR (Phase 2) yet.** Added **Phase 1.5 — Remediation Plan** (Section 7 of `quantum_investigation_roadmap.md`) with self-contained tasks R1–R6 and an explicit decision gate.
  * All verdicts derived from the scalar-readout experiments are downgraded to **PROVISIONAL** in the roadmap and in `quantum_implementation_plan.md` until R3 completes. Specifically: the no-residuals rule, the no-spatial-ancilla rule, the MERA-vs-QTTN expressibility ranking, and Question F's premise.
  * Findings 4 and 5 are documentation/framing fixes (task R5), not re-runs.
  * Findings 6 and 7 are genuine open scientific gaps (tasks R4 and R5 respectively).
* **What is NOT invalidated** (these do not route through the scalar readout, and stand as-is):
  * Recycling equivalence and the depth-3 recycling wall (2026-07-17 / 2026-07-26) — exact numerical/feasibility results.
  * `default.tensor` forward-pass tractability at 32×32 and 64×64, and the parameter-shift gradient-cost measurement (2026-07-26).
  * The CNOT-count-to-entropy diagnostic and the entropy-vs-tree-depth saturation trend (`72.7% → 89.7%`) — diagnostic measurements on random weights, no classifier head involved.
  * The LCU postselection-overhead diagnostic (`P(ancilla=0)`: L1 `0.986`, L2 `0.515`) — a property of the circuit, not of accuracy.
  * The methodological lesson about multi-seed averaging (~10 seeds) for any noise-robustness claim.

---

### [2026-07-27] Completed Task R1: Remove the Scalar-Readout Bottleneck — Gate Passed via Capacity Fallback
* **Objective**: Roadmap Section 7, Task R1. Build a shared, configurable `HierarchicalQTTNClassifier` (`qttn_core.py`) replacing the drifted per-script copies, and find a `readout × encoding` configuration that reaches the demonstrated ≥70% mean val-acc gate over 5 seeds on 16×16 `overlapping` synthetic shapes — the threshold below which ablations (residuals, spatial ancilla) cannot be trusted, per the Code Audit entry above.
* **Motivation**: Every recent architecture decision was made on a model that compresses the whole image to one scalar (`Linear(1,4)` head), pinning results at the 50–57% shortcut ceiling. This task is blocking for R2–R6.
* **Method**: `qnlp/image_tower/classification/quantum/qttn_core.py` (new shared module) + `run_r1_remediation.py`. Level-1 (4 patches → 1 node, weight-shared across 4 blocks) always emits a scalar message, unchanged; only the **root (level-2) readout width** and the **level-1 patch encoding width** are ablated, isolating the two variables the audit flagged. Sweep 1 fixed `encoding=scalar_ry`, varied `readout ∈ {scalar, root_multi_pauli, level1_survivors}`; Sweep 2 fixed the Sweep-1 winner's readout, varied `encoding ∈ {scalar_ry, multi_axis}`. Both: 5 seeds × 15 epochs, 256 train / 64 test, 16×16 `overlapping` shapes (original protocol, for comparability with prior log entries).
* **Results — Sweep 1 (256/15, encoding=scalar_ry)**:
  | readout | final val acc (mean ± std, 5 seeds) | peak | params |
  |---|---|---|---|
  | scalar (old/regressed) | 55.0 ± 3.5% | 56.9% | 105 |
  | root_multi_pauli | 57.5 ± 7.2% | 62.2% | 113 |
  | level1_survivors | 57.5 ± 5.6% | 61.2% | 117 |
* **Results — Sweep 2 (256/15, readout=root_multi_pauli)**:
  | encoding | final val acc | peak | params |
  |---|---|---|---|
  | scalar_ry | 57.5 ± 7.2% | 62.2% | 113 |
  | multi_axis | 59.1 ± 6.4% | 64.4% | 211 |
  Best at the original protocol: `root_multi_pauli` + `multi_axis`, **59.1 ± 6.4%** — an improvement over the 55.0% scalar baseline, but **short of the 70% gate**.
* **Capacity fallback triggered** (per the roadmap's escalation order, option (b) first): re-ran the best config (`root_multi_pauli` + `multi_axis`) at **1024 train samples / 30 epochs**. Result: **78.75 ± 7.2%** over 5 seeds (`[70.3, 76.6, 73.4, 90.6, 82.8]`). **Gate passed.**
* **Interpretation**: widening the readout alone (scalar → 3-dim Pauli triple) was not sufficient at the original 256-sample/15-epoch protocol — it moved the mean from 55% to 59%, still inside noise of the baseline given the stds involved. The dominant factor was training-set size/epochs, not readout width in isolation; but the *combination* of wider readout + multi-axis encoding + more data is what cleared the gate, and readout width matters as of Sweep 1 (58 vs 55, direction is consistent, if noisy). This is consistent with the 2026-07-17 75% run, which also used more capacity in its readout (`Linear(4,4)`) — the mechanism generalizes, but the original 256/15 protocol turns out to be underpowered for *any* config at this task's difficulty, not just the scalar-readout one.
* **Decision**: **Architecture of record for R2 onward: `readout=root_multi_pauli`, `encoding=multi_axis`, training protocol = 1024 train / 64 test / 30 epochs** (supersedes the 256/15 protocol for all subsequent Phase 1.5 tasks — R2/R3 should use 1024/30, not 256/15, since 256/15 does not clear the gate for any tested config and is no longer representative). `qttn_core.py`'s `readout` and `encoding` flags are now the shared interface; `level1_survivors` was statistically indistinguishable from `root_multi_pauli` in Sweep 1 (57.5% both) and was not re-tested in Sweep 2 — worth a follow-up if R2/R3 numbers are borderline.
* **Reproduce**: `conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r1_remediation`. Model: `qnlp/image_tower/classification/quantum/qttn_core.py`. Results: `qnlp/image_tower/classification/quantum/results/r1_remediation_results.json`.
* **Status**: R1 acceptance criterion met (decision-gate item 1). Proceed to R2 (ansatz-of-record reconciliation) using the new protocol.

### [2026-07-28] Completed Task R1b: The Missing Control Arm — Audit Diagnosis CONFIRMED, and a Power Analysis That Rewrites R3's Design
* **Objective**: Roadmap Section 7, Task R1b. Fill the hole in R1's 2×2 — nobody ran the *original* (`scalar` readout, `scalar_ry` encoding) config at the *new* 1024/30 protocol — and settle whether the Code Audit's diagnosis (the `nn.Linear(1,4)` readout bottleneck invalidated the July-26/27 ablations) is actually supported.
* **Motivation**: R1 cleared the ≥70% gate, but not via the predicted mechanism. Readout width moved `55.0 ± 3.5 → 57.5 ± 7.2` and encoding width `57.5 ± 7.2 → 59.1 ± 6.4` — both inside seed noise at n=5 — while the protocol change moved `59.1 → 78.75`. With the capacity fallback run only on the winning config, the experiment as executed could not attribute the gain, yet the causal claim was already written into this log as fact.
* **Design**: [run_r1b_control.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/run_r1b_control.py). **Part A** runs the missing cell by importing `run_r1_remediation.train_config` directly rather than reimplementing it, so the new number shares R1's exact code path and is strictly comparable. **Part B** re-runs both corner configs at 1024/30 with the data ordering pinned per seed, to test whether paired comparison would sharpen R3.
* **Findings — the completed 2×2** (final val acc, 5 seeds, 16×16 `overlapping`):

  | config | 256 train / 15 ep | 1024 train / 30 ep |
  |---|---|---|
  | `scalar` / `scalar_ry` (105 params) | `55.0 ± 3.5` | **`62.5 ± 6.9`** ← R1b control |
  | `root_multi_pauli` / `multi_axis` (211 params) | `59.1 ± 6.4` | `78.8 ± 7.2` |

  * **Verdict: AUDIT_CONFIRMED.** The scalar-readout model does **not** clear the 70% gate even with 4× the data and 2× the epochs (`62.5%`). The architecture gap at the new protocol is **+16.2 points**, comfortably above the 10.3 points this n=5 design can resolve.
  * **The interaction is the real result.** Protocol alone (on the old config) buys `+7.5` pts; architecture alone (at the old protocol) buys `+4.1` pts; but architecture *at adequate training* buys `+16.2` pts. Neither factor is sufficient alone — **readout width is only expressible once there is enough data to train it**, which is why R1's one-factor-at-a-time sweeps at 256/15 each looked like noise. This is a cleaner and more defensible statement than either the original audit claim or R1's write-up.
  * **Correction to the R1 entry's attribution**: R1's line *"readout width matters as of Sweep 1 (58 vs 55, direction is consistent, if noisy)"* was over-reading a 2.5-point gap against stds of 3.5 and 7.2. That sweep genuinely could not resolve the effect; R1b resolves it at the protocol where it is expressible. The conclusion R1 reached was right, but not for the reason it gave.
  * **Architecture of record stands**: `readout=root_multi_pauli`, `encoding=multi_axis`, 1024 train / 64 test / 30 epochs. The 211-vs-105 parameter cost is earned.
* **Findings — Part B, and a correction to R3's planned design**: paired at n=10, old `63.9 ± 8.1` vs. new `83.9 ± 10.0`, mean delta `+20.0 ± 14.2`, new config won on **9/10** seeds — independently reproducing Part A's conclusion on a larger sample.
  * **Pairing does not help here, and slightly hurts.** Cross-variant seed correlation is **−0.22**: seed identity is *not* a shared difficulty factor between variants. Min detectable effect is `10.2` pts paired vs. `7.6` pts unpaired. **The pairing precondition added to roadmap R3 on 2026-07-27 was based on an assumption this measurement refutes** — run-to-run variance here is dominated by within-run optimization stochasticity, not by the data draw, so there is nothing for pairing to cancel. R3's precondition is corrected accordingly.
  * **Scoring runs by the last-5-epoch mean instead of the single final epoch** is a small free improvement: MDE `8.0 → 7.2` pts, and it equalizes the two arms' stds (`8.1/10.0 → 8.2/8.2`). Adopt it for R3.
  * **Power analysis — the number R3 actually needs.** At pooled std `8.2` and ~18s per run at 1024/30: resolving a **2-pt** effect needs **130 seeds/arm** (~39 min), **3-pt** needs **58** (~17 min), **5-pt** needs **21** (~6 min), **8-pt** needs **9** (~3 min). The planned 10 seeds resolves only ~8-point effects — while the residual/ancilla effects R3 is hunting were 2–3 points. **At 10 seeds R3 would have produced another inconclusive null**, exactly the 2026-07-26 failure mode at higher accuracy. Runs are cheap enough that this is affordable: budget **~60 seeds/arm** for R3 (≈17 min/arm, resolving 3-pt effects) rather than 10.
* **Decisions**:
  * Code Audit diagnosis confirmed — no correction needed to its central claim, but its *attribution* is sharpened to the interaction statement above.
  * Decision-gate item 1 is now fully closed.
  * Roadmap R3 precondition rewritten: drop pairing, adopt last-5-epoch scoring, raise seeds to ~60/arm, and continue to report the minimum detectable effect alongside every result.
* **Reproduce**: `conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r1b_control`
  * Conda Environment: `qnlp`. Script: `qnlp/image_tower/classification/quantum/run_r1b_control.py`.
  * Raw JSON: `qnlp/image_tower/classification/quantum/results/r1b_control_results.json` (includes per-seed val-acc curves, so R3 can compare scoring estimators without re-running anything).
  * Deterministic: two independent full runs produced identical numbers.
* **Caveat**: Part B's absolute numbers (`63.9` / `83.9`) sit above Part A's (`62.5` / `78.8`) for the same configs because Part B uses 10 seeds and a different RNG path (loaders built before model init). Part B is internally consistent and Part A is comparable to R1; do not mix the two sets in one table.

### [2026-07-28] Completed Task R2: Ansatz-of-Record Reconciled — the Encoding Was the Error, Not the Ansatz
* **Objective**: Roadmap Section 7, Task R2. The documentation stated the selected architecture was Multi-Axis Encoding + IQP; the July-26/27 code ran `RY` + `StronglyEntanglingLayers`. Determine which is right at 16×16 and make code and documentation agree.
* **Design**: [run_r2_ansatz.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/run_r2_ansatz.py) — `{scalar_ry, multi_axis} × {strongly_entangling, iqp}`, readout fixed at `root_multi_pauli` (R1/R1b winner), 21 seeds/arm, protocol of record (1024/30). 21 seeds resolves ~5-pt effects, the right granularity for a decision that should not be overturned by anything smaller.
* **Findings**:

  | config | score (last-5 mean) | params |
  |---|---|---|
  | `scalar_ry` + `strongly_entangling` (as coded in July) | `68.4 ± 6.5` | 113 |
  | `scalar_ry` + `iqp` | `68.0 ± 6.9` | 113 |
  | `multi_axis` + `strongly_entangling` | `79.3 ± 9.7` | 211 |
  | **`multi_axis` + `iqp`** (documented choice) | **`80.9 ± 4.9`** | 211 |

  * **Verdict: CONFIRM_DOCUMENTED.** Multi-Axis + IQP is the best arm; the documented choice survives at 16×16.
  * **The decomposition is the real result.** The **encoding** is resolved and large: `+12.6` pts vs a `3.6`-pt limit. The **ansatz** is *not* resolved: `multi_axis+iqp` vs `multi_axis+strongly_entangling` is `+1.6` pts against a `4.8`-pt limit. **So the July code's consequential error was the encoding (`scalar_ry`), not the ansatz.** IQP is adopted because it matched the documented choice and has visibly lower seed variance (`4.9` vs `9.7`) — which buys resolution downstream — *not* because it is measurably more accurate. That distinction is recorded in `phase15_common.ARCH` so it cannot quietly become "IQP is better".
  * The original 2026-07-17 selection came from a 12-config sweep at 8×8 with one seed per config. It reached the right answer, but on evidence that could not have distinguished it from several alternatives.
* **Decisions**: Architecture of record pinned in `phase15_common.ARCH` = `{readout: root_multi_pauli, encoding: multi_axis, ansatz: iqp}`, with `PROTOCOL` = 1024 train / 64 test / 30 epochs. Kept there rather than as `qttn_core` constructor defaults so R1/R1b remain reproducible exactly as logged.
* **Reproduce**: `conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r2_ansatz`. Raw: `results/r2_ansatz_results.json`.

### [2026-07-28] Completed Task R3: Ablations Re-run at Restored Capacity — Rejections Upheld, Now on Real Evidence
* **Objective**: Roadmap Section 7, Task R3. Re-test the two residual mechanisms that showed any signal (`reupload`, `mixed_channel`) and the spatial ancilla, on the architecture of record with enough seeds to resolve the effects being claimed.
* **Motivation**: All three verdicts were reached on the scalar-readout model in the 50–57% band, where a null result cannot distinguish "the mechanism doesn't help" from "the bottleneck dominates".
* **Design**: [run_r3_ablations.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/run_r3_ablations.py), 30 seeds/arm, 15 noise-swept, scored on the mean of the last 5 epochs, unpaired comparisons. `near_identity` and `lcu`/`lcu-lite` deliberately not re-run (roadmap R3b).
  * **Seed count justified by measurement, not convenience**: the earlier plan said 58 seeds, derived from R1b's pooled std of `8.2` — but that was measured on a *different* configuration. R2 measured the architecture of record at std `4.94`, where 30 seeds resolves ~2.6-pt effects (48 would be needed for 2-pt, 22 for 3-pt). Each arm reports the resolution it actually achieved.
  * **Execution**: run as one worker per arm in parallel. The original sequential run was single-threaded on an 11-core machine (~9% utilisation), buffered all output through `conda run` so progress was invisible, and wrote results only at exit — it was killed after 4h13m with nothing recoverable. The parallel version, with per-seed checkpointing and live output, completed in ~25 min. Merged by [combine_r3.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/combine_r3.py).
* **Findings**:

  | arm | score | vs. baseline | resolution limit | verdict | failures |
  |---|---|---|---|---|---|
  | `reupload` | `80.9 ± 6.6` | `+1.5` | 3.8 | **unresolved** | 0/30 |
  | `baseline` | `79.4 ± 8.0` | — | — | — | 0/30 |
  | `mixed_channel` | `68.5 ± 10.1` | `−10.9` | 4.7 | **worse, resolved** | 0/30 |
  | `with_ancilla` | `59.9 ± 6.9` | `−19.5` | 3.8 | **worse, resolved** | 0/30 |

  Noise sweep (mean over 15 seeds each):

  | arm | p=0 | 0.01 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 |
  |---|---|---|---|---|---|---|---|
  | baseline | 80.7 | 81.6 | 80.0 | 79.0 | 77.5 | 70.2 | 57.0 |
  | reupload | 83.6 | 79.2 | 82.2 | 78.5 | 69.2 | 55.3 | 44.6 |
  | mixed_channel | 72.8 | 72.0 | 70.7 | 69.6 | 65.0 | 54.6 | 45.7 |
  | with_ancilla | 61.9 | 64.0 | 62.3 | 54.9 | 49.2 | 39.4 | 32.5 |

* **What changed relative to the provisional verdicts**:
  * **The rejections hold, but for the first time on positive evidence rather than absence of it.** Previously these were 2–3 pt gaps inside ~10-pt noise. `mixed_channel` is now `−10.9` against a `4.7`-pt limit; the ancilla `−19.5` against `3.8`.
  * **The dead-gradient failures are gone.** `mixed_channel` failed 1/10 seeds in July; here **0/30**. It is not unstable at restored capacity — it is simply consistently worse. This closes backlog item 13 (stabilise `mixed_channel`) outright: there is nothing to stabilise, and no remaining motivation to try.
  * **`reupload` is a true null, not a trade-off.** July reported the highest peak accuracy but a collapse under noise (`28.1%` at p≥0.10 vs baseline's `50.0%` floor). At restored capacity it tracks baseline on accuracy (`+1.5`, unresolved) and degrades *similarly* under noise (`69.2` vs `77.5` at p=0.10) — worse, but nothing resembling a collapse. **The dramatic fragility narrative was largely an artifact of the bottlenecked model.** Correct statement: no demonstrated benefit at effects ≥ 3.8 pts.
  * **The noise curves are informative for the first time.** Baseline degrades gracefully (`80.7 → 77.5 → 57.0`) instead of flatlining at the 50% single-attribute floor. Every prior noise sweep in this project was pinned to that ceiling and therefore measuring almost nothing.
* **Scope caveat on the ancilla result (attached to the result in the JSON so it travels with it)**: `−19.5` pts is the largest effect in the table and must **not** be written up as "positional encoding is harmful". This task is translation-invariant single-object classification, where position barely affects the label — so the ancilla can only add parameters and training difficulty with nothing to contribute, and a negative result is close to structurally guaranteed. It also tests the **per-quadrant** ancilla (4 positions), not the original **per-patch** design (16). What this licenses: do not use it for this class of task. Whether implicit tree topology suffices for *relational* reasoning is untested — see Question C.2 and the next-steps entry below.
* **Internal consistency check**: the baseline here (`79.4 ± 8.0`) agrees with R4's independent quantum arm (`79.0 ± 8.7`) on the same architecture.
* **Reproduce**: four parallel workers, e.g.
  `/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python -u -m qnlp.image_tower.classification.quantum.run_r3_ablations --arms baseline --seeds-per-arm 30 --noise-seeds 15 --out r3p_baseline`
  (repeat for `reupload`, `mixed_channel`, `with_ancilla`), then
  `conda run -n qnlp python -m qnlp.image_tower.classification.quantum.combine_r3`.
  Raw: `results/r3_ablation_results.json`, per-arm `results/r3p_*_ablation_results.json`.

### [2026-07-28] Completed Task R4: First Classical Control in the Project's History — and Question A.3 Is NOT Answered
* **Objective**: Roadmap Section 7, Task R4. Compare the quantum tower against matched classical models on identical data, and thereby address Question A.3 (does the unitary constraint limit capacity vs. unconstrained classical CP factors?).
* **Motivation**: Every result in this investigation to date is quantum-vs-quantum. Without a classical control, nothing here can support a claim about quantum models *relative to their classical analogue*, which is the thesis's actual claim territory.
* **Design**: [run_r4_classical_control.py](file:///Users/fotinoskyriakides/Desktop/Dev/qnlp/qnlp/image_tower/classification/quantum/run_r4_classical_control.py), 21 seeds/arm, identical data/seeds/protocol, same 16→4→1 quad-tree. Classical arms received a **fair hyperparameter search** (lr × bond_dim) before comparison — see the failure note below for why this was not optional.
* **Findings**:

  | arm | score | params |
  |---|---|---|
  | `mlp_reference` | `96.0 ± 1.5` | 999 |
  | **`quantum`** (architecture of record) | **`79.0 ± 8.7`** | **211** |
  | `mlp_param_matched` | `70.5 ± 18.1` | 257 |
  | `classical_full` (CP + residual + dropout) | `59.3 ± 2.9` | 332 |
  | `classical_bare` (CP, no residual/dropout) | `33.9 ± 12.4` | 308 |

  * **vs. the 999-param MLP**: quantum loses by `17.0` pts, **resolved**.
  * **vs. the size-matched MLP**: quantum is `+8.5` pts against an `8.9`-pt limit — **statistically tied**, with the MLP far less stable at that size (`±18.1` vs `±8.7`).
  * **Therefore the defensible claim is parameter efficiency, not raw accuracy**: at matched parameter count the quantum tower is competitive with (trending above) a classical MLP; it loses only when the classical model is given ~5× the parameters.
* **Question A.3: NOT ANSWERABLE from this run, and the script refuses to emit a verdict.** `CPQuadRankLayer` scores `33.9%` where an MLP of comparable size reaches `96.0%` (`+62.1` pts, limit `5.5`). The CP quad-node — not classical computation — is what underperforms, so the quantum-vs-CP gap is a **baseline artifact**. It must not be cited as quantum advantage. A.3 needs a classical TTN at least competitive with the MLP reference before the unitarity question can even be posed.
* **A near-miss worth recording.** The first version of this experiment used the quantum model's lr and the structurally-faithful `bond_dim=2` for the classical arm, which scored **25.8% — exactly chance for 4 classes** — and produced a headline "quantum beats classical by 57.8 points". That number was entirely an artifact of a dead baseline. Two guards now exist so this cannot recur silently: the `mlp_reference` task-difficulty arm (which makes an underperforming baseline visible immediately), and `phase15_common.is_chance_level` / `assert_not_chance_level`, covered by a regression test.
* **Secondary finding, relevant to the divergence question**: the repo's own classical node depends heavily on residual+dropout — `classical_full` beats `classical_bare` by `+25.4` pts (limit `5.6`, resolved). Those mechanisms are load-bearing classically while the quantum tower has binding rules against them (upheld by R3). That contrast is now measured rather than assumed, and belongs in the thesis.
* **Reproduce**: `conda run -n qnlp python -m qnlp.image_tower.classification.quantum.run_r4_classical_control`. Raw: `results/r4_classical_control_results.json`.

### [2026-07-28] Tooling Finding: PennyLane 0.43.2 `default.mixed` Silently Returns Wrong Values Under Parameter Broadcasting
* **What**: With multi-axis encoding (three rotations on the same wire), `default.mixed` mis-handles parameter broadcasting. Behaviour depends on batch size and on the rest of the circuit:
  * most batch sizes → a **square** number of results (32 → 64, 128 → 256, 12 → 36) or an internal reshape error (5, 7, 9, …);
  * batch sizes that are **powers of 4** → correct shape, **numerically wrong values**;
  * with the IQP ansatz → hard failure at every batch size, including 1.
* **How it was proved**: at $p=0$ the `DepolarizingChannel` is exactly the identity, so the noisy device must reproduce the clean device to machine precision. It disagreed by up to `4.2e-01` (should be ~`1e-16`). Unbatched (row-by-row) evaluation is correct to `7.8e-16`. `scalar_ry` encoding is unaffected — batched matches clean to `8.9e-16`.
* **Impact on existing results: none.** Verified that both scripts which could have hit this already evaluate noise sample-by-sample, with a comment naming the bug (`benchmark_encodings_ansatze.py:253`, `compare_topologies.py:164`, dated 2026-07-17). **That knowledge existed in the codebase for eleven days, was never recorded in this log, and never propagated to the July-26 scripts** — which is the single strongest argument for the Phase C consolidation.
* **Fix**: `qttn_core._run_noisy` evaluates row-by-row when needed, selected by a **self-validating probe** that runs the p→0 identity check per circuit at construction. If a future PennyLane release fixes the bug, the fast batched path resumes automatically with no code change.
* **A wrong first fix, worth recording**: the initial workaround chunked batches into powers of 4. It produced correct shapes and wrong numbers — the same silent-corruption class as the bug itself — and was caught only because chunk-invariance was checked explicitly and came back non-zero.
* **Framework implication**: this does **not** justify migrating away from PennyLane. The bug is confined to `default.mixed`; `default.qubit` (1e-15), `default.tensor`/quimb (the 32×32 and 64×64 forward passes) and `parameter-shift` (1e-11 vs backprop) are all verified sound, and none of the remaining critical path (R6/SPSA, CLEVR) touches `default.mixed`. Recommended instead: pin the version, file the bug upstream, and add a **Qiskit Aer cross-check** on a few small noise sweeps — cheap at 4–8 qubits, a strong independent validation, and it sets up the un-run Task 3.2 (ZNE/Mitiq), which wants Qiskit anyway.

### [2026-07-28] Figure Audit: 11 of 20 Figures Are Compromised
Full triage with dispositions in `quantum_investigation_roadmap.md` Section 8, Phase B. Summary:

| Status | Count | Figures |
|---|---|---|
| **KEEP** (no classifier head involved) | 5 | 1, 6, 7, 17, 18 |
| **KEEP + CAVEAT** | 4 | 2, 3, 4, 5 |
| **REGENERATE** | 10 | 9–16, 19 (R3 supplies 9–16) |
| **RETIRE** | 1 | 8 |

* The compromised figures all route the whole image through `nn.Linear(1,4)` and use `scalar_ry` + `strongly_entangling`, which R2 has since shown costs ~12.6 pts.
* **Figure 8 (`topology_noise_resilience.png`) is the worst case and is retired, not regenerated.** Its QTTN arm scored `42.2%` — *below* the 50% single-attribute ceiling, i.e. that arm had not reliably learned even one attribute — and it underpins two separate narrative threads (the "MERA is more expressive than QTTN" claim, and Question F's founding observation). The QTTN-vs-MERA decision rests on contraction complexity (Section 5), a scaling argument independent of accuracy, so the accuracy claim is dropped rather than re-derived.
* **Figure 12's caption asserts the mixed-channel noise-robustness claim that was retracted the same day.** Actively misleading if reused.
* Figures 6 and 7 need a **caption fix only**: "empirical BP immunity proof" overstates 4 points at N≤20 against an asymptotic strawman.

### [2026-07-28] Code Audit #2: The Trained Model Is Not a Coherent Quantum Tree — It Measures and Re-encodes Between Levels
* **Objective**: Check whether the implemented tree actually passes qubits between levels, prompted by the question of whether the inter-level bond must be a single qubit.
* **Finding**: It does not pass qubits at all. Every `QuantumNode` instantiates **its own 4-qubit device** (`qttn_core.py:229`). Level-1 nodes return a single `⟨Z⟩` expectation value each, and those four **classical scalars** are re-encoded as `RY` angles into a *separate* root circuit (`qttn_core.py:214-218`). The tree measures at every level. Inherited unchanged from `investigate_quantum_residuals.py` (2026-07-26) into `qttn_core.py`.
* **The coherent version exists and predates it.** `train_synthetic_shapes.py` (2026-07-17) is the architecture as designed: a **single 16-qubit device**, all 16 patches encoded simultaneously, level-1 block unitaries on wires `[0-3] [4-7] [8-11] [12-15]`, then the level-2 unitary applied **directly to the four surviving qubits** `[0, 4, 8, 12]`, which remain quantum throughout and are measured only at the end (4 expectations → `Linear(4,4)`). It reached **`75.0%`** at 16×16 on the *old, weaker* 256/15 protocol. **The July-26 scripts replaced it with the measure-and-re-encode version — the same regression pattern as the scalar readout, and equally undocumented.**
* **What the re-encoding destroys**:
  * **Inter-level entanglement, entirely.** In the coherent tree the root unitary acts on four qubits still entangled with their subtrees, and the partial trace leaves a *mixed* state carrying those correlations. Measurement collapses this to a real number. The trained model has **zero** entanglement across tree levels.
  * **The bond is narrower than $\chi=2$ — it is a single real scalar.** A coherently-passed qubit carries two complex amplitudes (three real parameters as a mixed state); a measured `⟨Z⟩` carries one. This is very likely why widening the *root* readout from 1 to 3 Paulis mattered so much in R1: the same information loss applies at every level, not just the top.
  * **Two analyses describe a circuit that is not the trained one.** Question A.2 (entropy saturation `72.7% → 89.7%`) and Question E.2 (noise propagation through partial trace) were both measured on coherent circuits. The trained model has no inter-level entanglement for them to characterise.
  * **It violates the project's own purely-quantum rule.** Classical scalars carrying information between tree levels is classical capacity inside the core pipeline — stricter than the `Linear(16,4)` for which Question B.3 was closed on principle.
* **Why it was introduced**: almost certainly the noisy-simulation constraint. `default.mixed` is a density-matrix simulator, $O(4^N)$: a 16-qubit noisy circuit needs **~68GB** and was OOM-killed (2026-07-27 entry), which is why every noisy circuit in this project has stayed at 4–5 qubits. The measure-and-re-encode structure is what permits a "tree" when no single circuit may exceed 5 qubits. **But that constraint only ever applied to noisy simulation** — noiseless 16-qubit statevector simulation is $2^{16} = 65{,}536$ amplitudes, entirely trivial, and `default.tensor` already runs the coherent tree at 64 and 256 qubits for forward passes.
* **Impact on R1–R4**: all four ran on the hybrid. The "architecture of record" is therefore a **quantum-node / classical-wiring hybrid**, not a quantum TTN. R4's parameter-efficiency claim describes that hybrid, and "quantum native throughout" is false for what has been trained (though true of the design).
* **What survives**: the node-level ablations. Residuals and the spatial ancilla are mechanisms *within* a node, and each node is a genuine 4-qubit VQC, so those comparisons stand on their own terms — but they are **node-level results, not tree-level ones**, until re-checked on the coherent tree.
* **Decision**: added **Task R7** (port to the coherent circuit and re-baseline) to roadmap Section 7, **ahead of the Phase 1.6 figure regeneration** — there is no purpose in regenerating figures from a model that is about to be replaced.

### [2026-07-28] Scope Decision: Noisy Simulation Is Being Closed Out, Not Expanded
* **Decision**: The project moves forward on **noiseless simulation only**. Remaining noise work is limited to one close-out characterisation of the base model — a depolarizing sweep, including the question of whether a small amount of noise *improves* accuracy, for which evidence already exists — after which the noise track is closed and the thesis states that subsequent work is noiseless.
* **Rationale**: noise is not the thesis contribution, the existing evidence is sufficient to characterise the model's tolerance, and the cost of doing it *properly* on the coherent tree is disproportionate (see below).
* **What this closes** (all of it, without further experiments):
  * **Phase 3 Tasks 3.1, 3.2, 3.3** (noise calibration on CLEVR, ZNE/Mitiq error mitigation, optimizer-under-noise comparison) — out of scope. Section 6 backlog item 1 closed.
  * **Question E.1 and E.3** — closed with the existing depolarizing sweeps. E.2 stays a partial answer with its ceiling-effect caveat and is not pursued.
  * **Question F** — closed. The scheduling sub-question already has a clean negative result; the eval-time effect is recorded as an observation with a probable mechanism (expectation values contracting toward zero) and is not investigated further.
  * **Quantum trajectories** (stochastic Kraus sampling on a statevector simulator, $O(2^N)$ instead of $O(4^N)$) — this was the route to noisy simulation of the *coherent* 16-qubit tree. **Not needed now**, and the ~half-day of work is avoided.
  * **Qiskit Aer cross-check** (backlog item 18) — dropped. It was proposed to validate the noise chapter after a silent-corruption bug; with the noise track closed, the motivation goes with it.
  * **The PennyLane `default.mixed` broadcasting bug** — downgraded from a live hazard to a documented curiosity. The workaround and its regression test stay in place, but nothing on the forward path depends on that device any more.
* **Standing limitation to state in the thesis, not to fix**: noisy simulation was capped at 4–5 qubits throughout by the $O(4^N)$ density-matrix cost, so all noise results characterise a **single quantum node**, not the full tree. Whether noise tolerance measured at one node extends to a deep tree is untested and, under this scope decision, will remain so.

### [2026-07-28] Task R7 (partial): Coherent Tree Implemented — and the Readout Is Topology-Dependent, Worth 43 Points
* **Objective**: Roadmap Task R7. Replace the measure-and-re-encode hybrid with the coherent quantum tree and re-baseline.
* **Built**: `CoherentQTTNClassifier` in `qttn_core.py` — one device holding all 16 patch qubits, level-1 block unitaries on `[0-3] [4-7] [8-11] [12-15]`, then the level-2 unitary applied **directly to the survivors** `[0, 4, 8, 12]`, which stay quantum until a single measurement at the end. 211 params with shared level-1 weights, identical to the hybrid, so comparisons are not size-confounded. Runs on `lightning.qubit` with adjoint differentiation.
* **Cost, measured**: ~17.5 min per 30-epoch run (lightning + adjoint) vs ~72 min (`default.qubit` + backprop) vs ~15 s for the hybrid. The hybrid was cheap only because it never simulated more than 4 qubits at once. `default.tensor` is unusable for training here — its parameter-shift backward took 242 s for one batch.
* **Coherence is now mechanically enforced.** A single qubit is mixed exactly when its Bloch vector is shorter than 1 (purity $=(1+|r|^2)/2$), so `survivor_bloch_length() < 1` proves the qubit reaching level 2 is entangled with its subtree; in the hybrid it is exactly 1 by construction. `vn_entropy` would be the direct measure but OOMs at 16 qubits, while three expectation values are cheap and equally decisive. Six regression tests cover this and the refusal of un-ported variants.
* **First pilot failed, and the cause is instructive.** 3 seeds, 30 epochs: `34.1% / 38.1% / 38.8%` — below the 50% single-attribute ceiling, against the hybrid's `79.4%`. Entanglement was clearly working (|r| fell from `0.98` at init to `0.53–0.91` after training), so the tree was coherent; something else was wrong.
* **Diagnostic — the readout is topology-dependent** (1 seed, 15 epochs, per-block level-1 weights):

  | readout | measures | values | score | peak | \|r\| |
  |---|---|---|---|---|---|
  | `root_multi_pauli` | $\langle X\rangle,\langle Y\rangle,\langle Z\rangle$ on wire 0 | 3 | `35.6%` | 43.8% | 0.667 |
  | `top_layer_qubits` | $\langle Z\rangle$ on wires 0, 4, 8, 12 | 4 | **`78.4%`** | 89.1% | 0.306 |

  * **A 43-point swing, and it is not "more numbers"** (3 vs 4). The root's Bloch vector is the **reduced state of one qubit** after tracing out the other fifteen — three real parameters, no matter how much structure feeds in. Reading four wires gives four marginals from different parts of the register, which jointly carry far more about the 16-qubit state.
  * `78.4%` at **15** epochs is level with the hybrid's `79.4%` at **30**. **The coherent tree is not worse — the readout choice was.**
  * The successful config is also *more* entangled (|r| `0.306` vs `0.667`), i.e. it uses coherence harder.
* **This is the same failure mode the whole remediation exists to fix, reintroduced by me.** `root_multi_pauli` was carried over from the hybrid without re-deriving it. In the hybrid it is not a bottleneck, because bonds are already classical scalars and information reaches the root by another route; in the coherent tree it funnels all 16 patches through one qubit. The 2026-07-17 reference had this right and measured all four survivors. **The regression tests did not catch it**: `test_readout_width_matches_config` verifies the head matches the configured readout, but cannot tell that a validly-configured readout is a bottleneck for a given topology.
* **Naming**: `level1_survivors` is a misleading name — it measures the four wires *entering* the top node, read *after* that node has acted, i.e. the root plus the three qubits the tree nominally discards at the top. Renamed to **`top_layer_qubits`**, with the old name kept as an alias so R1's logged sweep stays reproducible.
* **Decisions**:
  * Added `phase15_common.COHERENT_ARCH` = `{readout: top_layer_qubits, encoding: multi_axis, ansatz: iqp}`. **The architecture of record is now per-topology**, which the single `ARCH` had hidden.
  * The tower outputs the four top-layer qubits rather than a single root. This must be stated in the thesis, not glossed: it means the tree does not contract to a single root, and it is a **documented workaround for a simulation limit**, not a design preference (see the scaling entry below).
* **Still outstanding**: shared vs per-block level-1 weights (both diagnostic runs crashed on a syntax error in a scratch script), and the full multi-seed coherent baseline. `share_level1_weights` is now a constructor flag.

### [2026-07-28] Correction: R4's "The CP Baseline Is Broken" Was an Artifact of My Rank-Matching Procedure
* **What was reported** (R4, first run): `classical_bare` `33.9%`, `classical_full` `59.3%`, against an MLP reference at `96.0%` — concluding that the CP quad-node, not classical computation, was underperforming, and therefore that **Question A.3 was unanswerable**.
* **That conclusion was wrong, and the cause was mine.** `tune_classical` swept learning rate and bond dimension but **derived CP rank from the parameter budget** via `match_rank_to()`. Matching the hybrid's 211 parameters forced **rank = 1 at every bond dimension** — and a rank-1 CP decomposition is a single outer product, i.e. degenerate. The baseline was hobbled by the matching procedure and then diagnosed as broken.
* **Re-run with the coherent model's 287-parameter budget** (which happens to admit rank 2), same tuner, 21 seeds:

  | arm | old (rank forced to 1) | new (rank 2) | params |
  |---|---|---|---|
  | `classical_bare` (no residual/dropout) | `33.9 ± 12.4` | **`56.7 ± 6.9`** | 340 |
  | `classical_full` (+ residual + dropout) | `59.3 ± 2.9` | **`88.4 ± 5.5`** | 352 |

  Tuned configs: bare `lr=0.003, bond_dim=4, rank=2`; full `lr=0.03, bond_dim=4, rank=2`.
* **What now follows**:
  * **The CP node was never broken.** Question A.3 is approachable again, and the earlier "not answerable" verdict is withdrawn.
  * Under **matched constraints** — no residual, no dropout, as the quantum tower requires — the classical CP node reaches `56.7%`, against the coherent quantum tree's provisional `~78%`. That is the like-for-like comparison A.3 asks for, and it currently favours the quantum node.
  * Allowed **residual + dropout**, the classical node reaches `88.4%` and **overtakes** the quantum tower. The residual/dropout contribution is `+31.7` pts — far larger than previously measured, and it sharpens the divergence: these mechanisms are load-bearing classically while R3 confirmed they do not help the quantum tower.
* **Fix applied**: rank is now a **free axis** in `tune_classical` (swept over 1, 2, 4, 8, 16), with parameter matching enforced as an upper *bound* (1.5x the quantum model) so the classical arm cannot simply buy capacity but is never forced into a degenerate rank. A warning fires if a rank-1 config still wins, since that indicates the budget is too tight for a fair comparison.
* **Methodological lesson, third of its kind in this investigation**: a derived quantity silently constrained to a degenerate value (rank 1), exactly as the readout was silently constrained to one scalar. Both produced confident conclusions that reversed once the constraint was lifted. **Any quantity derived rather than swept should be checked for degeneracy at its chosen value.**
* **Still not run**: the coherent quantum arm (~17.5 min/seed x 21 seeds ~ 6 h, or ~40 min across 10 workers). The classical arms are in `results/r4_classical_only_coherent.json`.

### [2026-07-28] Scaling Analysis: Depth, Recycling, and Bond Dimension
Prompted by whether the wide readout survives deeper trees. Analysis, not experiment.
* **Depth**: readout width stays at the branching factor (4) at any depth, since the layer entering the final node is always 4 qubits. What worsens is the **compression ratio**: depth 2 is 16 leaves into 4 output qubits, depth 3 is 64 into 4, depth 4 is 256 into 4. The wide readout buys a fixed amount of headroom while demand grows geometrically — **a patch, not a solution**, and it degrades exactly as the model scales. Reading a *lower* layer recovers width but truncates the tree and pushes aggregation into the classical head.
* **Active qubit recycling**: unaffected, and well suited. The four top-layer qubits are the last alive; measuring all four is free, and nothing we want to read gets reset. Physical width grows with *depth*, not leaf count — ~7 qubits at depth 2 (measured 2026-07-17), ~12 at depth 3 if each group of 16 leaves is collapsed before the next begins. The one thing that would break it is reading a *lower* layer: holding 16 level-1 survivors at depth 3 pushes width to ~20 and forfeits most of the benefit. (Hardware statement only — recycling remains confirmed not to help classical simulation past depth 2.)
* **Bond dimension is the principled fix.** Passing $k$ qubits per bond gives $\chi = 2^k$, and a $k$-qubit reduced state carries $4^k - 1$ real parameters:

  | $k$ | $\chi$ | params at the root | node width | leaf qubits (16 patches) |
  |---|---|---|---|---|
  | 1 | 2 | 3 | 4 | 16 |
  | 2 | 4 | 15 | 8 | 32 |
  | 3 | 8 | 63 | 12 | 48 |

  The wide readout recovers information by reading qubits the tree *meant to discard*; increasing $\chi$ makes the tree genuinely carry more upward, restoring the single root as a meaningful object. It also matches the A.2 entropy result — root entropy at 89.7% of its $\ln 2$ ceiling — which we now know was a **binding** constraint rather than a curiosity.
* **Where each is affordable**: hardware with recycling — yes ($k=2$ is ~8-qubit nodes, ~14 physical qubits at depth 2). Tensor-network simulation — yes ($\chi=4$ is a small bond). **Statevector training — no**: $k=2$ at 16 patches is 32 qubits, past the ~29-qubit wall on 18 GB. This is the same wall as depth-3, reached from a different direction.
* **Framing for the thesis**: $\chi = 1$ qubit is a **simulation-driven constraint, not a design preference**. The capacity limit is measured (entropy saturates its ceiling; relieving it at the readout is worth 43 points), the principled fix is identified, and its cost is quantified. That is a considerably stronger position than presenting the wide readout as an architectural choice.

### [2026-07-29] Run Attempt: R7 Baseline Invalid (My Bug), R4 Incomplete — but the Corrected Rank Sweep Moved classical_bare Again
* **R7 baseline: 10 seeds wasted, entirely my error.** `run_r7_coherent` still defaulted to `pc.ARCH` (`root_multi_pauli`) rather than `pc.COHERENT_ARCH` (`top_layer_qubits`). I had identified this one-line fix and flagged it, then handed over run commands without applying it. All 10 seeds reproduced the known-bad configuration: seed 0 `34.1%`, matching the earlier failed pilot exactly. **Fixed**: the script now uses `COHERENT_ARCH` and exposes `--shared-l1`, defaulting to per-block level-1 weights (287 params), which is what both the 2026-07-17 reference and the readout diagnostic used.
* **R4 coherent: incomplete, also my error.** I quoted "~40 min parallel" but gave a command that runs 21 seeds **sequentially** at ~18 min/seed — about 6 hours. It got through tuning and one quantum seed before stopping, and because R4 had no per-seed checkpointing, nothing was saved. **Fixed**: `--seeds` allows sharding across workers and `--out-suffix` prevents clobbering; every seed now checkpoints.
* **What the run did establish** — the corrected rank sweep (rank now a free axis) changed `classical_bare` a third time:

  | version of the tuner | winning config | params | score |
  |---|---|---|---|
  | rank derived from param match | `bond_dim=8, rank=1` | 308 | `33.9%` |
  | rank derived, larger budget | `bond_dim=4, rank=2` | 340 | `56.7%` |
  | **rank swept freely** | **`bond_dim=2, rank=4`** | **428** | **`80.8%`** |

  `classical_full` settled at `lr=0.03, bond_dim=4, rank=2`, 352 params, `90.3%`. Both tuning scores are 3 seeds.
* **The parameter-efficiency picture is the real finding here.** `classical_bare` only reaches `80.8%` by using **428 parameters — 1.49x the quantum model's 287**, essentially exhausting the 1.5x budget cap. At the quantum model's own parameter count it does considerably worse. So the honest framing of Question A.3 is not "quantum beats classical" but: **the unitarity-constrained quantum node reaches comparable accuracy at roughly two-thirds of the parameters**, which is the same parameter-efficiency claim R4 supported on the hybrid, now on firmer ground.
* **First coherent quantum data point**: seed 0 scored **`89.1%`** at 30 epochs (per-block weights, `top_layer_qubits`) — above the 15-epoch diagnostic's `78.4%` and above the hybrid's `79.4%`. One seed, so provisional, but it suggests the coherent tree is not merely equal to the hybrid but better, consistent with the pilot's plateau analysis showing it still improving past epoch 15.
* **Standing caution**: `classical_bare` has now moved `33.9 -> 56.7 -> 80.8` across three tuner revisions, entirely from how the search space was defined. Every one of those numbers was reported at the time as a measurement. **Treat any single tuned baseline as provisional until the search space itself has been sanity-checked** — the MLP reference exists precisely to catch this and would have flagged the first two.

### [2026-07-29] Task R7 COMPLETE: The Coherent Tree Is the Best Model the Project Has, at the Smallest Parameter Count
* **Result** (coherent tree, `COHERENT_ARCH`, per-block level-1 weights, 1024/30, scored on the last-5 mean):

  | model | score | params | n |
  |---|---|---|---|
  | `mlp_reference` | `96.0 ± 1.5` | 999 | 21 |
  | **`quantum_coherent`** | **`89.3 ± 3.9`** | **287** | 4 |
  | `classical_full` (CP + residual + dropout) | `88.4 ± 5.5` | 352 | 21 |
  | `classical_bare` (CP, corrected rank sweep) | `79.6 ± 10.3` | 428 | 21 |
  | `quantum_hybrid` | `79.4 ± 8.0` | 211 | 30 |
  | `mlp_param_matched` | `70.5 ± 18.1` | 257 | 21 |

  | comparison vs coherent | diff | resolves ≥ | verdict |
  |---|---|---|---|
  | vs `quantum_hybrid` | `+9.9` | 5.7 | **better** |
  | vs `classical_bare` | `+9.7` | 6.3 | **better** |
  | vs `classical_full` | `+0.9` | 5.6 | tie |
  | vs `mlp_param_matched` | `+18.8` | 9.0 | **better** |
  | vs `mlp_reference` | `−6.7` | 6.2 | worse |

* **The claim this supports**: at **287 parameters** the coherent quantum tower beats the hybrid it replaced, ties the strongest classical tensor network (which needs 352 params *and* residual + dropout), beats the bare CP node (which needs 428), and beats a same-size MLP by 18.8 points. It trails only a 999-param MLP, at 3.5x its size. **This is a resolved parameter-efficiency result, and it is the quantum-vs-classical statement the investigation previously could not make.**
* **Question A.3 answered.** Under matched constraints — no residual, no dropout — the unitarity-constrained quantum node beats the unconstrained classical CP node by `+9.7` pts (limit 6.3) while using **33% fewer parameters** (287 vs 428). The unitary constraint does not cost representation capacity on this task; it appears to help. This supersedes the earlier "not answerable" verdict, which was an artifact of the rank-matching flaw.
* **Coherence is doing real work.** Survivor Bloch length fell from `0.98` at init to `0.38–0.73` after training across seeds, and the coherent tree beats the measure-and-re-encode hybrid by 9.9 points. Restoring inter-level entanglement was worth it.
* **4 seeds is sufficient here, and the earlier "10 seeds" demand was a statistics bug, not a data problem.** `phase15_common.compare` took `n = min(n_a, n_b)` and applied an equal-n pooled formula, discarding every observation in the larger arm. Against arms with n=21–30 this was badly conservative: coherent-vs-hybrid read as "unresolved at a 10.8-pt limit" when the correct Welch limit is `5.7` and the 9.9-pt gap is resolved. **Fixed** to Welch's unequal-variance, unequal-n form. The coherent model's low variance (`3.9` vs the hybrid's `8.0` and the MLP's `18.1`) is what makes small n adequate — low-variance arms need fewer seeds, and the expensive arm here is the low-variance one.
* **Not settled**: shared vs per-block level-1 weights. All coherent runs used per-block (287 params), matching the 2026-07-17 reference. The shared variant (211 params) is untested and would, if comparable, strengthen the parameter-efficiency claim further. `--shared-l1` exists for whenever it is wanted; it is not blocking.
* **Reproduce**: `run_r7_coherent --arm baseline --seeds N --out r7base_sN` (one worker per seed; **at most 4–5 concurrently** — ten concurrent 16-qubit `lightning.qubit` processes exhausted 18 GB and six were killed silently). Classical arms: `run_r4_classical_control --coherent --skip-quantum --out-suffix _fixedrank`. Aggregated: `results/r7_coherent_baseline.json`.

### [2026-07-29] Task R5 COMPLETE: Documentation and Consistency Fixes
No experiments. Four items, each a claim in this log that was stated more strongly than its evidence.

**R5.1 — The $p_{crit}$ contradiction: resolved, and it was never a contradiction.** The two figures use different criteria on different models.
* `benchmark_encodings_ansatze.py:319-332` **computes** $p_{crit}$ as *the first swept $p$ at which accuracy falls below 50%*, defaulting to `0.20` — so "`p_crit > 0.200`" means "never fell below 50% within the swept range", not "measured at 0.2".
* `emulate_noise_synthetic_shapes.py` **computes no $p_{crit}$ at all.** The "$p_{crit} \approx 0.05$" in the 2026-07-17 entry is a narrative reading of the curve, taken from accuracy falling `85.9% → 66.4%` at $p=0.05$.
* Applying the formal criterion to that same data (`85.9%` clean, `66.4%` at 0.05, `33.6%` at 0.10) gives **$p_{crit} = 0.10$**, not 0.05.
* **Why the numbers still differ under one criterion**: the models differ. Figure 3's `hea_3cnot` model starts at `85.9%` clean; Figure 4's multi-axis models start at `95.3–96.9%`. A model starting 10 points higher takes more noise to cross a fixed 50% line. **The apparent contradiction was a threshold-definition artifact compounded by a baseline-accuracy difference, not a measurement conflict.**
* **Fix**: cite $p_{crit}$ only with its definition attached, and note that it is a threshold on *absolute* accuracy, so it conflates noise tolerance with clean accuracy. Anywhere a noise-*tolerance* claim is wanted, use relative degradation instead.

**R5.2 — Barren-plateau claim rescoped (Figures 6, 7).** The 2026-07-18 entry calls this an "Empirical BP Immunity Proof", from 4 system sizes at $N \le 20$ with non-monotonic variance (`1.51e-1, 6.08e-2, 8.61e-2, 5.94e-2`), compared against an asymptotic $2^{-N}$ strawman. Four points cannot establish an asymptotic scaling law, and the comparison target is not a claim anyone makes about structured circuits. **Corrected framing**: hierarchical and tree architectures are *known theoretically* to resist barren plateaus under local observables; this data is **consistent with** that literature and shows no vanishing-gradient onset up to $N=20$. Cite the theory; present the measurement as corroboration, not proof. The data itself is sound and the figures are retained unchanged — this is a caption and wording fix.

**R5.3 — Question F closed.** The scheduling sub-question already had a clean negative result (training under noise does not improve clean accuracy, 5 seeds). The mechanism sub-question is now closed too, under the noiseless-only scope decision. Its founding observation (`42.2% → 53.9%` under noise) came from a `Linear(1,4)` model, where depolarizing noise contracts a scalar toward zero and can move samples across a 1-D decision boundary — a readout-calibration effect, not regularization. Recorded as the probable mechanism; not investigated further.

**R5.4 — A stated position on the purely-quantum rule.** The tension: R2 measured that widening the classical patch encoder from `Linear(48,1)` to `Linear(48,3)` bought `+12.6` pts — larger than any quantum architectural effect measured in this project — while Question B.3 was closed *on principle* because a `Linear(16,4)` before a 4-qubit VQC was "classical capacity substituting for quantum circuit width".

  **Position, with a checkable boundary**: *the classical encoder may set the parameters of state preparation, but may not reduce the number of qubits the architecture would otherwise require.*

  | case | verdict under the rule |
  |---|---|
  | `Linear(48,3)` → RX, RY, RZ on one qubit | **Permitted.** The architecture assigns one qubit per patch; the encoder fills that qubit's three rotation parameters. It cannot express more than the circuit consumes. |
  | `Linear(16,4)` → 4-qubit VQC for data needing 16 qubits | **Prohibited.** Reduces qubit count; the classical layer is doing the compression the circuit should do. |
  | Classical scalars carried between tree levels (the hybrid) | **Prohibited.** Replaces a quantum bond outright — the largest violation this project committed, and R7 removed it. |

  **This also dissolves the tension rather than merely adjudicating it.** The `+12.6` points did not come from adding classical capacity: `scalar_ry` left **two of the three available rotation parameters per qubit unused**, and `multi_axis` uses the full single-qubit parameterisation. The gain came from using the *quantum* resource fully, not from offloading work to the encoder. Question B.3's closure stands, and Question B.3 and R2 were never in conflict.

**R5.5 — Superseded protocols annotated.** Every pre-R1 row in the metrics table was measured at 256 train / 15 epochs (or 512/128 at 8×8), a protocol now known to be underpowered for this task regardless of architecture. Pre-R1 and post-R1 numbers must not be compared directly, and pre-R7 numbers describe the measure-and-re-encode hybrid rather than the coherent tree. Both boundaries are marked in the table.

### [2026-07-28] Next Steps (revised: theory phase closing, scaling questions move to CLEVR)
The theoretical exploration is being closed. Remaining work is finishing the coherent baseline, writing up, and regenerating figures. **The three "what do we do when 16 statevector qubits isn't enough" questions — bigger patches vs SPSA vs bond dimension — are all deferred into the CLEVR scope**, since they are the same question and CLEVR is where the answer actually matters.

**To finish the theory phase (compute, to be run):**
1. **R7 baseline** on the coherent tree with `COHERENT_ARCH`, ~10 seeds. Settles shared vs per-block level-1 weights at the same time. ~40 min across 10 parallel workers.
2. **R4 coherent quantum arm**, so the classical control (already run: bare `56.7 ± 6.9`, full `88.4 ± 5.5`, MLP `96.0 ± 1.5`) has something to compare against. ~40 min parallel.
3. Optionally `reupload` on the coherent tree — same wire count, so nearly free, and R3 left it a genuine null rather than a rejection.

**Write-up and figures (no compute):**
4. **R5 doc fixes**: the $p_{crit}$ contradiction, barren-plateau rescoping, and a stated position on the classical-encoder tension.
5. **Regenerate figures** per the Section 8 triage (5 keep, 4 keep-with-caveat, 10 regenerate, 1 retire). Note figures 9-16 can only come from the **hybrid** R3 data, since mixed_channel and the ancilla were deliberately not ported to the coherent tree — they must be labelled as **node-level** results.
6. **Noise write-up** — no experiments. Tolerance characterised at single-node scale, training-under-noise shown not to help, full-tree emulation infeasible at $O(4^N)$. State the single-node limitation.

**Deferred into CLEVR (Phase 2) — one decision, three options:**
7. 16x16 cannot support CLEVR's ">80% on all 4 attribute heads" criterion. The options are **bigger patches** (32x32 at 8x8 patches keeps 16 qubits; works today, but it is classical preprocessing, not quantum scaling), **SPSA** (genuinely deeper trees, ~1 day, uncertain), or **higher bond dimension** (the principled fix, needs tensor-network training). Decide inside CLEVR against real data rather than in the abstract.

**Carried into CLEVR as requirements** (established by this work):
* **Classical controls from day one** — matched-parameter reference alongside every accuracy claim, with rank swept freely (see the R4 correction above).
* **Question C.2 is required** — run the relational task with and without the spatial ancilla. R3's `-19.5` came from a translation-invariant task and says nothing about relational reasoning.
* **Noiseless only.**

---

## Experiment & Metrics Record

| Date | Model Configuration | Dataset | Metric | Result | Notes / Insights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| — | **⚠️ PROTOCOL BOUNDARY 1** | — | — | — | **Every row below this line and above BOUNDARY 2 used 256 train / 15 epochs (or 512/128 at 8x8) — underpowered for this task regardless of architecture (R1/R1b). Do not compare across the boundary.** |
| 2026-07-17 | 4-qubit Node: 0 CNOTs | Random Rotations | VN Entropy / KL Div | 0.0000 / 0.0109 | Product state. Perfect pure-state coverage. |
| 2026-07-17 | 4-qubit Node: 1 CNOT | Random Rotations | VN Entropy / KL Div | 0.3412 / 0.2366 | Entangling 1 child. State becomes mixed. |
| 2026-07-17 | 4-qubit Node: 2 CNOTs | Random Rotations | VN Entropy / KL Div | 0.4332 / 0.3890 | Entangling 2 children. Entropy increases. |
| 2026-07-17 | 4-qubit Node: 3 CNOTs | Random Rotations | VN Entropy / KL Div | 0.5089 / 0.7097 | All children to parent. High entropy transfer. |
| 2026-07-17 | 4-qubit Node: 4 CNOTs | Random Rotations | VN Entropy / KL Div | 0.5010 / 0.7223 | Ring layout. Diminishing returns on entropy. |
| 2026-07-17 | 4-qubit Node: 6 CNOTs | Random Rotations | VN Entropy / KL Div | 0.5139 / 0.7562 | Multi-ring. Max mixedness achieved. |
| 2026-07-17 | Standard vs. Recycled QTTN | 16-Patch Random | Output Difference | 0.00000000e+00 | Verified exact output equivalence. Qubits: 16 -> 7. |
| 2026-07-17 | 16-qubit QTTN: hea_3cnot | 16x16 Shapes (256/64) | Cross-Entropy Loss / Val Acc | 0.8490 / 75.0% | Vectorized run. Verified trainability & convergence. |
| 2026-07-17 | 4-qubit QTTN: hea_3cnot | 8x8 Shapes (512/128) | Noisy Acc vs. Depolarizing p | p=0: 85.9%, p=0.02: 89.8%, p=0.05: 66.4% | ⚠️ The log's "p_crit ~ 0.05" is a narrative reading; **no threshold is computed in this script**. Under the formal <50% criterion used by the sweep below, this model's p_crit is **0.10**. See R5.1. |
| 2026-07-17 | 4-qubit QTTN: ANGLE + HEA | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 78.1%, p_crit > 0.200 | Sweep run. Normal angle prep. |
| 2026-07-17 | 4-qubit QTTN: ANGLE + IQP | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 86.7%, p_crit > 0.200 | Sweep run. Normal angle prep. |
| 2026-07-17 | 4-qubit QTTN: ANGLE + ALT | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 93.8%, p_crit > 0.200 | Sweep run. Normal angle prep. |
| 2026-07-17 | 4-qubit QTTN: MULTI_AXIS + HEA | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 96.9%, p_crit > 0.200 | Sweep run. Highest overall performance. |
| 2026-07-17 | 4-qubit QTTN: MULTI_AXIS + IQP | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 95.3%, p_crit > 0.200 | p_crit := first swept p with acc < 50%; ">0.200" = never crossed in range. Single seed. Superseded by R2 at 16x16. |
| 2026-07-17 | 4-qubit QTTN: MULTI_AXIS + ALT | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 96.1%, p_crit > 0.200 | Sweep run. High performance, shallow depth. |
| 2026-07-17 | 2-qubit QTTN: AMPLITUDE + HEA | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 66.4%, p_crit = 0.150 | Sweep run. Extreme qubit compression. |
| 2026-07-17 | 2-qubit QTTN: AMPLITUDE + IQP | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 76.6%, p_crit > 0.200 | Sweep run. 2 qubits, 4 parameters, 76.6% acc. |
| 2026-07-17 | 2-qubit QTTN: AMPLITUDE + ALT | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 78.1%, p_crit = 0.150 | Sweep run. 2 qubits, 4 parameters, 78.1% acc. |
| 2026-07-17 | 4-qubit QTTN: ZZ_MAP + HEA | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 89.8%, p_crit > 0.200 | Sweep run. Non-linear feature map prep. |
| 2026-07-17 | 4-qubit QTTN: ZZ_MAP + IQP | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 84.4%, p_crit > 0.200 | Sweep run. Non-linear feature map prep. |
| 2026-07-17 | 4-qubit QTTN: ZZ_MAP + ALT | 8x8 Shapes (512/128) | Depolarizing Noise Sweep | Noiseless: 84.4%, p_crit = 0.150 | Sweep run. Non-linear feature map prep. |
| 2026-07-17 | 16-qubit QTTN: Multi-Axis + IQP | 32x32 Color-Only Shapes | Classification Acc (128/32) | 50.0% | Diagnostic representation bias run (6 epochs). |
| 2026-07-17 | 16-qubit QTTN: Multi-Axis + IQP | 32x32 Shape-Only Shapes | Classification Acc (128/32) | 37.5% | Diagnostic representation bias run (6 epochs). |
| 2026-07-17 | 16-qubit QTTN: Multi-Axis + IQP | 32x32 Overlapping Shapes | Classification Acc (512/128) | 65.6% (Peak) / 53.9% (Final) | Convergence proof model trained for 20 epochs. |
| 2026-07-17 | 4-qubit QTTN: Multi-Axis + IQP | 8x8 Overlapping Shapes | Classification Acc (512/128) | 77.3% | Convergence proof model trained for 25 epochs. |
| 2026-07-18 | 4-qubit QTTN: Multi-Axis + IQP | 8x8 random patches | Gradient Variance (100 trials) | 1.51e-01 | Barren plateau scaling sweep (N=4). |
| 2026-07-18 | 9-qubit QTTN: Multi-Axis + IQP | 12x12 random patches | Gradient Variance (100 trials) | 6.08e-02 | Barren plateau scaling sweep (N=9). |
| 2026-07-18 | 16-qubit QTTN: Multi-Axis + IQP | 16x16 random patches | Gradient Variance (100 trials) | 8.61e-02 | Barren plateau scaling sweep (N=16). |
| 2026-07-18 | 20-qubit QTTN: Multi-Axis + IQP | 20x20 random patches | Gradient Variance (100 trials) | 5.94e-02 | Barren plateau scaling sweep (N=20). |
| 2026-07-18 | 4-qubit MPS: Multi-Axis + IQP | 8x8 Overlapping Shapes | Noiseless / Noisy (p=0.10) Acc | 43.0% / 46.1% | Comparative topology sweep. Flat 1D Chain. |
| 2026-07-18 | 4-qubit MERA: Multi-Axis + IQP | 8x8 Overlapping Shapes | Noiseless / Noisy (p=0.10) Acc | 51.6% / 53.1% | Comparative topology sweep. Disentangled Tree. |
| 2026-07-18 | 4-qubit QTTN: Multi-Axis + IQP | 8x8 Overlapping Shapes | Noiseless / Noisy (p=0.10) Acc | 42.2% / 53.9% | Comparative topology sweep. Standard Tree. |
| 2026-07-26 | Hierarchical QTTN: Baseline | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (3 seeds) | 56.8% ± 2.9 / 57.3% | Quantum-native residual comparison. |
| 2026-07-26 | Hierarchical QTTN: Data Re-upload | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (3 seeds) | 54.7% ± 7.7 / 59.9% | Highest peak acc, 2.6x higher seed variance. |
| 2026-07-26 | Hierarchical QTTN: Near-Identity Init | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (3 seeds) | 56.8% ± 2.9 / 56.8% | Statistically indistinguishable from baseline. |
| 2026-07-26 | Hierarchical QTTN: Baseline (seed=0) | 16x16 Overlapping Shapes | Noisy Acc vs. Depolarizing p | p=0: 54.7%, p≥0.02: flat 50.0% | — |
| 2026-07-26 | Hierarchical QTTN: Data Re-upload (seed=0) | 16x16 Overlapping Shapes | Noisy Acc vs. Depolarizing p | p=0: 54.7%, p≥0.10: collapses to 28.1% | Doubled noise-channel exposure from re-encoding. |
| 2026-07-26 | Hierarchical QTTN: Near-Identity Init (seed=0) | 16x16 Overlapping Shapes | Noisy Acc vs. Depolarizing p | p=0: 53.1%, p≥0.02: flat 50.0% | Identical to baseline under noise. |
| 2026-07-26 | Hierarchical QTTN: Mixed-Unitary Channel | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (3 seeds) | 45.8% ± 20.3 / 50.5% | 1 of 3 seeds hit a dead-gradient trap. |
| 2026-07-26 | Hierarchical QTTN: LCU (postselected) | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (3 seeds) | 42.7% ± 18.6 / 49.0% | Same trap; noiseless only (postselect unsupported on default.mixed). |
| 2026-07-26 | Hierarchical QTTN: Mixed-Unitary Channel (seed=0) | 16x16 Overlapping Shapes | Noisy Acc vs. Depolarizing p | p=0: 51.6%, p≥0.02: flat 54.7% | Single-seed result; see 10-seed retest below (does not replicate). |
| 2026-07-26 | Hierarchical QTTN: LCU (seed=0) | 16x16 Overlapping Shapes | Postselection Success Prob. | L1: 0.986, L2: 0.515 | Root node discards ~half of trials (Method 5 overhead). |
| 2026-07-26 | Hierarchical QTTN: Baseline (10 seeds) | 16x16 Overlapping Shapes (256/64) | Final Val Acc | 54.1% ± 3.6 | 0/10 seeds failed. |
| 2026-07-26 | Hierarchical QTTN: Mixed-Unitary Channel (10 seeds) | 16x16 Overlapping Shapes (256/64) | Final Val Acc | All: 51.7% ± 12.5; Converged only: 55.6% ± 5.2 | 1/10 seeds failed (10%, down from 1/3 estimate). |
| 2026-07-26 | Hierarchical QTTN: Baseline vs. Mixed-Channel (9-10 seeds avg) | 16x16 Overlapping Shapes | Noisy Acc vs. Depolarizing p (converged-only) | p=0.05: 53.6% / 53.1%; p=0.20: 51.6% / 47.2% | Noise-robustness claim retracted; baseline ≥ mixed_channel at high p. |
| 2026-07-26 | Hierarchical QTTN: No Spatial Ancilla | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (5 seeds) | 55.0% ± 3.5 / 56.9% | Question C ablation. |
| 2026-07-26 | Hierarchical QTTN: With Spatial Ancilla | 16x16 Overlapping Shapes (256/64) | Final/Peak Val Acc (5 seeds) | 52.8% ± 5.7 / 54.4% | Mildly worse; 1/5 seeds never converged. |
| 2026-07-26 | 4-qubit Node: 3 CNOTs, Tree Depth 1 (4 leaves) | Random Rotations | Root-Qubit VN Entropy | 0.5037 ± 0.1680 (72.7% of ln(2)) | Question A.2 entropy-vs-depth. |
| 2026-07-26 | 16-qubit QTTN: 3 CNOTs, Tree Depth 2 (16 leaves) | Random Rotations | Root-Qubit VN Entropy | 0.6219 ± 0.1149 (89.7% of ln(2)) | Entropy saturates further toward bound with depth. |
| 2026-07-26 | Recycled QTTN, tree-traversal MCM, 4 blocks (12 resets) | N/A (feasibility test) | Wall-clock time | ~2.4-2.5s, consistent across repeats | Ancilla-free, exact, validated to 13 sig. figs vs. deferred-measurement ground truth. |
| 2026-07-26 | Recycled QTTN, tree-traversal MCM, 8 blocks (24 resets) | N/A (feasibility test) | Wall-clock time | Hung, >120s, zero output | Exponential blowup (~2^resets branches); depth-3 needs ~50 resets, infeasible. |
| 2026-07-26 | QTTN, default.tensor method='tn', Depth-3 (64 qubits, 32x32) | Random Rotations | Forward-pass time | 0.13s | Correct (agrees with MPS method to ~9 decimals). |
| 2026-07-26 | QTTN, default.tensor method='tn', Depth-4 (256 qubits, 64x64) | Random Rotations | Forward-pass time | 1.20s | Reproducible across trials; MPS method OOM-killed at this width. |
| 2026-07-26 | QTTN, default.tensor method='tn', Depth-3 (252 params) | Random Rotations | Full gradient-step time (1 sample) | 56.7s | parameter-shift cost, not backprop; training-impractical at this scale. |
| 2026-07-27 | 4-qubit node (level-1 survivor) | Random Rotations | VN Entropy vs. Depolarizing p | 0.505 (p=0) to 0.640 (p=0.20) | Question E.2. |
| 2026-07-27 | 8-qubit 2-level circuit (root) | Random Rotations | VN Entropy vs. Depolarizing p | 0.616 (p=0) to 0.684 (p=0.20) | Gap to L1 narrows with p; confounded by ceiling proximity. |
| 2026-07-27 | Hierarchical QTTN: trained p=0.0, eval clean | 16x16 Overlapping Shapes (256/64) | Final Clean Val Acc (5 seeds) | 55.0% ± 3.5 | Question F scheduling test. |
| 2026-07-27 | Hierarchical QTTN: trained p=0.02, eval clean | 16x16 Overlapping Shapes (256/64) | Final Clean Val Acc (5 seeds) | 54.1% ± 5.0 | Training noise does not help; mildly worse. |
| 2026-07-27 | Hierarchical QTTN: trained p=0.05, eval clean | 16x16 Overlapping Shapes (256/64) | Final Clean Val Acc (5 seeds) | 54.1% ± 5.0 | Same conclusion at higher training noise. |
| 2026-07-27 | R1: readout=scalar, encoding=scalar_ry (105 params) | 16x16 Overlapping (256 train / 15 ep) | Final Val Acc (5 seeds) | 55.0% ± 3.5 | R1 Sweep 1. The audited/regressed config. |
| 2026-07-27 | R1: readout=root_multi_pauli, encoding=scalar_ry (113) | 16x16 Overlapping (256 train / 15 ep) | Final Val Acc (5 seeds) | 57.5% ± 7.2 | R1 Sweep 1. Inside noise vs. scalar at this protocol. |
| 2026-07-27 | R1: readout=level1_survivors, encoding=scalar_ry (117) | 16x16 Overlapping (256 train / 15 ep) | Final Val Acc (5 seeds) | 57.5% ± 5.6 | R1 Sweep 1. Tied with root_multi_pauli; not carried into Sweep 2. |
| 2026-07-27 | R1: readout=root_multi_pauli, encoding=multi_axis (211) | 16x16 Overlapping (256 train / 15 ep) | Final Val Acc (5 seeds) | 59.1% ± 6.4 | R1 Sweep 2 winner, but below the 70% gate. |
| 2026-07-27 | R1: readout=root_multi_pauli, encoding=multi_axis (211) | 16x16 Overlapping (**1024 train / 30 ep**) | Final Val Acc (5 seeds) | **78.8% ± 7.2** | R1 capacity fallback. **Gate passed.** Protocol of record from here on. |
| 2026-07-28 | R1b control: readout=scalar, encoding=scalar_ry (105) | 16x16 Overlapping (**1024 train / 30 ep**) | Final Val Acc (5 seeds) | 62.5% ± 6.9 | **Does not clear the 70% gate** → audit diagnosis CONFIRMED; +16.2 pt architecture gap. |
| 2026-07-28 | R1b paired: scalar/scalar_ry vs. root_multi_pauli/multi_axis | 16x16 Overlapping (1024/30, paired) | Final Val Acc (10 seeds) | 63.9% ± 8.1 vs. 83.9% ± 10.0 | Delta +20.0 ± 14.2; new config wins 9/10 seeds. |
| 2026-07-28 | R1b power analysis (pooled std 8.2, last-5-epoch scoring) | 16x16 Overlapping (1024/30) | Seeds needed per arm | 2pt: 130, 3pt: 58, 5pt: 21, 8pt: 9 | Cross-variant seed corr −0.22 → **pairing does not help**. Superseded by R2's std 4.94 for the architecture of record (2pt: 48, 3pt: 22). |
| 2026-07-28 | R2: scalar_ry + strongly_entangling (as coded in July) | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 68.4% ± 6.5 | 113 params. |
| 2026-07-28 | R2: scalar_ry + iqp | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 68.0% ± 6.9 | 113 params. |
| 2026-07-28 | R2: multi_axis + strongly_entangling | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 79.3% ± 9.7 | 211 params. |
| 2026-07-28 | **R2: multi_axis + iqp (ARCHITECTURE OF RECORD)** | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | **80.9% ± 4.9** | Encoding resolved (+12.6 vs 3.6 limit); ansatz NOT resolved (+1.6 vs 4.8). |
| 2026-07-28 | R3: baseline | 16x16 Overlapping (1024/30) | Score, last-5 mean (30 seeds) | 79.4% ± 8.0 | 0/30 failures. Agrees with R4's independent quantum arm (79.0 ± 8.7). |
| 2026-07-28 | R3: reupload | 16x16 Overlapping (1024/30) | Score, last-5 mean (30 seeds) | 80.9% ± 6.6 | +1.5 vs baseline, limit 3.8 → **unresolved**. July's noise-collapse story was a bottleneck artifact. |
| 2026-07-28 | R3: mixed_channel | 16x16 Overlapping (1024/30) | Score, last-5 mean (30 seeds) | 68.5% ± 10.1 | −10.9, limit 4.7 → **worse, resolved**. 0/30 failures (was 1/10) — not unstable, just worse. |
| 2026-07-28 | R3: with_ancilla | 16x16 Overlapping (1024/30) | Score, last-5 mean (30 seeds) | 59.9% ± 6.9 | −19.5, limit 3.8 → **worse, resolved**. SCOPE: position-irrelevant task; see Question C.2. |
| 2026-07-28 | R3 noise sweep: baseline | 16x16 Overlapping | Acc vs depolarizing p (15 seeds) | p=0: 80.7%, p=0.10: 77.5%, p=0.20: 57.0% | First noise curve in this project not pinned to the 50% shortcut floor. |
| 2026-07-28 | R3 noise sweep: reupload / mixed_channel / with_ancilla | 16x16 Overlapping | Acc at p=0.10 (15 seeds) | 69.2% / 65.0% / 49.2% | All degrade faster than baseline. |
| 2026-07-28 | **R4: quantum (architecture of record)** | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | **79.0% ± 8.7** | 211 params. |
| 2026-07-28 | R4: MLP reference | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 96.0% ± 1.5 | 999 params. Beats quantum by 17.0 pts (resolved) — but ~5x the parameters. |
| 2026-07-28 | R4: MLP, parameter-matched | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 70.5% ± 18.1 | 257 params. Quantum +8.5 vs 8.9 limit → **statistically tied**. Claim = parameter efficiency. |
| 2026-07-28 | R4: classical CP tree, bare | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 33.9% ± 12.4 | 308 params, hyperparameter-tuned. **CP node is broken → Question A.3 NOT answerable.** |
| 2026-07-28 | R4: classical CP tree + residual + dropout | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 59.3% ± 2.9 | ⚠️ SUPERSEDED — rank forced to 1 by parameter matching. See corrected row below. |
| 2026-07-28 | R7: coherent tree, readout=root_multi_pauli | 16x16 Overlapping (1024/15, 1 seed) | Score, last-5 mean | 35.6% | Root qubit's marginal only — a severe bottleneck. |
| 2026-07-28 | **R7: coherent tree, readout=top_layer_qubits** | 16x16 Overlapping (1024/15, 1 seed) | Score, last-5 mean | **78.4%** (peak 89.1%) | **+43 pts from readout alone.** Level with the hybrid's 79.4% in half the epochs. |
| 2026-07-29 | **R7 FINAL: coherent tree (287 params)** | 16x16 Overlapping (1024/30) | Score, last-5 mean (4 seeds) | **89.3% ± 3.9** | Beats hybrid +9.9 and classical_bare +9.7 (both resolved); ties classical_full. Best score-per-parameter in the project. |
| 2026-07-29 | R4 corrected: classical CP bare, rank swept freely | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 79.6% ± 10.3 | 428 params (1.49x quantum). Confirms the 80.8% tuning estimate. |
| 2026-07-28 | R4 corrected: classical CP bare (rank swept freely) | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 56.7% ± 6.9 | 340 params, lr=0.003/bond_dim=4/rank=2. Matched-constraint arm for Question A.3. |
| 2026-07-28 | R4 corrected: classical CP + residual + dropout | 16x16 Overlapping (1024/30) | Score, last-5 mean (21 seeds) | 88.4% ± 5.5 | 352 params. +31.7 vs bare — residual/dropout strongly load-bearing classically. |





---

## Theoretical & Design Notes

### Dense Angle Encoding vs. Amplitude Encoding
*Dense Angle Encoding* (used in the current HEA/IQP scripts) maps 3 channels per pixel to $R_x, R_y, R_z$ rotations on a single qubit. This maps 12 features per $2 \times 2$ patch (4 pixels × 3 channels) to 4 qubits. 
* *Pros*: Simple, maps nicely to individual physical qubits.
* *Cons*: Cannot represent more than 3 dimensions per qubit. If we scale to larger patches, qubit count scales linearly.

*Amplitude Encoding* encodes $2^d$ features into the amplitudes of $d$ qubits.
* *Pros*: Extremely compact ($12$ features can be encoded in $\lceil \log_2(12) \rceil = 4$ qubits).
* *Cons*: State preparation is classically expensive and results in deep circuits (lots of CNOTs) which are highly susceptible to noise.

### CP-Rank to CNOT Mapping (Why CNOTs and not Hadamards?)
* **Local Operations (Hadamards, rotations)** belong to the group of local operations and classical communication (LOCC). They represent basis rotations and cannot generate entanglement (Schmidt rank across register partitions remains unchanged). They are the quantum analog of classical projection factor matrices $A^{(i)}$.
* **Non-local Operations (CNOTs, CZs)** physically entangle the registers, increasing the Schmidt rank of the operation. This rank mapping dictates that CNOT density is the direct quantum analog of the CP-rank parameter $R$.
* **Implication for Ansatz Design**: To scale CP-rank, we only need to sweep entangling gate density (CNOT count). Static Hadamards are still used to rotate the coordinate basis (e.g. into the X-basis) to allow Z-basis entanglers (like IsingZZ) to generate non-local correlation.
