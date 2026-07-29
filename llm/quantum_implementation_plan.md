## Implementation and Experimentation Plan

### Architecture Summary (What We're Building)

> **⚠️ The diagram below predates the quantum investigation and is superseded. The image tower as built and validated (Task R7, 2026-07-29) is described here; the corrected bullet list follows the diagram.**
>
> ### Image tower as built — the coherent quantum tree
>
> ```
> Image (16×16)  →  16 patches of 4×4×3
>        │
>        ▼   Linear(48, 3) per patch, tanh·π          [classical I/O boundary]
>   3 rotation angles per patch
>        │
>        ▼   ONE 16-qubit device; patch p on wire p
>   RX,RY,RZ per wire                                 [multi-axis encoding]
>        │
>        ▼   Level-1: IQP block unitaries on
>            [0-3] [4-7] [8-11] [12-15], per-block weights
>   survivors 0, 4, 8, 12 stay QUANTUM (no measurement)
>        │
>        ▼   Level-2: IQP unitary on [0, 4, 8, 12]
>        │
>        ▼   ⟨Z⟩ on wires 0, 4, 8, 12                 [readout = top_layer_qubits]
>   Linear(4, n_classes)                              [classical I/O boundary]
> ```
>
> **287 parameters. `89.3% ± 3.9` on 16×16 synthetic shapes** (4 seeds, 1024 train / 64 test / 30 epochs, scored on the last-5-epoch mean). Beats the measure-and-re-encode hybrid it replaced by `+9.9` pts and a matched classical CP tree by `+9.7` (both resolved); ties `classical_full` (88.4%) while using 65 fewer parameters and no residual or dropout; beats a same-size MLP by `+18.8`. Runs on `lightning.qubit` with adjoint differentiation, ~17.5 min per 30-epoch run.
>
> Two properties are load-bearing and were each established the hard way:
> * **The tree is coherent.** Survivors are passed as *qubits*, never measured mid-tree. Enforced by a regression test: a survivor's Bloch vector must be shorter than 1 (measured `0.38–0.73` after training, vs exactly 1 for a product state).
> * **The readout is four wires, not one.** Reading only the root qubit's Bloch vector scores `35.6%`; reading the four top-layer wires scores `78.4%` — a 43-point gap, because the root is one qubit's marginal of a 16-qubit state. This means the tower does **not** contract to a single root, and that is a documented workaround for a simulation limit (χ=1 bonds), not a design preference.

```
═══════════════════════════════════════════════════
TEXT TOWER                              IMAGE TOWER
═══════════════════════════════════════════════════
Tokens                              Image (64×64) [see image-size note]
  │                                      │
  ▼                                      ▼
lambeq parse → DisCoCat diagram     Bilinear patch embedding
  │           (variable topology)   4×4 patches → 256 patches
  ▼                                 [NO positional encoding --
EinsumModel contraction              rejected, see below]
(cotengra-optimised einsum)              │
  │                                      ▼
  │                                 TTN Level 0: CP-rank-32
  │                                      │
  │                                 TTN Level 1: CP-rank-32
  │                                      │
  │                                 TTN Level 2: CP-rank-32
  │                                      │
  │                                 TTN Level 3: CP-rank-32
  │                                      │
  ▼                                      ▼
Text embedding vector              Image embedding vector
  │                                      │
  ▼                                      ▼
Text Alignment Head                Image Alignment Head
(Linear + L2-norm)                 (Linear + L2-norm)
  │                                      │
  └────────────────┬────────────────────┘
                   ▼
       Shared Contrastive Space (embedding_dim, L2-normed)
                   │
                   ▼
       InfoNCE Loss + Triplet Loss (triplet_weight = 40000)
       One positive + one hard negative (ARO) per sample
```

Key design decisions:

- **Different alignment heads** for each tower (text-specific and image-specific)
- **No shared projection weights** between towers
- **CP-rank-32** at every TTN level in the image tower (this describes the classical CP-decomposition analogue; the quantum ansatz uses a 3-CNOT "all children to parent" entangling pattern per node instead — see `research_log.md` 2026-07-17 CNOT-to-entropy diagnostic)
- **Bilinear patch embedding** separates color and spatial structure (Hadamard product)
- ~~**Gated positional encoding** with learnable scale initialised near 0 (≈0.05)~~ — **REJECTED for this task class; re-validated 2026-07-28 (Task R3).** At the architecture of record, 30 seeds: `59.9% ± 6.9` with the ancilla vs `79.4% ± 8.0` without — `−19.5` pts against a `3.8`-pt resolution limit. (The original 2026-07-26 verdict, `52.8 ± 5.7` vs `55.0 ± 3.5` at n=5, was inside noise and did not actually support its conclusion.)
  **⚠️ Scope — do not generalise this to "positional encoding is harmful".** The synthetic-shapes task is translation-invariant single-object classification, where position barely affects the label, so an explicit position mechanism has nothing to contribute and a negative result is close to structurally guaranteed. It also tests a **per-quadrant** ancilla (4 positions), not the original **per-patch** design (16). **Question C.2 — whether implicit tree topology suffices for *relational* reasoning — is untested and is now a required part of CLEVR**: run the left/right/front/behind task with and without the ancilla, since that is the only place position is the label. See `quantum_investigation_roadmap.md` Question C.
- **⚠️ CLEVR resolution warning (2026-07-27)**: CLEVR images are 480×320. Downsampled to 16×16, an object occupies a handful of pixels — `material` (rubber vs. metal, essentially specular-highlight detection) and `size` become close to unlearnable. **Step 2's pass criterion below (>80% on all 4 attribute heads) is not reachable at 16×16.** Either Task 1.5/R6 (SPSA) lands and CLEVR runs at 32×32, or the task must be explicitly re-scoped (e.g. drop `material`, and state in the thesis that resolution, not architecture, is the limiting factor). Do not silently run at 16×16 against a criterion that resolution cannot support. See `quantum_investigation_roadmap.md` Section 7, task R6 and the decision gate.
- **Image size**: 16×16 (16 patches → 16 qubits) is the validated size. Deeper trees need more qubits than statevector simulation allows — a depth-3 quad-tree is 64 qubits, and the practical ceiling on 18 GB is ~29 (~24 with batch broadcasting). **Three routes exist and the choice is deferred into CLEVR**, since they are one question best answered against real data: (a) **bigger patches** — 32×32 at 8×8 patches keeps 16 qubits and works today, but the extra pixels are absorbed by the classical encoder, so it is resolution scaling, not quantum scaling; (b) **SPSA** on `default.tensor`, which enables genuinely deeper trees; (c) **higher bond dimension** (χ=2^k), the principled fix for the readout bottleneck, affordable on hardware and under tensor-network simulation but not for statevector training.
- **Triplet loss dominates** (weight 40000); hard negatives are curated ARO syntactic perturbations
- **Three forward passes per sample:** image × 1, true caption × 1, false caption × 1
- **DisCoCat text tower** (EinsumModel): sentence structure determines diagram topology, not sequence length

> **NOTE — Non-Linear Contractions:** We will **never** use non-linear contractions (NLC) in this quantum implementation. All tensor network contractions are strictly linear. Do not add GELU activations, learnable gates, or any non-linearity between contraction steps. The EinsumModel's `non_linear_contractions` flag must always be `False`.

> **NOTE — NaN / Infeasibility Handling:** This quantum implementation does **not** require NaN/infeasibility handling. Unlike the classical EinsumModel which can encounter OOM-infeasible contractions and returns NaN as a sentinel, quantum circuits have fixed topology and bounded resource usage. There is no need for NaN guards, `drop_nonfinite_rows`, or the `_safe_text_embed` pattern — omit all of this machinery.

> **NOTE — Classical Hybrid Shortcuts:** This project's direction is a purely quantum implementation. Do **not** insert classical components inside the core quantum pipeline to substitute for quantum circuit capacity — e.g. a classical compression layer before a variational circuit to reduce the qubit width it would otherwise need (`qnlp/image_tower/classification/quantum/hybrid_trainer.py`'s `Linear(16,4)` before a 4-qubit VQC is the rejected example; that file is deprecated as an architecture direction, 2026-07-26). Qubit-budget problems should be solved by genuinely quantum means (active qubit recycling, tensor-network simulators), not by offloading work to classical layers. The only classical components that belong in these models are the unavoidable I/O boundary: encoding raw pixel/token data into rotation angles at the start, and projecting measured expectation values into class logits or an embedding space at the end — both are necessary interfaces, not research targets, and should stay as minimal as possible (a single `Linear` layer, no additional capacity). Do not propose experiments that tune, grow, or characterize these boundary layers as though they were part of the quantum architecture question; close such proposals on principle rather than running them. See `llm/quantum_investigation_roadmap.md` Section 6 (backlog items 5 and 7, both closed 2026-07-26) for the specific questions this ruled out.
>
> **CHECKABLE BOUNDARY (added 2026-07-29, Task R5.4)** — the rule above was previously enforced by judgement, which let a larger violation through unnoticed. The boundary is: **the classical encoder may set the parameters of state preparation, but may not reduce the number of qubits the architecture would otherwise require.**
>
> | case | verdict |
> |---|---|
> | `Linear(48,3)` → RX, RY, RZ on one qubit | **Permitted** — the architecture assigns one qubit per patch; the encoder fills that qubit's three rotation parameters and cannot express more than the circuit consumes. |
> | `Linear(16,4)` → 4-qubit VQC for data needing 16 qubits | **Prohibited** — reduces qubit count. This is `hybrid_trainer.py`, rejected. |
> | Classical scalars carried between tree levels | **Prohibited** — replaces a quantum bond outright. This was the hybrid tree, the largest violation this project committed; R7 removed it. |
>
> This also **dissolves** the apparent conflict with R2, which measured `+12.6` pts from widening the encoder. That gain was not added classical capacity: `scalar_ry` left two of the three available rotation parameters per qubit unused, and `multi_axis` uses the full single-qubit parameterisation. The improvement came from using the **quantum** resource fully.

> **✅ CODE AUDIT #2 — RESOLVED by Task R7 (2026-07-29).** The finding, retained because the failure mode is instructive: **the trained model was not a coherent quantum tree.** Every `QuantumNode` builds its own 4-qubit device; level-1 nodes return a single `⟨Z⟩` and those **classical scalars** are re-encoded as rotation angles into a separate root circuit. There is **zero entanglement across tree levels**, and the inter-level bond is one real scalar rather than the $\chi=2$ qubit the design specifies. The coherent version exists (`train_synthetic_shapes.py`, 2026-07-17, one 16-qubit device, survivors passed as qubits) and was replaced undocumented — the same regression pattern as the scalar readout. **R7 ported back to it and the coherent tree is now the model of record**, scoring `89.3%` against the hybrid's `79.4%` — restoring inter-level entanglement was worth `+9.9` pts. "Quantum native throughout" is now true of what has been trained, not only of the design. The node-level ablations below were measured on the hybrid and stand as **node-level** results; `mixed_channel` and the spatial ancilla were deliberately not ported to the coherent tree (each ancilla doubles the statevector, and both were rejected decisively).
>
> **⚠️ SCOPE (2026-07-28): the project proceeds NOISELESS ONLY.** Task R8 was **dropped without an experiment** (2026-07-28): tolerance is already characterised, training under noise was shown not to help, and full-tree emulation is infeasible. Phase 3 (ZNE/Mitiq, optimizer-under-noise, CLEVR noise calibration) is closed out of scope, and CLEVR is run without noisy emulation. Standing limitation to state in the thesis: all noise results characterise a single 4–5 qubit node, not the tree, because `default.mixed` is $O(4^N)$ and a 16-qubit noisy circuit needs ~68GB.
>
> **✅ NOTE STATUS UPDATE (2026-07-28): the residual and positional-encoding rules are RE-VALIDATED and binding again.** They were downgraded to provisional on 2026-07-27 after a code audit found every supporting experiment ran through a scalar-readout bottleneck (`nn.Linear(1, 4)` — the whole image compressed to one number, all results pinned in a `50–57%` band against the `50.0%` single-attribute shortcut ceiling). Task R3 re-ran them at the architecture of record with 30 seeds:
>
> * **Residuals**: `mixed_channel` `−10.9` pts vs baseline (limit `4.7`) — rejection confirmed on positive evidence. `reupload` `+1.5` (limit `3.8`) — a true null; state it as "no demonstrated benefit at effects ≥ 3.8 pts", **not** as harmful.
> * **Positional encoding**: `−19.5` pts (limit `3.8`) — confirmed for this task class, with the scope caveat in the bullet above.
>
> **Two justifications in the residual NOTE below are now known to be bottleneck artifacts and must not be repeated**: mixed-channel's "~10% training-failure rate" (**0/30** failures at restored capacity — it is not unstable, just worse) and re-uploading's "collapses under noise to 28.1%" (does not reproduce; it degrades comparably to baseline). The rejections stand; those two reasons do not.
>
> The **Classical Hybrid Shortcuts** NOTE remains **UNRESOLVED**, for a new reason: R2 measured that widening the classical patch encoder from `Linear(48,1)` to `Linear(48,3)` bought `+12.6` pts — a larger effect than any quantum architectural change measured anywhere in this project. The rule needs a quantitative boundary or an explicit defence. See `quantum_investigation_roadmap.md` §8.5 item 1 and backlog item 17.
>
> **NOTE — Residual Connections:** Do **not** add residual/skip connections to any quantum tree node, in this image tower or in any future quantum model in this project. Five quantum-native mechanisms were tested and rejected (2026-07-26, see `llm/research_log.md` and `llm/quantum_investigation_roadmap.md` Section 5): data re-uploading (higher noiseless peak accuracy but collapses under depolarizing noise — worse than no residual at $p \ge 0.10$), near-identity ansatz init (no measurable effect), mixed-unitary channel (small noiseless edge but a confirmed ~10% training-failure rate and no noise-robustness benefit once tested on 10 seeds — an earlier single-seed result claiming a benefit did not replicate), and LCU/LCU-lite (inherits the same instability, adds real shot overhead, and could only be tested noiseless due to a `default.mixed` postselection limitation). Unlike classical CP-tensor layers (which do use an explicit residual, see `qnlp/discoviz/models/cp_node.py`), quantum tree nodes here should use a plain ansatz with no skip-connection mechanism. This is a closed investigation, not an open question — do not re-propose residual connections without a genuinely new hypothesis distinct from the five already ruled out.

---

## Iterative Build Plan

Every step follows a mandatory two-stage execution protocol before moving on:

> **⚠️ The original two-stage (Simulation → Emulation) protocol is superseded.** Per the 2026-07-28 scope decision the project proceeds **noiseless only**, so Stage 2 is removed from every step below. Noise tolerance is already characterised at single-node scale (stable to p≈0.02, degrading past p≈0.05), training under noise was shown not to help, and full-tree emulation is infeasible: `default.mixed` costs $O(4^N)$ and a 16-qubit noisy circuit needs ~68 GB.

**Every step now requires, before moving on:**
1. **Noiseless simulation** — the circuit trains, gradients reach every parameter, and accuracy clears the stated criterion.
2. **A matched-parameter classical control**, reported alongside. Established by R4: without it, "our model scores X%" is uninterpretable, and a mis-specified baseline once produced a fake 57.8-point quantum-advantage result.
3. **A stated resolution limit** for any comparison. "No significant difference" without the minimum detectable effect is not a finding.

---

### ✅ Step 1: Image Tower — Synthetic Shapes (16×16) — COMPLETE 2026-07-29

**Task:** Classify synthetic 16×16 images into N classes (color × shape combinations).  
**Why:** Fully controlled dataset, fast iteration, minimal qubit budget. Validates the quantum TTN circuit structure and CP layer design in isolation before touching real data.

**Result**: `89.3% ± 3.9` (4 seeds), 287 params — above every matched control except a 999-param MLP. Question A.3 answered: under matched constraints (no residual, no dropout) the unitarity-constrained quantum node beats an unconstrained classical CP node by `+9.7` pts while using 33% fewer parameters.

**Also settled here**: no residual connections (five mechanisms rejected), no spatial ancilla *for translation-invariant classification*, encoding resolved (multi-axis) but ansatz not (IQP vs StronglyEntangling is inside noise), and noise closed at single-node scale.

---

### Step 2: Image Tower — CLEVR Single-Object Attribute Classification

**Dataset:** `dpdl-benchmark/clevr` on HuggingFace.

```python
import polars as pl
splits = {'train': 'data/train-*.parquet', 'test': 'data/test-*.parquet'}
df = pl.read_parquet('hf://datasets/dpdl-benchmark/clevr/' + splits['train'])
```

Each row has per-object attribute arrays (`color`, `shape`, `material`, `size`, `3d_coords`, `pixel_coords`, `rotation`) where index `i` corresponds to the i-th object in the scene.

**Task:** Filter to scenes with exactly one object. Classify that object by its full attribute tuple: `color × shape × material × size`, as 4 independent prediction heads.  
**Why:** Directly tests compositional attribute binding — the same capacity that ARO contrastive tasks require. A model that can discriminate "large red rubber cube" from "small red metal sphere" has the representational dimensions needed for VLM tasks. Synthetic images suit the TTN spatial hierarchy (geometric structure, no texture shortcuts). Coarse classification benchmarks like MNIST or CIFAR do not test this capacity.

**Pass criteria:** All 4 attribute heads reach >80% accuracy in simulation.

- Simulation (noiseless): all 4 attribute heads converge; per-attribute accuracy >80%
- **Matched-parameter classical control alongside every head** (required, R4)
- **Resolution limit reported** for every comparison (required, R1b)
- ~~Stage 2 — Emulation~~ — **removed 2026-07-28**: CLEVR is run noiseless (scope decision). See `quantum_investigation_roadmap.md` Task R8.

---

### Step 3: Image Tower — CLEVR Multi-Object (Relational)

**Dataset:** Same as Step 2 (`dpdl-benchmark/clevr`). Filter to scenes with exactly two objects.

Spatial relations are derived from `3d_coords` — no additional annotations needed:

```python
def get_spatial_relation(obj_a_coords, obj_b_coords):
    ax, ay = obj_a_coords[0], obj_a_coords[1]
    bx, by = obj_b_coords[0], obj_b_coords[1]
    if abs(ax - bx) > abs(ay - by):
        return "left" if ax < bx else "right"
    else:
        return "front" if ay < by else "behind"
```

**Task:** Predict the spatial relation between the two objects (left/right/in-front/behind) in addition to the attribute tuple of each object.  
**Why:** Tests whether the TTN encodes spatial structure and object relations, not just local patch features. Spatial relation understanding is the hardest subtest in ARO (relation perturbations). Passing this is a very strong signal that the image tower has the expressivity needed for VLM contrastive learning.

**Pass criteria:** Spatial relation accuracy significantly above chance (>60%) in simulation.

- Simulation (noiseless): spatial relation head learns; object attribute accuracy maintained from Step 2
- **Run with AND without the spatial ancilla** (required — Question C.2). R3's `−19.5` ancilla result came from a translation-invariant task where position barely affects the label and says nothing about relational reasoning. This is the only place the question can be settled.
- Matched-parameter classical control alongside
- ~~Stage 2 — Emulation~~ — **removed 2026-07-28**: CLEVR is run noiseless (scope decision).

---

### Step 4: Multimodal — Flickr8k Binary Match (image + text)

**Task:** Binary classification: does this (image, caption) pair match?  
**Why:** First end-to-end image + text pass without hard negatives. Tests both towers jointly with a simple supervised signal before introducing triplet structure and curated ARO negatives.

- Stage 1 — Simulation: both towers produce aligned embeddings; binary cross-entropy converges
- ~~Stage 2 — Emulation~~ — **removed 2026-07-28**: noiseless only (scope decision).

---

### Step 5: Contrastive Learning — ARO with Triplet Loss

**Task:** Full ARO contrastive training: (image, true\_caption, false\_caption) triplets with InfoNCE + Triplet loss (weight 40000).  
**Why:** The target task. Hard negatives are curated syntactic perturbations; the triplet term dominates the loss. Requires three forward passes per sample and produces the `hard_neg_acc` metric used to evaluate VLM quality.

- Stage 1 — Simulation: reproduce classical ARO `hard_neg_acc` baseline
- ~~Stage 2 — Emulation~~ — **removed 2026-07-28**: noiseless only (scope decision).
