## Implementation and Experimentation Plan

### Architecture Summary (What We're Building)

> **⚠️ Diagram below predates the quantum investigation (2026-07-17 to 2026-07-27) and is now superseded in two ways — see corrected bullet list beneath it.**

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
- **Image size**: 16×16 is the only size with end-to-end validated *training* (every quantum experiment to date used this size). 64×64 (and 32×32) have validated *forward-pass* simulation only, via `default.tensor(method='tn')` — training at these sizes is currently blocked on gradient cost (parameter-shift is exact but ~57s/gradient-step at 32×32; SPSA is the proposed fix, not yet implemented — see `quantum_investigation_roadmap.md` Task 1.5). **Start CLEVR implementation at 16×16** unless Task 1.5 is completed first.
- **Triplet loss dominates** (weight 40000); hard negatives are curated ARO syntactic perturbations
- **Three forward passes per sample:** image × 1, true caption × 1, false caption × 1
- **DisCoCat text tower** (EinsumModel): sentence structure determines diagram topology, not sequence length

> **NOTE — Non-Linear Contractions:** We will **never** use non-linear contractions (NLC) in this quantum implementation. All tensor network contractions are strictly linear. Do not add GELU activations, learnable gates, or any non-linearity between contraction steps. The EinsumModel's `non_linear_contractions` flag must always be `False`.

> **NOTE — NaN / Infeasibility Handling:** This quantum implementation does **not** require NaN/infeasibility handling. Unlike the classical EinsumModel which can encounter OOM-infeasible contractions and returns NaN as a sentinel, quantum circuits have fixed topology and bounded resource usage. There is no need for NaN guards, `drop_nonfinite_rows`, or the `_safe_text_embed` pattern — omit all of this machinery.

> **NOTE — Classical Hybrid Shortcuts:** This project's direction is a purely quantum implementation. Do **not** insert classical components inside the core quantum pipeline to substitute for quantum circuit capacity — e.g. a classical compression layer before a variational circuit to reduce the qubit width it would otherwise need (`qnlp/image_tower/classification/quantum/hybrid_trainer.py`'s `Linear(16,4)` before a 4-qubit VQC is the rejected example; that file is deprecated as an architecture direction, 2026-07-26). Qubit-budget problems should be solved by genuinely quantum means (active qubit recycling, tensor-network simulators), not by offloading work to classical layers. The only classical components that belong in these models are the unavoidable I/O boundary: encoding raw pixel/token data into rotation angles at the start, and projecting measured expectation values into class logits or an embedding space at the end — both are necessary interfaces, not research targets, and should stay as minimal as possible (a single `Linear` layer, no additional capacity). Do not propose experiments that tune, grow, or characterize these boundary layers as though they were part of the quantum architecture question; close such proposals on principle rather than running them. See `llm/quantum_investigation_roadmap.md` Section 6 (backlog items 5 and 7, both closed 2026-07-26) for the specific questions this ruled out.

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

- **Stage 1 — Simulation:** Run the quantum circuit on a noiseless classical simulator (`default.qubit` in PennyLane or `AerSimulator` statevector mode in Qiskit). The circuit must produce correct results here before proceeding. This validates architecture and gradient flow.
- **Stage 2 — Emulation:** Run the same circuit on a noise-emulated backend (PennyLane `default.mixed` with depolarizing noise, or Qiskit Aer with a calibrated IBM noise model). Measure the performance gap vs. simulation. This determines whether the circuit is noise-resilient enough to be worth running on real hardware.

Do not move to the next step until both stages of the current step pass.

---

### Step 1: Image Tower — Synthetic Shapes (16×16)

**Task:** Classify synthetic 16×16 images into N classes (color × shape combinations).  
**Why:** Fully controlled dataset, fast iteration, minimal qubit budget. Validates the quantum TTN circuit structure and CP layer design in isolation before touching real data.

- Stage 1 — Simulation (`default.qubit`): verify loss decreases, gradients flow through CP layers
- Stage 2 — Emulation (noise model): measure accuracy degradation vs. simulation; tune circuit depth

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

- Stage 1 — Simulation: all 4 attribute heads converge; per-attribute accuracy >80%
- Stage 2 — Emulation: identify which attributes degrade fastest under noise; use to guide circuit depth decisions

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

- Stage 1 — Simulation: spatial relation head learns; object attribute accuracy maintained from Step 2
- Stage 2 — Emulation: measure which TTN levels are most noise-sensitive for relational vs. attribute tasks

---

### Step 4: Multimodal — Flickr8k Binary Match (image + text)

**Task:** Binary classification: does this (image, caption) pair match?  
**Why:** First end-to-end image + text pass without hard negatives. Tests both towers jointly with a simple supervised signal before introducing triplet structure and curated ARO negatives.

- Stage 1 — Simulation: both towers produce aligned embeddings; binary cross-entropy converges
- Stage 2 — Emulation: measure noise impact on each tower separately by swapping one tower back to classical to isolate the degradation source

---

### Step 5: Contrastive Learning — ARO with Triplet Loss

**Task:** Full ARO contrastive training: (image, true\_caption, false\_caption) triplets with InfoNCE + Triplet loss (weight 40000).  
**Why:** The target task. Hard negatives are curated syntactic perturbations; the triplet term dominates the loss. Requires three forward passes per sample and produces the `hard_neg_acc` metric used to evaluate VLM quality.

- Stage 1 — Simulation: reproduce classical ARO `hard_neg_acc` baseline
- Stage 2 — Emulation: assess whether quantum noise degrades syntactic discriminability specifically, and whether the degradation is uniform across ARO subtask types (attribute, relation, action)
