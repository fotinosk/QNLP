## Implementation and Experimentation Plan

### Architecture Summary (What We're Building)

```
═══════════════════════════════════════════════════
TEXT TOWER                              IMAGE TOWER
═══════════════════════════════════════════════════
Tokens                              Image (64×64)
  │                                      │
  ▼                                      ▼
lambeq parse → DisCoCat diagram     Bilinear patch embedding
  │           (variable topology)   4×4 patches → 256 patches
  ▼                                 + gated positional encoding
EinsumModel contraction                  │
(cotengra-optimised einsum)              ▼
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
- **CP-rank-32** at every TTN level in the image tower
- **Bilinear patch embedding** separates color and spatial structure (Hadamard product)
- **Gated positional encoding** with learnable scale initialised near 0 (≈0.05)
- **Triplet loss dominates** (weight 40000); hard negatives are curated ARO syntactic perturbations
- **Three forward passes per sample:** image × 1, true caption × 1, false caption × 1
- **DisCoCat text tower** (EinsumModel): sentence structure determines diagram topology, not sequence length

> **NOTE — Non-Linear Contractions:** We will **never** use non-linear contractions (NLC) in this quantum implementation. All tensor network contractions are strictly linear. Do not add GELU activations, learnable gates, or any non-linearity between contraction steps. The EinsumModel's `non_linear_contractions` flag must always be `False`.

> **NOTE — NaN / Infeasibility Handling:** This quantum implementation does **not** require NaN/infeasibility handling. Unlike the classical EinsumModel which can encounter OOM-infeasible contractions and returns NaN as a sentinel, quantum circuits have fixed topology and bounded resource usage. There is no need for NaN guards, `drop_nonfinite_rows`, or the `_safe_text_embed` pattern — omit all of this machinery.

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
