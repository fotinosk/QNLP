# TTN Image Encoder on CIFAR-10 — Research Log

**Goal:** get `TTNImageModel` to learn real-photo semantics *at all*,
measured on CIFAR-10 supervised classification, before any further
vision-language work. SVO-Probes is blocked behind this: nineteen
SVO runs sat at chance (`SVO_EXPERIMENTS.md`), the ARO result is provably
image-invariant, and the tower is now at chance on CIFAR-10 too.

**Constraint (the point of the project):** the encoder is a *quantum-inspired*
tensor network. It stays **multilinear** — no learned non-linear activations
inside the network. Stage E is the only exception and only if everything
before it fails.

## What "multilinear" does and does not forbid

Worth stating up front, because it determines whether Stage B is allowed:
in quantum machine learning the non-linearity lives in **state preparation**,
not in the circuit. A pixel is encoded into a qubit state by an angle
encoding — `|ψ(x)⟩ = cos(πx/2)|0⟩ + sin(πx/2)|1⟩` — which is a non-linear
function of the pixel value. The circuit acting on those states is then
linear (unitary), and the network stays multilinear in the *encoded
features*.

So a fixed, non-learned local feature map is **not** a violation of the
constraint — it is the standard quantum-inspired construction, and it is
what the tensor-network image-classification literature uses (Stoudenmire &
Schwab's MPS classifier and its TTN successors all apply a fixed local map
before the network). The current model does not have one: it feeds raw
normalised pixel values into a learned bilinear map. That is the single
biggest deviation from every TN image classifier that works.

What the constraint *does* forbid: GELU/ReLU gates inside nodes, MLP heads,
learned non-linear activations between layers. Those are Stage E.

## Set expectations before starting

Multilinear TN classifiers are strong on MNIST and comparatively weak on
CIFAR-10 — published TN-only CIFAR-10 numbers are well below CNN parity.
*(Recalled, not verified — worth a literature check before fixing a target.)*
Do not aim at 90%. Proposed thresholds:

| Reference point | CIFAR-10 acc |
|---|---|
| chance | 0.10 |
| logistic regression on raw pixels | ~0.40 |
| **minimum bar: TTN must beat raw-pixel logistic regression** | **>0.40** |
| small CNN / ResNet-18 (context, not a target) | ~0.85-0.93 |

If the tower cannot beat a linear model on raw pixels, it is not encoding
anything a tensor network is uniquely doing, and that itself is the result.

---

## Stage 0 — harness and gates (do first, blocks everything)

Nothing below is interpretable without this.

1. **CIFAR-10 at 32×32**, not 64×64. Fewer leaves, faster iteration, and it
   is the standard reference point. With `patch_size=2` this gives a 16×16
   leaf grid = 256 leaves, depth 4 — same tree shape as today.
2. **Verify the probe itself before trusting "TTN is at chance."**
   `TTNImageModel.forward` ends in `F.normalize(x, p=2, dim=-1)`. For
   contrastive training that is correct; for classification it discards the
   magnitude, and in a TN classifier the output magnitude carries class
   signal. Confirm the probe removes it (or reads the pre-normalisation
   activation) and that the linear head sees raw tower output. A probe built
   on the L2-normalised output could plausibly report chance on its own.
3. **Overfit gate: can it fit 500 images?** Train on 500 CIFAR images with
   augmentation and regularisation off. A model that cannot reach ~100%
   train accuracy on 500 examples has an *optimisation/conditioning*
   problem, not a capacity problem — and that distinction decides whether
   Stage A or Stage C is the productive direction. Run this gate after every
   change below; it is far more informative than a val number near chance.
4. **Spread trace at init and after training**, via
   `qnlp/discoviz/diagnostic/tower_spread_trace.py --parquet <cifar>`. The
   trace already showed the 64×64 tower separates images fine at init
   (pairwise cos 0.146) and collapses to 0.9984 after ARO training. Watch
   whether CIFAR training collapses it the same way.

---

## Stage A — cheap defects (each one line; do all before anything structural)

Hours, not days. Any of these could be the whole story.

### A1. Per-node initialisation is not doing what it looks like
`CPQuadRankLayer._initialize` calls `nn.init.orthogonal_` on tensors shaped
`[num_nodes, rank, in_dim]`. Torch flattens dims `[1:]`, so this makes
**different nodes orthogonal to each other**, and gives each node's entire
factor matrix Frobenius norm **1** — not per-node isometry. Verified:

```
per-node Frobenius norm:        1.0000   (isometric would be sqrt(rank)=5.66)
node 0, rows W W^T diagonal:    0.028    (orthonormal rows would be 1.0)
cross-node dot products:        1.0, ~0, ~0, 0   <- orthogonality is ACROSS nodes
mean abs entry:                 0.0176
```

Each node is therefore a heavily down-scaled random projection, not a
canonical-form TTN tensor. **Fix:** loop over nodes and orthogonalise each
node's `[rank, in_dim]` matrix individually. This is the TN-canonical
prescription (isometric tensors, as in a TTN/MERA in canonical form) and is
squarely within the multilinear constraint. Highest-suspicion item on this
list, and a two-line change.

### A2. `dropout=0.3` inside a product network
`merged = p_tl * p_tr * p_bl * p_br`, then dropout on the product. In an
additive network dropout removes one contribution; in a multiplicative one
it removes a whole rank channel's product. Default is 0.3 at every layer.
Set `IMAGE_MODEL_DROPOUT=0` and re-measure. Regularisation is not what is
limiting a model sitting at chance.

### A3. Hand-tuned per-layer gains
`gains = [2.0, 1.5, 1.0, 1.0]` with a learnable per-node gain on top reads
as someone fighting variance explosion/decay through the product tree.
Once A1 lands, re-derive these rather than keeping the tuned values —
with isometric tensors the natural gain is 1.0 everywhere.

### A4. Dataset mean-centring before the network
The init trace: raw ImageNet-normalised pixels have mean pairwise cosine
**0.1754**; dataset-mean-centred, **0.0069**. A multiplicative network
amplifies a shared DC component multiplicatively across four levels, so the
between-image signal becomes an ever smaller fraction of the activation.
Subtract the dataset mean (per-channel, per-pixel) before the feature map.

### A5. Positional embedding ablation
`x = x + positional_embedding * pos_scale` injects position *additively*
into a multiplicative network. The quantum-side investigation already found
a positional ancilla mildly harmful and concluded the tree topology encodes
position implicitly (`llm/quantum_implementation_plan.md`). Try
`pos_scale=0` — consistent with that finding and one fewer thing to explain.

---

## Stage B — the representation (the main event)

If Stage A does not clear the bar, this is where I would put the money.

### B1. A proper local feature map (state preparation)
**Replace** the bilinear patch embedding — currently
`c_feat = patch · color_factor`, `p_feat = patch · pixel_factor`,
`x = c_feat * p_feat` — with a fixed local map applied per pixel:

```
φ(x) = [cos(πx/2), sin(πx/2)]        x ∈ [0,1]
```

then a single learned linear map from the per-patch stack of φ-features
into `bond_dim`. Why this matters, three separate reasons:

- **It is the construction that works.** Every TN image classifier in the
  literature applies a fixed local map first. The current model has none.
- **It makes the polynomial inhomogeneous.** `cos(πx/2) ≈ 1` for small `x`,
  so the map carries an effective constant channel. Without one, the network
  computes a *homogeneous* degree-2^depth polynomial and cannot represent
  low-degree interactions at all. With one, it represents every degree up to
  2^depth. This is the classic `φ(x) = [1, x]` trick and it is a large
  expressivity difference for zero cost.
- **The current bilinear product is degenerate.** `c_feat * p_feat` is a
  rank-1 quadratic form in the patch — it multiplies two learned linear
  projections of the *same* patch, which is a very restrictive way to spend
  the one non-linearity the model has.

It also stays strictly within the constraint: fixed map, no learned
non-linearity, direct angle-encoding analogue.

### B2. Leaf granularity
With B1, try leaves = single pixels (32×32 = 1024 leaves, depth 5) as well
as 2×2 patches. Per-pixel leaves with a cos/sin map is the most
literature-faithful configuration. Note the current patch embedding sums
over the 16 pixels *within* a patch, destroying intra-patch spatial
structure before the tree ever sees it.

### B3. Colour handling
`color_factor` linearly mixes the 3 channels into the bond space before the
product. Cleaner TN treatment: keep channel as its own tensor index, or map
each channel through φ and contract. Cheap variant to try once B1 lands.

---

## Stage C — network structure and conditioning

### C1. Replace the 4-way CP node with pairwise binary contractions
`CPQuadRankLayer` factorises a 4-way node as a rank-32 CP decomposition. The
standard TTN contracts children *pairwise* with an explicit bond dimension.
Two binary contractions per node is the textbook structure, is exactly
representable, and has much better-understood conditioning than a rank-32 CP
approximation of a 4-way tensor.

### C2. `cp_rank` as the capacity knob
`cp_rank=32` against `in_dim` up to 512 is a severe bottleneck. **Note this
is a different parameter from the `bond_dim` that was ruled out** — that
ruling was about the *text* tower's bond dimension in the COCO campaign
(`COCO_EXPERIMENTS.md`), not the image tower's CP rank, which has never been
swept. Sweep it only after A1 and B1; capacity is not the binding constraint
while initialisation and the feature map are wrong.

### C3. Normalisation strategy through the product tree
`_rms_norm` standardises each child's rank vector before the 4-way product.
A product of four unit-RMS zero-mean vectors is heavy-tailed — most
components land near zero and a few dominate. Worth measuring the kurtosis
of `merged` per layer, and considering log-domain accumulation or explicit
per-node canonicalisation instead.

### C4. Residual branches
`out + res_proj(x.mean(dim=2))` on layers 2-3 makes the function
inhomogeneous (degree 4 mixed with degree 1) — which may be doing real work,
and overlaps with what B1's constant channel provides more principledly. The
quantum side ruled residuals out for the *quantum* tower after testing five
mechanisms; here they exist and are untested. Ablate once B1 lands, and
expect B1 to make them redundant.

---

## Stage D — training algorithm

### D1. DMRG-style sweep optimisation
Global Adam over all tensors of a multilinear network is not what the TN
literature does. Optimising **one tensor at a time** holding the rest fixed
is a *linear least-squares problem per tensor* — convex, well-conditioned,
and the standard TN training method. This is a genuinely quantum-inspired
training scheme rather than a borrowed deep-learning one, which makes it
worth real estate in the thesis independently of whether it wins. Attempt
only after the representation is right; a sweep optimiser will faithfully
converge a badly-parameterised model to a bad answer.

---

## Stage E — last resort: relax multilinearity

Only if A-D all fail to beat raw-pixel logistic regression. Ordered by how
much of the quantum-inspired motivation survives:

1. **Gated non-linearity mirroring the text tower's NLC** —
   `out = contraction + gate * GELU(contraction)`, one learnable scalar gate
   per node, initialised to 0 so the multilinear model is recovered exactly
   at gate=0. The text tower learned to use this (gate rose 0.044 → 0.393 in
   the COCO campaign), and initialising at 0 makes the gate's learned value
   a *measurement* of how much non-linearity the task demands — publishable
   either way.
2. Classical convolutional stem feeding the TN. Abandons most of the
   motivation; a control experiment, not a model.

---

## Recommended order, compressed

```
0  harness: CIFAR-10 @32x32, verify probe isn't reading L2-normalised output,
   overfit-500 gate, spread trace                              [blocks all]
A1 per-node isometric init (verified defect)                   [hours]
A2 dropout 0.3 -> 0                                            [minutes]
A4 dataset mean-centring                                       [minutes]
A5 pos_scale = 0 ablation                                      [minutes]
A3 re-derive layer gains after A1                              [hours]
B1 cos/sin local feature map + constant channel  <- main event [1-2 days]
B2 leaf granularity (per-pixel vs 2x2)                         [1 day]
B3 colour as a tensor index                                    [1 day]
C1 pairwise binary contractions instead of 4-way CP            [2-3 days]
C2 cp_rank sweep (NOT the ruled-out text bond_dim)             [1 day]
C3 product-tree normalisation / kurtosis study                 [1-2 days]
C4 residual ablation                                           [hours]
D1 DMRG-style sweep optimisation                               [1 week]
E1 gated non-linearity, gate init 0                            [last resort]
```

Gate between stages: **does it beat raw-pixel logistic regression (~0.40)?**
Report the overfit-500 result alongside every val number — it separates
"cannot optimise" from "cannot represent", and those need different fixes.

---

## Results

### Stage 0 — harness and gates (2026-09-18, in progress)

**0.1 — CIFAR-10 @ 32×32.** Controlled via `IMAGE_MODEL_IMAGE_SIZE=32`,
`IMAGE_MODEL_PATCH_SIZE=2` (env-var-driven singleton config, no code
change needed) — gives `num_patches=256`, `depth=4`, matching this
document's target tree shape exactly.

**0.2 — probe was reading the L2-normalised output; fixed.** Confirmed the
suspected defect: `TTNImageModel.forward` unconditionally returned
`F.normalize(head_output, p=2, dim=-1)`, and
`qnlp/discoviz/diagnostic/ttn_supervised_probe.py`'s `TTNClassifier` fed
that normalised vector straight into `nn.Linear` for softmax
classification. Fixed both:
- `TTNImageModel.forward` gained a `normalize: bool = True` parameter
  (default preserves every existing caller — contrastive training is
  scale-invariant under cosine similarity, so this changes nothing for
  SVO/ARO/COCO).
- `TTNClassifier` now calls `self.backbone(x, normalize=False)`, reading
  the raw pre-norm head output, where output magnitude can carry class
  signal a softmax classifier needs.
- This means the CIFAR-10 result already logged above (TTN 0.1136 vs. CNN
  0.7554 / ResNet18 0.8001) was measured on the L2-normalised output — the
  defect this stage exists to catch. **That number needs to be re-measured
  post-fix before being treated as a capacity conclusion**; re-running is
  part of Stage 0's harness validation, not an optional extra.

**0.3 — overfit-500 gate, and 0.4 — spread trace at 32×32 init:** launched
together via `scripts/submit_ttn_cifar_stage0.sh` (job pending as of this
entry — `IMAGE_MODEL_DROPOUT=0` for the overfit gate per this stage's
explicit ask to remove regularisation, `--overfit-n 500 --overfit-epochs
200` for ttn/cnn/resnet18, plus the spread trace at the same 32×32/patch=2
config with `-n 128` random-init images). Results pending.
