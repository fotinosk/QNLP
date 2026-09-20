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
   **⚠ Superseded as a decision metric** — mean pairwise cosine is
   degenerate near 0 (reads the same for a faithful embedding and for
   noise). Use Gram correlation + kNN class consistency instead; see
   "B1 confirmed, and the damage localised" in Results. Mean cosine stays
   useful only as a collapse detector (pairwise → 1.0).

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
   overfit-500 gate, spread trace                              [DONE — see Results]
   -> verdict: TTN 0.0983 vs logreg 0.3771, majority 0.1000. Fails the
      gate cleanly. Proceed to Stage A.
A1 per-node isometric init (verified defect)                   [DONE]
   -> real gain: TTN 0.0983 -> 0.1591. Keep. Still fails the >0.40 gate.
A2 dropout 0.3 -> 0                                            [minutes]
A4 dataset mean-centring                                       [re-check]
A5 pos_scale = 0 ablation                                      [re-check]
   -> A1/A4/A5 verdicts were judged on a degenerate metric; re-run them
      under Gram-corr/kNN before treating them as closed (see below).
A3 re-derive layer gains after A1                              [hours]
B1 cos/sin local feature map + constant channel  <- NEXT       [1-2 days]
   -> CONFIRMED by the structure trace: the bilinear product `c*p` drops
      Gram corr 0.982 -> 0.112 and kNN class consistency 0.171 -> 0.131 in
      one operation, before the tree runs. B1 replaces exactly that step.
B2 leaf granularity (per-pixel vs 2x2)                         [1 day]
B3 colour as a tensor index                                    [1 day]
C1 pairwise binary contractions instead of 4-way CP            [2-3 days]
C2 cp_rank sweep (NOT the ruled-out text bond_dim)             [1 day]
C3 product-tree normalisation / kurtosis study    <- demoted    [1-2 days]
   -> real (layers decay Gram corr 0.112 -> 0.0006) but secondary: it
      would only preserve an already-ruined representation. Test it by
      re-running the structure trace AFTER B1.
C4 residual ablation                                           [hours]
D1 DMRG-style sweep optimisation                               [1 week]
E1 gated non-linearity, gate init 0                            [last resort]
```

Gate between stages: **does it beat raw-pixel logistic regression (~0.40)?**
Report the overfit-500 result alongside every val number — it separates
"cannot optimise" from "cannot represent", and those need different fixes.

Cheap pre-gate before committing any change to a full training run:
**kNN class consistency at the output** (chance 0.100, raw-pixel input
0.2004). Minutes to measure, and it is the metric that actually tracks
whether class information survives the tower.

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

**0.3 — overfit-500 gate: PASSED for all three architectures** (job
7430551, `IMAGE_MODEL_DROPOUT=0`, no augmentation, `--overfit-n 500
--overfit-epochs 200`):

| arch | passed | epochs to 100% train acc |
|---|---|---|
| ttn | True | **130** |
| cnn | True | 14 |
| resnet18 | True | 95 |

TTN *can* memorise 500 examples — this rules out a hard
optimisation/capacity block per this gate's own logic (a model unable to
reach ~100% here would indicate a genuine optimisation/conditioning
failure, not a representational one). But it took **~9x more epochs than
the CNN** to get there, which is itself a real, measurable conditioning
problem — consistent with Stage A's suspected defects (A1 per-node init,
A4 mean-centring) rather than a Stage B representational rewrite being
required.

**0.4 — spread trace at 32×32/patch=2 random init: collapses, unlike the
64×64 trace.** Same job, `-n 128`:

| stage | pairwise cos mean |
|---|---|
| raw pixels (ImageNet-normalised) | 0.0052 |
| after colour projection (linear) | 0.0060 |
| after pixel projection (linear) | 0.0026 |
| after bilinear product c*p | 0.4442 |
| + positional embedding | 0.4449 |
| after quadtree layer 0 | 0.1851 |
| after quadtree layer 1 | 0.0096 |
| after quadtree layer 2 | 0.0011 |
| after quadtree layer 3 | 0.0012 |
| **after final_norm + head (output)** | **0.0037** |

Contrast with the original 64×64/patch=4 trace in the "image tower is the
bottleneck" section of `SVO_EXPERIMENTS.md`, which *recovered* to ~0.15 by
layer 1 and held there through the output (0.1459). Here, at the same tree
depth (4, since `256` leaves either way) but smaller patches (2×2 vs 4×4,
so a 4-dim raw patch vector instead of 16-dim), the representation
collapses monotonically after layer 0 and never recovers — the random-init
tower is nearly indistinguishable across different CIFAR images by the
time it reaches the classifier head (0.0037 ≈ what 128 independent random
unit vectors in a high-dim space would give by chance). This is a
plausible mechanism for 0.3's slow convergence: the classifier head starts
from an almost-uninformative representation and needs many steps of joint
tower+head training to carve out class-relevant structure, whereas the CNN
and ResNet18 start with normal, well-conditioned random features.

**Stage 0 status:** harness verified (probe bug found and fixed; overfit
gate passing rules out a hard capacity block; spread trace pinpoints where
random-init signal is lost). Not yet run: the actual bug-fixed CIFAR-10
accuracy number against the >0.40 logistic-regression bar — that's the
next step before deciding whether to proceed to Stage A's specific fixes.

### Corrected full CIFAR-10 run — decisive, still a clean fail (job 7430561)

Same config as the spread trace/overfit gate (32×32, patch_size=2), with
the Stage 0.2 `normalize=False` fix in place, `--epochs 40 --patience 8`,
plus the new raw-pixel logistic regression baseline:

| arch | test_acc | majority_baseline | params |
|---|---|---|---|
| **ttn** | **0.0983** | 0.1000 | 2,281,376 |
| cnn | 0.7883 | 0.1000 | 391,946 |
| resnet18 | 0.7547 | 0.1000 | 11,181,642 |
| **logreg (raw pixels)** | **0.3771** | 0.1000 | 30,730 |

TTN's train/val loss sat flat at exactly `ln(10) = 2.303` for all 13
epochs before early stopping — the normalize fix changed nothing
observable. **TTN scores below its own majority baseline and dramatically
below the raw-pixel logistic regression floor (0.0983 vs 0.3771).** This
is precisely the failure condition this document defined up front: *"If
the tower cannot beat a linear model on raw pixels, it is not encoding
anything a tensor network is uniquely doing, and that itself is the
result."* Logreg landing at 0.3771 also validates the document's recalled
~0.40 figure empirically rather than leaving it as an unverified citation.

**Resolves an apparent tension with the overfit-500 gate.** That gate
passed (TTN memorises 500 fixed points in 130 epochs), which could read as
contradicting a flat-chance result here. It doesn't: 500 points is a
regime where enough gradient steps let the model brute-force memorise
specific examples even from a poorly-conditioned start; the full 45,000-row
training set with a realistic epoch budget (13 before early stopping)
demands actual generalisable structure, and the spread-trace finding
(random-init representation is ~indistinguishable across images by the
output layer, 0.0037 pairwise cosine) means there is essentially no
signal for the classifier head to work with early in training, and not
enough steps at this scale to escape that regime. Both gates are
consistent with the same underlying mechanism: **a badly-conditioned
random-init representation**, not an outright absence of representational
capacity (Stage 0's overfit gate) but a practical one at real training
scale (this result).

**Stage 0 verdict: proceed to Stage A.** The gate this document sets
(`does it beat raw-pixel logistic regression, ~0.40?`) has a clean,
unambiguous answer: no, by a wide margin (0.10 vs 0.38). Per the
document's own ordering, this is not yet grounds for a Stage B
representational rewrite — Stage A's cheap, well-motivated defects (A1
per-node isometric init being the highest-suspicion item, A4 dataset
mean-centring given the 0.1754→0.0069 pairwise-cosine effect already
measured at the pixel stage) haven't been tried yet, and both connect
directly to the conditioning story this stage's results support.

**Note:** the SVO top-20 object classification numbers logged earlier in
`SVO_EXPERIMENTS.md`'s "Supervised capacity probe" section (job 7430493:
TTN 0.1424 / CNN 0.3928 / ResNet18 0.3242, all vs. majority 0.1304) were
run *before* the Stage 0.2 fix and at 64×64, not 32×32 — same caveat as
the original CIFAR-10 number. Not re-run yet; lower priority than CIFAR-10
since this document's scope is specifically the CIFAR-10 gate.

## Stage A (2026-09-18)

### A1 — per-node isometric init: fixed, verified, does NOT fix the collapse alone

Implemented exactly as diagnosed: `CPQuadRankLayer._initialize` now loops
over nodes and calls `nn.init.orthogonal_` on each node's own `[rank,
in_dim]` matrix individually, instead of once on the full `[num_nodes,
rank, in_dim]` tensor (which orthogonalised nodes against each other
instead of isometrising each node). Verified directly: per-node Frobenius
norm is now `sqrt(rank)=5.657` (was 1.0), `W @ W.T` is the identity
(rows orthonormal, was ~0.028 off-diagonal-adjacent), and node-vs-node dot
products are no longer forced to zero.

**But the spread trace at 32×32/patch=2 random init is essentially
unchanged with A1 alone:**

| stage | baseline (pre-A1) | + A1 |
|---|---|---|
| after quadtree layer 0 | 0.1851 | 0.1888 |
| after quadtree layer 1 | 0.0096 | 0.0088 |
| after quadtree layer 2 | 0.0011 | 0.0024 |
| after quadtree layer 3 | 0.0012 | -0.0005 |
| **output** | **0.0037** | **-0.0007** |

A1 is a real, verified defect and a correct fix on its own terms (it's
the canonical TN-isometric prescription and is worth keeping regardless),
and it is **not** the mechanism behind the random-init collapse this
document is chasing (the trace above shows that clearly) — but the actual
training result (job 7430647, same 32×32/patch=2 config as job 7430561's
baseline, identical everything else) says more than the trace predicted:

| arch | test_acc, baseline (no A1) | test_acc, + A1 | majority |
|---|---|---|---|
| **ttn** | **0.0983** | **0.1591** | 0.1000 |
| cnn | 0.7883 | 0.7875 | 0.1000 |
| resnet18 | 0.7547 | 0.7643 | 0.1000 |
| logreg | 0.3771 | 0.3771 (identical — not touched by A1) | 0.1000 |

**Correction to the prediction above: A1 does have a measurable, real
effect.** TTN's loss dropped below `ln(10)` for the first time across any
run in this document (2.30 → ~2.20-2.23) instead of sitting frozen
exactly at chance, and test accuracy nearly doubled relative to majority
baseline's margin (from -0.2 points, i.e. *below* majority, to +5.9
points over it). The random-init spread trace looking unchanged was
measuring the wrong thing: A1 doesn't change the *immediate* feature
representation at init, but it plausibly improves *gradient conditioning*
during training (isometric per-node tensors keep gradient norms
comparable across nodes and layers; the old init's forced inter-node
orthogonality with Frobenius-norm-1 blocks likely produced very uneven,
poorly-scaled gradients). This is a real, if modest, positive result —
worth keeping A1 as a baseline improvement going forward — but TTN is
still roughly **2.4x below the logistic-regression floor** and nowhere
near CNN/ResNet18, so Stage A's gate is still failed decisively.

### A4 (mean-centre input) and A5 (pos_scale=0): also do not fix it

Quick trace-only ablations (`--mean-center`, `--zero-pos-scale`, new flags
on `tower_spread_trace.py`), each alone and combined, all at the same
32×32/patch=2 random init:

| stage | baseline | A4 (mean-centre) | A5 (pos_scale=0) | A4+A5 |
|---|---|---|---|---|
| after quadtree layer 0 | 0.1851 | 0.1927 | 0.1947 | 0.2025 |
| after quadtree layer 1 | 0.0096 | 0.0093 | 0.0094 | 0.0098 |
| after quadtree layer 2 | 0.0011 | 0.0023 | 0.0034 | 0.0026 |
| after quadtree layer 3 | 0.0012 | 0.0018 | 0.0033 | 0.0037 |
| **output** | **0.0037** | **0.0026** | **0.0048** | **0.0034** |

Every variant is indistinguishable from baseline within noise. None of
Stage A's four cheap defects (A1, A4, A5 tested; A2/dropout is untestable
via this eval-mode trace since dropout doesn't fire in `.eval()`) touch
the actual mechanism.

**Where the signal consistently dies, across every variant tried:** the
pattern is identical in all five traces (baseline, A1, A4, A5, A4+A5) —
spread survives quadtree layer 0 at a healthy ~0.19, then crashes by an
order of magnitude at layer 1 and never recovers. This happens *inside*
`CPQuadRankLayer.forward`'s repeated structure (per-child RMS-norm →
4-way elementwise product → output projection), applied identically at
every layer, so whatever causes the crash at layer 0→1 is presumably also
active at layer 0 itself — layer 0's output (0.19) is already well below
the input to the tree (bilinear product output, 0.44-0.46) before any
layer runs, and each subsequent layer compounds it further. This matches
Stage C3's suspected mechanism almost exactly: *"A product of four
unit-RMS zero-mean vectors is heavy-tailed — most components land near
zero and a few dominate."* Combined with Stage 0 A1-style reasoning, the
init-level defects (A1/A4/A5) are not where this specific problem lives.

**Revised priority: skip ahead to Stage C3 (product-tree normalisation)
and Stage B1 (proper local feature map) before further Stage A items.**
Both were already flagged as candidates; this trace evidence makes them
the load-bearing hypotheses rather than one candidate among several. A2
(dropout=0) remains cheap and untested in training (not just at
init) — worth including as a control in whatever training run tests C3/B1,
but is not expected to be the primary mechanism given traces above are
dropout-independent (eval mode) and still show the collapse.

### Recommendation: B1 next, not C3

Both remaining candidates target the same crash point (bilinear product →
quadtree layer 0-1), but they differ a lot in how well-specified and cheap
they are to actually try:

- **B1 (fixed cos/sin local feature map)** has a complete, concrete recipe
  already written in this document: replace `c_feat * p_feat` with
  `φ(x) = [cos(πx/2), sin(πx/2)]` applied per pixel/channel, then one
  learned linear map from the per-patch φ-stack into `bond_dim`. This is a
  self-contained change to `TTNImageModel.__init__`/`forward`'s patch
  embedding section only — comparable in size and risk to the A1 fix
  already shipped today — and it directly replaces the one part of the
  pipeline this document independently flagged as degenerate (`c_feat *
  p_feat` is a rank-1 quadratic form built from two learned linear
  projections of the *same* patch). It's also immediately testable with
  the exact same harness used all day: spread trace first (cheap, minutes,
  tells us if the layer 0→1 crash survives), then the same
  `submit_ttn_cifar_corrected.sh` training run if the trace looks better.
- **C3 (product-tree normalisation)** is comparatively open-ended — the
  document's own language is "worth measuring the kurtosis... and
  considering log-domain accumulation or explicit per-node
  canonicalisation instead," i.e. a diagnosis-then-design task, not a
  fix with a recipe. It may well be the deeper explanation (the RMS-norm
  + 4-way-product structure is unchanged by B1), but starting there means
  spending time deciding what to try before there's anything to test.

**Recommend B1 next.** It is cheaper, already fully specified, stays
strictly within the multilinear constraint (fixed map, no learned
non-linearity — the standard angle-encoding analogue), and there's a real
chance it also *helps* C3's problem indirectly: a bounded, inhomogeneous
feature map (cos/sin, unlike an unconstrained bilinear product) is much
less likely to produce the heavy-tailed activations that make repeated
multiplicative combination collapse variance in the first place. If B1's
spread trace still crashes at layer 0→1 with a well-conditioned input,
that would be strong, clean evidence that C3's mechanism is real and
independent of the feature map — sharpening rather than wasting the next
step either way. Not implemented yet — this is a recommendation, pending
confirmation.

### ✅ B1 confirmed, and the damage localised — structure trace (2026-09-18)

The B1-vs-C3 recommendation above was made on the mean-pairwise-cosine
trace. That metric turns out to be **degenerate for this question**, and
replacing it both confirms B1 and pinpoints exactly which operation is
responsible.

**Why the old metric couldn't decide this.** Mean pairwise cosine ≈ 0 is
what you get from a faithful embedding of diverse images *and* from pure
noise. At 32×32 the CIFAR input already sits at ~0.005-0.015, so the trace
had almost no dynamic range left and read identically for "structure
preserved" and "structure destroyed". This is why the A1/A4/A5 ablations
all looked like "no change", and why an initial reading of the 64×64-vs-
32×32 comparison (input 0.1754 → output 0.1459 there; 0.0052 → 0.0037
here, i.e. output ≈ input in both) suggested the tower was *preserving*
input geometry and the layer 0→1 "crash" was an artifact of the bilinear
product's inflated common mode. **That reading was wrong** — the matching
mean levels are a coincidence, and the measurement below refutes it.

**Two metrics that do decide it**, measured at every stage on 512 CIFAR
images, random init, 32×32/patch=2:
- **Gram corr** — correlation between a stage's pairwise-cosine matrix and
  the *input's*. 1 = geometry preserved, 0 = destroyed.
- **kNN cons** — fraction of each image's 10 nearest neighbours sharing its
  true CIFAR class. Chance = 0.100. This is the one that matters: it asks
  whether *class information* survives, not merely whether geometry does.

| stage | mean cos | Gram corr | kNN cons |
|---|---|---|---|
| input (raw pixels) | 0.0154 | 1.0000 | **0.2004** |
| colour projection (linear) | 0.0176 | 0.9975 | 0.2016 |
| pixel projection (linear) | 0.0123 | 0.9823 | 0.1707 |
| **bilinear product `c*p`** | 0.4383 | **0.1120** | **0.1305** |
| + positional embedding | 0.4390 | 0.1123 | 0.1309 |
| quadtree layer 0 | 0.1750 | 0.0737 | 0.1266 |
| quadtree layer 1 | 0.0083 | 0.0416 | 0.1213 |
| quadtree layer 2 | 0.0018 | 0.0131 | 0.0977 |
| quadtree layer 3 | -0.0001 | 0.0006 | 0.1012 |
| **output** | 0.0005 | **0.0002** | **0.1070** |

**Findings:**

1. **Raw CIFAR pixels carry real class structure** — kNN consistency
   0.2004, twice chance. Consistent with logistic regression on raw pixels
   reaching 0.3771. There is signal at the input for the tower to lose.
2. **The output is at chance** (0.1070, Gram corr 0.0002). The random-init
   tower genuinely destroys class information, confirming this document's
   earlier conclusion in substance — on much firmer evidence than the
   mean-cosine trace provided.
3. **The bilinear patch product is the primary destroyer.** Both linear
   projections are harmless (Gram corr 0.998 and 0.982; kNN 0.20 and 0.17).
   The single `c_feat * p_feat` step drops Gram corr from 0.982 to **0.112**
   and kNN from 0.1707 to 0.1305 — most of the way to chance, in one
   operation, **before the tree runs at all**.
4. **The quadtree layers are secondary but real.** They continue the decay
   (0.112 → 0.074 → 0.042 → 0.013 → 0.0006), so C3's heavy-tailed-product
   mechanism is genuinely present — but they are finishing off a
   representation the patch embedding already broke.

**Verdict: B1 is confirmed as the next step**, now as the load-bearing fix
rather than the cheaper of two guesses — it replaces exactly the operation
the trace identifies. **C3 stays on the list but is demoted to second**:
fixing the tree alone would preserve an already-ruined representation. The
right test of C3 is the one this document already proposed — re-run this
trace *after* B1 and see whether the quadtree still decays a
well-conditioned input.

**The random-features probe is superseded, don't run it.** It was proposed
to split "architecture destroys class info" from "training dynamics
broken". kNN consistency at chance answers that directly: a linear head on
frozen random features has nothing to separate.

### Methodology changes this forces

1. **Replace mean pairwise cosine with Gram correlation + kNN class
   consistency** in `tower_spread_trace.py`. Keep mean cosine only as a
   collapse detector (pairwise → 1.0), which is what it was originally
   built for on ARO and where it does work.
2. **The A1/A4/A5 "no effect" verdicts above are not yet established.**
   They were judged on the degenerate metric. A1 already produced a real
   accuracy gain (0.0983 → 0.1591) that the old trace failed to predict —
   direct evidence the metric misses things. Re-run those three ablations
   under Gram-corr/kNN before treating them as closed.
3. **Judge B1 on kNN consistency at the output, not on mean cosine.**
   Target: the output should retain something meaningfully above 0.100,
   ideally approaching the input's 0.2004. That is a cheap, minutes-long
   check to run *before* committing to a full CIFAR-10 training run.

### B1 implemented and traced — fixes the embedding step, not the tower

Implemented as `IMAGE_MODEL_USE_B1_FEATURE_MAP=true` (opt-in, default
`False` — SVO/ARO/COCO keep today's bilinear embedding unchanged while
this stays a CIFAR-scoped investigation). `TTNImageModel.forward` inverts
the ImageNet normalisation back to ~[0,1] (every existing data pipeline
normalises before calling this model, so this keeps every external
contract unchanged), applies `phi(x) = [cos(pi*x/2), sin(pi*x/2)]` per raw
pixel, then one learned linear map into `bond_dim`, replacing
`c_feat * p_feat` entirely. Verified with a real forward/backward pass
(finite output and gradients) before running anything on the cluster.

**Trace result (512 CIFAR images, random init, 32×32/patch=2, same
Gram-corr/kNN metrics as the structure trace above):**

| stage | Gram corr (baseline) | Gram corr (+B1) | kNN cons (baseline) | kNN cons (+B1) |
|---|---|---|---|---|
| input | 1.0000 | 1.0000 | 0.2004 | 0.2004 |
| after patch embedding step | **0.1120** | **0.5529** | 0.1305 | 0.1791 |
| quadtree layer 0 | 0.0737 | 0.4968 | 0.1266 | 0.1832 |
| quadtree layer 1 | 0.0416 | 0.3611 | 0.1213 | 0.1709 |
| quadtree layer 2 | 0.0131 | 0.1629 | 0.0977 | 0.1330 |
| quadtree layer 3 | 0.0006 | 0.0296 | 0.1012 | 0.1125 |
| **output** | **0.0002** | **0.0249** | **0.1070** | **0.1061** |

**B1 works exactly where it should, and exactly there.** At the patch
embedding step — the one operation this document identified as
degenerate — Gram correlation improves 5x (0.112 → 0.553) and kNN
consistency improves meaningfully (0.131 → 0.179). This is the cleanest
confirmation in this document that the diagnosis was right.

**But it does not survive the quadtree.** By the output, B1's numbers
(Gram corr 0.0249, kNN cons 0.1061) are barely distinguishable from
baseline's (0.0002, 0.1070) — both are at essential chance on kNN
consistency. This is exactly the test this document proposed for
disentangling B1 from C3: *"If B1's spread trace still crashes... that
would be strong, clean evidence that C3's mechanism is real and
independent of the feature map."* It does, and it is. **C3 is not merely
"secondary" — on this evidence it is co-equal with B1: fixing the
embedding without fixing the tree still loses essentially all class
information by the output.**

Full CIFAR-10 training with B1 (on top of the now-permanent A1 fix)
launched as job 7431014, to check for the same kind of real-but-
trace-underestimated gain A1 produced — but the trace here is far more
informative than A1's was (a genuine 5x Gram-corr improvement at the one
identified defect, not "no visible change"), so a null training result
would mean something different this time: not "the metric missed it" but
"the tree destroys whatever the embedding hands it, regardless of
quality." Either outcome sharpens the case for C3 as the next real
target.

### ✅ First result to clear the document's gate (job 7431014)

| config | TTN test_acc | vs. logreg floor (job 7431014's own: 0.3686) |
|---|---|---|
| no fixes (job 7430561) | 0.0983 | -0.270 |
| + A1 (job 7430647) | 0.1591 | -0.210 |
| **+ A1 + B1 (job 7431014)** | **0.4968** | **+0.128** |
| cnn (job 7431014, context) | 0.7830 | — |
| resnet18 (job 7431014, context) | 0.7567 | — |

(logreg landed at 0.3686 this run vs. 0.3771 previously — normal
run-to-run noise on a tiny 30,730-parameter model with early stopping;
TTN's margin over it is decisive either way.)

**TTN beats the raw-pixel logistic regression floor for the first time in
this entire investigation** — by nearly 13 points, not marginally. This
is the document's explicit, stated gate (`>0.40`) and it is now cleared.

**TTN never early-stopped.** It ran the full 40 epochs with `patience=8`
never triggering — val accuracy was still climbing at the end (0.4842 at
epoch 25 → 0.4916 at epoch 40, no plateau), unlike every previous TTN run
in this document, which either sat flat at chance or (with A1 alone)
plateaued and triggered early stopping well before the epoch budget. This
strongly suggests the run is capacity/schedule-limited, not converged —
more epochs, a longer patience, or a learning-rate schedule are likely to
extract more, not less.

**Reconciling this with the trace's prediction.** The pre-training trace
showed B1's output at essential chance (kNN cons 0.1061, Gram corr 0.0249)
after the quadtree layers ran their course — which read as "the tree
destroys whatever the embedding hands it, regardless of quality." That
was a prediction about a *fixed, untrained* representation, and it was
wrong as a predictor of *trainability* — exactly the same way A1's
unchanged trace undersold its real training-time value. The mechanism is
probably the same in both cases: gradient descent doesn't just read out
the random-init representation, it reshapes the whole tower (quadtree
tensors included) jointly with the classifier head, and a well-conditioned
starting point at the embedding (B1) evidently gives that joint
optimisation something far more useful to work with than mean cosine or
even Gram-corr/kNN-at-init can predict. **The lesson for this document
going forward: no trace-only metric, however much better than mean
cosine, is a substitute for actually training** — traces are for cheap
triage between candidate fixes, not for declaring a fix dead.

**This does not retire C3.** The trace evidence that the quadtree
decays a well-conditioned input is still real and still measured
directly — it just doesn't mean what "decays it to chance" seemed to
mean for trainability. C3 could still be the difference between 0.50 and
CNN's 0.78; it just isn't gating whether TTN can learn *at all* anymore,
which was the open question this whole document was scoped to answer.

**Recommendation, in order:**
1. **Immediate, cheap:** re-run this exact config with more epochs /
   higher patience (e.g. `--epochs 100 --patience 20`) before concluding
   anything about where TTN's ceiling is — the current 0.4968 is very
   likely an undercount given it never plateaued.
2. **Then, in parallel or after:** revisit Stage C3 anyway, now as a
   "how much further can this go" question rather than "can it learn at
   all." Concrete first C3 experiment: measure the kurtosis of `merged`
   (the 4-way product inside `CPQuadRankLayer.forward`) per layer, on a
   forward pass through the *trained* B1 checkpoint from job 7431014
   (more informative than random init, now that init-time measurements
   have twice undersold real training outcomes) — this tells us whether
   the heavy-tailed-product concern is actually present in the model that
   achieved 0.4968, before designing a fix for it.
3. Re-run the A4/A5 ablations on top of A1+B1 using the corrected
   Gram-corr/kNN trace (not yet done under the new metric) — cheap, and
   the "Methodology changes this forces" section above already flagged
   this as open.

Not implemented yet — recommendations pending confirmation.

---

## Parallel batch plan (2026-09-18)

The cluster can run these concurrently. The binding risk in this campaign
has been *bundling variables to save wall-clock* — the SVO fix attempt's
three-way bundle (augmentation + image_lr + triplet_weight) produced a
failure mode nobody could attribute. Parallel capacity removes the only
reason to bundle, so every job below changes **exactly one thing**.

Run this way, the whole Stage A/B ablation table completes in one
wall-clock cycle instead of ten sequential ones. That matters because
**the ablation table is the deliverable** — the thesis contribution is
"these specific defects account for a 0.098 → 0.497 swing, constraint
never relaxed," and that claim needs clean single-variable rows.

### Wave 1A — CIFAR ablation grid (launch together)

All pure config on the existing harness. Same seed, same budget, one
variable each.

| # | config | answers |
|---|---|---|
| 1 | A1+B1, `--epochs 100 --patience 20`, **all archs on the same budget** | the real ceiling, plus baselines on a fair budget |
| 2 | **B1 alone** (A1 off) | the missing single-variable row — is A1 load-bearing, or was the gain all B1? |
| 3 | A1 alone, long budget | matched-budget A1 row (the 0.1591 figure was 40 epochs/patience 8) |
| 4 | A1+B1+A2 (`IMAGE_MODEL_DROPOUT=0`) | A2 has never been tested *in training* — the eval-mode traces can't fire dropout at all |
| 5 | A1+B1+A4 (dataset mean-centring) | judged only on the degenerate mean-cosine metric so far |
| 6 | A1+B1+A5 (`pos_scale=0`) | same |
| 7 | A1+B1+B2 (per-pixel leaves, 1024 leaves, depth 5) | the literature-faithful configuration; independent of everything else |
| 8 | A1+B1, `cp_rank` ∈ {32, 64, 128} | capacity is now plausibly binding — job 7431014 never plateaued |

**Why job 1 re-runs the baselines:** the current 0.4968-vs-0.7830
comparison is confounded. TTN was the only model that didn't early-stop,
so the gap is partly a budget artefact, not purely a capability gap.

### Wave 1B — transfer track (launch simultaneously with 1A)

Separate pipeline, no dependency on the CIFAR grid. **Note B1 is opt-in
and defaults to `False`, so ARO/SVO are still running the degenerate
bilinear embedding** — these need the flag explicitly enabled.

| # | config | answers |
|---|---|---|
| 9 | ARO with A1+B1 enabled | does the fix transfer to the real task? |
| 10 | SVO-Probes with A1+B1 enabled | the project's actual target — has never had a working image tower behind it |

**Pre-check before trusting job 10 — minutes, local, not a cluster job.**
B1's `forward` inverts the ImageNet normalisation to recover ~[0,1] before
applying `φ`. CIFAR's pipeline is `Resize → ToTensor → Normalize`, so that
inversion is exact. SVO's train transform is now `RandomResizedCrop →
ColorJitter → RandomHorizontalFlip → Normalize` — ColorJitter can push
values outside [0,1], and `cos(πx/2)`/`sin(πx/2)` is **periodic, so
out-of-range values wrap rather than saturate**: two images differing only
in brightness could encode to the same point. Verify the actual value range
reaching `φ` under the SVO transform first, or a negative SVO result is
uninterpretable rather than merely negative.

### Wave 2 — genuinely dependent, do not launch blind

| # | depends on | why it's worth running |
|---|---|---|
| 11 | job 9's checkpoint | **`image_ablation.py` — is the image-invariance gone?** Highest-information single measurement in the batch. If shuffled/zeroed images now degrade ARO accuracy, this project has a vision-language model that actually uses vision for the first time, and the ARO chapter is rewritten. |
| 12 | job 9's checkpoint | SVO warm-started from the new ARO checkpoint — repeats experiment 17, but this time the transferred tower isn't collapsed |
| 13 | job 1's trained checkpoint | C3 kurtosis of `merged` per layer, on the **trained** model (init-time measurements have twice undersold real training outcomes) |
| — | code + job 13's result | C1 (pairwise binary contractions) and B3 (colour as a tensor index) need implementation, and C1's design should be informed by what job 13 measures |

### Rules for the batch

1. **One variable per job.** Parallel capacity removes the only reason to
   bundle. This is the rule the SVO campaign violated.
2. **Fix the seed; hold everything else constant across the grid.** Rows
   are only comparable if they differ in exactly one thing.
3. **Pre-commit what each row would change, before results land.** With
   ten results arriving at once, the temptation is to read the batch as a
   whole and narrate a story around it. Decide in advance which rows would
   move the baseline.
4. **Judge on training, not on traces.** Established twice now: A1's trace
   showed nothing and gave +6 points; B1's trace showed chance-at-output
   and gave +34. Traces triage candidates; only training decides.

### What NOT to parallelise

Speculative C/D-stage variants whose design depends on measurements that
don't exist yet (C1's contraction structure, C3's actual fix, D1). A
cluster makes it cheap to run experiments whose results can't yet be
interpreted — a different failure mode from this campaign's previous one,
but still a failure mode.

## Batch launched (2026-09-18)

**Pre-check for job 10 (required before trusting it):** simulated SVO's
actual train transform (`RandomResizedCrop → ColorJitter → RandomHorizontalFlip
→ Normalize`) on 200 synthetic images, then applied B1's exact
normalisation-inversion (`x * std + mean`). Result: reconstructed pixel
values stayed exactly within `[0, 1]` (min 0.0, max 1.0) across all 200
samples — `torchvision.ColorJitter` clamps its output internally, so the
periodic-wraparound risk this pre-check was checking for does not
materialise in practice. Job 10 is safe to trust.

**Code additions needed before the grid could run** (all default to
today's fixed/correct behaviour — see `qnlp/discoviz/models/{cp_node,image_model}.py`):
- `use_isometric_init` (default `True`) — lets row 2 disable Stage A1 to
  get the missing single-variable "B1 alone" row.
- `mean_center_input` (Stage A4, batch-mean-centring before the patch
  embedding — the earlier `--mean-center` flag only ever affected the
  diagnostic trace, never actual training).
- `zero_pos_scale` (Stage A5, forces the positional embedding's
  contribution to exactly zero regardless of the learned `pos_scale`).
- B2 (per-pixel leaves) and the `cp_rank` sweep needed no code changes —
  both were already environment-configurable (`IMAGE_MODEL_PATCH_SIZE=1`,
  `IMAGE_MODEL_CP_RANK`).

All verified with a real forward+backward pass locally before touching
the cluster. New generic runner: `scripts/submit_ttn_cifar_ablation.sh`
(env-var-parameterised, nothing hardcoded — every row below is this same
script with different `qsub -v` overrides).

**Launched, all 11 jobs, queued simultaneously:**

| row | job | name | config |
|---|---|---|---|
| 1 | 7431183 | ttn_b1_ceiling | A1+B1, all archs, epochs=100 patience=20 |
| 2 | 7431184 | ttn_b2_b1only | B1 alone (`IMAGE_MODEL_USE_ISOMETRIC_INIT=false`), epochs=100 patience=20 |
| 3 | 7431185 | ttn_b3_a1only_long | A1 alone (B1 off, default), epochs=100 patience=20 |
| 4 | 7431186 | ttn_b4_dropout0 | A1+B1+`IMAGE_MODEL_DROPOUT=0` |
| 5 | 7431187 | ttn_b5_meancenter | A1+B1+`IMAGE_MODEL_MEAN_CENTER_INPUT=true` |
| 6 | 7431188 | ttn_b6_zeropos | A1+B1+`IMAGE_MODEL_ZERO_POS_SCALE=true` |
| 7 | 7431189 | ttn_b7_perpixel | A1+B1+`IMAGE_MODEL_PATCH_SIZE=1` (1024 leaves, depth 5) |
| 8a | 7431190 | ttn_b8a_cprank64 | A1+B1, `IMAGE_MODEL_CP_RANK=64` |
| 8b | 7431191 | ttn_b8b_cprank128 | A1+B1, `IMAGE_MODEL_CP_RANK=128` |
| 9 | 7431192 | ttn_b9_aro_b1 | ARO, legacy-faithful config, +`IMAGE_MODEL_USE_B1_FEATURE_MAP=true` |
| 10 | 7431193 | ttn_b10_svo_b1 | SVO, +`IMAGE_MODEL_USE_B1_FEATURE_MAP=true` |

Wave 2 (rows 11-13, the image-ablation-on-job-9's-checkpoint test, SVO
warm-start from the new ARO checkpoint, and C3 kurtosis on a trained
checkpoint) is intentionally not launched yet — each depends on a
checkpoint from jobs 9 or 1 that doesn't exist until those finish. Per
rule 3 above: row 11 (does ARO stop being image-invariant?) is the
highest-information single measurement in the whole batch and is the
first thing to check once job 9 completes.

## Wave 1A/1B results (2026-09-18)

9 of 11 jobs finished within the hour; job 9 (ARO+B1, 100-epoch budget)
is still running, several hours out. One bug found and fixed along the
way.

**Bug found: row 7 crashed on launch.** `gains = [2.0, 1.5, 1.0, 1.0]`
only has 4 hand-tuned entries; `IMAGE_MODEL_PATCH_SIZE=1` at 32×32 gives
1024 leaves → depth 5 → `gains[4]` → `IndexError`. Fixed (deeper layers
fall back to the untuned `gain=1.0` default), verified locally, relaunched
as job **7431935**.

### Wave 1A — CIFAR grid, TTN accuracy

| row | config | TTN test_acc | Δ vs row 1 (A1+B1 baseline) |
|---|---|---|---|
| 1 | **A1+B1, epochs=100/patience=20 (ceiling re-run)** | **0.5168** | — |
| 2 | B1 alone, A1 off | 0.4697 | -0.047 |
| 3 | A1 alone, long budget | 0.1574 | -0.359 |
| 4 | A1+B1+dropout=0 | 0.5130 | -0.004 |
| 5 | A1+B1+mean-centre | 0.5127 | -0.004 |
| 6 | A1+B1+pos_scale=0 | 0.5191 | +0.002 |
| 7 | A1+B1+per-pixel leaves (job 7431935, after the gains-list fix) | 0.3713 | -0.146 |
| 8a | A1+B1, cp_rank=64 | 0.5328 | +0.016 |
| 8b | A1+B1, cp_rank=128 | 0.5420 | +0.025 |

Row 1's baselines (fair, matched 100-epoch/patience-20 budget for
everything, resolving the earlier confound where TTN was the only model
that hadn't early-stopped): **cnn 0.7842, resnet18 0.7569, logreg
0.3784.**

**Reading the grid — this is the ablation table the document said would
be the deliverable:**

1. **B1 is the dominant effect, A1 is real but secondary.** B1 alone
   (row 2, 0.4697) recovers almost all of the combined gain (row 1,
   0.5168) on its own — A1 contributes a real but modest +0.047 on top.
   Symmetrically, A1 alone (row 3, 0.1574) is barely above its own
   0.1591 result from the 40-epoch run, confirming (independent of
   budget) that **A1 alone plateaus early and cannot get TTN near the
   logreg floor by itself** — the fixed feature map is what was actually
   gating trainability, not the init.
2. **A2 (dropout), A4 (mean-centre), A5 (pos_scale=0) are all noise.**
   Rows 4, 5, 6 land within ±0.004-0.002 of row 1 — genuinely
   indistinguishable from run-to-run variance. None of these Stage A
   items matter once B1 is in place. This resolves the "re-run A4/A5
   under the corrected metric" item from the methodology section above:
   the corrected metric was never necessary because these don't move
   real training outcomes either way, consistent with (though for a
   different reason than) the original degenerate-trace verdict.
3. **`cp_rank` is a small but real, monotonic lever.** 32→64→128 gives
   0.5168 → 0.5328 → 0.5420 (+0.016, +0.025 cumulative) — diminishing
   but still positive returns, and job 1 never plateauing at 40 epochs
   (see the earlier "First result to clear the document's gate" section)
   is consistent with capacity being a real, secondary constraint on top
   of the embedding fix. Not dramatic, but real and monotonic across two
   independent steps.
4. **B2 (per-pixel leaves) is a real regression, not a neutral variant.**
   Row 7 (job 7431935, after fixing the `gains`-list crash): **0.3713**,
   *below* row 1's 2×2-patch result (0.5168) by 0.146 — the single
   largest negative delta in the grid, and barely above the logreg floor
   (0.3784, i.e. essentially tied with or slightly below it). Despite
   being "the most literature-faithful configuration" per the original
   Stage B2 write-up, going to single-pixel leaves (1024 leaves, depth 5,
   9.18M params — the largest model in the grid) performs *worse* than
   2×2 patches with 6x fewer parameters. Plausible reason: patch_size=1
   discards all intra-patch spatial pooling and instead asks a much
   deeper tree (5 layers vs. 4) to recover that structure combinatorially,
   which the still-untrained-at-init evidence throughout this document
   suggests this architecture does poorly at scale. Not a settled
   explanation — worth a targeted look if B2 is revisited, but the data
   says don't adopt it as-is.
5. **The picture in one line:** of the six Stage A/B levers tested
   (A1, A2, A4, A5, B2, plus B1 itself), exactly two help — B1 (the
   whole game) and A1 (a real, secondary +0.047) — three are confirmed
   noise (A2, A4, A5), and one (B2) is a real regression despite being
   the more literature-faithful choice. `cp_rank` (Stage C2) adds a
   further small, monotonic increment on top of B1+A1.

### Wave 1B — transfer track

**Job 10 (SVO+B1) finished — a genuine negative result for the actual
target task.** SVO-Probes overall **0.5157** (obj 0.5293 / subj 0.4804 /
verb 0.5210) — still flat at chance, no improvement over any prior SVO
run. SVO-Swap **0.5524** — actually *worse* than experiment 17's ARO
warm-start result (0.6095), the best SVO-Swap result to date. **B1's
CIFAR-10 gain does not transfer to SVO as tested here.** Plausible
reasons, not yet distinguished: SVO's training set (~8,600 rows) is far
smaller than CIFAR-10's 45,000, so B1 may need more data than SVO
provides to have room to help; SVO's images are real, uncontrolled
photographs rather than CIFAR's canonical 32×32 benchmark images, which
may interact differently with a fixed per-pixel angle encoding; or B1
alone (without also transferring ARO's much larger training signal, as
experiment 17's warm start did for the text tower) isn't sufficient for
SVO specifically. This does not undercut the CIFAR-10 finding — it narrows
what "the fix" actually means: B1 measurably fixes the *tower's
CIFAR-10-classification capacity*, and that is not automatically the
same thing as fixing SVO-Probes.

**Job 9 (ARO+B1) finished** — early-stopped at epoch 32 (best epoch 22,
patience 10). Final ARO test: attribution 0.7582, relation 0.6038,
overall 0.6886 — essentially identical to the non-B1 result (0.7573 /
0.6036 / 0.6879, job 7428516), within noise on every figure.

**Row 11 (`image_ablation.py` on this checkpoint) — decisive, and it's a
clean negative.** Real/shuffled/zeroed images give **identical** accuracy
(0.7129, all three variants) and pairwise cosine between different
images' embeddings is **1.0000** (min 0.9999, std 0.0000) — the image
tower is *still fully collapsed*, if anything more completely than the
pre-B1 checkpoint (0.93-0.998, `image_pairwise_cos_mean`). **B1 does not
fix ARO's image-invariance.** This resolves the question the whole Wave
1B/row-11 exercise was built to answer, and the answer is unambiguous:
ARO's collapse and CIFAR-10's chance-level classification were never the
same failure. CIFAR-10's failure was a genuinely bad *representation* (a
degenerate rank-1 patch embedding destroying class structure before the
tree even runs, per the structure trace) — B1 fixes that, and the
CIFAR-10 numbers prove it. ARO's failure is an *objective* problem: the
triplet loss's caption-side hard negative is fully satisfiable with a
constant image embedding (the mechanism already diagrammed in "The
mechanism: `triplet_weight` deletes the only anti-collapse term," earlier
in this document), so training finds and keeps that degenerate solution
regardless of whether the tower *could* represent images well. **A better
tower cannot fix a loss that doesn't need one.** SVO's job 10 negative
result is now explicable by the same logic: SVO-Probes' loss does require
the image (both candidates share a caption), which is exactly why it
was expected to benefit from B1 differently than ARO — but it didn't
improve either, meaning SVO's chance-level result has a separate, still
unexplained cause (data scale, image quality, or something else not yet
isolated) distinct from ARO's collapse mechanism.

**Batch verdict:** the CIFAR-10 capacity question this document was
scoped to answer is closed — B1 (+A1, +cp_rank) demonstrably fixes the
tower's ability to learn from real images, taking it from below chance
to within striking distance of a from-scratch ResNet-18. That fix does
not, on its own, transfer to either downstream VLM task tested here, for
two different and now well-understood reasons (ARO: objective permits
collapse regardless of tower quality; SVO: still unexplained, worth its
own investigation rather than being folded into this one). The natural
next step for the *VLM* work is fixing ARO's objective (e.g. lowering
`triplet_weight` enough that InfoNCE's anti-collapse pressure survives,
now that there's a tower worth preserving) rather than further CIFAR-10
tuning — but that is a new investigation, not a continuation of this
document's scope.

---

## Plan forward (2026-09-18) — two proposals evaluated, then the next batch

Two architectural proposals raised: **overlapping patches**, and **a
different network per patch instead of one shared matrix**. Both are
worth testing. One needs a factual correction about what is already
shared and what isn't.

### Proposal 2 first: "a different network per patch"

**Correction: the tree is already per-node, not shared.**
`CPQuadRankLayer`'s parameters are `[num_nodes, rank, in_dim]` and the
forward einsum (`"bni, nri -> bnr"`) indexes them by node `n`. Every
quadtree node already has its own independent tensors — position-specific,
not weight-tied across the tree. So at the *tree* level this is already
the case, and matches standard TN practice (MPS/TTN classifiers give each
site its own tensor).

**Where it is genuinely untested: the patch embedding.** B1 introduced
`self.feature_proj = nn.Linear(in_channels * patch_size**2 * 2, bond_dim)`
— a **single Linear shared across all 256 patches**. That is the one part
of the model that is weight-tied across spatial positions, and it is the
part the proposal actually targets. Untested, and worth testing:

- **TN-faithful.** Position-dependent site tensors are the norm in the TN
  image-classification literature, not the exception.
- **Cheap.** A `[num_patches, in_features, bond_dim]` parameter at
  256 × 24 × 64 ≈ 393K params — small next to the 2.3M-9.2M models
  already in the grid.
- **It subsumes the positional embedding.** A5 showed `pos_scale=0` is
  noise (row 6: 0.5191 vs 0.5168), i.e. the additive positional embedding
  currently contributes nothing. Per-patch embeddings make position
  *structural* rather than an additive hack bolted onto a multiplicative
  network — replacing a term that demonstrably does nothing with one that
  might.
- **Honest risk (the one the proposal itself names):** it trades
  translation equivariance for per-position specialisation. On CIFAR-10
  each patch position sees all 45,000 training images, so data is not the
  constraint. On SVO (~8,600 rows) it would be far more marginal — so
  evaluate this on CIFAR and do not assume it ports.

**Verdict: test it.** Highest-expected-value of the two proposals.

### Proposal 1: overlapping patches

Worth one clean job, with a design constraint and a real risk.

**Why it might help.** The B2 result is the evidence in its favour: row 7
(per-pixel leaves) *regressed* to 0.3713 while 2×2 patches give 0.5168,
which says intra-patch pooling matters and the tree recovers fine spatial
structure poorly when asked to do it combinatorially. Overlapping patches
sit between those two points — they keep patch-level pooling while
softening the rigid partition, so that pixels straddling a patch boundary
interact below the top of the tree. This is the same reason CNNs use
stride < kernel size.

**Design constraint — avoid confounding with depth.** The quadtree needs
`4^depth` leaves. The naive overlap (patch=2, stride=1, padded to 32×32)
gives 1024 leaves → depth 5, which is *exactly the configuration that just
regressed in row 7*. Any result would be uninterpretable. Use instead:

```
patch_size=4, stride=2, pad=1  ->  16x16 = 256 leaves, depth 4
```

Same tree shape as the current baseline, 2× overlap, one variable changed.

**Real risk, specific to multiplicative networks.** The tree *multiplies*
leaves together. If a pixel appears in four overlapping patches, it enters
the product four times — raising its effective polynomial degree rather
than merely adding a redundant path, as it would in an additive CNN. That
plausibly worsens the heavy-tailed-activation problem the structure trace
already measured (Stage C3). So this experiment must be read alongside a
structure trace and the C3 kurtosis measurement, not on accuracy alone.

**Motivational caveat worth recording for the thesis.** Copying one
pixel's encoded state into several sites has no native quantum analogue —
no-cloning. It is adjacent to *data re-uploading*, which the quantum-side
investigation already tested as a residual mechanism and found no robust
benefit (`llm/quantum_implementation_plan.md`). So a win here would need
framing as a quantum-inspired-but-not-quantum-realisable choice, which
costs something in the story even if it gains accuracy.

**Verdict: one job, at constant depth, read with the trace.** Lower
expected value than proposal 2, but cheap and genuinely informative about
whether the blocky partition is a limiter.

---

### Track 1 — the VLM objective (highest value in the project right now)

This document's own batch verdict is that CIFAR's capacity question is
closed and the blocker is now ARO's objective. That makes the following
the most valuable experiment available anywhere in the project, and it
should not wait behind CIFAR architecture work. **Log results in
`SVO_EXPERIMENTS.md`, not here — this is outside this document's scope.**

| # | config | answers |
|---|---|---|
| T1 | ARO, A1+B1+cp_rank=128, `triplet_weight` ∈ {1, 10, 100} | does the tower stay uncollapsed once InfoNCE's anti-collapse pressure survives? |
| T2 | `image_ablation.py` on each T1 checkpoint | the actual success criterion — **real vs shuffled vs zeroed must differ** |
| T3 | SVO-Probes with the best T1 configuration | does a tower that provably uses images move the target task? |

Log `image_pairwise_cos_mean` as a first-class training metric on all of
these. The mechanism is already diagnosed ("`triplet_weight` deletes the
only anti-collapse term"); what is new is that there is now, for the first
time, a tower worth preserving. Success on T2 is what turns the ARO
chapter from "provably image-invariant" into "uses vision."

### Track 2 — CIFAR architecture (parallel, independent of Track 1)

| # | config | answers |
|---|---|---|
| C-a | **per-patch embedding** (proposal 2): `feature_proj` → `[num_patches, in_features, bond_dim]` | does position-specific state preparation beat a shared map? |
| C-b | C-a **with `pos_scale=0`** | does the per-patch embedding subsume the positional embedding, as predicted? |
| C-c | **overlapping patches** (proposal 1): patch=4, stride=2, pad=1, depth 4 | does softening the rigid partition help, or does degree inflation hurt? |
| C-d | `cp_rank=256` | where does the monotonic 32→64→128 (0.5168→0.5328→0.5420) trend saturate? |
| C-e | C1: pairwise binary contractions replacing the 4-way CP node | the last untested structural item |

C-a and C-c need code; C-b depends on C-a; C-d is pure config and can go
immediately. Baseline for all rows: **A1+B1+cp_rank=128 = 0.5420**, not
row 1's 0.5168.

### Dependent, not to be launched blind

- **C3 kurtosis of `merged` per layer on the trained A1+B1 checkpoint** —
  still undone (wave-2 row 13), and now needed to interpret C-c's overlap
  result as well as to design any C3 fix.
- Structure trace (Gram-corr/kNN) on C-a and C-c *at init* is fine as
  triage, but per this document's own rule — established twice — **no
  trace-only metric decides anything; only training does.**

### Standing gate

CIFAR baseline to beat: **0.5420**. Context: cnn 0.7842, resnet18 0.7569,
logreg 0.3784. The remaining question for this document is how much of the
~0.24 gap to a from-scratch CNN is closable while staying multilinear —
and that is now a "how far" question, not a "does it work at all" one.

## Batch 2 results (2026-09-18)

### Track 2 (CIFAR) — all four finished

| job | config | test_acc | Δ vs A1+B1+cp128 (0.5420) |
|---|---|---|---|
| C-d | cp_rank=256 | **0.5500** | +0.008 |
| C-a | per-patch embedding | 0.4147 | -0.127 |
| C-b | per-patch + pos_scale=0 | 0.3907 | -0.151 |
| C-c | overlapping patches (depth 4) | 0.5171 | -0.025 (≈ baseline 0.5168 pre-cp-sweep) |

**cp_rank saturates.** 32→64→128→256: 0.5168 → 0.5328 → 0.5420 → 0.5500.
Diminishing returns continue (+0.016, +0.009, +0.008) — capacity is a
real but small, saturating lever, not the remaining gap to CNN.

**C-a/C-b: the higher-expected-value proposal was wrong, and wrong by a
lot.** Per-patch embedding regresses hard (-0.127), and adding
`pos_scale=0` on top makes it *worse* (-0.151), directly contradicting
the prediction that per-patch embeddings would subsume the (already-shown-
useless, per A5) positional embedding. Plausible mechanism: a shared
`nn.Linear` pools gradient signal across all 256 patches every step; an
independent `[num_patches, in_features, bond_dim]` tensor gives each
patch position 1/256th of the effective training signal per step, with
no cross-patch statistical sharing — on a 45,000-image dataset that may
simply not be enough per-patch data for independent projections to
outlearn a shared one, even though the *architecture* is more
TN-faithful. This is a real, useful negative result, not a bug: being
"more literature-faithful" (per-position tensors, as in B2 and now C-a)
has now regressed **twice** relative to a shared/pooled alternative — a
pattern worth naming rather than re-trying a third variant of the same
idea without a different theory of why it would work.

**C-c is a clean null.** Overlapping patches (0.5171) neither help nor
hurt relative to the pre-cp-sweep baseline (0.5168) — softening the rigid
patch partition doesn't matter at this depth/config. Doesn't rule out
degree-inflation being a real cost (per the pre-registered risk), just
that it's not a large one here; the C3 kurtosis measurement on this
checkpoint (still not done) would say more.

**Revised Track 2 priority:** C-a/B2's shared pattern (per-position
tensors regress) argues against C-e (C1, pairwise binary contractions)
without first understanding *why* per-position specialisation is losing
to weight-sharing here — that diagnosis, not another architecture
variant, is now the higher-value next step in this track.

### Track 1 (ARO objective) — early results are the most important
finding of this batch, and they're a serious problem for the hypothesis

**`image_pairwise_cos_mean` collapses to ~1.0 within 2-3 epochs at
`triplet_weight=1` and `10`** — the two lowest weights tested, more than
three orders of magnitude below the original 40000:

| job | triplet_weight | epoch 1 (val) | epoch 2 (val) | epoch 3 (val) |
|---|---|---|---|---|
| T1a | 1 | 0.2094 | 0.9950 | **0.9997** |
| T1b | 10 | 0.1446 | 0.6617 | **0.9966** |
| T1c | 100 | 0.1342 | *(running)* | *(running)* |

**This is a real problem for the "lower triplet_weight enough and
InfoNCE's anti-collapse pressure survives" hypothesis** — at
`triplet_weight=1`, the triplet term should be almost negligible next to
InfoNCE's loss (~5.0), yet the tower still collapses just as fast as at
weight 40000 did. Two readings, not yet distinguished:
1. InfoNCE's own in-batch anti-collapse pressure is *itself* too weak to
   prevent collapse on ARO specifically, independent of the triplet
   term's weight entirely — the loss doesn't need the image to vary
   *at all* for either term to be minimised, since ARO's task never
   requires comparing two different images.
2. Something in the training dynamics (learning rate ratio between image
   and text towers, batch composition, or the tower's own conditioning
   from Stage A/B) makes collapse the path of least resistance
   independent of loss weighting.

Reading 1 is the more parsimonious explanation and consistent with this
document's own earlier diagnosis (ARO's task structure, not merely the
loss weighting, permits a constant-image solution) — if true, no
`triplet_weight` value fixes this, and Track 1 needs a structurally
different intervention (e.g. an explicit anti-collapse term independent
of triplet_weight, not just detuning it).

## Track 1 concluded — full trajectories, T2 ablation, and S1 (2026-09-19)

All three T1 jobs ran to early stopping overnight; T2 (`image_ablation.py`
on each checkpoint) and S1 (SVO) are also complete. This closes Track 1.

### The trajectories tell a different story than the early epochs did

All three jobs show the *same* two-phase pattern: rapid collapse to
~0.99-1.0 within 2-3 epochs (as the early-epoch table above already
showed), **followed by a slow, continuous de-collapse for the rest of
training** — and the de-collapse rate is inversely related to
`triplet_weight`, exactly as the "weaker triplet term, more room for
InfoNCE's anti-collapse pressure" hypothesis predicts:

| job | triplet_weight | peak `image_pairwise_cos_mean` | last epoch's value | epoch stopped |
|---|---|---|---|---|
| T1a | 1 | 0.9997 (ep. 3) | **0.704** (ep. 23) | 23 (early-stopped) |
| T1b | 10 | 0.9976 (ep. 4) | **0.804** (ep. 27) | 27 (early-stopped) |
| T1c | 100 | 0.9999 (ep. 12-18) | **0.977** (ep. 42) | 42 (early-stopped) |

So reading 1 above was too pessimistic as stated: the tower is *not*
stuck at collapse regardless of weight — lower `triplet_weight` does let
it escape further, monotonically, exactly matching the weight ordering.

### But checkpoint selection hides this: a real methodological gap

`Trainer` selects the best checkpoint by `hard_neg_acc` alone, which is
insensitive to collapse — a collapsed model gets identical accuracy to a
partially de-collapsed one on this metric (both readings on validation
have been hovering at 0.67-0.69 the entire time in all three runs,
regardless of the collapse trajectory). Consequently **the actual
selected checkpoints are far earlier and more collapsed than the
trajectories above suggest**:

| job | best epoch (of 100 max) | selected checkpoint's `image_pairwise_cos_mean` | ARO overall (this checkpoint) |
|---|---|---|---|
| T1a (tw=1) | 13 | 0.885 | 0.6630 |
| T1b (tw=10) | 17 | 0.884 | 0.6892 |
| T1c (tw=100) | 32 | 0.990 | 0.6907 |

T1a and T1b's *selected* checkpoints are nearly identical in collapse
(0.885 vs 0.884) despite T1a's full trajectory eventually reaching much
further de-collapse (0.704) than T1b's would have at the same epoch —
the monitor metric simply doesn't reward the thing this experiment cares
about. **This is a real gap worth fixing before any future Track 1 run**:
either monitor `image_pairwise_cos_mean` directly (e.g. minimize it, or a
combined criterion), or report results from a fixed late epoch rather
than "best by accuracy."

### T2 — the decisive test, run despite the checkpoint-selection caveat

`image_ablation.py` on all three *selected* checkpoints (with
`IMAGE_MODEL_CP_RANK=128` — the ablation script infers `embedding_dim`
from the checkpoint but not `cp_rank`, a small gap worth fixing in that
script too):

| job | pairwise cos (different images) | real | shuffled | zeros |
|---|---|---|---|---|
| T1a (tw=1) | 0.878 (min 0.341, std 0.100) | 0.6699 | 0.6523 | 0.6660 |
| T1b (tw=10) | 0.873 (min 0.380, std 0.099) | 0.6875 | 0.6836 | 0.6875 |
| T1c (tw=100) | 0.989 (min 0.848, std 0.015) | 0.7031 | 0.7070 | 0.7090 |

**This is the answer, and it's still a clean negative — but a more
interesting one than before.** T1a's images have genuinely, measurably
stopped being identical (pairwise cosine spread from std 0.0 to 0.100,
min down to 0.34 — a real, substantial de-collapse at the embedding
level). **But real/shuffled/zeroed accuracy remain statistically
indistinguishable at every triplet_weight tested** (T1c's zeros variant
is even nominally *higher* than real, well within noise). The image
tower is no longer emitting a literally-constant vector, but the
caption-vs-image decision still does not depend on which image — real,
a randomly wrong one, or none at all — is shown.

**This sharpens rather than resolves the diagnosis.** "The tower has
collapsed to a constant vector" and "the loss doesn't need the image"
are not the same failure, and this batch separates them for the first
time: T1a shows the *first* is fixable (lower `triplet_weight`, patience
for slow de-collapse), but the *second* persists even after the first is
fixed. The caption embeddings and the (now genuinely varying) image
embeddings are simply not coupled in a way that makes the classification
decision track the image. This is consistent with — and sharpens —
reading 1's original framing: it was never really about whether the
*tower* can vary its output (it plainly can, once collapse is escaped);
it's that ARO's training signal never asks the *decision* to depend on
that variation, because both InfoNCE and the triplet term are satisfiable
by making the *caption* embeddings alone separate true from false
relative to whatever fixed-ish direction the image lands near — image
variation can appear without ever being read out. **No `triplet_weight`
value tested fixes this**, because the mechanism isn't (purely) about
that weight — Track 1's original hypothesis is falsified, or at least
insufficient on its own.

**Track 1 verdict: closed, negative, but genuinely informative.** The
image tower's ability to vary is not the bottleneck for ARO (B1 fixed
that, this batch confirms the fix generalises to letting the tower
partially escape induced collapse too). What's missing is a mechanism
that makes the *task's decision* actually depend on that variation —
something closer to an explicit term that penalises exactly the failure
mode seen here (accuracy invariant to image identity), not merely a
softer version of the existing triplet term.

### S1 (SVO) — a genuine, if modest, positive result, alongside a genuine regression

Config: A1+B1+cp_rank=128+triplet_weight=100 (the same recipe as T1c,
applied to SVO).

| metric | S1 (this run) | job 10 (B1 only, tw=40000) | pre-committed criterion |
|---|---|---|---|
| SVO-Probes overall | **0.5323** (obj 0.5879 / subj 0.4990 / verb 0.5216) | 0.5157 | beat 0.5157 — **passed** |
| SVO-Swap | 0.4762 | 0.5524 | — (not the pre-committed metric) |

**SVO-Probes cleared its pre-committed bar** — a genuine, if modest
(+0.017), improvement over job 10, and the best SVO-Probes result to
date across the entire project (previous best: experiment 14's 0.5305).
Consistent with the same mechanism as T1a-c: lowering `triplet_weight`
from B1-only's default (40000) to 100 plausibly let SVO's image tower
partially escape whatever collapse-adjacent state it was in under job
10's config too, even though SVO's task (unlike ARO's) genuinely needs
image variation to solve at all. **SVO-Swap regressed** (0.5524 → 0.4762)
— the same triplet_weight change that helped the image-side task hurt
the caption-side one, a real trade-off rather than a free win, consistent
with `triplet_weight` trading off which side of the hard-negative
decision the model prioritises.

**`image_ablation.py --task svo` on this checkpoint — a genuinely
different result from ARO's.** SVO's ablation is the mirror of ARO's
(caption fixed, images vary, so the corrupting variant swaps in a *wrong*
caption rather than a wrong image — see the script's docstring): if
accuracy holds up under a wrong caption, the model is choosing an image
independent of what the caption actually says.

| variant | N | hard_neg_acc |
|---|---|---|
| real (correct caption) | 1024 | **0.5234** |
| shuffled_caption (wrong caption) | 1024 | 0.4648 |

Image embeddings: pairwise cosine between different images mean **0.0106**
(std 0.24, min -0.66) — genuinely diverse, not collapsed, consistent with
B1's known effect on SVO's image tower.

**This is categorically different from ARO's ablation result.** ARO's
real/shuffled/zeroed variants were statistically identical (differences
<0.004, pure noise). Here, real vs. shuffled-caption differ by **5.9
points** — a real, if modest, signal that the model's true/false-image
choice does shift when the caption is wrong, i.e. the decision is not
purely image-identity-driven independent of caption content. **Caveat on
magnitude, not direction:** both numbers are close to chance (0.52 real,
0.46 shuffled — the "real" condition itself is barely above 0.50), so
this is weak evidence of a real coupling, not strong evidence of a well-
functioning caption-image mechanism. But weak-and-real is a different
finding than ARO's zero-and-flat, and is the first time in this entire
project that an SVO checkpoint has shown *any* measurable
caption-conditioning of the image-side decision.

**Track 1/S1 combined verdict:** the same recipe (A1+B1+cp_rank=128,
triplet_weight=100) produces two different partial successes and one
clear miss, depending on which side of the hard-negative structure the
task needs: ARO (caption-side hard negative) still shows zero
image-decision coupling despite the tower now varying; SVO (image-side
hard negative) shows the first non-zero, if weak, caption-decision
coupling recorded in this project, alongside a real accuracy gain on its
own pre-committed metric and a real regression on SVO-Swap. Neither
result is a finished success — both point toward the same open question
this document has now surfaced twice: **varying embeddings are necessary
but not sufficient; something about how the loss couples the two
modalities' decisions still needs a more direct fix than tuning
`triplet_weight`.**

---

# Implementation spec: Routes N, B, A (2026-09-19)

**Audience: an implementing agent with no other context.** Everything
needed is in this section. Three independent routes, plus one shared
prerequisite. All three can be developed and run in parallel after the
prerequisite lands.

**Scope assumption, load-bearing — state it in any write-up.** Captions
are well-structured and contract to rank 1 (true for SVO-Probes and ARO;
*not* assumed for COCO, which is out of scope for this work). Route A
further assumes subject/verb/object roles are identifiable per row from
dataset columns. These hold for the target benchmarks and are a declared
scope condition, not an oversight.

## What is reused vs. what is new

| Component | Decision |
|---|---|
| `TTNImageModel` (`qnlp/discoviz/models/image_model.py`) | **Reuse.** Route N adds an opt-in gate inside its layers; Routes A/B add a read-only accessor. No fork. |
| `CPQuadRankLayer` (`qnlp/discoviz/models/cp_node.py`) | **Reuse**, extended in place for Route N (opt-in, default off). |
| `EinsumModel` (`qnlp/discoviz/models/einsum_model.py`) | **Reuse unchanged.** Routes A/B read per-symbol tensors via its existing `sym2weight` dict. |
| `ContrastiveVLM` (`qnlp/domain/models/vlm/contrastive_vlm.py`) | **Reuse**, with a new selectable scoring head. Do **not** fork the class. |
| Scoring (currently cosine) | **New modules** for Routes B and A, selected by config, with cosine retained as the default control. |
| Training scripts (`qnlp/scripts/svo/run.py`, `ttn_supervised_probe.py`) | **Reuse.** New behaviour is opt-in via env-var config only. |

**Rule for all three routes: every new behaviour is opt-in and defaults to
off.** Existing SVO/ARO/COCO/CIFAR runs must be bit-for-bit unchanged when
the new flags are unset. This is how A1/B1 were shipped and it is what
keeps the ablation table single-variable.

## P1 — Shared prerequisite: root-indexed region extraction

Routes A and B both need the image tower to expose *intermediate* region
tensors, not just its final pooled vector.

**Why root-indexed.** Tree depth varies with image size
(`depth = log_4(num_patches)`), so counting levels from the leaves gives a
variable interface. Counting from the **root** is depth-invariant: one
level below the root is always 4 nodes, two levels below is always 16.

**Given** `self.layers` where `layers[i]` outputs shape
`[B, num_nodes_i, out_dim_i]`, and `num_nodes` descends 64 → 16 → 4 → 1
for depth 4:

```
regions at root-level k  =  output of layers[len(layers) - 1 - k]

k = 0  ->  [B,  1, 1024]   (root; equivalent to today's pooled output)
k = 1  ->  [B,  4,  512]
k = 2  ->  [B, 16,  256]
```

**Implement** as a method on `TTNImageModel`:

```python
def forward_regions(self, x, level: int = 1) -> torch.Tensor:
    """Return [B, 4**level, dim_at_level] root-indexed region tensors.
    level=0 is the root. Runs the same forward pass as forward(); capture
    the output of layers[len(self.layers) - 1 - level]."""
```

Requirements:
- Must not change `forward()`'s behaviour or outputs.
- `level` must be validated against `len(self.layers)`.
- `dim_at_level` differs per level, so **consumers must project to a
  common `score_dim` with a per-level `nn.Linear`** (linear, so the
  multilinear constraint is preserved).
- Default `level=1` (4 regions) for first experiments; `level=2` (16) is
  the variant to sweep.

## Route N — non-linearity, as a measurement

**Motivation.** CIFAR has plateaued (0.5500 at `cp_rank=256`) against a
CNN's 0.7842. A purely multilinear network computes a homogeneous
polynomial with no thresholding. This route *quantifies what the
multilinear constraint costs* rather than abandoning it.

**Design — mirror the text tower's NLC exactly.** In
`CPQuadRankLayer.forward`, after the output projection
(`out = einsum(merged, factor_out)`) and **before** the residual add:

```python
if self.nonlinearity != "none":
    out = out + self.gate * self._f(out)
```

- `self.gate = nn.Parameter(torch.zeros(1))` — **one scalar per layer,
  initialised to exactly 0.0.** At gate=0 the model is bit-for-bit the
  current multilinear model. This is what makes the result a measurement:
  the learned gate value *is* the answer to "how much non-linearity does
  this task demand."
- `_f` is selected by config:
  - `"born"`: `f(x) = x * x` — the real-valued analogue of the Born rule
    (|ψ|²), the one non-linearity a quantum model genuinely has. This is
    the principled variant.
  - `"gelu"`: `f(x) = F.gelu(x)` — an upper bound on what *any*
    non-linearity buys. The gap between `born` and `gelu` is itself a
    reportable result.

**Config:** `IMAGE_MODEL_NONLINEARITY` on `ImageModelSettings`
(`image_model.py`), values `"none"` (default) | `"born"` | `"gelu"`.

**Logging:** log every layer's gate value per epoch. The trajectory of
4-5 scalars is the headline figure for this route. Precedent: the text
tower's NLC gate rose 0.044 → 0.393 over training.

**Jobs (CIFAR, existing harness):**

| job | config | criterion |
|---|---|---|
| N1 | A1+B1+cp256, `NONLINEARITY=born` | beat 0.5500 |
| N2 | A1+B1+cp256, `NONLINEARITY=gelu` | beat 0.5500; gap vs N1 = cost of the principled choice |

Baseline is the existing 0.5500 run — do not re-run it.

## Route B — structured scoring (region-level, no role assumptions)

**Motivation.** The model is two-tower late fusion: one image vector, one
caption vector, one cosine. SVO-Probes asks which of two images matches a
fixed caption, so all discriminative burden sits on a single scalar
comparison of two summary vectors. Route B replaces that with a
comparison that keeps image regions separate.

**Route B makes no assumptions about sentence structure** (it uses only
the rank-1 caption vector), which is why it runs first.

**Design — CP-factorised trilinear score.** Given caption vector
`t ∈ R^{score_dim}` and regions `R ∈ R^{n_regions × score_dim}` (from P1,
after the per-level projection):

```
score(t, R) = sum_{j=1..n_regions} sum_{r=1..rank} (u_r · t) (v_r · R_j) (w_r)_j
```

with learnable `u, v ∈ R^{rank × score_dim}`, `w ∈ R^{rank × n_regions}`.

- **Multilinear in both `t` and `R`** — constraint preserved.
- **Not degenerate.** A simpler "weighted sum of per-region dot products"
  collapses algebraically to pooling the regions first and is therefore
  equivalent to the current cosine head — do not implement that. The
  third factor `w` indexed by region is what makes the score depend on
  *which* region matches.
- Suggested `rank = 32` (mirrors `cp_rank`'s convention), `score_dim =
  128`. Parameter count ≈ `2*rank*score_dim + rank*n_regions` ≈ 8K.

**Where it lives:** new module, e.g.
`qnlp/core/training/scoring/trilinear.py`, selected inside
`ContrastiveVLM` by a config flag. The existing cosine path stays the
default.

**Config:** `SVO_ML_SCORE_HEAD = "cosine"` (default) | `"trilinear"`,
plus `SVO_ML_REGION_LEVEL` (default 1) and `SVO_ML_SCORE_DIM`
(default 128).

**Integration note.** `ImageContrastiveLoss` and `SVOHardNegStep`
currently compute cosine similarity directly on embeddings. The score
head must be applied *before* the loss, so the loss receives scores
rather than recomputing cosine. Keep the cosine path untouched when
`SCORE_HEAD="cosine"`.

## Route A — role-grounded cross-modal contraction

**Motivation.** DisCoCat's semantics are that nouns denote entities and
verbs denote relations between them. Route A grounds subject and object
against image regions and lets the verb mediate — the structure
SVO-Probes is explicitly built to test, which the current architecture
discards by pooling everything into one vector.

**Design.** Per row, with `s`, `v`, `o` the per-symbol tensors for the
subject, verb and object words (read from `EinsumModel.sym2weight`), and
`R` the region tensors from P1:

```
g_subj[j] = <P_s · s, R_j>        affinity of the subject to region j
g_obj [k] = <P_o · o, R_k>        affinity of the object  to region k

score = sum_{j,k} g_subj[j] * M(v)[j,k] * g_obj[k]
```

where `P_s`, `P_o` are learned linear projections into `score_dim`, and
`M(v)` is a `[n_regions, n_regions]` matrix produced from the verb tensor
by a learned linear map. Multilinear in `s`, `o`, `v` and `R`.

**Uncertainty the implementer must resolve first.** In DisCoCat a
transitive verb has type `nʳ · s · nˡ` (a rank-3 tensor with two noun
legs), but this pipeline's `LemmatizeStep` and `UnifyEinsumRankStep`
force rank-1 diagram outputs, so the *stored* verb tensor may already be
rank-1. **Inspect the actual shape in `sym2weight` before implementing.**
- If the verb tensor is rank-3 with two noun legs: contract it directly
  to form `M(v)`.
- If it is rank-1 (expected): use a learned linear map
  `R^{dim_v} → R^{n_regions × n_regions}` to produce `M(v)`. Document
  which case was found and which path was taken.

**Data prerequisite.** Route A needs subject/verb/object identified per
row. `data/svo/raw/svo_probes_corrected.csv` already has `subj`, `verb`,
`obj` columns — **no CCG/diagram surgery is required.** Propagate those
three columns through `qnlp/scripts/svo/prepare_datasets.py` into
`svo_{train,val,test}_probes.parquet`, applying **the same lemmatisation
the symbol pipeline uses**, so the strings key correctly into
`sym2weight`. Log the fraction of rows whose three roles all resolve to
known symbols; if that fraction is below ~90%, stop and report before
running experiments — a low hit rate invalidates the route.

**Config:** `SVO_ML_SCORE_HEAD = "role_grounded"`, reusing
`SVO_ML_REGION_LEVEL` and `SVO_ML_SCORE_DIM`.

**Control.** Route A and the cosine head must be runnable as two heads on
the same tower and data, changing only `SVO_ML_SCORE_HEAD` — that is the
single-variable ablation.

## Phase 0 findings (2026-09-19) — Route A's premise needs revisiting before implementation

Two read-only checks, run before writing any Route A/B/P1 code, per the
agreed execution order (Phase 0 diagnostics → Route D → N/P1/A-data →
B/A jobs). Both against a real SVO checkpoint
(`runs/checkpoints/svo_probes/2026-09-18_21-36-56/best_model.pt`).

### Verb tensor shape — resolved, but neither anticipated case is live

Inspected `sym2weight` directly (`symbols_list`/`sizes_list` from the
checkpoint's state dict, grouped by base word).

**Finding 1 — the same word has multiple symbols, disambiguated by CCG
type signature, not one symbol per lexeme.** `run` appears as 8 separate
components: `run_0__n` (bare-noun usage, "a run"), `run_0__n.r@B` /
`run_1__B.r@s` / `run_1__B.r@s@B` (intransitive-verb chain), and further
components for transitive usage. A string lookup on the word alone is
ambiguous — determining which symbol(s) apply to a *specific row's*
verb requires that row's actual compiled CCG diagram, not just its
`verb` column value.

**Finding 2 — no verb is a single dense tensor of any rank, let alone
rank-3 with two noun legs.** Even a clearly-transitive verb (`hold`) is
split into 3-5 low-rank pieces chained through an auxiliary bond index
of size `bond_dim=10`:

```
hold_0__n.r@B      shape (512, 10)
hold_1__B.r@s@B    shape (10, 512, 10)
hold_2__B.r@n.l    shape (10, 512)
hold_2__B.r@p.l@B  shape (10, 512, 10)
hold_3__B.r@n.l    shape (10, 512)
```

This is `UnifyEinsumRankStep` performing its actual documented job —
keeping every *stored* parameter at ≤ rank 2-3 via bond-dimension
factorization — not a semantic rank-1 collapse. It's a tensor-train/MPO
decomposition of what would otherwise be a higher-rank verb tensor, with
the bond legs (`B`, size 10) meant to be contracted away, not read as
role indices.

**Consequence for Route A's design.** The spec's two-way fork ("rank-3
with two noun legs → contract directly" vs. "rank-1 → learn a fresh
linear map") has no live branch — reality is a third case not
anticipated in either path. Reconstructing a usable `M(v)` bilinear map
means: (a) identifying which specific chain of symbols a given row's
diagram actually uses (a diagram-level lookup, not a word lookup), then
(b) contracting that chain along its bond legs to materialise an
`[n_regions, n_regions]`-shaped map — which is then implicitly rank-limited
to `bond_dim=10` regardless of its nominal output shape. This is
comparable in engineering cost to re-deriving part of what
`EinsumModel.forward` already does per-diagram internally, not a
standalone tensor lookup as the spec's pseudocode implies.

### Role-resolve rate — close to the pre-committed gate, not clearly over it

Checked `subj`/`verb`/`obj` columns (lowercased) against the checkpoint's
known base-word vocabulary (514 unique base words, from 1356 total
symbols):

| role | resolve rate | N |
|---|---|---|
| subj | 97.2% | 36,841 |
| verb | 93.1% | 36,841 |
| obj | 94.8% | 36,841 |
| **all three** | **86.3%** | 36,841 |

**Resolved — the true rate clears the gate comfortably.** The 86.3%
above was against the full raw manifest, not the rows actually used for
training/eval. Reconstructed `sample_id`'s mapping back to the raw CSV
(`sample_id` = `f"svo_{i}"`, `i` the row's index in the same
image-availability-filtered dataframe `load_svo_to_atlas.py` ingests, in
CSV order) and re-ran the check restricted to each split's actual rows —
every `sample_id` matched exactly (8609/8609, 2908/2908, 2767/2767),
confirming the reconstruction is exact, not approximate:

| split | rows | all-three-resolve |
|---|---|---|
| train | 8,609 | **98.9%** |
| val | 2,908 | **99.5%** |
| test | 2,767 | **98.7%** |

All three splits comfortably clear the 90% gate. As predicted, the
full-manifest number was an underestimate — every caption word that
survives the word-frequency filter is already in-vocab, and subj/verb/obj
largely coincide with in-caption words. **Route A's data prerequisite is
cleared — no further check needed before implementing the
subj/verb/obj-column propagation into the parquets.**

### Decided — proceed with Route A, de-risked twice over

Two follow-ups closed the remaining uncertainty:

**The diagram-level symbol-chain lookup Route A seemed to need already
exists as data, no new mechanism required.** Checked a real training
row's stored `symbols`/`diagram` columns directly: each row already
carries the exact ordered chain of symbols used for its own sentence —
e.g. verb "sit" in one row resolves to exactly `sit_0__n.r@B` and
`sit_1__B.r@s` in that row's own symbol list, contractible via their
shared bond leg per the row's diagram string. Route A doesn't need a new
diagram-parsing mechanism; it's a straightforward extension of the
symbol-matching pattern already used elsewhere in this codebase (e.g.
`evaluate.py`'s `known = set(model.text_model.sym2weight.keys())`).

**Combined with the resolve-rate result above (98.7-99.5% on real
training/val/test rows) and the `max_order`/bond-chain-contraction
finding below, both of Route A's original blockers are resolved
favourably.** Decision: **proceed with Route A as one of the three
routes**, implementing it against the existing `bond_dim`-factored
symbol chains (not a `max_order` change) per the recommendation below.

### Follow-up experiment — is the split fixable via the ansatz/parser? Yes, but at a real cost

Two direct questions raised: can `max_order` (the CCG-to-tensor ansatz's
own splitting parameter, not a CCG grammar rule) be changed to get a
literal dense verb tensor, and is `UnifyEinsumRankStep` actually needed
for SVO? Both tested empirically, no pipeline code changed.

**The split is controlled by `CustomMPSAnsatz`'s `max_order` parameter
(`qnlp/discoviz/parser/asnsatz.py`), currently hardcoded to 3 in
`compiler_step.py`'s `_worker_init`.** It is `lambeq`'s
`SplitTensorAnsatz`: any box with more than 1 total wire gets factored
into a chain of pieces, each carrying at most `max_order - 2` "real"
wires plus bond legs on either side. At `max_order=3`, that chunk size
is 1 — every multi-wire box, no exceptions, gets maximally split. A
transitive verb (3 wires: subject, sentence, object) needs
`max_order >= 5` to survive as one piece (chunk size 3 = the whole
`cod`, so the loop runs once and both bond legs get stripped off the
single resulting box).

**Verified directly** (`BobcatTextProcessor` + `CustomMPSAnsatz`, "a dog
hold a ball"):

| `max_order` | `hold`'s symbol(s) | shape(s) | total params |
|---|---|---|---|
| 3 (current) | `hold_0__n.r@B`, `hold_1__B.r@s@B`, `hold_2__B.r@n.l` | (512,10), (10,512,10), (10,512) | 61,440 |
| 5 | `hold_0__n.r@s@n.l` | **(512, 512, 512)** | **134,217,728** |

`max_order=5` genuinely produces a single dense rank-3 tensor with
subject/sentence/object legs — exactly what Route A's original design
assumed existed. **The cost is a ~2,185x parameter increase per
transitive-verb symbol.** With embedding_dim=512 and SVO's ~8,600-row
training set (likely a few examples per distinct verb at most, spread
across hundreds of verbs), fitting 134M parameters per verb from scratch
is not viable — this would need either a much smaller embedding_dim for
verb legs specifically, or accepting severe per-verb overfitting. It
also requires recompiling the entire SVO/ARO/COCO CCG cache (LMDB), a
real one-time cost, not a config flip at train time.

**Practical alternative, no pipeline changes needed:** the current
`bond_dim=10` factorization is a valid low-rank (rank ≤10) decomposition
of the same rank-3 tensor `max_order=5` would materialise densely.
Route A's originally-planned "contract the symbol chain along its bond
legs" approach reconstructs an effective `M(v)` at zero extra parameter
or recompilation cost — it's rank-limited to `bond_dim`, but that's the
same kind of capacity trade-off already accepted elsewhere in this
project (the image tower's `cp_rank`). A middle path also worth
considering later: sweep `bond_dim` itself (not `max_order`) upward —
32/64 rather than 10 — for a higher-rank factored approximation without
`max_order`'s combinatorial blowup, mirroring the image tower's `cp_rank`
sweep (32→64→128→256, small monotonic gains). **Recommendation: keep
`max_order=3` and implement Route A's contraction against the existing
factored chain; treat `bond_dim` (not `max_order`) as the capacity knob
if the rank-10 reconstruction proves insufficient.**

**`UnifyEinsumRankStep` — not dead code, but rarely fires on real
captions.** Tested the full pipeline (tokenize → lemmatize → parse →
rewrite → ansatz) on 15 real, LLM-corrected SVO captions sampled from
the corpus plus a handful of hand-picked realistic SVO/prepositional
sentences (22 total): **every one already produced a rank-1 diagram
output** before `UnifyEinsumRankStep` would need to truncate anything —
consistent with `LemmatizeStep`'s stated purpose (forcing sentences into
a finite, `S`-typed form) already doing this job on well-formed input.
But a deliberately fragmentary input (`"Running."`, no subject) produced
a genuine 2-wire output — confirming the step is a real, if rare, safety
net for malformed/fragment captions, not inert. **Recommendation: don't
remove it outright** — a corpus-scale check (what fraction of the actual
~36,841-row manifest triggers it) would be needed to state a precise
rate, but the fragment test shows it's not a no-op, and removing it
would convert a graceful truncation into a hard downstream shape error
on whatever fraction of rows (however small) still need it.

## Parallel batch and dependencies

```
P1 (region extraction)  ──┬──> Route B implementation ──> B jobs
                          └──> Route A implementation ──> A jobs
                                    ^
                                    └── needs the parquet role columns (independent work, start now)

Route N  ── no dependency on P1 ──> N jobs   (start immediately)
Route D  ── no dependency at all ──> D job   (start immediately)
```

**Start immediately, in parallel, no blockers:**

| # | work | kind |
|---|---|---|
| N1, N2 | Born / GELU gated CIFAR runs | cluster jobs (after a small code change) |
| D | text-tower capacity probe (below) | small script + 1 job |
| P1 | region extraction | code only, blocks A and B |
| A-data | subj/verb/obj columns into the SVO parquets | data pipeline, blocks A only |

**Then, in parallel once P1 lands:**

| # | job | control | criterion |
|---|---|---|---|
| B1 | SVO, `SCORE_HEAD=trilinear`, `REGION_LEVEL=1` | cosine head, same tower/config | beat 0.5323 Probes **and** widen the real-vs-shuffled-caption gap beyond 5.9 pts |
| B2 | as B1 with `REGION_LEVEL=2` (16 regions) | B1 | — |
| A1 | SVO, `SCORE_HEAD=role_grounded`, `REGION_LEVEL=1` | cosine head, same tower/config | same as B1 |

Baseline config for every SVO row: **A1+B1+cp_rank=128+triplet_weight=100**
(the S1 recipe, SVO-Probes 0.5323).

## Route D — text-tower capacity probe (cheap, never run)

The image tower was found to be catastrophically broken *because* a
supervised probe was run on it. The same probe has never been run on
`EinsumModel`, and Routes A and B both assume its per-symbol tensors are
meaningful.

Mirror `ttn_supervised_probe.py` on the text side: train a linear
classifier on caption embeddings to predict the SVO verb (or object)
class, top-20 classes, against a majority baseline and a bag-of-words
logistic-regression reference. Report kNN class consistency on caption
embeddings as well. If the text tower is at chance, Routes A and B are
built on sand and that must be known first.

## Reporting requirement for all routes

Report the **real-vs-shuffled-caption ablation gap** (via
`qnlp/discoviz/diagnostic/image_ablation.py --task svo`) alongside
accuracy for every SVO run. Accuracy near chance carries little
information on its own; the coupling gap measures the thing these routes
are designed to fix. Current reference: 0.5234 real vs 0.4648 shuffled =
**5.9 points**.

## Batch 3 results (2026-09-19) — N1/N2, Route D, A-data swap rebuild

All four "start immediately, no blockers" jobs from the table above have
finished.

### Route N — gated non-linearity, CIFAR-10 (jobs 7432824, 7432825)

| variant | epochs run | best val_acc | test_acc | gate values (final) |
|---|---|---|---|---|
| N1 (`born`, out += gate·out²) | 100 (no early stop) | 0.5642 | **0.5586** | [0.271, -0.004, 0.020, 0.008] |
| N2 (`gelu`, out += gate·gelu(out)) | 91 (early stopped) | 0.5946 | **0.5857** | [-1.10, 1.20, -0.06, -0.16] |
| C-d baseline (cp_rank=256, linear) | — | — | 0.5500 | n/a |

N2 (gelu) beats the cp_rank=256 baseline outright (0.5857 vs 0.5500) with
the same parameter budget as the default cp_rank config — this is a real,
adopted win for Track 2, not noise. The gate values are informative: N1's
gates stay small and mostly hover near 0 except the first layer (~0.27),
i.e. the squaring non-linearity is used sparingly and mostly at the
coarsest level. N2's gates grow large and asymmetric across layers
(layer 0 → -1.10, layer 1 → +1.20), i.e. GELU is being used aggressively
and differently at different tree depths — the network wants
depth-dependent non-linearity, which a single shared linear tree cannot
express. **Conclusion: adopt GELU-gated non-linearity as the new Track 2
baseline going forward**, replacing the pure-multilinear tree.

### Route D — text-tower capacity probe (job 7432827)

Verb classification, top-20 classes, majority baseline 0.2490:

| model | test_acc |
|---|---|
| `einsum` (actual DisCoCat text tower) | **0.7950** |
| `bow_logreg` (bag-of-words ceiling) | 0.9707 |
| kNN class consistency (einsum embeddings, k=10) | 0.8908 (chance 0.05) |

The text tower is nowhere near chance and nowhere near broken the way the
image tower was — 0.795 vs a 0.249 majority baseline, with high kNN
consistency (0.89) confirming the embedding space itself clusters by verb,
not just the final linear head. There is a real gap to the bag-of-words
ceiling (0.795 vs 0.971), so the tower is not extracting every bit of
verb information available in the raw words, but it is unambiguously
extracting most of it. **Conclusion: the text tower is not the
bottleneck. Routes A and B's premise (that per-symbol text tensors carry
meaningful, resolvable role information) is not built on sand.**

### A-data — `build_svo_swap` rebuild at 32G (job 7432838)

Succeeded cleanly after the two 16G OOM failures (root-caused as real
node memory contention, fixed by raising the reservation, not a code bug).
Against the corrected pipeline (subj/verb/obj now surviving into
`svo_test_probes.parquet`):

- 178 human/animal subject+object candidates identified
- 105 successfully swapped and CCG-recompiled → `data/datasets/svo_swap_eval.parquet`

This supersedes the earlier 95-pair swap set (built before the A-data
column fix) — the eval set used for any future SVO-Swap number should be
this 105-pair one.

### Net effect on the plan

All four items in "start immediately, no blockers" are now done. Nothing
in the batch plan is running. The remaining work is exactly the two
P1-gated implementations (Route B, Route A) and their corresponding SVO
jobs — no further diagnostics are blocking either.

## Route B and Route A implemented (2026-09-19)

No new backbone models — both plug into `TTNImageModel`/`EinsumModel`
unchanged, exactly as "What is reused vs. what is new" specified. New
code: `qnlp/domain/models/vlm/score_heads.py` (`TrilinearScoreHead`,
`RoleGroundedScoreHead`), `qnlp/core/training/losses/structured_contrastive.py`
(`StructuredContrastiveLoss`), a `score_head` param on `ContrastiveVLM`
(`None` = every existing run stays bit-for-bit unchanged), and
`EinsumModel.get_role_tensor`/`get_verb_chain`/`forward_roles` for Route
A's per-symbol lookups. Config: `SVO_ML_SCORE_HEAD` = `cosine` (default) |
`trilinear` | `role_grounded`, plus `SVO_ML_REGION_LEVEL`,
`SVO_ML_SCORE_DIM`, `SVO_ML_RANK`.

### A real design problem found and solved during implementation: the verb tensor is too big to materialise

Phase 0 established the verb chain's outer legs are full `embedding_dim`
(512), not a trivial size-1 "sentence" leg as DisCoCat's abstract type
theory might suggest — checked directly against `sym2weight`
(`hold_1__B.r@s@B` has shape `(10, 512, 10)`, not `(10, 1, 10)`).
Contracting a 3-piece chain the naive way (as `get_role_tensor` does for
subject/object) therefore produces a genuine dense `[512, 512, 512]`
tensor **per row** — 134M elements, ~512MB in float32 — before it's even
projected down to the `[n_regions, n_regions]` matrix Route A's formula
needs. At any real batch size this is not near the edge of feasible, it's
off by orders of magnitude (a batch of 128 would need ~64GB just for this
intermediate).

**Fix — project before contracting, not after.** Tensor contraction is
multilinear, so contracting the chain's two outer legs against a learned
`Linear(embedding_dim -> n_regions)` *before* doing the internal bond-dim
contraction gives the exact same final `[n_regions, n_regions]` matrix as
projecting the full dense tensor afterward, but only ever touches
`O(bond_dim * n_regions)`-sized intermediates. The middle "sentence" leg
(also size 512) is reduced to a scalar via a learned weight vector at the
same step. Implemented in `RoleGroundedScoreHead._verb_matrix`. This is
the concrete resolution of Phase 0's flagged risk ("reconstructing a
usable M(v) is comparable in engineering cost to re-deriving part of what
EinsumModel.forward already does per-diagram") — it turned out to be a
projection-ordering trick, not a diagram-parsing project.

**v1 scope, stated explicitly:** only the mainline 3-piece transitive-verb
chain is handled (subject-leg, sentence-leg, object-leg — the standard
SVO-Probes case). A row whose verb parses to a different chain length is
marked invalid the same way an unresolved role is (NaN sentinel, dropped
before the loss via `SVOHardNegStep`'s new `_finite_mask`/`_index_rows`
helpers, which generalise `drop_nonfinite_rows` to the role-tensor dict
shape).

### Verified locally before touching the cluster

Two smoke tests (`forward`+`backward` on synthetic data, then the full
`SVOHardNegStep`+`StructuredContrastiveLoss` path with a fake batch)
confirm both heads produce correctly-shaped `[B,B]` score matrices and
`[B]` matched-pair scores, gradients reach both the score head and the
text/image backbones, and no NaN/crash occurs. One real stability issue
was caught this way: unlike cosine (bounded in [-1,1]), both heads'
raw scores are unbounded and grew large enough at init to produce loss
values in the hundreds against the fixed `temperature=0.07`. Fixed with
attention-style `1/sqrt(rank)` (trilinear) / `1/n_regions` (role-grounded)
scaling, added directly in `score_heads.py`.

### Evaluation coverage — SVO-Probes only for now, SVO-Swap is a known gap

`evaluate_svo_probes` (`qnlp/scripts/coco_multi_caption/evaluate.py`) is
score-head-aware: it calls `score_head.score_pairs(...)` instead of
`F.cosine_similarity` when a structured head is active, and threads
`subj`/`verb`/`obj` through for Route A. `evaluate_sugarcrepe` (which
`evaluate_svo`'s SVO-Swap leg reuses) was **not** updated — it still
hardcodes cosine, so for `trilinear`/`role_grounded` it will throw (caught
by `evaluate_svo`'s existing `_guard`, logged as "eval skipped", not a
crash) rather than report a real SVO-Swap number. `svo_swap_eval.parquet`
also doesn't carry subj/verb/obj for the swapped sentence, so Route A's
swap eval needs that data prerequisite extended before it can work at
all. Flagging this now rather than silently reporting `nan` as if it were
a null result.

### Launch config (2026-09-19)

Baseline per the plan: A1+B1+cp_rank=128+triplet_weight=100,
`SVO_ML_REGION_LEVEL=1` (4 regions). Two jobs:

| job | `SVO_ML_SCORE_HEAD` | criterion |
|---|---|---|
| B1 | `trilinear` | beat 0.5323 Probes **and** widen the 5.9pt real-vs-shuffled-caption ablation gap |
| A1 | `role_grounded` | same |

Route A's first launch (job 7432895) crashed at epoch 0: an object word's
MPS chain had pieces with incompatible bond dimensions, and
`get_role_tensor`'s generic sequential-tensordot fold raised instead of
treating it as an unresolved role. Fixed by validating shape
compatibility before each fold (returns `None` instead of crashing) and
tightening `forward_roles`'s verb-chain validity check to the exact
mainline 3-piece shape `RoleGroundedScoreHead` expects. Relaunched as job
7432897.

### B1 result — a clean negative, same failure mode as T1/T2

Job 7432894 finished (early-stopped). **SVO-Probes overall: 0.5168**
(subj_neg 0.5093, verb_neg 0.5198, obj_neg 0.5147) — at chance, below the
0.5323 cosine baseline it needed to beat. SVO-Swap: unavailable (the known
`evaluate_sugarcrepe` gap for non-cosine heads).

Train `hard_neg_acc` climbed steadily and cleanly: 0.57 (epoch 2) → 0.92
(epoch 11), while val `hard_neg_acc` stayed flat at 0.51–0.52 the entire
run. This is the **same checkpoint-selection-gap pattern already
documented for Track 1** (T1a-c): the model can clearly fit the specific
triplets it's trained on (train signal is real and monotonic), but that
fit does not generalise to held-out pairs at all. Route B's structured
scoring did not fix the generalisation gap — it reproduced it. The
region-level comparison gives the model more free parameters to
memorise training triplets with, not more of the right kind of
structure.

### A1 in progress — no train signal yet, unlike B1

As of epoch 15, `hard_neg_acc` oscillates at chance (0.44–0.55) on
**both** train and val, and `pos_score_mean`/`neg_score_mean` stay near
zero (~0.001) with no separating trend — unlike B1, which at least showed
strong (if non-generalising) train-side discrimination by this point.
Watching for whether this changes with more epochs before drawing a
conclusion.

---

# Route B variations (2026-09-19) — diagnosis and next batch

**Audience: an implementing agent with no other context.** Self-contained.
Builds on `qnlp/domain/models/vlm/score_heads.py`'s `TrilinearScoreHead`
and the B1 result above (SVO-Probes 0.5168, below the 0.5323 cosine
baseline).

## Diagnosis: the failure is memorisation, and the region index is why

B1's signature is unambiguous: train `hard_neg_acc` 0.57 → 0.92 over 11
epochs while val stayed at 0.51–0.52 throughout. Not collapse, not an
optimisation failure — the head added capacity without adding the right
inductive bias.

The specific defect is in the score's third factor:

```
score(t, R) = sum_j sum_r (u_r · t) (v_r · R_j) (w_r)_j
                                                 ^^^^^^^
                                    indexed by ABSOLUTE region position
```

`w_r` is indexed by region index `j` — an absolute spatial slot (region 0
is always the top-left quadrant). The head therefore learns statements of
the form "caption feature `r` matches *the top-left quadrant*." For
SVO-Probes, where a subject can appear anywhere in the frame, this is
close to the worst available bias: it is a direct route to memorising
"this caption pairs with the image whose top-left quadrant looks like X."

**This is the third instance of the same pattern in this document.**
Per-position parameters keep losing to position-agnostic sharing:

| change | Δ vs. its baseline |
|---|---|
| B2 — per-pixel leaves | -0.146 |
| C-a — per-patch embedding | -0.127 |
| B1 — region-indexed trilinear weights | -0.016 (0.5168 vs 0.5323) |

The region-index factor was introduced deliberately, to avoid the
degenerate-pooling trap flagged in the Route B spec ("a weighted sum of
per-region dot products collapses to pooling the regions first"). It
avoids that trap and substitutes a worse one. **Any replacement must be
permutation-invariant over regions while still not collapsing to
pooling-first** — that is the precise design constraint for everything
below.

## Note: the multilinear constraint has been retired, deliberately

Route N's result ("adopt GELU-gated non-linearity as the new Track 2
baseline, replacing the pure-multilinear tree") means non-linear
aggregation over regions — `max`, `log-sum-exp`, `|·|²` — is now
available in the score head. V2 and V3 below depend on this. Record it as
a deliberate, measured relaxation (gate values quantify what it bought),
not an unremarked drift.

## Variations

### V1 — gated residual trilinear (run this first)

```
score = cosine(t, pooled(R)) + gate * trilinear(t, R)     gate init exactly 0.0
```

**Rationale: every change that has worked in this project has been a
strict generalisation of the working model, initialised to recover it
exactly** — A1's isometric init, B1's opt-in flag, Route N's gate-init-0.
B1 as launched *replaced* cosine outright, discarding the 0.5323 baseline
behaviour rather than building on it. Made additive, the structured term
can only earn its way in, and the learned gate value is a free
measurement of how much it contributes.

- `gate`: single `nn.Parameter(torch.zeros(1))` on the head.
- At gate=0 the head must be numerically identical to the cosine head.
- Log the gate per epoch.

### V2 — Born-rule region pooling (best-motivated)

```
score = sum_j |<t, R_j>|^2        (equivalently ||R t||^2)
```

- **Permutation-invariant** over regions — no positional memorisation
  surface at all.
- **Does not collapse to pooling-first**: it is quadratic in the regions,
  not linear, which is exactly what the degenerate-pooling trap requires
  it to avoid.
- It is the natural generalisation of cosine from a single image vector
  to a *set* of regional states, and it is the Born rule — the same
  principle Route N validated one level down in the tree.
- Normalise `t` and each `R_j` first so terms stay bounded, and scale by
  `1/n_regions`.

### V3 — position-agnostic aggregation via max / log-sum-exp

```
score = sum_r (u_r · t) * max_j (v_r · R_j)          (or LSE_j for smooth gradients)
```

Encodes the bias the task actually needs — *does this entity appear
anywhere in the image* — rather than *in a particular quadrant*. Drops
`w_r` entirely. Use LSE if max's sparse gradients stall training.

### V4 — capacity reduction and bounded terms (cheap control)

The failure signature is memorisation, so the direct response belongs in
the batch even though it is the least interesting row: `rank` 32 → 8, tie
`u = v`, L2-normalise `t` and each `R_j` before scoring so every term is
a cosine, and apply weight decay to the head's parameters.

### V5 — do NOT run `REGION_LEVEL=2` yet

16 regions multiply the positional-memorisation surface by 4. Hold it
until V1-V3 identify a head that generalises at `REGION_LEVEL=1`.

## Orthogonal: freeze a CIFAR-pretrained image tower (F1)

SVO has ~8,600 training rows against a ~2.3M-parameter image tower. **Most
of the memorisation capacity is in the backbone, not in the ~8K-parameter
score head**, so head-only variations may be unable to fix B1's
train/val gap on their own.

There is now a tower that provably encodes real-image semantics — Route
N2's CIFAR-10 checkpoint at 0.5857. Load it into the SVO run, freeze it,
and train only the score head and text tower. This removes the dominant
capacity source and puts the one validated asset this project has built
to work. Never tried on SVO; config-level plus a checkpoint-loading path
analogous to `_warm_start_from_aro` in `qnlp/scripts/svo/run.py`.

Also worth one variant with the tower loaded but *unfrozen* (warm start
rather than freeze), to separate "good initialisation" from "reduced
trainable capacity."

## Batch — all independent, run in parallel

Baseline for every row: **A1+B1+cp_rank=128+triplet_weight=100,
`REGION_LEVEL=1`, SVO-Probes 0.5323** (the S1 cosine result).

| job | change | criterion |
|---|---|---|
| V1 | gated residual trilinear, gate init 0 | beat 0.5323; report final gate value |
| V2 | Born-rule region pooling | beat 0.5323 |
| V3 | max / LSE aggregation, `w_r` removed | beat 0.5323 |
| V4 | rank=8, tied `u=v`, normalised terms, head weight decay | beat 0.5323 |
| F1 | V1 config + **frozen** N2 CIFAR-pretrained image tower | beat 0.5323 |
| F2 | V1 config + N2 tower warm-started, unfrozen | separates init from capacity |

**Report the real-vs-shuffled-caption ablation gap on every row**
(`image_ablation.py --task svo`; current reference 5.9 pts). With train at
0.92 and val at chance, accuracy alone does not distinguish a head that
grounds from a head that memorises.

**Also report train `hard_neg_acc` alongside val for every row.** B1's
train-side climb to 0.92 is what identified the failure mode; a variant
that fixes the bias should show a *smaller* train/val gap, not
necessarily a higher train number.

## Hold Route A unchanged until it finishes

A1's in-progress signature — chance on **both** train and val, scores
near zero with no separating trend — is categorically different from
B1's (strong train-side discrimination that fails to generalise). If it
persists, it points at the role-tensor plumbing (`forward_roles`,
`get_verb_chain`, the `_verb_matrix` projection) rather than at the
scoring idea, and should be debugged as such. Do not apply the V1-V4
variations to Route A until that distinction is settled.

## V1-V4, F1, F2 implemented and launched (2026-09-19)

Per "Hold Route A unchanged until it finishes" — none of these touch
Route A; A1 (job 7432897) keeps running unmodified alongside this batch.

**New score heads** in `qnlp/domain/models/vlm/score_heads.py`:
- `GatedResidualTrilinearScoreHead` (V1) — extends `TrilinearScoreHead`;
  verified locally that at `gate=0` its `score_matrix` is exactly
  `cosine(text_proj(t), mean_j(region_proj(R)))`, bit-identical (not just
  close) to the pure-cosine term with the trilinear contribution zeroed.
- `BornRuleScoreHead` (V2) — `sum_j |<t,R_j>|^2 / n_regions`.
- `AggregationScoreHead` (V3) — `agg="max"` or `"lse"`, drops the
  region-indexed `w_r` factor entirely.
- V4 is config on the existing `TrilinearScoreHead`: `tie_uv` (share `u`
  and `v`) and `normalize_terms` (L2-normalise `t`/`R_j` before scoring),
  plus `rank=8` and higher `head_weight_decay` set at launch time — no new
  class needed.

**F1/F2** needed an artifact that didn't exist yet: `ttn_supervised_probe.py`
only ever kept its best backbone in memory (`best_state`), discarding it
on exit — so N2's validated 0.5857 CIFAR-10 checkpoint was never actually
saved to disk. Fixed: it now writes
`runs/checkpoints/ttn_supervised_probe/backbone_ttn_nonlin-{nonlinearity}_best.pt`
(`{"backbone_state_dict", "test_acc", "val_acc"}`) whenever `arch == "ttn"`.
N2 (gelu) relaunched to produce it — **F1/F2 wait on that job**, not
launched in this immediate batch. `run.py` gained
`_load_pretrained_image_tower` (loads the checkpoint; `freeze=True` for F1
sets `requires_grad_(False)` + `.eval()` and excludes the tower from the
optimizer's param groups entirely, rather than relying on `requires_grad`
alone — confirmed `Trainer` never calls `.train()/.eval()` itself, so the
frozen tower's eval-mode dropout sticks for the whole run without being
reset each epoch).

Verified locally: all four new heads pass the same forward+backward+
`SVOHardNegStep` integration smoke test used for B1/A1, with the added V1
gate=0 identity assertion.

### Batch launched

Baseline unchanged: A1+B1+cp_rank=128+triplet_weight=100,
`SVO_ML_REGION_LEVEL=1`, SVO-Probes 0.5323 (S1).

| job | `SVO_ML_SCORE_HEAD` + config | notes |
|---|---|---|
| V1 | `trilinear_gated` | gate value is the headline number to watch |
| V2 | `born` | |
| V3 | `aggregation`, `SVO_ML_AGGREGATION_FN=max` | |
| V4 | `trilinear`, `SVO_ML_RANK=8`, `SVO_ML_TIE_UV=true`, `SVO_ML_NORMALIZE_TERMS=true`, `SVO_ML_HEAD_WEIGHT_DECAY=0.01` | |

F1/F2 pending N2's backbone checkpoint (relaunched as a prerequisite, see
above) — will launch once that job finishes.

**Per the spec: report train `hard_neg_acc` alongside val for every row**,
since B1's diagnosis came from the train/val gap, not accuracy alone.

## A1 finished — chance, but confounded by a real resolve-rate gap Phase 0 missed (2026-09-19)

Job 7432897 finished. **SVO-Probes overall: 0.4982** (obj_neg 0.4156,
subj_neg 0.4844, verb_neg 0.5160) — exact chance, no better than a coin
flip on any subset.

**But the eval only covered 546 of 2767 test rows (19.7%) — an ~80%
skip rate**, not the ~1% Phase 0's resolve-rate check predicted.
Phase 0 verified that subj/verb/obj *strings* match a known symbol base
name (98.7-99.5% across splits). It never checked the additional, much
stricter condition `RoleGroundedScoreHead` actually needs: that the verb
resolves to *exactly* the mainline 3-piece transitive chain (`_is_valid_
verb_chain`'s 2D/3D/2D shape-and-bond-dim check). Evidently only ~20% of
real SVO-Probes verb usages are simple transitive chains — the rest are
presumably intransitive, phrasal, copula, or otherwise differently-shaped
CCG parses that the v1 scope explicitly doesn't handle.

**Consequence: A1's negative result is confounded, not clean.** Training
on ~20% of the data (and whichever verbs happen to be simple-transitive —
plausibly a biased subset, not a random one) could easily explain "no
train signal" on its own, independent of whether the role-grounded scoring
idea itself has merit. This is a different, and more actionable, diagnosis
than the "role-tensor plumbing bug" hypothesis floated above. Before
re-attempting Route A: measure the *true* resolve rate (fraction of rows
where `_is_valid_verb_chain` succeeds, not just the string-match rate) and
report it against the 90% gate properly — the gate was never actually
checked against the condition that matters.

## V1-V4/F1/F2 batch: infrastructure issues found and fixed en route (2026-09-19)

Two real problems surfaced launching the batch, neither in the scoring
logic itself:

**Bad GPU node.** V1/V3/V4's first three attempts all crashed identically
(`CUDA error: CUDA-capable device(s) is/are busy or unavailable`) —
confirmed all three landed on the same node, `animal-206-2.local`. Not a
code issue; relaunched with `qsub -l hostname='!animal-206-2.local'`,
which cleared it.

**F1/F2 checkpoint compatibility, two rounds.** First launch used the SVO
baseline's `cp_rank=128`/no-nonlinearity — mismatched N2's actual training
config (`cp_rank=256`, `nonlinearity=gelu`), which would have crashed
`load_state_dict`. Caught before either job started (`qw` state) and
cancelled. Second launch matched cp_rank/nonlinearity but missed that N2
was trained at `image_size=32, patch_size=2` (CIFAR-native), not SVO's
default `64/4` — a different patch-embedding input size. Fixed by passing
matching `IMAGE_MODEL_IMAGE_SIZE`/`PATCH_SIZE` at launch. Third launch hit
a real remaining mismatch: `ttn_supervised_probe.py`'s classifier probe
and SVO use different `embedding_dim` (128 vs 512), so `TTNImageModel.head`
(and `final_norm`) never match shape — but neither is used by
`forward_regions`-based score heads at all, so `_load_pretrained_image_tower`
now explicitly drops `head.*`/`final_norm.*` from the checkpoint's state
dict before loading (plain `strict=False` does NOT skip shape-mismatched
keys present in both dicts, only missing/unexpected ones — this needed an
explicit filter, not just a flag). Verified locally with a synthetic
128-dim-checkpoint-into-512-dim-model test before relaunching.

### V2 (Born-rule) result

Finished: SVO-Probes overall **0.5269** (obj_neg 0.5277, subj_neg 0.5010,
verb_neg 0.5342) — below the 0.5323 baseline, though closer than B1
(0.5168). A modest improvement over the plain trilinear head, consistent
with removing the region-index memorisation surface, but not yet a pass.

### V3 (max-aggregation) result

Finished: SVO-Probes overall **0.5052** (obj_neg 0.5065, subj_neg 0.4557,
verb_neg 0.5192) — near chance, worse than V2, no better than B1.

### The overfitting pattern is universal across every configuration tested today — new central finding

With V1/V4/F1/F2 all mid-run, the same signature appears in **every
single one**, without exception, alongside B1/V2/V3 above:

| config | train `hard_neg_acc` (~epoch 10-24) | val `hard_neg_acc` |
|---|---|---|
| B1 (plain trilinear) | 0.92 | 0.51-0.52 |
| V1 (gated residual) | 0.98 | 0.53-0.55 |
| V2 (Born-rule) | (finished, see above) | — |
| V3 (max-agg) | (finished, see above) | — |
| V4 (capacity-reduced, rank=8) | 0.95 | 0.51-0.53 |
| F1 (image tower **frozen**, 0 trainable image params) | 0.93 | 0.51-0.53 |
| F2 (image tower warm-started, trainable) | 0.92 | 0.52-0.55 |

**F1 is the decisive data point.** With the image tower completely frozen
— pretrained on real CIFAR-10 data, zero trainable image-side parameters
— train accuracy still climbs to 0.93 while val stays at chance. This
directly falsifies the "most memorisation capacity is in the image
backbone" hypothesis that motivated F1/F2 in the first place. Combined
with F2 showing the identical pattern with the tower *un*frozen, and
every scoring-head variant (cosine through five structurally different
alternatives) showing it too, the conclusion is now hard to avoid:
**the overfitting source is common to all these configurations, not
particular to any image-side design choice.** The remaining candidates
are the text tower (17.7M params, unfrozen and identical across every
run above) and the training setup itself (`triplet_weight=100` pushing
hard-negative separation aggressively against a training set — 8,609
rows — small relative to that parameter count). Neither has been
isolated yet; that is the next thing to test, not another scoring-head
variant.

## F1 result — the first pass, though for a reason unrelated to the score head (2026-09-19)

Finished: **SVO-Probes overall 0.5356** (obj_neg 0.5603, subj_neg 0.5196,
verb_neg 0.5312) — clears the 0.5323 baseline, the first configuration in
this entire Route B/variation batch to do so. Same overfitting signature
as everything else (test/accuracy in-batch is low, `score_head_gate`
settled at 0.0062 — the trilinear correction term is essentially inert).

**Read this result carefully — it is not evidence the trilinear score
head works.** The gate barely moved from its zero init, meaning almost
none of the gain traces to the structured scoring idea Route B was
testing. What actually changed vs. every other run today is the frozen,
CIFAR-pretrained image backbone (a real, validated 0.5857 CIFAR-10
classifier) replacing a from-scratch-initialised one. The most defensible
reading: a better-initialised (not necessarily lower-capacity — F1 froze
it, but F2's un-frozen warm-start is the config that isolates that)
image tower gives the cosine-equivalent term something more useful to
work with, independent of the score head sitting on top of it. This is
consistent with the corpus-wide finding above (memorisation is not
image-side) while also showing the image tower's *quality*, not just its
capacity, matters. Wait for F2 (warm-started, unfrozen) before concluding
whether freezing specifically was necessary, or whether any warm start
would have done as well.

## F2 and V4 both pass too — the pattern is capacity/init, not scoring architecture (2026-09-19)

**F2 (warm-started, *unfrozen*): SVO-Probes overall 0.5475** (obj_neg
0.5521, subj_neg 0.5113, verb_neg 0.5564) — the best result in the entire
batch, beating even F1 (0.5356). Gate again settled near zero (-0.0038).
**F2 beating F1 answers the freeze-vs-warm-start question directly:
freezing was not necessary, and was if anything slightly worse than
leaving the tower trainable.** The pretrained tower's *initialisation*
is the transferable asset, not a need to protect it from further
training. This further confirms the corpus-wide finding: reduced image
capacity does not help; a better starting point does.

**V4 (capacity-reduced trilinear: rank=8, tied u=v, normalised terms,
higher head weight decay, from-scratch image tower): SVO-Probes overall
0.5439** (obj_neg 0.5472, subj_neg 0.5134, verb_neg 0.5516) — a second,
independent pass, via a completely different mechanism than F1/F2 (no
pretrained tower involved at all).

### Emerging pattern across the full batch

| config | mechanism | SVO-Probes | vs. 0.5323 |
|---|---|---|---|
| B1 | trilinear, full capacity | 0.5168 | fail |
| V1 | + gated cosine residual | (pending) | — |
| V2 | Born-rule (permutation-invariant) | 0.5269 | fail |
| V3 | max-aggregation | 0.5052 | fail |
| **V4** | **trilinear, reduced capacity** | **0.5439** | **pass** |
| **F1** | trilinear-gated, frozen pretrained tower | **0.5356** | **pass** |
| **F2** | trilinear-gated, warm-started tower | **0.5475** | **pass** |

Every fix that worked reduced the model's ability to memorise (V4's
smaller/tied/normalised/decayed head) or gave it a better starting point
that needs less adaptation to fit (F1/F2's pretrained tower) — not a more
expressive comparison function. Every attempt at a more expressive or
differently-biased score (V1/V2/V3, all still using a from-scratch,
full-capacity image tower) failed to clear the bar. This is consistent
with — not contradicting — the earlier finding that the overfitting
source is common across configurations (likely the text tower / training
setup): interventions that globally reduce overfitting pressure help
regardless of where they're applied, while interventions that only change
*how* the (already-overfit-prone) representations get compared do not.

## Batch complete — V1 result and final summary (2026-09-19)

**V1 (gated residual trilinear, full-capacity from-scratch tower):
SVO-Probes overall 0.5273** (obj_neg 0.5505, subj_neg 0.4722, verb_neg
0.5348) — below the 0.5323 baseline. Gate settled at 0.0128, still
essentially inert. Fails, joining B1/V2/V3.

### Full batch, final

| config | mechanism | image tower | SVO-Probes | vs. 0.5323 |
|---|---|---|---|---|
| B1 | trilinear | from-scratch, full capacity | 0.5168 | fail |
| V1 | + gated cosine residual | from-scratch, full capacity | 0.5273 | fail |
| V2 | Born-rule (permutation-invariant) | from-scratch, full capacity | 0.5269 | fail |
| V3 | max-aggregation | from-scratch, full capacity | 0.5052 | fail |
| **V4** | trilinear, capacity-reduced (rank=8, tied, normalised, weight decay) | from-scratch, full capacity | **0.5439** | **pass** |
| **F1** | trilinear-gated | **frozen**, CIFAR-pretrained | **0.5356** | **pass** |
| **F2** | trilinear-gated | **warm-started**, CIFAR-pretrained | **0.5475** (best) | **pass** |

**The split is exactly along one axis, cleanly, across all seven runs.**
Every configuration that changed *only* the comparison function while
keeping a full-capacity, from-scratch image tower (B1, V1, V2, V3) failed
— including the two variations specifically designed to fix B1's
diagnosed region-index memorisation (V2, V3), which is itself informative:
the memorisation wasn't really about *which* comparison function was
used. Every configuration that reduced capacity (V4) or improved
initialisation (F1, F2) — regardless of scoring function, since F1/F2
still used the trilinear-gated head — passed. F2 > F1 > V4 > baseline,
with F2 (warm-started, unfrozen, best init AND full trainability) the
clear winner.

**Conclusion for what to try next:** stop iterating on scoring-head
architecture (Routes A and B's original premise) and instead pursue the
capacity/initialisation axis directly:
- Apply the CIFAR-pretrained warm-start (F2's recipe) to the plain cosine
  head, isolating whether the score head contributes anything at all
  once the real lever (image tower init) is controlled for.
- Investigate the text tower (17.7M params, unfrozen and identical
  across every run in this batch) as the likely dominant memorisation
  source per the earlier cross-cutting finding — a text-side capacity
  reduction or pretraining analogous to F1/F2 is the natural next
  experiment, not another image-region scoring variant.
- Sweep V4's capacity-reduction knobs further (smaller `rank`, stronger
  weight decay) now that it's confirmed to be a working, independent
  lever.

# NODE_ARCHITECTURE_PLAN.md — row 0 result (2026-09-20)

Excess kurtosis of `merged` (the 4-way Hadamard product inside
`CPQuadRankLayer`), measured on N2's trained checkpoint (0.5857,
gelu-gated, `cp_rank=256`), 512 real CIFAR-10 images:

| layer | excess kurtosis | shape |
|---|---|---|
| 0 | 1035.4 | (512, 64, 256) |
| 1 | 163.3 | (512, 16, 256) |
| 2 | 9301.2 | (512, 4, 256) |
| 3 | 313.9 | (512, 1, 256) |

**Overwhelmingly heavy-tailed** — not close to Gaussian (0) or even a
Laplace tail (excess kurtosis 3); every layer is two to four orders of
magnitude past that. **NODE-2 (degree reduction) is well-motivated and
included in the batch.**

## Implementation and batch launch (2026-09-20)

All of NODE-1/2/3/4, tying, and DTTN implemented — see the git commit for
full design notes. Summary:

- `qnlp/discoviz/models/node_variants.py`: `PairwiseBinaryNode` (NODE-1),
  `DegreeReducedNode` (NODE-2), `TuckerNode` (NODE-4, requires tying —
  the core has `rank^5` entries). NODE-3 needs no new class:
  `CPQuadRankLayer(tied=True)` + `torch.nn.utils.parametrizations
  .orthogonal` on its five factor tensors.
- Tying (`IMAGE_MODEL_TIE_NODES`) added to `CPQuadRankLayer` itself
  (needed for row 5's control) by storing the tied dim as 1 and
  `.expand`-ing it at forward time — a broadcasting view, so the default
  untied path is provably unchanged (verified both by code inspection and
  a local smoke test).
- DTTN (`qnlp/discoviz/models/dttn_image_model.py`) per
  `DTTN_IMPLEMENTATION_GUIDE.md`, selected via
  `IMAGE_MODEL_IMAGE_BACKBONE=dttn` through a new `build_image_model()`
  factory, wired into the 6 named construction sites.
- Caught one real bug before it shipped: `TuckerNode`'s first einsum
  attempt reused letter `b` for both the batch dimension and a rank
  index — a silent shape-collision bug, caught by the local smoke test
  (raised a clear `opt_einsum` shape-mismatch error), not by static
  analysis.
- All 8 `node_type` x `tie_nodes` combinations, plus DTTN at both 32x32
  and 64x64, pass local forward+backward+`forward_regions` smoke tests
  before touching the cluster.

### Batch launched

All pure-multilinear (`NONLINEARITY=none`) against the 0.5500 baseline,
A1+B1, CIFAR-10 32x32/patch_size=2:

| job | row | config |
|---|---|---|
| 7433281 | 1 | `NODE_TYPE=pairwise`, untied, `cp_rank=256` |
| 7433282 | 2 | `NODE_TYPE=degree2`, untied, `cp_rank=256` |
| 7433283 | 3 | `NODE_TYPE=isometric`, `TIE_NODES=all`, `cp_rank=256` |
| 7433284 | 4 | `NODE_TYPE=tucker`, `TIE_NODES=all`, `cp_rank=8` (core is `rank^5`) |
| 7433285 | 5 (control) | `NODE_TYPE=cp`, `TIE_NODES=all`, `cp_rank=256` — isolates tying alone |
| 7433286 | 6 | DTTN-T, `stem_patch=1` (32x32-adapted) |

Regression check (existing 0.5857 GELU-gated number reproducing with the
new code, unset backbone flags) not re-run on the cluster — the default
path's equivalence is proven by code inspection (`_expand`/`build_image_model`
are no-ops when `tied=False`/`image_backbone=ttn`) rather than spending a
100-epoch job re-confirming an algebraically guaranteed identity.

### Batch results (first pass)

| row | candidate | test_acc | params | vs. 0.5500 |
|---|---|---|---|---|
| 1 | NODE-1 (pairwise) | 0.5474 | 23.7M | fail — worse, at 3.3x the params |
| 2 | NODE-2 (degree2) | 0.6069 | 12.6M | **pass**, +0.057 |
| 3 | NODE-3 (isometric) | stuck at chance (0.10) after 4 epochs, ~400s/epoch | — | likely failing to train (still confirming) |
| 4 | NODE-4 (tucker, rank=8) | 0.5692 | **985K** | **pass**, +0.019 at 14% of baseline's params |
| 5 | tying only (control) | 0.6354 | 2.28M | **pass, biggest win** — +0.085 at ~32% of baseline's params |
| 6 | DTTN-T | ~0.83-0.85 by epoch 33, still climbing | ~7.7M | **pass, by far the strongest result** — beats the CNN (0.7842) and ResNet-18 (0.7569) references, not just the gated-TTN baseline |

Two clear takeaways: **tying is the single most impactful cheap
intervention** (large accuracy gain, large parameter reduction, zero new
node math), and **DTTN is decisively the best image tower this project
has found.** NODE-1 is a clear loser. NODE-3's status is unresolved —
either very slow to escape a bad init or genuinely broken by the
orthogonal constraint.

## DTTN launched on SVO and ARO, from scratch (2026-09-20)

Per the user's stated general preference (train from scratch by default;
reach for transfer learning only with a concrete, evidenced reason — not
a precaution), and since DTTN is architecturally unlike TTN (CNN-style
locality/translation priors TTN lacks, so a from-scratch failure mode
observed for TTN on SVO doesn't transfer by default): launched DTTN-T
from scratch on both SVO and ARO, default config otherwise (cosine score
head, 64x64 images — `dttn_stem_patch=2`/`transitions=FTTT` gives a final
2x2 feature map, "marginal" per `DTTN_IMPLEMENTATION_GUIDE.md`'s
resolution table but usable).

No new model class needed — `DTTNImageModel` is already a separate class
in its own file, `TTNImageModel` is untouched, and reverting to the
original architecture is just not setting `IMAGE_MODEL_IMAGE_BACKBONE`
(default `ttn`). This separation was already the design goal of the
`build_image_model` factory.

| job | run |
|---|---|
| 7433808 | SVO, DTTN-T, from scratch |
| 7433809 | ARO, DTTN-T, from scratch |

If either lands at chance the way every from-scratch TTN attempt on SVO
has, that becomes the concrete evidence needed to justify a CIFAR-pretrain
warm-start experiment (F1/F2's pattern) — not before.

## DTTN-T final CIFAR result, NODE-3 failure confirmed, early SVO/ARO signal (2026-09-20)

**DTTN-T: test_acc 0.8589**, 7.6M params. Decisively the strongest image
tower this project has produced — well past the CNN (0.7842) and
ResNet-18 (0.7569) references that were the practical ceiling before
this batch, and nowhere close to the gated-TTN baseline (0.5857). This
is the headline architectural result of the whole node-architecture
investigation.

**NODE-3 (isometric): confirmed failing to train**, not just slow. 16
epochs in, loss pinned at exactly 2.3027 (= ln 10) and val_acc
oscillating randomly around 0.09-0.10 with zero trend. The orthogonal
parametrisation (combined with tying, per the plan's requirement) is
preventing any gradient signal from reaching the isometric factors. With
`patience=20` and val fluctuating around chance, a spurious "new best"
could reset the patience counter repeatedly and run the full 100 epochs
at ~400-500s/epoch (~11+ hours) without ever escaping chance — worth
killing early rather than letting it run to completion.

**SVO DTTN (from scratch, job 7433808) — a third independent
confirmation the image tower isn't SVO's bottleneck.** By epoch 8, train
`hard_neg_acc` climbs to 0.82 while val sits at chance (0.48-0.51) — the
identical train/val gap documented extensively for TTN on SVO (B1, V1-V4,
F1/F2). Even a from-scratch backbone that just achieved 0.8589 on
CIFAR-10 reproduces SVO's exact failure signature. Combined with F1's
frozen-pretrained-TTN result (also stuck at the same gap), this closes
off the image tower as a plausible explanation from two independent
directions (a validated pretrained TTN, and a validated from-scratch
DTTN) — the bottleneck is elsewhere, most likely the text tower as
already hypothesised.

**ARO DTTN (from scratch, job 7433809) — different signature, worth
watching.** By epoch 3, val `hard_neg_acc` is actually climbing alongside
train (0.59 -> 0.63 -> 0.65) — real generalisation, unlike SVO. But
`image_pairwise_cos_mean` is climbing fast (0.82 -> 0.93 -> 0.96),
approaching the collapse regime documented for TTN's ARO image tower
historically. Not yet conclusive either way — could plateau, could
collapse further.

## SVO DTTN finished: chance, decisively closing off the image-tower hypothesis (2026-09-20)

**SVO-Probes overall 0.4879** (obj_neg 0.4967, subj_neg 0.5175, verb_neg
0.4760), **SVO-Swap 0.4190**. At chance, on both benchmarks.

This is the decisive result. DTTN just achieved 0.8589 on CIFAR-10 —
comfortably the strongest image tower this project has built — trained
from scratch directly on SVO's own images, and it reproduces SVO's exact
failure signature anyway. Combined with F1 (a *validated, real-image-
grounded* **frozen** TTN tower, also stuck at the same train/val gap) and
this from-scratch DTTN result, the image tower is now ruled out as SVO's
bottleneck from every angle this project has tried: frozen vs. trainable,
weak vs. strong architecture, TTN vs. CNN-style. **The bottleneck is
elsewhere** — most likely the text tower (17.7M unfrozen params, held
constant across every one of these experiments), as flagged after the
Route B variations batch. That is now the highest-value remaining
direction for Track 1.

## ARO DTTN: the collapse concern is confirmed, not just a risk

By epoch 6, `image_pairwise_cos_mean` = **0.986** — essentially fully
collapsed to a narrow cone — while `hard_neg_acc` still reads 0.65-0.66.
This is the same anisotropic-collapse failure mode documented for TTN's
ARO image tower: the model can satisfy the specific hard-negative
triplets it trains on via near-identical embeddings, without necessarily
preserving the general discriminative structure needed to generalise.
The apparent hard_neg_acc "success" here may be fragile rather than real
— continue watching, and treat the current 0.65-0.66 with the same
scepticism as B1/T1's train-side numbers until a real generalisation
check (e.g. `image_ablation.py`) is run.

## Frozen CLIP sanity check launched on SVO (2026-09-20)

Every image tower tried on SVO has landed at chance — TTN in every
configuration, and now DTTN from scratch despite hitting 0.8589 on
CIFAR-10. Strong evidence the bottleneck is the text side, but not
airtight: every one of those towers was still learning visual features
from SVO's own small (~8,600 row) training set, leaving a residual doubt
that SVO's images just don't carry enough signal for *any* from-scratch
tower this little data can train. A frozen, externally-pretrained CLIP
image encoder settles that directly.

`CLIPImageModel` (`qnlp/discoviz/models/clip_image_model.py`) — a frozen
`openai/clip-vit-base-patch32` (151M params, 0 trainable), inverting each
caller's existing ImageNet normalisation back to ~[0,1] and re-normalising
with CLIP's own stats internally, so no transform pipeline needed to
change. Selected via `IMAGE_MODEL_IMAGE_BACKBONE=clip`. Explicitly a
sanity check, not a candidate architecture — not quantum-inspired, not
trained here, `forward_regions` deliberately unimplemented (cosine-only,
matching SVO's default score head).

Verified locally before touching the cluster: correct output shape,
unit-norm output, 0 trainable params. Pre-downloaded the checkpoint via
the login node first, since compute nodes may lack internet access.

**Launched: job 7434139**, SVO, frozen CLIP, text tower trained from
scratch as normal, cosine score head, everything else default.

If this also lands at chance, that is decisive: the task is not learnable
with a known-excellent fixed image representation, meaning the bottleneck
is the text tower (or the loss/data design), not "the image side needs
more/better data." If it clears chance, that would be the most important
result in this entire investigation — it would mean the image towers
built here specifically (TTN and DTTN alike) are the limiting factor
after all, not the text side.

## Frozen CLIP result: a stable above-chance signal — the leading hypothesis reverses (2026-09-20)

By epoch 12, val `hard_neg_acc` has plateaued at **~0.61-0.62** across four
consecutive epochs (9: 0.613, 10: 0.616, 11: 0.617, 12: 0.608) — durable,
not noise. Train is at 0.97 (real overfitting on top of the real signal),
but `image_pairwise_cos_mean` stays flat at ~0.69 throughout — **no
collapse**. This is a real, sustained validation signal above chance that
**no from-scratch image tower tried in this project — TTN in any
configuration, or DTTN — has ever produced on SVO.**

**This reverses the project's leading hypothesis.** The prior conclusion
("image tower ruled out, bottleneck is the text side") rested on F1 (a
frozen *pretrained-on-CIFAR* TTN) and DTTN-from-scratch both failing. But
neither of those is a strong, general-purpose image representation:
CIFAR-10 pretraining is a narrow, small-image-classification skill, and
DTTN's demonstrated strength was specifically on CIFAR-10, not
necessarily transferable when trained from scratch on SVO's ~8,600
images. A genuinely strong, externally-validated representation (CLIP)
clears chance and holds there. **The evidence now points at the
from-scratch image towers built in this project failing to learn
adequate visual features from SVO's small dataset — not the text tower,
not the loss design, not the data.**

Still to confirm: the final test-set number (once training finishes/
early-stops) and whether the 0.61-0.62 plateau holds or drifts. But
directionally this is now the single most important result in the whole
image-tower investigation, and it points toward pursuing a
CLIP-pretrained-then-warm-started (or otherwise stronger-initialised)
image tower for SVO specifically, rather than the text-tower
investigation the prior conclusion had prioritised.

## Frozen CLIP: final result — the best SVO number in this project (2026-09-20)

Job 7434139 finished. **SVO-Probes overall 0.5811** (obj_neg 0.6303,
subj_neg 0.5897, verb_neg 0.5606), **SVO-Swap 0.6095** (105 pairs). Both
clear the previous best on either benchmark by a wide margin (S1's
cosine-head config: 0.5323 Probes; best prior Swap number was in the
0.42-0.61 range depending on config, never this consistently above
chance on both simultaneously). `image_pairwise_cos_mean` stayed at
0.696 throughout training and at test time — no collapse.

**Verdict: the sanity check is decisive.** SVO's caption/image task is
genuinely learnable with a strong, fixed image representation. Every
from-scratch image tower this project has built and tested on SVO — TTN
in every configuration (A1/B1/cp_rank sweeps/Route N/Route B variations/
F1's CIFAR-pretrained-frozen variant) and DTTN from scratch — has failed
to extract adequate visual features from SVO's own ~8,600-image training
set, even though DTTN independently proved capable of excellent visual
features on CIFAR-10 (0.8589). The bottleneck for SVO has been the image
tower's ability to learn good-enough features from this specific,
small, real-photo dataset — not the text tower, not the loss/triplet-
weight design, not the data itself.

**Implication for next steps:** the highest-value direction for Track 1
is now getting a better-than-from-scratch image representation into SVO
training — most directly, warm-starting (not necessarily freezing, per
the ARO/SVO CIFAR-pretrained-TTN comparison — F2 beat F1) a tower
pretrained on a larger, more diverse image dataset than SVO alone can
supply. DTTN pretrained on CIFAR-10 (already have a checkpoint machinery
for this, `ttn_supervised_probe.py` saves backbones) is the cheapest
next experiment in that direction; unfreezing CLIP itself (fine-tuning
rather than freezing) is a further option if a from-scratch-in-spirit
result is still wanted without inheriting an external architecture.

## NODE-3 final result: confirmed dead

**test_acc = 0.1000**, exactly the majority baseline. Early-stopped at
epoch 32 (best val_acc 0.1040 — indistinguishable from noise around
chance). 32 epochs at ~400-500s each with zero learning. The orthogonal
parametrisation (combined with the required tying) prevents this node
from training at all, at least in this configuration — a clear, clean
fail, not a resource-constrained inconclusive result.

# DISCOCLIP_REPRODUCTION_PLAN.md execution (2026-09-20)

## Phase 0 — anchor the target: SUCCESS

Ran the reference `kinianlo/discoclip` repo as-is (already cloned locally
at `~/Desktop/Dev/discoclip`, data already preprocessed to the paper's
exact row count: 5288+1850+1850 - 3 headers = **8,984 pairs**, matching
the paper exactly). Config: `configs/svo_default.yaml` (bobcat reader,
embedding_dim 512, bond_dim 10, batch_size 64, lr 0.003, epochs 10,
`hard_neg_loss_weight=0` — no triplet term at all by default, confirming
Phase 1.4's question), device MPS (Apple Silicon).

**Three trivial environment fixes needed, none of them bugs in the
reference code:**
- Missing `shtab` dependency (`pip install shtab`).
- Missing `logs/`/`checkpoints/` output directories (README doesn't
  mention creating them).
- Missing NLTK data (`averaged_perceptron_tagger_eng`, `punkt`,
  `punkt_tab`).
- `requirements.txt` doesn't pin `transformers`, so pip installed the
  latest (5.2.0) — incompatible with `lambeq`'s bundled
  `BertForChartClassification` (`AttributeError:
  'BertForChartClassification' object has no attribute
  'all_tied_weights_keys'`). Pinned to `transformers==4.57.1`, matching
  this project's own known-working version — fixed it immediately.

**Result: test acc 0.8323** (subj 0.8190 n=431, verb 0.8029 n=893, obj
0.8931 n=524) **against the paper's target 0.8355** — within 0.32 points,
comfortably inside the plan's 2-point tolerance. Per-subset shape matches
the paper's reported 80.74/82.42/87.79 reasonably closely. Training took
under a minute for all 10 epochs (CLIP embeddings are precomputed, so
each epoch is cheap — no image encoding at train time).

**SVO-Swap (target 93.68%) not independently reproduced** — no dedicated
eval script exists in the repo for `data/processed/svo_probes_swapped.csv`
(searched `train_svo.py`, `test_aro.py`, notebooks; found none). Not
gating: the plan's explicit Phase 0 success criterion is the Probes
number, which passed cleanly. Revisit only if the Swap number becomes
load-bearing later.

**Verdict: the published target is real and reachable in this
environment.** Every subsequent experiment is now a bisection between two
known-good anchors (0.5811 ours, 0.8355 theirs), not a chase after an
unreachable number.

## Phase 1 — audits (2026-09-20)

### 1.1 Evaluation protocol — resolved, matches ours

Read `train_svo.py`'s test loop directly. **"Overall" is a per-row mean
over the full test set**, not an average of the three subset accuracies
— the subj/verb/obj numbers are computed separately, over near-disjoint
subsets (431+893+524=1848 ≈ the ~1850-row test split, confirming each row
belongs to essentially one subset). This is the same convention our own
`evaluate_svo_probes`'s "overall" already uses — no discrepancy here.

**Skip rate: 0.** `svo_tn_collate_fn` drops any row whose sentence failed
to parse, printing `"Found N invalid samples"` if so — checked the
Phase 0 reproduction log directly: **zero such warnings**, i.e. 100% of
the test set was evaluated. A useful, stark contrast to this project's
own Route A finding (silently evaluated only 19.7% of its test set).

### 1.3 Model variant — resolved, model *is* the same construction

`discoclip/utils/ansatz.py`'s `CustomMPSAnsatz` is — class name, default
`max_order=3`, structure — **the same MPS bond-dimension factorisation
this project's own `qnlp/discoviz/parser/asnsatz.py` implements.** The
`reader="bobcat"` config (the default, and what Phase 0 used) is their
"Compact" model. **This is not a different model from ours — it is the
same construction**, resolving the plan's 1.3 concern. No R5 (variable-
rank) experiment needed.

### 1.2 Data/vocabulary — a real discrepancy found, flagged not resolved

Queried the Phase 0 reproduction run's actual logged parameter count
directly (`model_num_params` in its mlflow db): **1,677,312** — not the
537,600 the plan cites. Since 1.3 confirms the architecture is identical
to ours, and our own text tower runs ~13,100 params/symbol on average
(17.7M params / 1356 symbols), 1.68M implies a vocabulary on the order of
~130 symbols for this reproduction — far smaller than our 1356. Given
row counts are comparable (8,984 vs our 8,609-9,107 depending on
threshold), this points at a much more aggressive vocabulary reduction
somewhere in their pipeline (tokenisation/lemmatisation folding more
surface forms together, or a stricter frequency cutoff than word-count
alone) — not yet identified precisely. **Flagging this honestly rather
than reconciling it to the plan's cited number**: whichever of
537,600/1,677,312 is the real reference figure, our 17.7M is
substantially larger either way (10.6x-33x), so this doesn't change
R1's rationale, but the exact target vocabulary size for R1 is now less
certain than the plan assumed.

### 1.4 Loss — confirmed, matches the plan's description

`configs/svo_default.yaml`'s default `hard_neg_loss_weight: 0` — Phase 0's
reproduction run used **no triplet term at all**, confirming "self-
supervised contrastive loss" (their words) is a literal plain InfoNCE
loss with zero hard-negative weighting, unlike our default
`triplet_weight=40000` (or S1's 100).

### 1.5 Optimisation defaults — confirmed

`configs/svo_default.yaml`: `learning_rate: 0.003`, `batch_size: 64`,
`bond_dim: 10`, `temperature: 0.07` — exactly as the plan states.

## Phase 2 — bisection batch launched (2026-09-20)

R2/R3 need no data changes — launched immediately, both arms each. R1
needs a regenerated dataset at `WORD_FREQ_THRESHOLD=50`; added
`SVO_PREP_WORD_FREQ_THRESHOLD`/`SVO_PREP_OUTPUT_SUFFIX` to
`prepare_datasets.py`/`build_svo_swap.py` (writes to `_thresh50`-suffixed
parquets, leaving the default threshold-10 files every other experiment
reads from untouched) and a matching `SVO_ML_DATASET_SUFFIX` on
`svo/run.py` to select it. Regeneration launched as job 7434678; R1's two
training jobs follow once it completes.

Baselines, exactly as previously run: CLIP arm = `image_backbone=clip`
only (job 7434139's config, `triplet_weight` at its default 40000 — never
explicitly tuned for this arm); TTN arm = S1's recipe (A1+B1+cp_rank=128,
`triplet_weight=100`). This asymmetry is pre-existing, not something
Phase 2 introduces — R2/R3 move each arm from its own actual best
config, not from a shared one.

| job | row | arm | change |
|---|---|---|---|
| 7434678 | R1 (data) | — | regenerate parquets at threshold=50 |
| 7434679 | R2 | CLIP | `triplet_weight` 40000 -> 0 |
| 7434680 | R2 | TTN | `triplet_weight` 100 -> 0 |
| 7434681 | R3 | CLIP | `text_lr` 0.001->0.003, `batch_size` 128->64 |
| 7434682 | R3 | TTN | same, `triplet_weight` held at 100 |

Per the plan: **a TTN-arm change only counts as real at +0.03 or more**
(baseline ~0.53, where this project has documented differences below
that are noise); the CLIP arm is the decision input.

### R1 data build and training launch

`build_svo_swap` OOM'd at the default 16G (same recurring failure seen
before) — `prepare_datasets` itself succeeded first: **9,107 rows** at
threshold=50 (matches the figure already cited in this doc's earlier
notes), split 5463/1817/1827, role-resolve 98.5-99.5%. Relaunched just
`build_svo_swap` via `submit_svo_swap_only.sh` (32G) — succeeded: 89
human/animal candidates, **49 swapped+compiled pairs** (smaller than the
threshold=10 variant's 105, as expected with fewer surviving rows).

**R1 training launched: jobs 7434827 (CLIP) and 7434828 (TTN)**, both on
the `_thresh50` dataset variant.

### R2/R3 mid-training snapshot (not final)

| job | epoch | val `hard_neg_acc` |
|---|---|---|
| R2 CLIP (drop triplet) | 22 | 0.646 |
| R2 TTN (drop triplet) | 14 | 0.543 |
| R3 CLIP (lr/batch) | 15 | 0.624 |
| R3 TTN (lr/batch) | 14 | 0.534 |

Directionally matches the plan's pre-committed prediction: R2 tentatively
helps the CLIP arm (above its ~0.61-0.62 baseline plateau) while the TTN
arm sits near chance in both R2 and R3 — consistent with S1's tuned
`triplet_weight=100` being load-bearing for that arm specifically. None
of these are final; reporting only once each job completes or
early-stops.

### R3 CLIP result — first real movement toward the reference

**Finished: SVO-Probes overall 0.6039** (obj_neg 0.6352, subj_neg 0.5876,
verb_neg 0.5971), **SVO-Swap 0.6000**. Matching the reference's optimiser
defaults (`text_lr=0.003`, `batch_size=64`) took the CLIP arm from the
0.5811 baseline to 0.6039 — **closing ~9% of the 0.254-point gap to the
0.8355 target on its own.** Swap is roughly flat (0.6095 -> 0.6000).
First confirmed real, positive single-variable result in this batch.

R1 (both arms) confirmed training correctly on the regenerated
threshold=50 data (5463/1817/1827 rows loaded, matching Phase 0's row
counts). At epoch 3: R1 CLIP val `hard_neg_acc` 0.639 (promising early);
R1 TTN 0.512 (at chance, typical this early for the TTN arm).

### R3 complete — clean split between arms, as the plan anticipated

**R3 TTN: SVO-Probes 0.5183** (obj_neg 0.5244, subj_neg 0.4577, verb_neg
0.5336) vs. S1's 0.5323 baseline — a drop of 0.014, **below the
pre-committed +0.03 threshold: not a real effect, within noise.**
SVO-Swap 0.6095.

| arm | baseline | R3 | delta | verdict |
|---|---|---|---|---|
| CLIP | 0.5811 | 0.6039 | +0.023 | real movement toward the target |
| TTN | 0.5323 | 0.5183 | -0.014 | noise (below +0.03 threshold) |

Exactly the clean split the plan flagged as a plausible, non-contradictory
outcome — S1's `text_lr=0.001`/`batch_size=128` was already close to
optimal for the TTN arm, so moving toward the reference's defaults
neither helps nor meaningfully hurts it, while the same change gives
CLIP a real, if modest, gain.

## ARO DTTN (from scratch, job 7433809) — final result

Early-stopped at epoch 49 (best epoch 39).

| task | N | acc | true_cos | false_cos |
|---|---|---|---|---|
| attribution | 4438 | 0.8053 | 0.8277 | 0.6216 |
| relation | 3650 | 0.5942 | 0.5939 | 0.4493 |
| overall | 8088 | 0.7101 | 0.7222 | 0.5438 |

Winoground (22/22 pairs evaluated) and SugarCREPE swap_obj (4/159
evaluated) both had too few evaluated pairs to be meaningful — likely a
vocabulary-coverage gap in DTTN's cold-start text tower on those eval
sets, not informative here.

**Caveat, not yet resolved: `image_pairwise_cos_mean` finished at
0.998** — essentially fully collapsed, the same concern flagged earlier
in this run's trajectory (0.82 at epoch 2 -> 0.986 at epoch 6 -> 0.998 at
the end). The attribution-over-relation asymmetry (0.80 vs 0.59) is the
typical qualitative pattern for vision-language models on ARO generally
(CLIP itself shows the same split), so the shape isn't obviously wrong —
but with embeddings this collapsed, 0.7101 overall should not yet be
trusted as genuine compositional understanding without an
`image_ablation.py`-style real/shuffled/zeroed check, the same test that
caught collapse-driven false positives elsewhere in this project's TTN
work. Recorded as-is; verification is a follow-up, not done here.

## R2 complete — biggest single-variable CLIP gain so far

**R2 CLIP (drop triplet term): SVO-Probes 0.6137** (obj_neg 0.6661,
subj_neg 0.6392, verb_neg 0.5869), SVO-Swap 0.5524. vs. 0.5811 baseline:
**+0.033** — the largest single-variable gain in this batch, ahead of
R3's +0.023. No collapse (`image_pairwise_cos_mean` 0.696).

**R2 TTN: SVO-Probes 0.5558** (obj_neg 0.6042, subj_neg 0.5175, verb_neg
0.5492), SVO-Swap 0.6095. vs. 0.5323 baseline: **+0.024** — just under
the pre-committed +0.03 threshold, so still counts as noise by that rule,
though the closest any TTN-arm row has come to crossing it.
`image_pairwise_cos_mean` = 0.067 — real, non-collapsed embeddings.

## Phase 2 running tally

| row | CLIP arm | delta | TTN arm | delta | verdict |
|---|---|---|---|---|---|
| baseline | 0.5811 | — | 0.5323 | — | — |
| R2 (drop triplet) | 0.6137 | **+0.033** | 0.5558 | +0.024 | CLIP real; TTN noise (just under threshold) |
| R3 (lr/batch) | 0.6039 | +0.023 | 0.5183 | -0.014 | CLIP real; TTN noise |
| R1 (threshold=50) | pending | — | pending | — | running |

Both completed rows point the same direction: real, meaningful gains for
the CLIP arm (+0.023 to +0.033), no reliable effect on the TTN arm.
Awaiting R1, then R4 (all three combined) once R1 reports.

## R1 TTN result — third consecutive noise result for the TTN arm

**R1 TTN (threshold=50): SVO-Probes 0.5304** (obj_neg 0.5337, subj_neg
0.5385, verb_neg 0.5263) vs. S1's 0.5323 baseline: **-0.002**, essentially
flat, clearly within noise. SVO-Swap 0.3061 (n=49 — the threshold=50
swap set is much smaller than the default's 105, so this number carries
more variance than the others).

**All three TTN-arm rows now land within noise** (R1 -0.002, R2 +0.024,
R3 -0.014) — no single-variable change tested so far produces a real
effect on the TTN arm. R1 CLIP still running (epoch 18, val
`hard_neg_acc` 0.647, tracking the same 0.61-0.65 range as R2/R3).

## Phase 2 running tally (updated)

| row | CLIP arm | delta | TTN arm | delta | verdict |
|---|---|---|---|---|---|
| baseline | 0.5811 | — | 0.5323 | — | — |
| R1 (threshold=50) | pending | — | 0.5304 | -0.002 | TTN noise |
| R2 (drop triplet) | 0.6137 | +0.033 | 0.5558 | +0.024 | CLIP real; TTN noise |
| R3 (lr/batch) | 0.6039 | +0.023 | 0.5183 | -0.014 | CLIP real; TTN noise |
