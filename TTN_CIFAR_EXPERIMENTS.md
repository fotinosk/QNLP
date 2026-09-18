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
of triplet_weight, not just detuning it). **Not yet concluded** — T1c
(triplet_weight=100, running) and the full trajectories of T1a/T1b
(checking whether collapse is truly permanent or partially reversible
later in training) are needed before treating this as settled. Log
`image_pairwise_cos_mean`'s full trajectory, not just early epochs, before
drawing a final conclusion — a metric that spikes early but recovers
would tell a different story than one that stays at 1.0 throughout.

**S1 (SVO) still loading data as of this entry.**
