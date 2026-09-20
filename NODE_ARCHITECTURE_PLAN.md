# Node architecture plan: four candidates plus DTTN

**Audience: an implementing agent with no prior context.** Self-contained
except for DTTN, which has its own document (`DTTN_IMPLEMENTATION_GUIDE.md`).

## Context and baselines

The image tower (`TTNImageModel`, `qnlp/discoviz/models/image_model.py`)
is a quadtree of `CPQuadRankLayer` nodes (`qnlp/discoviz/models/cp_node.py`).
Two defects have been found and fixed (isometric per-node init; a fixed
cos/sin patch feature map), taking CIFAR-10 from 0.0983 to 0.5500. Adding
a GELU gate inside each node took it to 0.5857.

Benchmarks to beat, all CIFAR-10 at 32x32, `patch_size=2`:

| reference | acc |
|---|---|
| current best (GELU-gated, `cp_rank=256`) | **0.5857** |
| current best, pure multilinear (`cp_rank=256`, no gate) | **0.5500** |
| small CNN | 0.7842 |
| ResNet-18 | 0.7569 |
| raw-pixel logistic regression (floor) | 0.3784 |
| DTTN-S, reported in its paper (224x224, long schedule) | 0.95 |

## The experimental question that frames this batch

DTTN reaches 95% on CIFAR-10 **with no nonlinear activations**. That means
multilinearity is almost certainly not this project's ceiling — the node
is. The GELU gate adopted earlier may have been compensating for a poor
node rather than lifting a real constraint.

**Therefore: run every candidate below in its pure-multilinear form
first** (`IMAGE_MODEL_NONLINEARITY=none`), against the 0.5500 baseline.
A candidate that beats 0.5857 *without* the gate is a much stronger
result than one that needs it. Add the GELU arm only for whichever
candidate wins.

## The current node, for reference

`CPQuadRankLayer.forward`, per node, for children `x_tl, x_tr, x_bl, x_br`:

```
p_c      = W_c x_c                       W_c in R^{rank x in_dim}, separate per child AND per node
p_c      = RMSNorm(p_c)                  normalises across the rank dimension
merged   = p_tl * p_tr * p_bl * p_br     elementwise, i.e. degree 4 in the children
merged   = gain * merged                 learnable per-node scalar
out      = V merged                      V in R^{rank x out_dim}, per node
out      = out + res_proj(mean(children))    layers 2-3 only
```

Two structural properties matter for what follows:

- **Degree 4 per node.** Over a depth-4 tree that is degree 4^4 = 256 in
  the input pixels.
- **A rank-`r` CP decomposition** of the 5-way node tensor. CP forces a
  single shared rank index across all four children, so different rank
  components cannot interact — there are no cross-terms.

## Prerequisite measurement (no cluster job, do first)

**Kurtosis of `merged`, per layer, on a trained checkpoint.** A product of
four unit-RMS vectors is heavy-tailed in expectation: most components
collapse toward zero and a few dominate. This has been hypothesised four
times in the research log and never measured.

It decides whether NODE-2 is well motivated or pointless. If `merged` is
heavy-tailed, degree reduction is the right fix; if it is near-Gaussian,
NODE-2's rationale evaporates and the batch should weight NODE-1 and
NODE-3 instead. Report excess kurtosis per layer alongside the existing
per-layer Gram-correlation trace.

## Cross-cutting option: tie node tensors within a level

`W_c` and `V` are currently `[num_nodes, rank, in_dim]` — every node in a
level has its own parameters (64 nodes in layer 0, then 16, 4, 1).

**Tying** means one shared tensor per level, dropping the `num_nodes`
index from the einsums. This:

- cuts image-tower parameters by roughly an order of magnitude,
- adds **translation equivariance**, which the tower currently lacks,
- matches DTTN, whose abstract explicitly names "the parameter-sharing
  property", and which uses depthwise convolutions (shared across space).

Per-position parameterisation has lost four times in this project's logs
(per-pixel leaves -0.146, per-patch embedding -0.127, region-indexed
scoring, per-symbol text tensors). The tree is the one place the lesson
has not been applied.

**The four child factors `factor_tl/tr/bl/br` stay separate** — tying is
across *nodes*, not across child positions. Within-receptive-field
structure is preserved; only global position-specificity is removed.

Config: `IMAGE_MODEL_TIE_NODES` = `"none"` (default) | `"all"` |
`"fine"` (tie layers 0-1 only, leaving the 4-node and 1-node layers
per-node, where position plausibly carries meaning and parameters are few).

Tying composes with all four candidates. NODE-4 **requires** it (see below).

---

## NODE-1 — pairwise binary contraction (the standard TTN node)

**Motivation.** The textbook TTN contracts children *pairwise* with an
explicit bond dimension. A rank-32 CP approximation of a 5-way tensor is
the non-standard choice, and its conditioning is poorly understood. This
is the "use the standard construction" candidate.

**Design.** Replace the single 4-way product with two nested binary
contractions, normalising between them:

```
a   = RMSNorm( (A_tl x_tl) * (A_tr x_tr) )      first binary contraction
b   = RMSNorm( (A_bl x_bl) * (A_br x_br) )
out = V( (B_a a) * (B_b b) )                     second binary contraction
```

Still degree 4 overall, but built from binary operations with an
intermediate normalisation, which is the conditioning difference.

**Implementation.** `A_*` and `B_*` are `[rank, in_dim]` (plus a leading
`num_nodes` index when untied). Parameter count is comparable to today's
node. Keep the existing residual and gain.

**Risk.** Low. Closest change to the current model.

## NODE-2 — reduce per-node polynomial degree

**Motivation.** Degree 4 per node compounds to degree 256 over the tree.
DTTN uses **degree 2 per block** (a single Hadamard product of two
branches) and composes higher order across depth — it reaches 2^L order
with L blocks, and reports 95% on CIFAR-10. That is direct external
evidence that low per-module degree with depth-wise composition is the
better structure.

**Design.** Replace the 4-way product with a sum over pairwise products:

```
merged = sum_{c < c'} p_c * p_c'        6 pairs, degree 2
out    = V merged
```

Optionally learn a weight per pair (6 scalars per node, or per level when
tied).

**Implementation.** Straightforward; keep RMSNorm on each `p_c` as today.

**Gating.** Run this only if the kurtosis measurement shows `merged` is
heavy-tailed. If it is near-Gaussian, the motivation is gone.

**Risk.** Reducing degree reduces expressiveness per node; the tree may
need more depth to compensate. Watch the per-layer Gram-correlation
trace — if decay improves but accuracy does not, depth is the limiter.

## NODE-3 — isometry maintained during training

**Motivation.** The A1 fix made each node's projection isometric **at
initialisation** and produced a real gain (+0.047, the second-largest
single win in this campaign). But the constraint is not maintained — after
one optimiser step the tensors drift off the isometric manifold. In
canonical-form TTN/MERA the tensors *stay* isometric.

Benefits: norm preservation through the tree, which attacks the measured
signal decay directly (Gram correlation 0.553 -> 0.030 across four
levels), and no exploding or vanishing products.

**Design.** Constrain `W_c` (and optionally `V`) to the Stiefel manifold
throughout training.

**Implementation — use PyTorch's built-in parametrisation, do not
hand-roll:**

```python
torch.nn.utils.parametrizations.orthogonal(module, name="weight")
```

This reparameterises the tensor so it remains semi-orthogonal under
ordinary gradient descent. Note it operates on 2D matrices, so with the
untied `[num_nodes, rank, in_dim]` layout each node's matrix must be
registered separately, or the layout refactored to a `ModuleList`. **This
makes NODE-3 and node-tying natural partners** — tied tensors are 2D
already.

**Risk.** Low-to-moderate. Orthogonal parametrisation slows each step
somewhat. If it is too slow untied, run it tied only.

**Thesis note.** This is the most quantum-faithful candidate: isometric
tensors are exactly the canonical form of a TTN, and unitary maps are
what a quantum circuit applies.

## NODE-4 — Tucker core instead of CP

**Motivation.** CP is Tucker with a diagonal core. A full core allows
cross-terms between rank components that CP structurally forbids, so it is
strictly more expressive per rank.

**Design.**

```
merged_r = sum_{r1,r2,r3,r4} G[r1,r2,r3,r4,r] p_tl[r1] p_tr[r2] p_bl[r3] p_br[r4]
```

**Implementation constraint — read this before starting.** The core has
`r^5` entries. At `r=8` that is 32,768 parameters **per node**; with 64
nodes in layer 0 that is 2M parameters for one layer. **NODE-4 is only
affordable with node-tying enabled**, and even then use small `r` (4 to
8). Implement with `torch.einsum` and verify memory on a single batch
before launching.

**Risk.** Highest of the four. Small `r` may lose more than the core gains.
The saving grace is that small-`r`-plus-rich-core is exactly the
"smaller but more expressive" trade this batch is testing.

## DTTN — see `DTTN_IMPLEMENTATION_GUIDE.md`

A separate, validated architecture rather than a node variant: a 4-stage
stack of Antisymmetric Interaction Modules, fully free of nonlinear
activations, reported at 95.0% on CIFAR-10 and 77.9-82.4% on ImageNet-1K.

It is **not** a drop-in node — AIM operates on a whole feature map, not on
four children — so it replaces `TTNImageModel` behind the same interface
rather than replacing `CPQuadRankLayer`. The guide covers the full spec,
the resolution adaptation (critical: the reference downsamples 32x, which
collapses a 32x32 input to 1x1), the interface contract, and validation.

Treat DTTN as **both a candidate and the batch's upper reference point**:
it tells you how much of the gap to a CNN is attributable to the node
versus to everything else.

---

## Batch

All rows are independent. Baseline: **0.5500** (pure multilinear) and
**0.5857** (GELU-gated), CIFAR-10 at 32x32.

| # | config | notes |
|---|---|---|
| 0 | kurtosis measurement on the trained checkpoint | no job; gates row 2 |
| 1 | NODE-1, untied, multilinear | standard TTN construction |
| 2 | NODE-2, untied, multilinear | only if row 0 shows heavy tails |
| 3 | NODE-3, tied, multilinear | tied because orthogonal parametrisation prefers 2D |
| 4 | NODE-4, tied, `r=8`, multilinear | requires tying; check memory first |
| 5 | current node **+ tying only**, multilinear | isolates tying from every node change |
| 6 | DTTN-T | per the separate guide |

Row 5 is the control that makes rows 3 and 4 interpretable — without it,
any tied candidate confounds the node change with the tying.

**Follow-up, after the batch:** take the single best candidate and run one
GELU-gated arm. If the gate no longer helps, the pure-multilinear result
stands as the headline and the earlier "adopt GELU" conclusion should be
revised in the log.

## Reporting requirements

For every row:

1. **CIFAR-10 test accuracy**, against both baselines.
2. **Per-layer Gram correlation and kNN class consistency** at init and
   after training (`qnlp/discoviz/diagnostic/tower_spread_trace.py`).
   Reference: input kNN 0.2004, chance 0.100; the current tower decays to
   ~0.107 at the output.
3. **Parameter count.** Several candidates change it by an order of
   magnitude; accuracy comparisons without it are not interpretable.
4. **Excess kurtosis of `merged` per layer** where the candidate retains a
   `merged` tensor.

Established rule in this project, learned twice: **init-time traces do not
predict trainability.** A1's trace showed no change and it gained +0.047;
B1's trace showed chance-at-output and it gained +34 points. Traces are
for triage and mechanism, never for declaring a candidate dead.

## Out of scope

Do not wire any candidate into SVO or ARO training in this batch. The SVO
transfer question is worth asking only of whichever candidate wins on
CIFAR, and asking it of all six wastes cluster time on configurations that
will not be adopted.
