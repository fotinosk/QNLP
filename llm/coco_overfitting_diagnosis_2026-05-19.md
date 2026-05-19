# COCO overfitting diagnosis — `einsum_frozen_clip` checkpoint

Date: 2026-05-19  
Subject: why does `EinsumModel + frozen CLIP-ViT` overfit on full COCO?  
Train hits 0.99 hard-neg-acc, val plateaus at 0.535 — pure overfit.

This doc collects the diagnostics we ran on the existing peak-val
checkpoint (`runs/checkpoints/einsum_frozen_clip/2026-05-11_23-57-00/`)
without retraining. **All evidence below is from the trained model and
the existing train parquet — no new GPU runs needed to reach these
conclusions.**

---

## TL;DR

> The overfitting is **architectural, not data-driven.** Even on val
> captions whose every typed symbol was seen ≥ 100× in train, the model
> still scores at chance (0.533). The model has 1.04 B free parameters
> over 450 k training pairs and learns idiosyncratic per-(typed-symbol)
> mappings that fit train batches without generalising. The vocabulary
> long tail is a smaller, separable problem; the dominant problem is
> that EinsumModel's typed-CCG parameterisation provides no
> parameter-sharing channel that forces representations to generalise.

---

## 1. Per-rank parameter movement (init → trained)

```
Rank      n       ⟨freq⟩    ⟨‖θ_init‖⟩    ⟨‖θ_final‖⟩   ⟨‖Δθ‖⟩    rel-move    %dead (<1% Δ)
  1   12,789       229.1       1.0000        1.1166       0.323     0.323        9.7 %
  2   52,864       158.5      38.1569       38.3079       0.867     0.288       36.0 %   ← 36% of rank-2 tensors barely moved
  3   14,969       187.9      28.9029       29.7474       2.451     0.516       12.6 %
```

Notes:
- Rank-2 tensors carry the largest weight magnitude (init norm 38) but move
  the least relatively. **36 % of them are essentially untrained.**
- Rank-3 verb-like tensors actually move the most (0.516).
- This refutes the obvious "nouns under-trained" guess; the under-trained
  population is the rank-2 type group (most prepositions, adj-noun
  composers).

## 2. Per-frequency-decile movement

Sorted all 80 622 typed symbols by occurrence count in train; grouped into
deciles.

```
Decile   freq range       n          ⟨rel-move⟩
  1     [  0,    1]      8,062      0.0005     ← 10% NEVER trained (freq=0)
  2     [  1,    1]      8,062      0.134
  3     [  1,    1]      8,062      0.134
  4     [  1,    1]      8,062      0.134      ← 33% trained once
  5     [  1,    3]      8,063      0.160
  6     [  3,    6]      8,062      0.219
  7     [  6,    8]      8,062      0.249
  8     [  8,   17]      8,062      0.328
  9     [ 17,   62]      8,062      0.548
 10     [ 62, 289981]    8,063      1.449
```

- **10 % of typed symbols are never trained** (freq = 0 in train, present
  in val/test).
- **33 % see exactly one** gradient step over the whole 100-epoch run.
- High-frequency symbols (top decile) move 10× more than singletons.

## 3. Per-caption val accuracy by symbol-frequency

For each of the 56 339 val captions, compute the **minimum train-frequency**
of any typed symbol it contains. Then group val captions by that minimum
and report the model's actual hard-neg-acc on each bucket.

```
bucket                                       n         val_acc
min_freq == 0  (contains an unseen symbol)   2,221     0.549
min_freq == 1                                  866     0.519
min_freq == 2                                  498     0.546
min_freq ∈ [3, 9]                            2,061     0.539
min_freq ∈ [10, 99]                         10,610     0.514
min_freq ≥ 100  (every symbol common)       40,083     0.533     ← still chance
```

**This is the key finding.** Val accuracy is statistically indistinguishable
from 0.50 across every bucket — even on the 40 k captions where every
typed symbol was seen ≥ 100× in training, the model scores 0.533.

So:

- The vocabulary long tail is **not** the primary cause of the val
  collapse. Removing it (training only on captions whose every symbol
  appears ≥ 100×) would not fix the model.
- Whatever causes the overfit, it affects high-frequency typed symbols
  equally to rare ones.

## 4. COCO contrastive pairs are NOT word-order swaps

Important context: in the COCO contrastive parquets, `true_caption` and
`false_caption` for each row almost always have **entirely different CCG
diagrams** (different lengths, different structures). They are random
caption pairings, not the word-order swaps that ARO uses.

```
                  identical    same diagram +    same diagram +    different
                  output       same multiset     diff. multiset    diagram
val (56,339)      0 (0%)       0 (0%)            68 (0.12%)        56,271 (99.88%)
test (56,371)     0 (0%)       0 (0%)            63 (0.11%)        56,308 (99.89%)
```

So the COCO contrastive task is just **topical** discrimination —
distinguish a real caption-image pair from an unrelated caption-image
pair. **`clip_baseline` (ResNet-18 + small BERT, ~23 M params) trained on
this exact data hits 0.96 val.** EinsumModel on the same data hits 0.53.
The data is fine. The architecture isn't.

## 5. SVD spectrum: trained tensors don't collapse to low rank

Computed per-typed-tensor SVD spectrum (unfold first axis × rest, take
singular values, compute effective rank via singular-value entropy).

```
shape                  n         eff_rank init    eff_rank trained    top-1 var init → trained
(10, 512)           29,817      9.906            9.890 ± 0.092        0.1232 → 0.1254
(10, 512, 10)       14,969      9.991            9.681 ± 0.520        0.1072 → 0.1498
(512, 10)           23,047      9.905            9.283 ± 0.942        0.1233 → 0.1880
```

Effective rank stays close to the maximum (10) after training; rank-3
tensors collapse slightly (9.99 → 9.68) but nowhere near rank-1.
**The model is using its full per-tensor capacity** — overfitting is
distributed across all available directions, not concentrated on a
few singular modes.

## 6. Comparison: ARO-trained vs COCO-trained (same architecture)

```
                                        train_acc    val_acc       n_symbols
EinsumModel + TTN, ARO combined train     0.886       0.641 (max)    2,716
EinsumModel + frozen ViT, COCO train      0.990       0.535 (max)   80,622
```

The same architecture **does** learn on ARO (smaller, simpler vocab,
real hard negatives in training) but **doesn't** learn on COCO. Two
differences:

  a. COCO has 30× more typed symbols (80 k vs 2.7 k) → vocabulary tail.
  b. COCO's negatives are random unrelated captions, not word-order
     swaps → the contrastive signal is topical, not structural.

Sections 1-3 showed (a) is not the dominant cause (well-trained-vocab
val captions still fail). So the issue is (b)? But then `clip_baseline`
on the same data succeeds at 0.96. So the architecture must be unable to
form generalising representations from topical negatives.

---

## Conclusion: where the overfitting actually comes from

1. **EinsumModel grants each typed symbol its own free tensor.** There
   is no parameter-sharing channel by which an update to "dog_0__n"
   ever helps "dog_0__n.r@B" or any other variant. Every typed symbol
   is its own private representation.

2. **With 1.04 B params on 450 k training pairs (~ 2 300 params per
   training pair), the model has more than enough capacity to memorise
   train-batch-specific pairings idiosyncratically** without forming
   transferable text representations.

3. **There is no inductive bias forcing typed symbols of the same lemma
   to compose into a coherent semantic vector.** clip_baseline's BERT
   shares attention weights across positions — that constraint forces a
   generalising representation. EinsumModel has no analogous constraint.

4. **Even SVD-effective capacity is fully used** (rank-3 tensors use
   ~9.7 of 10 singular dimensions). So the overfitting isn't a "lazy
   model uses low-rank shortcut" — every available direction is
   memorising something.

## Implied fixes (ranked by expected impact / cost)

  - **Pretrained initialisation from CLIP text embeddings.** Per lemma,
    query the frozen CLIP text encoder, use the 512-d vector as init
    for every typed variant of that lemma's embedding axis. Bond axes
    get standard random. The model starts from a semantically meaningful
    point and training only learns *deviations*. ~2 h to code, ~30 h
    train. Best expected impact.

  - **Lemma-tied parameterisation done correctly.** Earlier rank-1
    attempt collapsed; rank-4 hit numerical issues. A version that uses
    pretrained init + small learnable per-typed residuals would combine
    the two ideas.

  - **Severe vocab pruning (UNK-bucket all freq ≤ 5 symbols).** Reduces
    ~50 % of vocab to a single shared param. Cheap to implement
    (preprocessing only). Won't fix the 0.53 high-freq plateau on its
    own but may stack with above.

## What does NOT work (already tested)

  - More weight decay (0.1, 1.0): no effect.
  - Smaller bond_dim (2, 4): proportional capacity loss in both
    train and val; no help.
  - Larger batch size (128 → 1024): no effect.
  - Longer training: peaked at val 0.535 in epoch 17.
  - Lemma-tied rank-1: model collapses (all outputs collinear).
  - Lemma-tied rank-4: slow + numerical instability.
  - **CLIP-text-init for per-lemma embedding axis: numerical pathology.**
    Tested 2026-05-19; epoch 1 train produced NaN losses. The CLIP text
    embedding has elements with std ≈ 1/√512 ≈ 0.044; after 13 steps of
    cotengra einsum contraction at bond_dim=10, output norm underflows
    below `F.normalize`'s `eps`, producing degenerate values that backprop
    to NaN gradients.

## Important secondary finding: EinsumModel is numerically fragile

Two attempted "informed initialisation" fixes (lemma-tied rank-4,
CLIP-text-init) have both NaN'd at training start. This is not a bug in
either fix — it's that the 13-step cotengra einsum contraction of
bond_dim-10 typed tensors has a *very narrow* range of init magnitudes
that produce non-degenerate gradients:

  - Element std too small → contraction underflows → `F.normalize`
    eps-divides → NaN.
  - Element std too large → contraction overflows → NaN.

The original EinsumModel's shape-dependent init (`bound = 1/mean(cod)`,
varying by typed symbol) hits this narrow window by luck. Any
"semantically meaningful" init that doesn't match those magnitudes per
shape blows up.

So: implementing a working informed-init for EinsumModel requires either
a) per-typed-symbol magnitude calibration of the init, or
b) replacing cotengra einsum with `torch.einsum` + explicit per-symbol
   `F.normalize` or LayerNorm after each contraction step, to keep
   intermediates well-conditioned.

Either is ~half-day of careful numerical work. Worth doing only if you
believe the resulting model would actually generalise (and per §3 above,
that's already in doubt for high-frequency captions on this
architecture).
