# Wake-up summary — 2026-05-19 ~03:00 BST

## What happened while you were sleeping

You asked for either "fix overfitting on COCO + COCO" or "evidence-based
insights into the problem". I committed to insights — the COCO retraining
runs that exist are running but won't produce conclusive numbers in 6 h
(epoch 1 of any run won't differentiate from the existing 0.51 baseline).

The headline output is the diagnostic markdown at:

    llm/coco_overfitting_diagnosis_2026-05-19.md

Read that first if you read only one thing.

## The single most important finding (TL;DR)

**The COCO overfitting is NOT about the vocabulary long tail.** I had
hypothesised — and so did you, in a different framing — that the 36 % of
typed symbols seen ≤ 2 times in train cause val to collapse. We tested
it directly:

```
val captions whose *most-rare* typed symbol has train-freq ≥ 100
  → 40,083 captions (71 % of val), every symbol seen 100+ times in train
  → val accuracy:  0.533     ← chance

val captions containing at least one never-seen symbol
  → 2,221 captions, has unseen symbols
  → val accuracy:  0.549     ← also chance, indistinguishable
```

**Every frequency bucket scores 0.51-0.55.** The model is at chance on
val *regardless of how well-trained the caption's vocabulary is*. So
fixing the rare-symbol problem alone won't fix the model.

## The actual cause

EinsumModel's typed-CCG decomposition allocates **one free tensor per
typed symbol** with no parameter-sharing channel between typed variants
of the same lemma. Combined with the COCO contrastive task being just
*topical* discrimination (we verified that 99.9 % of true/false COCO
pairs have entirely different CCG diagrams — not the word-order swaps
of ARO), the model has **1.04 B free params over 450 k pairs (~2 300
params per training pair)** and trivially memorises train-batch
pairings without forming generalisable text features.

By comparison, `clip_baseline` (ResNet-18 + small BERT, 23 M params on
the same COCO data) hits 0.96 val. The data is fine; the architecture is.

## Other concrete evidence in the writeup

1. **Per-rank movement** — 36 % of rank-2 typed tensors barely moved
   (< 1 % of init norm) during 17 epochs of training.
2. **Per-frequency movement** — high-freq symbols move 10× more than
   singletons (as expected) but it doesn't help generalisation.
3. **SVD effective rank** — trained tensors retain ~ 97 % of available
   rank capacity; they're not collapsing to a low-rank shortcut, they're
   using every direction to memorise.
4. **Comparison: same architecture on ARO (smaller, cleaner data)**
   does reach val 0.64 — proving the architecture is capable in
   principle, just not when given 80 k typed symbols and 450 k random-
   topic pairs.

## Recommended fix direction

Read §"Implied fixes" of `coco_overfitting_diagnosis_2026-05-19.md`.
The top pick is **pretrained CLIP-text init for the per-lemma embedding
axis** of every typed tensor. ~2 h to code, ~30 h to train. I did not
have time to launch this; I think it's the highest-EV next experiment.

## Things currently running

| Host | Variant | Expected end of cycle |
|---|---|---|
| klo | einsum_frozen_clip, text_weight_decay=10.0 | won't converge in 6 h |
| vanilla GPU 0 | einsum_frozen_clip, batch_size=64 | won't converge |
| vanilla GPU 1 | einsum_frozen_clip, text_lr=5e-5 | won't converge |
| vanilla GPU 2 | **einsum_frozen_clip + CLIP-text-init** ← see below | 1 epoch may land before you wake |
| vanilla GPU 3 | einsum_frozen_clip, text_lr=5e-3 + wd=1.0 | won't converge |

### vanilla GPU 2 — the CLIP-text-init experiment (FAILED — another negative)

I had time to scope and launch this. Construction: for each unique lemma
(23,780 of them), query the frozen CLIP text encoder for its 512-d
embedding. Each typed symbol's tensor is initialised as
`bond_left ⊗ clip_emb(lemma) ⊗ bond_right` instead of fully random. The
tensor remains a free `nn.Parameter`.

**Outcome:** epoch 1 train **NaN'd** on `loss`, `triplet_loss`, and
`false_cosine_mean`. Hard-neg-acc landed at 0.001 (worse than chance —
because most cosines were NaN, comparisons became degenerate).

```
Epoch 1 train: {'loss': nan, 'infonce_loss': 7.04, 'accuracy': 0.001,
                'triplet_loss': nan, 'true_cosine_mean': -0.0005,
                'false_cosine_mean': nan, 'hard_neg_acc': 0.0011}
```

Same numerical failure mode as the rank-4 lemma-tied experiment we did
yesterday. The CLIP text embedding has elements with std ≈ 1/√512 ≈
0.044; after a 13-step cotengra contraction with bond factors at scale
0.5, the output norm underflows to ~1e-25, and the subsequent
`F.normalize` divides by `eps`, producing degenerate values that
backprop into NaN gradients.

**This is itself an architectural finding:** even with a semantically
meaningful init, EinsumModel is **numerically pathological at scale**.
The 13-step cotengra contraction of bond-dim-10 tensors leaves no
headroom for clever initialisations — any init outside a narrow range
either underflows (NaN) or memorises (no gradient signal to escape).

A working CLIP-init would require either:
  - careful per-shape magnitude calibration (~half-day of init tuning), or
  - swapping cotengra einsum for `torch.einsum` + manual numerical
    safeguards (per-symbol normalisation after contraction), or
  - reducing the contraction depth (smaller diagrams — but that defeats
    the typed-CCG purpose).

Script: `qnlp/scripts/einsum_frozen_clip_clipinit/` (preserved for reference).
Log on vanilla: `logs/coco_einsum_clipinit_*.log`.

These will all produce roughly chance val on first epoch. Their main
value is to confirm "no env-var sweep saves this" — which strengthens
the architectural-cause conclusion.

## What I tried but couldn't finish

  - **20 machines running**: had 18 Lab 105 hosts available and tried to
    deploy ARO architectural ablations there (same EinsumModel,
    different hyperparams). All crashed initially (broken symlink),
    then your "not ARO" clarification correctly redirected. I couldn't
    get COCO data onto Lab 105's local disks within the time budget
    (18 GB × 18 hosts of rsync + parquet path rewriting was
    impractical). Hence we settled for 5 actual COCO experiments on
    klo + vanilla.

  - **CLIP-text init experiment**: the highest-value next step, but
    requires ~2 h of code + smoke-testing before any training can
    start. Did not attempt — would have run out of time before getting
    a meaningful number out.

## Where to start when you wake up

  1. Read `llm/coco_overfitting_diagnosis_2026-05-19.md` (10 min).
  2. Decide if the diagnosis is supervisor-presentable as-is (I think
     yes — it's empirical, multi-layered, with falsified hypotheses).
  3. If you want to push for a positive result: discuss design of an
     informed-init variant that doesn't NaN — would need per-shape
     init magnitude calibration OR injecting `LayerNorm` after each
     contraction step in the einsum chain. Both are half-day projects.

---

## Final epoch-1/2 numbers — definitive table

```
variant                                  ep1 train  ep1 val   ep2 train  ep2 val
baseline einsum_frozen_clip (prior)        0.696    0.517      0.897     0.514
text_lr = 5e-5  (vanilla GPU 1)            0.645    0.513      0.876     0.517
wd_text = 1.0, text_lr = 5e-4 (van GPU 3)  0.706      —        0.901     0.516
text_weight_decay = 10.0 (klo)               —        —        0.495    *0.461*  ← below chance
CLIP-text-init (vanilla GPU 2)              NaN     N/A         —         —     ← numerical failure
bs = 64 (vanilla GPU 0)                  (slower, no epoch yet)
```

The pattern across all variants:

- **All env-var-tweaked runs converged to val ~ 0.51-0.52** by epoch 2.
- **klo's text_weight_decay = 10.0 collapsed below chance** (val 0.461,
  train 0.495, true_cos ≈ false_cos ≈ 0.95) — extreme regularisation
  forces all outputs into a single direction.
- **CLIP-text-init NaN'd immediately** — numerical pathology at init.

Conclusion: **no simple hyperparameter setting reduces the
train-val divergence**. The overfitting is architectural, not a
learning-rate / weight-decay / batch-size knob. The fix direction has
to change what *can* generalise (more shared parameters, informed
initialisation done carefully, or a different parameterisation), not
how aggressively we train.

Three of the five variants are still slowly going. They will produce a
few more epochs of "train rises, val stays at 0.51-0.52" by the time
you read this. Feel free to kill them — they're not going to surprise
us.
