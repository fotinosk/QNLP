# DisCoCLIP reproduction plan

**Status: gating milestone for the thesis.** Until the published baseline
is reproduced, "our tensor network improves on DisCoCLIP" is not
measurable — any comparison would be against our own weakened
reimplementation rather than the published result.

## The gap

DisCoCLIP (Lo, Hawashin, Abbaszadeh, Limback-Stokin, Wazni, Sadrzadeh;
*SEM 2025; arXiv:2509.21287; code at `github.com/kinianlo/discoclip`)
uses a **frozen CLIP ViT-B/32** image encoder plus a DisCoCat tensor
network text encoder. Our frozen-CLIP run (job 7434139) uses the **same
image encoder**.

| configuration | SVO-Probes | SVO-Swap |
|---|---|---|
| best from-scratch TTN (S1) | 0.5323 | — |
| **ours, frozen CLIP** | **0.5811** | **0.6095** |
| **DisCoCLIP, frozen CLIP** | **0.8355** | **0.9368** |

Swapping a from-scratch tower for CLIP buys **+0.049**. The remaining
**+0.254** is not image-side — the image encoder is already identical.
The image tower accounts for roughly **16%** of the total gap; the shared
text/loss/data path accounts for **84%**.

**Why this lifts TTN too:** every divergence below sits in the text
tower, loss, data pipeline, or evaluation — all shared with every TTN
run. Closing them raises the floor for the whole project, and the TTN-vs-
CLIP comparison only becomes meaningful once both sit on a correct
pipeline.

## Known evidence pointing at the text/data path

- **Train 0.97 / val 0.61 with CLIP frozen.** With the image tower
  frozen, every trainable parameter is text-side, so this gap is
  unambiguously text-tower overfitting. No inference required.
- **Text tower size: ours 17.7M parameters vs the paper's 537,600** for
  the same task — 33x larger. Verified by computing the per-symbol shape
  distribution from our own parquets; our per-symbol counts are actually
  *at or below* the paper's MPS formula, so the excess is vocabulary
  size, not a missing decomposition.
- **Row counts:** the paper filters to **8,984** image-caption pairs. Our
  threshold-50 pipeline produced **9,107** (a 1.4% difference — almost
  certainly the same protocol). Our current threshold-10 pipeline
  produces **14,284**. The "Round 2: recover more training data" change
  (`WORD_FREQ_THRESHOLD` 50 -> 10) took us off the paper's protocol, and
  `SVO_EXPERIMENTS.md`'s own data-pipeline section had recorded that 50
  was "matching the paper".

---

## Phase 0 — anchor the target (do first, blocks interpretation of everything else)

**Run the reference implementation as-is**, on its own data prep and
configs, and confirm it produces ~83.55 / 93.68.

This is not optional. Two outcomes, both essential to know before
spending any further compute:

- **It reproduces** -> the target is real and reachable in this
  environment, and every subsequent experiment is a bisection between two
  known-good anchors.
- **It does not reproduce** -> the published number depends on something
  not in the public repo (data prep, a checkpoint, an unlisted setting).
  That must be discovered now, not after a week of bisecting toward a
  target that cannot be hit.

Record: exact commands, config file used, data preparation steps, package
versions, and the resulting per-subset numbers.

## Phase 1 — audits (no training, hours not days)

Cheap, and each could independently account for a large share of the gap.

### 1.1 Evaluation protocol
Confirm our SVO-Probes accuracy is computed the same way theirs is.
Specifically:
- Is their headline **83.55% "overall"** an average over subsets or over
  rows? Our `evaluate_svo_probes` reports both; make sure we compare
  like with like. (Their verb figure, 82.42%, is the one that maps to our
  `verb_neg` subset.)
- **Skip rates.** Count rows dropped during evaluation for any reason
  (unknown symbols, parse failures, non-finite embeddings). Route A
  silently evaluated only 19.7% of the test set and nobody noticed for a
  full run. Report evaluated/total explicitly for the frozen-CLIP job.
- Which split, and how it was constructed. Ours is a custom
  `split_by_groups` partition; theirs is a 60/20/20 split of 8,984 pairs.

### 1.2 Data and vocabulary
- Symbol count and total text-tower parameters at `WORD_FREQ_THRESHOLD`
  **50 vs 10**, plus mean training examples per symbol at each.
- Row counts at each threshold against the paper's 8,984.
- Whether their filtering is word-frequency based at all, or something
  else that happens to yield a similar count.

### 1.3 Model variant
Their 83.55% is the **"Compact"** model — CCG-based with **variable-rank**
tensors. Our scheme is uniform bond dimension 10. Determine from their
code what "variable-rank" selects per symbol and whether our uniform
scheme is a different model entirely rather than a reimplementation of
the same one.

### 1.4 Loss
Theirs: a self-supervised contrastive loss. Ours: InfoNCE **plus a
triplet term at weight 100** (40,000 by default). This project has
already documented that the triplet term drives degenerate solutions on
ARO. Read their loss and record exactly what it is.

### 1.5 Optimisation defaults
Their repo defaults: **lr 0.003, batch 64, bond 10, temperature 0.07**.
Ours: `text_lr=0.001`, `batch_size=128`, bond 10, temperature 0.07.
Note that our experiment 1 used 0.003 before the "corrected" config moved
it to 0.001.

## Phase 2 — bisect, one variable at a time (parallel batch)

All rows are frozen-CLIP + our pipeline, changing **one** thing toward
the reference. Baseline: **0.5811**.

| # | change | hypothesis |
|---|---|---|
| R1 | `WORD_FREQ_THRESHOLD` 10 -> 50 | restores the paper's protocol; shrinks vocabulary and parameters, raises examples-per-symbol |
| R2 | drop the triplet term (plain contrastive, `triplet_weight=0`) | removes a documented driver of degenerate solutions, absent from the reference |
| R3 | `text_lr` 0.001 -> 0.003, `batch_size` 128 -> 64 | matches reference defaults |
| R4 | R1 + R2 + R3 combined | if the single-variable rows each move it partway, the combination is the reproduction candidate |
| R5 | variable-rank tensors per 1.3, if it turns out to differ | the model-identity question |

Run R1-R3 in parallel; R4 after they report; R5 only if 1.3 finds a real
difference.

### Paired TTN arm — run every row twice

Each row above runs **twice in parallel**: once with frozen CLIP, once
with the current best TTN configuration (S1: A1+B1+cp_rank=128), changing
only `IMAGE_MODEL_IMAGE_BACKBONE`. Same seed, same data version, same
everything else — the moment the arms drift, the comparison stops meaning
anything.

**Why this is worth the extra jobs.** It measures the thesis's central
quantity at every step. The CLIP-vs-TTN gap is currently +0.049 against a
broken pipeline:
- if that gap **narrows** as fixes land, part of the tower's apparent
  deficit was an artifact of the broken pipeline;
- if it stays **flat at ~0.05** while both arms rise, the tower deficit is
  independent and additive — a much cleaner claim to write up.

Neither can be observed by fixing the pipeline first and re-running the
tower afterwards. It also makes Phase 3 redundant rather than merely
faster.

**Pre-committed interpretation rule.** The TTN arm sits at ~0.53, close
to chance, where this project has repeatedly documented that differences
are noise (experiment 15: even checkpoint selection is noise-driven at
this level). **A TTN-arm change counts as real only at +0.03 or more.**
Decided now, not when the numbers arrive. The TTN arm is confirmatory —
does this fix transfer? — and never the decision input for whether a fix
is adopted; that decision belongs to the frozen-CLIP arm, which has the
dynamic range to show it.

**Expect R2 to split the arms.** It drops the triplet term, but the TTN
arm's best configuration (S1) *uses* `triplet_weight=100` — that was its
winning setting. R2 plausibly helps CLIP and hurts TTN. That is
informative, not contradictory, and should not be read as the fix being
wrong.

**Report for every row:** train and val `hard_neg_acc` per epoch (the
0.97/0.61 gap is the diagnostic), per-subset test accuracy (subj/verb/obj
separately, to compare against 80.74/82.42/87.79), SVO-Swap, evaluated-
rows/total, and text-tower parameter count.

## Phase 3 — TTN on the corrected pipeline

Largely **subsumed by the paired arm above**, which produces the TTN
number for every configuration as it lands. What remains for Phase 3 is
only the final confirmation run at whatever configuration Phase 2 settles
on, plus any TTN-specific retuning (for example, re-optimising
`triplet_weight` for the tower if R2 splits the arms).

The comparison that matters for the thesis is then:

| | SVO-Probes |
|---|---|
| DisCoCLIP (frozen CLIP), reproduced | target ~0.8355 |
| ours, TTN image tower, same corrected pipeline | the thesis result |

## Success criteria and stopping rule

- **Phase 0 success:** reference code reproduces within ~2 points of
  83.55.
- **Phase 2 success:** our pipeline with frozen CLIP reaches within ~3
  points of the reference number.
- **Stopping rule:** if Phase 2 closes less than half the 0.254 gap, stop
  bisecting and diff our text encoder against theirs line by line
  instead. At that point the divergence is structural rather than
  configurational, and further single-variable runs will not find it.

## What not to do

- Do not run **exploratory** image-tower work (node candidates, DTTN
  variants, new pretrained towers) until Phase 2 concludes. The
  frozen-CLIP result bounds what any image tower can achieve on the
  current pipeline at ~0.58; a better tower cannot exceed that bound
  while the remaining 84% of the gap is unaddressed. This does **not**
  apply to the paired TTN arm above, which carries the existing best
  configuration forward rather than exploring new ones.
- Do not report any current SVO number as a comparison against DisCoCLIP.
  Until the baseline reproduces, those comparisons are against a
  weakened reimplementation.

---

# Phase 4 — structural divergence (2026-09-21)

## Status

| row | CLIP Probes | CLIP Swap | TTN Probes |
|---|---|---|---|
| baseline | 0.5811 | 0.6095 | 0.5323 (S1) |
| R1 (threshold 50) | 0.6229 | 0.5510 | noise |
| R4 (**confounded**, see below) | 0.6262 | 0.6327 | 0.5506 |
| **R6 (first clean combined run)** | **0.6649** | **0.7692** | 0.4864 |
| DisCoCLIP target | 0.8355 | 0.9368 | — |

**0.084 of the 0.254-point Probes gap is closed (33%).** Swap gained
+0.14 in a single step at R6.

## Correction: the stopping rule fired on confounded data — treat it as un-triggered

Phase 2's stopping rule was invoked on R4's "sharply sub-additive"
result. R4 was subsequently found to have run with `batch_size` silently
clobbered to 128 by an unconditional `export` in both `submit_svo.sh` and
`submit_svo_frozen.sh`, so one of its three combined variables was never
applied. The sub-additivity that justified stopping was measured on a
partly-broken combination.

R6 — the first run with all three genuinely applied — gained **+0.039**
on top of R4. That is the fixes compounding, not sub-additivity.

**Therefore: the stopping rule is un-triggered and bisection continues.**
The revised rule is at the end of this phase.

## The primary remaining lead: the parameter gap survived every configurational fix

| | text-tower parameters |
|---|---|
| ours, original | 17.7M (1,356 symbols) |
| ours, threshold 50 (R1) | 5.67M (465 symbols) |
| **Phase 0 reference reproduction** | **1.68M** |

Still **3.4x the reference** after the single largest configurational fix.
And the `lemmafix` recompile raised surviving rows from 9,107 to 15,199
(merging inflected forms lifts word frequencies, so more words clear the
threshold), so R6's vocabulary is plausibly *larger* again, not smaller.

No configurational change — data protocol, loss shape, learning rate,
batch size — has moved this. It is the one measured, quantified
difference from the reference that remains unexplained, and it is the
natural suspect for the remaining ~0.17.

## A — Audit: variable-rank tensors (blocking, no cluster time)

DisCoCLIP's 83.55% model is **"Compact" — CCG-based with variable-rank
tensors**, per their README. Ours uses a uniform bond dimension of 10.

A per-symbol rank scheme would produce exactly the observed signature: a
3-4x parameter difference that persists after every configurational
setting is matched.

This audit was scheduled as Phase 1.3 and has now been deferred through
two phases. **It is the highest-value hour available and blocks R7.**
Determine from `github.com/kinianlo/discoclip`:

1. How per-symbol ranks are assigned — by CCG category, by word
   frequency, by tensor order, or learned.
2. The resulting parameter count, and whether it accounts for 1.68M.
3. Whether our uniform bond-10 scheme is a reimplementation of the same
   model with a different setting, or **a different model entirely**.

Outcome (3) matters for the thesis independently of accuracy: if we have
been comparing against a different model, that needs stating plainly
rather than discovered in review.

## R7 — implement variable-rank, both arms

Conditional on A finding a real difference. This is the **first
structural rather than configurational change** in the campaign, and it
targets the one quantified divergence that survives.

Both arms, one variable, on top of R6's configuration.

## R8 — attribute R6's gain (optional, only if the cluster is idle)

R6 bundles three changes: `lemmafix` data, genuinely-applied
`batch_size=64`, and weight-norm off. Which drove the +0.039 is unknown.
Three single-variable rows against R6 would attribute it.

**Priority: below A and R7.** Attribution is good bookkeeping; closing
the gap is the objective. Do not let R8 delay A.

## Measurement requirements for every row from here

In addition to the Phase 2 reporting list:

1. **Text-tower symbol count and parameter count**, reported against the
   reference's 1.68M. This is now a primary metric, not a footnote — it
   is the quantity the remaining lead is about.
2. **Evaluated-rows / total** for both Probes and Swap. The lemmafix run
   reports 104/104 on Swap; keep reporting it, since a silent skip rate
   has already gone unnoticed for a full run once in this project.
3. **Verify the intended config in the job log before letting a run go
   unattended.** Three bugs in one batch — a hardcoded `derived_name`, a
   clobbered `batch_size`, and a device-moving `load_state_dict` — all
   silently produced runs that did not match their documented settings.
   R6 was launched only after confirming `Batch size: 64` in the log;
   make that the standing practice.

## Revised stopping rule

Bisection continues while it is closing the gap at a useful rate.

- **Continue** if A finds a real structural difference and R7 closes a
  further 0.05 or more.
- **Stop and diff the text encoders line by line** if R7 closes less than
  0.03, or if A finds no meaningful difference between the rank schemes.
  At that point the divergence is neither configurational nor in the
  tensor representation, and further single-variable rows will not find
  it.
- **Hard stop regardless of progress** once the campaign has consumed the
  time budgeted for it — the thesis needs the reproduction *documented*,
  not necessarily *completed*. A reproduction that closes 40% of the gap,
  with the remaining divergence identified and quantified, is a
  legitimate and reportable outcome.

## A — Audit result: no variable-rank scheme exists; same model

Read `github.com/kinianlo/discoclip` directly (local clone at
`/Users/fotinoskyriakides/Desktop/Dev/discoclip`):

- `discoclip/utils/ansatz.py`'s `CustomMPSAnsatz` is **line-for-line the
  same class we already have** (`_split_ar`, `BOND_TYPE`, `max_order`
  chunking). `bond_dim` is a single scalar passed once into
  `ob_map[self.BOND_TYPE] = Dim(bond_dim)` — there is no per-symbol or
  per-CCG-category rank parameter anywhere in the class.
- `discoclip/models/tn_models.py`'s `EinsumModel` is likewise
  near-identical to ours (same `reset_parameters` uniform-init formula,
  same `sym2weight`, even the same pre-existing `load_state_dict`
  CPU-device bug found in our own model this session).
- `configs/svo_default.yaml` (the actual config used for training):
  `embedding_dim: 512`, `bond_dim: 10` — both already exactly matched in
  every one of our runs. Also confirms `hard_neg_loss_weight: 0`
  (R2), `batch_size: 64`, `learning_rate: 0.003` (R3),
  `weight_decay: 0.01` (already applied at R5/R6), `temperature: 0.07`
  (already matched) — nothing here is unmatched.

**Verdict: our uniform bond-10 MPS scheme is the reference's model, not
a different one.** "Compact... variable-rank tensors" (the paper's
README wording) describes the MPS decomposition's per-symbol *number of
cores*, which varies with each symbol's CCG type complexity via
`_split_ar`'s chunking — exactly what our ansatz already does, since we
copied the same class. It does not describe a per-symbol bond-dimension
value. **No model-identity difference exists. R7 is not warranted.**

**Byproduct finding, not the smoking gun**: `svo_default.yaml` also sets
`epochs: 10, patience: 5`, far more aggressive than our
`max_epochs=100, patience=10`. Checked against R6 CLIP's actual
checkpoint selection (job 7435541): best epoch was **4** — well inside a
10-epoch budget regardless — so this difference is unlikely to be
material, though it hasn't been isolated as a single-variable row.

**Per the revised stopping rule** ("Stop and diff the text encoders line
by line... if A finds no meaningful difference between the rank
schemes"): **proceed to a full line-by-line structural diff** of the two
text encoders/training loops, extending the partial one already done in
Phase 4's earlier structural-comparison pass (which found the
weight-norm gauge-fix and `weight_decay` — both already applied). R7 and
R8 (attribution) are deprioritized under this outcome.

## Line-by-line diff: found the real driver, retracted a wrong "fix"

Requested full structural diff, in progress. Two findings so far:

1. **`SymbolLemmatizeStep` (added for the "corrected fix") was
   redundant.** Our own `BobcatTextProcessor` already tokenises raw text,
   WordNet-lemmatises separately, and relabels the CCG tree's leaves with
   lemmas before diagram conversion — the exact mechanism discoclip's
   processor uses. Verified: the *original* default LMDB already stores
   `hold_0__n.r@B` for "holds", not `holds_0`. The parse-order concern
   was based on a wrong premise; the step is a harmless no-op, and the
   earlier "structural fix" framing is retracted.
2. **The real driver of R6's data-side gain: a corrupted, unscoped
   Bobcat parser cache silently discarded 31% of the SVO corpus** in the
   default `derived_v1` population (17,782/26,189 valid vs. 26,180/26,189
   after a fresh cache). Root cause and fix documented in full in
   `TTN_CIFAR_EXPERIMENTS.md`'s "Correction" section — `svo/pipeline.py`,
   `winoground/pipeline.py`, and `compile_shard.py` all fixed.
   **This affects every past default-config SVO experiment, not just
   this reproduction plan** — a clean recompile of the default (non-
   suffixed) atlas is a separate, project-wide follow-up.

Diff continues on the remaining pieces (loss internals, evaluation
methodology, training-loop specifics) not yet compared line by line.

## R9 candidate found: InfoNCE anchor direction

Their `InfoNCE` (`discoclip/models/criteria.py`) is byte-identical to
ours (`qnlp/domain/models/other/loss.py`) — same formula, same
docstring, evidently copy-pasted at some point. But the two call sites
pass arguments in **opposite order**:

- theirs (`train_svo.py`): `contrastive_criterion(pos_image_embeddings,
  sentence_embeddings)` — `text_emb=image`, `pos_emb=caption` — **image
  is the anchor/query row**.
- ours (`ImageContrastiveLoss.__call__`, and the legacy ARO loss too):
  `self._infonce(caption_emb, true_emb)` — **caption is the anchor**.

Since `InfoNCE`'s in-batch similarity matrix is `text_emb @ pos_emb.T`
and cross-entropy softmaxes row-wise, swapping which side is the anchor
is not cosmetic — `S` and `S.T` give different losses and gradients for
a non-symmetric batch of (caption, image) pairs (verified: same batch,
`image_as_anchor=False` gives loss 4.349, `=True` gives 4.109). Their
hard-negative *evaluation* (`pos_sim > neg_sim`) is still caption-
anchored regardless of which direction training used, same as ours — so
this is purely a training-objective difference, not an eval one.

Added `ImageContrastiveLoss(image_as_anchor: bool = False)` (default
preserves every existing run) and wired it through
`SVOExperimentConfig.image_as_anchor` / `SVO_ML_IMAGE_AS_ANCHOR`.

## R9 result: no real effect on either arm

Jobs 7435931 (CLIP) / 7435932 (TTN), `image_as_anchor=true` on top of
R6's full configuration, both arms.

| | SVO-Probes | delta vs R6 | SVO-Swap | delta vs R6 |
|---|---|---|---|---|
| R6 CLIP | 0.6649 | — | 0.7692 | — |
| R9 CLIP | 0.6722 (obj_neg 0.7746, subj_neg 0.6811, verb_neg 0.6234) | **+0.0073** | 0.7692 | **+0.0000** |
| R6 TTN | 0.4864 | — | 0.4519 | — |
| R9 TTN | 0.5059 (obj_neg 0.5125, subj_neg 0.4861, verb_neg 0.5105) | **+0.0195** | 0.4615 | +0.0096 |

Both arms move less than the plan's +0.03 real-effect threshold on every
metric — CLIP's SVO-Swap is exactly unchanged. **The InfoNCE anchor
direction is not a real driver of the remaining gap.** `image_as_anchor`
stays off by default; not adopted.

This closes the InfoNCE-direction lead. R6 remains the best confirmed
configuration (CLIP 0.6649 SVO-Probes / 0.7692 SVO-Swap), gap to the
DisCoCLIP target (0.8355) still ~0.163.

## Thesis framing note — applies now, not at the end

The TTN arm sits at chance (R6: 0.4864) while CLIP reaches 0.6649 on the
**identical** pipeline. That within-pipeline comparison is clean and
supports the conclusion that the from-scratch image tower is the TTN
arm's bottleneck.

It does **not** support "our tower costs 0.17 versus DisCoCLIP", because
our pipeline is still 0.17 short of the reference. The honest statement
is: *on our pipeline, frozen CLIP reaches 0.66 where our from-scratch
tower reaches 0.49.* The gap decomposition only becomes a claim about the
tower once the baseline reproduces. Write it the first way until then.

---

# Phase 5 — next batch (2026-09-21)

Three items. One cluster job, two desk tasks. Nothing blocks anything
else, so all three start immediately.

## 5.1 — S1's TTN configuration on clean data (one job, highest value)

**The question this settles.** The project's central negative claim is
that the from-scratch image tower cannot learn adequate visual features
from SVO. Every run supporting it was either (a) trained on the
corrupted `derived_v1` population — 17,782 valid rows of 26,189, **68%**
— or (b) trained on clean data but in the **CLIP-optimal** configuration
rather than the tower's own best one.

R6's TTN arm is case (b): it ran `triplet_weight=0, text_lr=0.003,
batch_size=64` — settings tuned on the CLIP arm. The tower's best-known
configuration is S1's `triplet_weight=100`. Phase 2 explicitly predicted
this split: *"R2 plausibly helps CLIP and hurts TTN. That is
informative, not contradictory."* R6 TTN (0.4864) landing below S1 on
corrupted data (0.5323) is consistent with the predicted split, not with
a demonstrated tower failure.

**So the tower has never been tested on clean data in its own best
configuration.** This run does that.

**Config:** S1 (A1 isometric init + B1 feature map + `cp_rank=128`,
`triplet_weight=100`) on `SVO_ML_DATASET_SUFFIX=_thresh50_lemmafix`.
Everything else at S1's settings, not R6's.

**Comparisons — this run sits at the intersection of two
single-variable axes:**

| against | isolates | prior number |
|---|---|---|
| S1 on corrupted data | **data quality** (corrupt -> clean) | 0.5323 |
| R6 TTN on clean data | **configuration** (CLIP-optimal -> tower-optimal) | 0.4864 |
| R6 CLIP on clean data | **image tower** (CLIP -> TTN), same pipeline | 0.6649 |

**Criterion:** the plan's standing +0.03 threshold against 0.5323.

**Outcomes:**
- **Clears 0.5323 meaningfully** -> the tower's failure was partly an
  artefact of corrupted data and/or a mistuned configuration, and the
  central negative claim needs rewriting before it goes in the thesis.
- **Stays at ~0.53 or below** -> the negative claim survives its
  strongest challenge, and can be stated with confidence rather than
  hedged. That is a genuinely useful outcome too, and the reason this
  run is worth its slot regardless of direction.

## 5.2 — Variable-rank audit — SUPERSEDED, already resolved

This item duplicates the audit already completed earlier in this
document ("A — Audit result: no variable-rank scheme exists; same
model", above). Re-confirmed rather than re-litigated: `discoclip`'s
`CustomMPSAnsatz`/`EinsumModel` are line-for-line the same classes we
have, its `svo_default.yaml` sets `bond_dim: 10` as a single uniform
scalar, and no per-symbol rank parameter exists anywhere in the code.
"Compact... variable-rank tensors" describes the MPS decomposition's
per-symbol *core count* (already replicated via `_split_ar`), not a
bond-dimension value. **Verdict stands: same model, R7 not warranted.**
Skipped as redundant rather than re-run.

Original text, kept for reference below.

Scheduled as Phase 1.3, deferred through three phases. It remains the
largest unexplained structural divergence from the reference.

**The gap it targets:** our text tower is **3.4x** the reference's
parameter count (5.67M at threshold 50, vs Phase 0's reproduced 1.68M),
and no configurational change — data protocol, loss shape, learning
rate, batch size — has moved it. DisCoCLIP's 83.55% model is
**"Compact": CCG-based with variable-rank tensors**; ours uses a uniform
bond dimension of 10. A per-symbol rank scheme would produce exactly this
signature.

**Determine from `github.com/kinianlo/discoclip`:**

1. How per-symbol ranks are assigned — by CCG category, by word
   frequency, by tensor order, or learned.
2. The resulting parameter count, and whether it accounts for 1.68M.
3. Whether our uniform bond-10 scheme is the same model with a different
   setting, or **a different model entirely**.

Outcome (3) matters for the thesis independently of accuracy. If we have
been comparing against a different model than the one that produced
83.55%, that must be stated plainly rather than found in review.

**Decision gate:**
- Real difference found -> implement as **R7**, both arms, one variable,
  on top of R6's configuration.
- No meaningful difference -> the Phase 4 stopping rule applies: stop
  bisecting, diff the text encoders line by line, and if that finds
  nothing either, close the reproduction at ~33% of the gap with the
  residual documented.

## 5.3 — Winoground data soundness check (cheap script)

**Why.** `qnlp/preprocessing_pipelines/winoground/pipeline.py` has the
**identical bug** that corrupted the SVO atlas: it never passes
`cache_path`, so it defaults to the single shared
`~/.cache/lambeq/bobcat/diskcache` and is exposed to the same
concurrent-write-over-NFS corruption. It was flagged as suspect when the
SVO corruption was diagnosed and has not been checked.

**Check.** Mirror the diagnosis already used for SVO:
1. Run `enrich_atoms()` over the Winoground manifest and record the
   valid-diagram/symbols rate. SVO's corrupted population showed **68%**
   against **99.97%** for a fresh cache — anything materially below
   ~99% indicates the same problem.
2. Inspect the LMDB's `error` fields for the sqlite corruption
   signatures: `"database disk image is malformed"`, `"file is not a
   database"`, and the downstream `"not enough values to unpack
   (expected 1, got 0)"`.

**If corrupted:** every Winoground number in this project is unsound and
must not be reported. Recompile with `cache_path` passed (the fix already
applied to `svo/pipeline.py`), and re-run anything that depended on it.

**Result: clean.** `enrich_atoms()` over the full 800-row manifest:
**784/800 (98%) valid**. Inspected the LMDB `error` field for all 800
unique hashes: 16 failures, **all** genuine Bobcat parse errors
(`"Bobcat failed to parse 'Theres...'"` — an unrelated apostrophe-
stripping issue in Winoground's captions), **zero** instances of the
sqlite corruption signatures (`"database disk image is malformed"`,
`"file is not a database"`) that identified SVO's corruption. Winoground
never hit the concurrent-write bug in practice — presumably because it
was never compiled via a concurrent SGE array job the way SVO's
`compile_shard.py` was. Existing Winoground numbers are sound; the
`cache_path` fix already applied to `winoground/pipeline.py` prevents
this from becoming a problem in any future recompile, but no re-run is
required.

**Also audit the remaining pipelines** for the same omission.
`coco/pipeline.py` and the ARO loader already pass `cache_path`
correctly; SVO and Winoground did not. Confirm no others do, so this
class of bug is closed rather than patched twice.

## Note on the clean default-atlas recompile

A full clean recompile of the default (unsuffixed) SVO atlas is queued
separately. It is not in this batch's critical path — 5.1 uses the
already-clean `_thresh50_lemmafix` data — but every unsuffixed SVO number
in the project sits on the corrupted 68% population until it lands, and
no prior SVO result should be reported without noting that.
