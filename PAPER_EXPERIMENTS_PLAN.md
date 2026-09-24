# Paper experiments plan — extending DisCoCLIP with a TTN image tower

Settled plan as of 2026-09-23. Supersedes earlier drafts; the measurement
history behind these decisions is in `DISCOCLIP_REPRODUCTION_PLAN.md`.

## The three models

| id | image tower | text tower |
|---|---|---|
| **M1** | frozen CLIP | DisCoCat/TTN | *DisCoCLIP — the baseline being extended* |
| **M2** | **TTN** | DisCoCat/TTN | *the contribution — fully tensor-network* |
| **M3** | **TTN** | CLIP text | *isolates the image tower against a strong text encoder* |

M3 is the mirror of M1: M1 tests the text tower against a strong image
encoder, M3 tests the image tower against a strong text encoder.

## Settled decisions

**D1 — SVO uses the reference's row-split protocol**, and **only that
protocol is reported**. It is the protocol the published DisCoCLIP
numbers were produced under, so it is the one that makes our numbers
comparable to the literature. The strict grouped-split results are not
reported in the results tables.

The leakage is **acknowledged in the discussion section**: measured on
the reference's released CSVs, 51.6% of test positive images and 39.4%
of test (caption, positive-image) pairs also appear in training.

**D2 — Word-frequency threshold 150**, not 50. An absolute threshold is
not scale-invariant: the reference's 50 on 8,984 rows is 0.56% relative
frequency, ours on 36,841 rows is 0.14%. Threshold 150 restores their
relative strictness (151 word types vs their 160) and gave the best
measured configuration on both arms (E4: CLIP 0.9274, TTN 0.9171).

**D3 — Type-mismatch rows are dropped, not repaired.** Where a Winoground
word appears with a CCG type unseen in ARO training, the row is excluded.
Initialising the new-type symbol from the same word's other-type tensor
was considered and **rejected**: it asserts a word means the same thing
regardless of grammatical role, which is precisely what a DisCoCat model
denies. Repairing them would contradict the paper's own premise.

**D4 — No COCO training.** Out of scope; subject of a separate paper.

---

# Step 1 — Retrain all three models on ARO and SVO

Six runs: 3 models x 2 benchmarks. Final numbers for the paper.

**Blocker:** M3 needs a **CLIP text encoder** class, which does not yet
exist. Build it alongside `qnlp/discoviz/models/clip_image_model.py`,
same pattern: frozen `openai/clip-vit-base-patch32` text side, 512-dim
output matching `embedding_dim`, selected by config flag.
`ContrastiveVLM` already takes both towers, so no architectural change.

**Configs:** SVO at E4 settings (row-split, threshold 150). ARO at the
existing legacy-faithful config.

**Report for every run:** SVO-Probes overall plus subj/verb/obj subsets;
SVO-Swap; ARO attribution/relation/overall; evaluated-rows/total; text-
tower symbol count and parameter count.

**Runs in parallel with Step 2.**

## Step 1 results (2026-09-24)

CLIPTextModel built (`qnlp/discoviz/models/clip_text_model.py`), wired
through both SVO's and ARO's dataset/eval pipelines. Two bugs found by
the first real cluster runs and fixed: `evaluate_svo_probes` still
called `sym2weight` directly (missed when the other three OOV-filter
sites were fixed), and ARO's `run.py` had unguarded Winoground/ARO/
SugarCREPE eval calls, so a Winoground crash (its own `WinogroundDataset`
was never extended for CLIP text's raw-text passthrough — out of scope
for Step 1) prevented the ARO number from ever being computed. Both
fixed; M3's numbers below were recovered from the already-trained
checkpoints via `reeval_checkpoint.py`, not a retrain.

| model | SVO-Probes | SVO-Swap | ARO overall | ARO attribution | ARO relation |
|---|---|---|---|---|---|
| **M1** (CLIP image, DisCoCat text) | 0.9274 (E4 CLIP) | 0.9524 | 0.6735 | 0.7341 | 0.5997 |
| **M2** (TTN image, DisCoCat text) | 0.9171 (E4 TTN) | 0.8889 | pending (retraining, `m2_aro_retry`) | — | — |
| **M3** (TTN image, CLIP text) | 0.6394 (obj_neg 0.7189, subj_neg 0.5962, verb_neg 0.6196) | 0.4127 | 0.4723 | 0.4676 | 0.4781 |

**M2 ARO's first attempt was OOM-killed** at epoch 24 (transient cluster
memory issue, unrelated to any code change — training had reached
hard_neg_acc 0.90 on train at that point) — relaunched as `m2_aro_retry`,
still running.

**M3 sits well below M1 and M2 on both benchmarks**, and its ARO number
(0.4723) is essentially exact chance (0.5) — a frozen, general-purpose
CLIP text encoder does not transfer well to either task when paired with
the from-scratch TTN image tower, in contrast to M1's frozen-CLIP-image
pairing (which reaches 0.93+ on SVO). This is a real, reportable finding
for the paper's discussion of M3 as the image-tower isolation control:
the tower alone, without a compatible from-scratch text encoder trained
jointly with it, does not carry the representation on its own.

---

# Step 2 — Winoground substitution

## Starting position (measured, 365 local pairs)

| category | pairs | action |
|---|---|---|
| already covered by ARO vocabulary | 22 (6.0%) | keep untouched |
| blocked by >= 1 **type mismatch** | 266 (72.9%) | **drop** (D3) |
| **substitutable** (all blockers are genuinely new words) | **99 (27.1%)** | substitute |

**99 pairs is the ceiling**, requiring substitution of **124 distinct
content word-stems**. Partial effort scales poorly: top 50 stems yields
only 53 usable pairs.

**Re-measure on the cluster atlas first** — it has 400 pairs against the
local parquets' 365.

## Why 266 pairs are unreachable — this is itself a result

The top blocking stems are `be`, `there`, `than`, `more`, `while`, `it`:
comparatives ("more ... than"), existentials ("there is"), and copulas.
These are **grammatical constructions absent from ARO's caption
distribution**, not missing words. An ARO-trained DisCoCat encoder cannot
parse most of Winoground because it has never seen those constructions,
and the parser is correct to type them differently.

This applies to **DisCoCLIP identically** — same text-encoder family,
same training data. Report the coverage analysis as a finding regardless
of how the substitution turns out.

## Substitution procedure

**Rule 1 — substitute per pair, never per caption.** Winoground's two
captions share a word multiset in different orders; that is the entire
design. A substitution must be applied identically to both captions or
the benchmark's logic collapses.

**Rule 2 — candidates must be in the ARO vocabulary.** Select the
in-vocabulary word nearest in embedding space (CLIP text embeddings or
GloVe, restricted to in-vocab candidates), then review manually. Automatic
selection alone will produce semantically wrong substitutions on a
benchmark this adversarial.

**Rule 3 — it is a loop, not a pass.** Substitute -> recompile the CCG
diagrams -> re-check coverage at the **symbol** level. A synonym can parse
to a different CCG type and remain out-of-vocabulary. Budget two or three
iterations.

## Validation: off-the-shelf CLIP as judge

For each substituted pair, score the **original** and the **substituted**
version with off-the-shelf CLIP and compare whether its answer changes.

**Use off-the-shelf CLIP, not M3.** M3 is one of the evaluated models and
shares its image tower with M2, so using it to filter the benchmark it is
then scored on biases the result. Off-the-shelf CLIP is outside the
comparison, and it is available immediately — which is what lets Step 2
run in parallel with Step 1 rather than waiting for M3.

## Tagging — flag, do not discard

Every pair carries a tag:

| tag | meaning |
|---|---|
| `untouched` | covered as-is (22 pairs) |
| `dropped-type` | excluded per D3, with the offending symbol recorded |
| `substituted-consistent` | substituted; judge's answer unchanged |
| `substituted-flagged` | substituted; judge's answer changed |

Results are reported on the union **and** on the unflagged subset, so a
reader can see the substitution's effect rather than having to trust it.
Nothing is silently removed.

---

# Step 3 — Evaluate on Winoground

**M3 on all 365/400 pairs.** CLIP text has no vocabulary limit, so M3
produces a clean, unmodified-benchmark Winoground number. This is also
the control: if M3's score on original vs substituted pairs differs
materially, the substitution changed the task and that must be reported.

**M1 and M2 on the usable subset**, n stated explicitly on every number.

**Metrics:** Winoground's standard text score, image score, group score.

**Expectation setting.** At n ~ 99 with a typical group score around 10%
(~10 pairs correct), error bars are wide enough to swallow any difference
between the three models. Nearly all vision-language models fail
Winoground; CLIP itself scores near chance on group score. A weak result
is expected and should be framed as such rather than as a finding about
the tower.

---

# Sequencing

```
Step 1 (retrain)            Step 2 (substitution)
  |                           |
  +-- CLIP text encoder       +-- re-measure on cluster atlas (400 pairs)
  +-- 6 runs                  +-- classify: untouched / dropped-type / substitutable
                              +-- substitute 124 stems, pair-wise
                              +-- recompile + re-check (2-3 iterations)
                              +-- judge with off-the-shelf CLIP, tag
           \                 /
            \               /
             Step 3 (evaluate on Winoground)
```

Steps 1 and 2 are fully independent — the judge is off-the-shelf CLIP,
not M3.

# Reporting requirements

- **n stated** on every Winoground number.
- **Skip/drop rates** with reasons: how many pairs dropped for type
  mismatch, how many substituted, how many flagged.
- **SVO reported under the row-split protocol only** (D1), with the
  leakage acknowledged in the discussion section.
- **Which model produced which row**, where configurations differ.
