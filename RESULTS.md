# Experiment results (compiled from cluster job logs, 2026-07-20)

Source: `/SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/<job_name>.o<job_id>`.
Every row lists its job id so it can be verified against the raw log.

Conventions used below:

- **Wino (clean)** = the Winoground by-tag report's `normal` row (pairs with no
  ambiguity/difficulty tags). There is no separately named "clean" benchmark in the code.
- **Chance baselines**: Winoground text/image 0.25, group 0.125 (but ~0.16 under the
  skip-handling here); ARO / SugarCREPE / SugarCREPE++ hard-neg accuracy 0.50;
  COCO retrieval R@1 ≈ 0.0002 (1/5000).
- All Winoground evals in the current generation skip 36 pairs; the NLC eval skipped 115.
- Repeated rows with identical configs are deliberate parallel/repeat submissions.

---

## A. COCO training → full benchmark suite

### A1. Current generation — `tree_no_type` parser, linear contractions (emb 512, bond 10)

All trained on `coco_*_nlc_tree_no_type_*.parquet` (452,783 train rows), July 18–20.
"ep" = training epochs reached (early stopping).

| Job id | Variant | ep | COCO i2t R@1 | COCO t2i R@1 | Wino t/i/g | Wino clean t/i/g | ARO all (attr/rel) | SC full | SC++ |
|---|---|---|---|---|---|---|---|---|---|
| 7074582 | SC non-frozen | 8 | 0.0004 | 0.0002 | .123/.083/.043 | .111/.078/.033 | .493 (.498/.486) | 0.346 | 0.343 |
| 7075225 | SC non-frozen | 9 | 0.0002 | 0.0004 | .231/.143/.083 | .255/.092/.072 | .499 (.501/.497) | 0.497 | 0.501 |
| 7075413 | SC non-frozen | 10 | 0.0002 | 0.0004 | .214/.125/.060 | .216/.098/.046 | .500 (.501/.498) | 0.480 | 0.501 |
| 7076696 | SC non-frozen | 8 | 0.0002 | 0.0002 | .262/.165/.091 | .268/.163/.092 | .498 (.503/.493) | 0.528 | 0.498 |
| 7077130 | SC non-frozen | 7 | 0.0004 | 0.0000 | .259/.117/.060 | .209/.072/.020 | .497 (.496/.499) | 0.507 | 0.499 |
| 7077206 | SC non-frozen | 8 | 0.0004 | 0.0002 | .202/.103/.031 | .222/.065/.033 | .499 (.501/.496) | 0.494 | 0.498 |
| 7077282 | SC non-frozen | 8 | 0.0004 | 0.0002 | .202/.103/.031 | .222/.065/.033 | .499 (.501/.496) | 0.494 | 0.498 |
| 7077131 | SC frozen | 14 | 0.0004 | 0.0004 | .145/.140/.051 | .144/.124/.046 | .500 (.503/.496) | 0.511 | 0.515 |
| 7077207 | SC frozen | 9 | 0.0002 | 0.0002 | .182/.057/.029 | .163/.026/.007 | .503 (.504/.501) | 0.503 | 0.488 |
| 7077283 | SC frozen | 15 | 0.0002 | 0.0000 | .208/.154/.046 | .190/.150/.039 | .501 (.506/.494) | 0.505 | 0.505 |
| 7076697 | SC frozen | 34 | 0.0000 | 0.0006 | .114/.057/.026 | .065/.039/.007 | .529 (.527/.531) | 0.540 | 0.502 |
| 7076698 | MC non-frozen | 16 | 0.0002 | 0.0002 | .143/.157/.029 | .157/.157/.033 | .492 (.492/.491) | 0.512 | 0.502 |
| 7077132 | MC non-frozen | 16 | 0.0002 | 0.0002 | .143/.157/.029 | .157/.157/.033 | .492 (.492/.491) | 0.512 | 0.502 |
| 7077209 | MC non-frozen | 16 | 0.0002 | 0.0002 | .168/.094/.034 | .137/.078/.020 | .500 (.501/.498) | 0.476 | 0.492 |
| 7077281 | MC non-frozen | 11 | 0.0004 | 0.0000 | .228/.074/.037 | .196/.052/.020 | .494 (.498/.490) | 0.505 | 0.510 |
| 7077133 | MC frozen | 19 | 0.0002 | 0.0000 | .168/.086/.037 | .170/.078/.026 | .496 (.498/.494) | 0.490 | 0.492 |
| 7077210 | MC frozen | 20 | 0.0002 | 0.0000 | .208/.154/.046 | .190/.150/.039 | .501 (.506/.494) | 0.505 | 0.505 |
| 7077280 | MC frozen | 20 | 0.0002 | 0.0000 | .208/.154/.046 | .190/.150/.039 | .501 (.506/.494) | 0.501 | 0.505 |

**Takeaway:** every current-generation linear run sits at chance on retrieval, ARO,
SugarCREPE and SugarCREPE++; Winoground group scores 0.03–0.09 are below the 0.16
skip-adjusted baseline.

Incomplete / failed runs in this family (no final report): SC frozen 7073756, 7074583,
7075226, 7075412 (crashed — traceback in log); SC non-frozen 7073757 (crashed at epoch 1);
7073492/7073510/7073530/7073531/7073685/7073686 (died at startup).
Still running at time of writing: 7076699 (MC frozen, epoch 34+), and the first
speed-optimized-build submissions 7078454 (MC linear) and 7078455 (frozen), both
started 2026-07-20 ~10:00.
(7076697 completed 2026-07-20 23:06 — row added above. Longest-trained SC frozen run:
checkpoint epoch 34, 23,953 symbols. Its ARO 0.529 and SC full 0.540 are the only
current-generation numbers marginally above the 0.50 chance line, though retrieval is
still at chance and its Winoground clean scores are the family's worst. ARO eval skipped
362 pairs; SugarCREPE full/++ skipped 852/1,565 pairs with unknown/NaN symbols.)

### A2. Previous generation — Bobcat parser datasets

**Multi-caption, non-linear contractions (NLC).** Checkpoint
`coco_multi_caption/2026-06-19_12-38-07`, evaluated by dedicated eval job **7028962**
(training job's own eval suite was incomplete at the time):

| Metric | Value |
|---|---|
| Winoground t/i/g | .210/.233/.142 (115 pairs skipped) |
| Wino clean t/i/g | .264/.300/.191 |
| ARO all (attr/rel) | .495 (.499/.491) |
| SugarCREPE full | **0.763** |
| SugarCREPE++ | **0.626** |

⚠️ The SugarCREPE numbers in 7028962 are far above every other run's; the SugarCREPE
eval pipeline changed after this date, so treat them as unverified until re-run on the
current eval suite. Earlier ARO-only evals of NLC-era checkpoints: 6980358 (.499),
6980363 (.495), 6988753 (.483), 7028512 (.495).

**Multi-caption, linear.** Trained by job **7034530** (25 epochs, crashed at final eval);
checkpoint `coco_multi_caption/2026-07-07_00-19-21`, supplemented by dedicated evals.
The three evals of the *same checkpoint* disagree because the eval suite evolved between
them — the latest (7042425) is the authoritative one:

| Eval job | COCO i2t/t2i R@1 | Wino t/i/g | Wino clean t/i/g | ARO all | SC full | SC++ |
|---|---|---|---|---|---|---|
| 7042425 (latest) | 0.0004 / 0.0002 | .235/.225/.107 | .194/.239/.082 | .497 | 0.587 | 0.545 |
| 7042301 | — | .235/.225/.107 | .194/.239/.082 | .497 | 0.587 | 0.545 |
| 7038618 | — | .166/.249/.104 | .179/.254/.105 | .498 | 0.506 | 0.510 |

**Single-caption, linear (old parser):** all training jobs failed or were killed before
eval (7028983, 7029212, 7034514, 7040346, 7041771, 7042312, 7042763) — no results.

**Single-caption, frozen (old parser):** training jobs crashed at eval (6980321, 6981679,
6982020 etc.); dedicated frozen evals give only retrieval + SugarCREPE swap_obj:
6982931 (i2t R@1 0.0027, swap_obj 0.578), 6988752 and eval_frozen 6993054
(i2t R@1 0.0021, swap_obj 0.481).

---

## B. ARO training → ARO evaluation

All ARO trainings use the NLC EinsumModel + InfoNCE+triplet loss (bond 10).
Datasets: default ARO, `_rtl` (right-to-left contraction paths), `_random` (random paths).

### Non-frozen (trainable image tower)

Held-out **test-split** hard-neg accuracy (the honest numbers):

| Job id | Dataset | Epochs (best) | ARO attr | ARO rel | ARO overall (test) |
|---|---|---|---|---|---|
| 7002030 | rtl | 85 (75) | 0.803 | 0.606 | 0.716 |
| 6978187 | default | 43 (33) | — | — | 0.711 |
| 7000951 | rtl | 50 (40) | — | — | 0.699 |
| 7002031 | random | 38 (28) | 0.786 | 0.591 | 0.698 |
| 7000952 | random | 62 (52) | — | — | — (eval crashed) |
| 7000687 | rtl | killed @13 | — | — | (val 0.627) |
| 7000695 | (rtl-era) | killed @12 | — | — | (val 0.626) |

⚠️ **Correction / provenance note:** job 7000951's log also prints a table showing
attr 0.929 / rel 0.742 / overall **0.844** — but its N=52,187 reveals it was computed on
the **full ARO benchmark, which includes the model's own training pairs** (train
contamination). The eval script was fixed before 7002030/7002031, whose tables use the
proper ~8K held-out test split (N=7,806 / 8,088). 7000951's honest held-out number is
0.699, from its test-metrics line. Per-task breakdowns for 7000951/6978187 don't exist on
the test split (only the contaminated/absent tables).

The SugarCREPE swap_obj numbers these jobs print are meaningless — e.g. 7000951's "0.750"
is 4 evaluated pairs with 159 skipped (unknown symbols); Winoground likewise (22 pairs,
312 skipped). Omitted for that reason.

### Frozen (CLIP image tower)

| Job id | Dataset | Epochs | ARO attr | ARO rel | ARO overall | Note |
|---|---|---|---|---|---|---|
| 7004254 | random | 39 | 0.726 | 0.597 | 0.668 | final eval crashed; numbers from dedicated eval **7006015** (last val in-training: 0.662) |
| 7009966 | default | 26 | — | — | (val 0.657) | no final eval; both re-eval attempts failed |

Failed dedicated frozen evals: 7006008 (CUDA ECC error — node fault), 7009929 / 7009930
(checkpoint path `aro_frozen/2026-06-12_22-32-36/best_*.pt` does not exist).

**Takeaway:** ARO training works — clearly above the 0.50 chance line, with held-out test
overall remarkably consistent at **0.70–0.72** across dataset variants (best verified:
0.716, rtl, job 7002030); frozen tops out around 0.67. This contrasts sharply with
COCO-trained models, which sit at ARO chance.
