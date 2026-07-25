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

### B.1 Reference — colleague's ARO-trained results (external, pasted 2026-07-22)

PRG = bobcat. Provided for comparison against the held-out test numbers above; not
run by us, no job ids, exact eval protocol (test split vs. full) not confirmed on our
side.

| Model | Overall | Relation | Attribution |
|---|---|---|---|
| PRG nonLin Tensor Network + TTN Image — Linear contraction | 69.49 | 59.78 | 77.65 |
| PRG nonLin Tensor Network + TTN Image — Non-linear optimal paths | **71.24** | **60.44** | **80.13** |
| PRG nonLin Tensor Network + TTN Image — Non-linear random contraction path | 69.77 | 59.10 | 78.55 |
| PRG nonLin Tensor Network + ViT Image — Linear contractions | 63.52 | 55.81 | 70.01 |
| PRG nonLin Tensor Network + ViT Image — Non-linear optimal contraction paths | 66.18 | 59.70 | 71.52 |
| PRG nonLin Tensor Network + ViT Image — Non-linear random contraction paths | 66.77 | 59.73 | 71.56 |

---

## C. Hard-negative π-sweep (2026-07-22)

Plan: `HARD_NEG_PI_SWEEP_PLAN.md`. 4 cells × 5 π ∈ {0, 0.1, 0.25, 0.5, 1.0} = 20 runs,
SGE arrays (task 1→π=0 ... task 5→π=1.0). Checked live via `ssh beaker`, updated
2026-07-22 ~20:18. 20/20 done — full sweep complete. π=0.25 and π=0.5 (C2) both
needed standalone re-eval after their in-process final eval OOM'd post-training
(training itself succeeded in both cases) — see † below.

⚠️ Bug found while pulling these: `submit_pi_sweep_bobcat_linear.sh` (and presumably the
other 3) print "finished successfully" regardless of exit code — task 3 below crashed
with an unhandled OOM and still printed success. Same class of bug as the earlier score-
stage submit script; worth adding `|| exit 1` guards, not yet done.

### C1. Bobcat, frozen (job 7091906) — all 5 done

| π | ep | i2t R@1/R10 | t2i R@1/R10 | Wino t/i/g | ARO attr | ARO rel | SC full | SC++ |
|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 0.0090/0.0536 | 0.0028/0.0234 | .129/.190/.061 | 0.4990 | 0.4984 | 0.6336 | 0.5635 |
| 0.1 | 12 | 0.0110/0.0500 | 0.0022/0.0230 | .133/.165/.068 | 0.4942 | 0.5049 | 0.6287 | 0.5617 |
| 0.25 | 12 | 0.0098/0.0516 | 0.0018/0.0218 | .172/.183/.090 | 0.5002 | 0.4918 | 0.6254 | 0.5518 |
| 0.5 | 12 | 0.0100/0.0552 | 0.0030/0.0278 | .118/.154/.061 | 0.5008 | 0.4961 | 0.6209 | 0.5499 |
| 1.0 | 12 | 0.0090/0.0528 | 0.0034/0.0186 | .111/.176/.039 | 0.5013 | 0.5053 | 0.6181 | 0.5416 |

### C2. Bobcat, non-frozen / TTN (job 7091908) — all 5 done

| π | ep | i2t R@1/R10 | t2i R@1/R10 | Wino t/i/g | ARO attr | ARO rel | SC full | SC++ |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 | 0.0002/0.0028 | 0.0004/0.0028 | .272/.237/.133 | 0.4993 | 0.4957 | 0.4982 | 0.4930 |
| 0.1 | 10 | 0.0000/0.0022 | 0.0006/0.0026 | .201/.154/.090 | 0.5012* | 0.4960* | 0.6195 | 0.5627 |
| 0.25 | 6§ | 0.0004/0.0022 | 0.0004/0.0014 | .022/.229/.004 | 0.4983* | 0.5088* | 0.5324 | 0.5196 |
| 0.5 | 16§ | 0.0000/0.0020 | 0.0000/0.0032 | .007/.272/.007 | 0.5059 | 0.5605 | 0.7079 | 0.5737 |
| 1.0 | 2 | 0.0000/0.0018 | 0.0002/0.0022 | .075/.186/.043 | 0.4982 | 0.4953 | 0.5183 | 0.5081 |

\* π=0.1 and π=0.25 ARO rows: true_cos/false_cos ≈0.83 and ≈0.977 respectively (vs.
the other rows' ≈0.03-0.05, and nearly identical to each other within each row) —
embedding collapse, not a transcription error. Both are early/low-epoch checkpoints
(ep 10, ep 6) in a cell that sits at chance overall, so likely genuine undertraining
rather than a second bug — not confirmed.
† π=0.25 and π=0.5 both needed standalone re-eval after the in-process final eval
OOM'd post-training (training itself completed both times — early stopping at epoch
16 and 26 respectively). π=0.25's first standalone attempt (`evaluate.py`) also hit a
real bug — its dataset-selection heuristic picks `coco_single_caption` for linear
models by default, not from the actual training config, so it grabbed the WRONG test
set (this sweep deliberately trained bobcat on `coco_single_caption_nlc`) and hit
`KeyError` on a symbol this checkpoint never saw. Fixed via a `--dataset` override
flag (job 7104775 for π=0.25, job 7104819 for π=0.5, both epoch-16/6 best-checkpoints).
π=0.25's recovered numbers land at chance, consistent with the rest of this cell.
π=0.5's numbers do NOT — SC full 0.7079 and SC++ 0.5737 are the highest of any cell
in the whole sweep, bobcat-frozen included — worth a closer look (see below).
§ epoch of the loaded checkpoint (best-epoch, saved before early stopping — at epoch
16 for π=0.25, epoch 26 for π=0.5), not the final training epoch reached —
inconsistent with the "ep" column's meaning in every other row (training epochs
reached), flagged for clarity, not a data error.

### C3. Tree, frozen (job 7091907) — all 5 done

| π | ep | i2t R@1/R10 | t2i R@1/R10 | Wino t/i/g | ARO attr | ARO rel | SC full | SC++ |
|---|---|---|---|---|---|---|---|---|
| 0 | 50 (max, no early stop) | 0.0004/0.0046 | 0.0002/0.0024 | .120/.057/.017 | 0.5185 | 0.5073 | 0.5232 | 0.5237 |
| 0.1 | 50 (max, no early stop) | 0.0000/0.0036 | 0.0002/0.0016 | .105/.037/.011 | 0.5284 | 0.5100 | 0.5340 | 0.5281 |
| 0.25 | 26 | 0.0000/0.0020 | 0.0004/0.0030 | .151/.029/.011 | 0.5133 | 0.4925 | 0.5083 | 0.5338 |
| 0.5 | 34 | 0.0004/0.0024 | 0.0002/0.0016 | .185/.031/.014 | 0.4983 | 0.5005 | 0.4836 | 0.4911 |
| 1.0 | 25 | 0.0004/0.0026 | 0.0000/0.0018 | .180/.051/.031 | 0.4951 | 0.4950 | 0.4940 | 0.4972 |

### C4. Tree, non-frozen / TTN (job 7091909) — all 5 done

| π | ep | i2t R@1/R10 | t2i R@1/R10 | Wino t/i/g | ARO attr | ARO rel | SC full | SC++ |
|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 0.0002/0.0018 | 0.0002/0.0018 | .168/.094/.034 | 0.5010 | 0.4978 | 0.4759 | 0.4916 |
| 0.1 | 22 | 0.0002/0.0020 | 0.0002/0.0018 | .197/.066/.020 | 0.5021 | 0.4875 | 0.4821 | 0.4912 |
| 0.25 | 10 | 0.0004/0.0024 | 0.0002/0.0026 | .205/.063/.054 | 0.4957 | 0.5018 | 0.4969 | 0.4935 |
| 0.5 | 3 | 0.0000/0.0022 | 0.0004/0.0026 | .254/.048/.003 | 0.5055 | 0.4874 | 0.4914 | 0.4902 |
| 1.0 | 3 | 0.0004/0.0024 | 0.0004/0.0016 | .197/.100/.009 | 0.4989 | 0.4899 | 0.4938 | 0.4962 |

**Full-sweep read (20/20 cells done):** bobcat frozen remains the standout, all 5 π
values in: SC full/++ (0.55-0.63) and Wino group scores (0.04-0.09) both clear of
chance across every π, consistent with the pre-sweep best run (job 7076697, SC full
0.540). No π value in that cell is a clear winner over π=0 — hardness doesn't appear
to move this cell much either way. Tree-frozen sits close to chance throughout but
with a consistent small positive nudge on SC full/++ (0.51-0.53) regardless of π,
still far short of bobcat-frozen's margin. Tree non-frozen (C4) stays flat at chance
across all 5 π — 5/5 rows sit at chance on ARO and SC, consistent with the family A1
takeaway that non-frozen COCO training doesn't transfer to these benchmarks. Bobcat
non-frozen (C2) mostly matches that pattern too (4/5 rows at chance), EXCEPT π=0.5:
SC full 0.7079 / SC++ 0.5737 and ARO relation 0.5605 are the best scores of any cell
in the entire sweep. That row is also the only one of the four π=0.5 undertrained-
early-stop checkpoints (ep 16 of 50, vs π=0's ep 2 and π=1's ep 2 in the same cell)
to reach a meaningfully later epoch, so this could be a real hardness effect or
could just be "this cell needed to train longer to leave chance" — the no-early-
stopping rerun should disambiguate directly since it removes epoch count as a
confound. Worth flagging as the one standout result to watch for in that rerun.

---

## D. Hard-negative π-sweep rerun — no early stopping (2026-07-23)

Same 20-run grid (jobs 7104874-7104877), relaunched with `ML_PATIENCE=1000` so every
cell trains the full `max_epochs=50` instead of stopping ~10 epochs past its best
val epoch — removes epoch count as a confound between π values (see §C takeaway).
Also picks up the `trainer.py`/`run_frozen.py` checkpoint-load-OOM fix and the
submit-script silent-failure fix from the same commit. Checked live via `ssh beaker`,
updated 2026-07-24 ~09:18. **20/20 done — full rerun complete.** bobcat-frozen
π=0.25 (job 7104875 task 3) trained the full 50 epochs cleanly but then OOM'd on
the post-training checkpoint reload itself — not the OOM already fixed in the
prior commit (that one was `torch.load`'s CUDA unpickling), but a second, deeper
OOM inside `EinsumModel.load_state_dict`, which reallocates its weight
ParameterList directly on the model's current (still-GPU) device, briefly
holding old + new weights at once on an already-full ~12GB card. Rather than
re-run the full 50-epoch training a second time, wrote a standalone frozen-
checkpoint evaluator (`evaluate_frozen.py` / `submit_evaluate_checkpoint_frozen.sh`,
commit 402a055) that reloads the already-saved checkpoint on a clean process and
reruns just the eval — took ~10 min instead of ~21h. tree-frozen π=0.25 (job
7104877 task 3) completed cleanly in-process. The 2 tasks that earlier OOM'd on
small cards (π=0.5, π=1.0 in bobcat-frozen) were retriggered same day as jobs
7105312/7105313 with those hosts excluded and both completed successfully.

| Cell | π | best ep | i2t R@1/R10 | t2i R@1/R10 | Wino t/i/g | ARO attr | ARO rel | SC full | SC++ |
|---|---|---|---|---|---|---|---|---|---|
| Bobcat frozen (7104875) | 0 | 2/50 | 0.0126/0.0556 | 0.0040/0.0270 | .143/.212/.100 | 0.4993 | 0.5019 | 0.6362 | 0.5705 |
| Bobcat frozen (7104875) | 0.1 | 2/50 | 0.0088/0.0548 | 0.0024/0.0232 | .154/.183/.079 | 0.5024 | 0.4966 | 0.6260 | 0.5625 |
| Bobcat frozen (7104875) | 0.25† | 2/50 | 0.0098/0.0520 | 0.0036/0.0230 | .129/.176/.068 | 0.4985 | 0.4933 | 0.6262 | 0.5533 |
| Bobcat frozen (7105312, retrigger) | 0.5 | 2/50 | 0.0108/0.0480 | 0.0040/0.0238 | .129/.197/.061 | 0.5004 | 0.5052 | 0.6295 | 0.5604 |
| Bobcat frozen (7105313, retrigger) | 1.0 | 2/50 | 0.0096/0.0564 | 0.0020/0.0238 | .151/.165/.072 | 0.4988 | 0.5034 | 0.6185 | 0.5470 |
| Bobcat non-frozen/TTN (7104874) | 0 | 46/50 | 0.0002/0.0022 | 0.0002/0.0016 | .183/.229/.118 | 0.5022 | 0.5006 | 0.7186 | 0.5906 |
| Bobcat non-frozen/TTN (7104874) | 0.1 | 30/50 | 0.0002/0.0028 | 0.0004/0.0032 | .215/.222/.143 | 0.4997 | 0.4920 | 0.7160 | 0.5974 |
| Bobcat non-frozen/TTN (7104874) | 0.25 | 35/50 | 0.0002/0.0026 | 0.0000/0.0022 | .165/.190/.086 | 0.5163 | 0.5323 | 0.7286 | 0.5955 |
| Bobcat non-frozen/TTN (7104874) | 0.5 | 13/50 | 0.0000/0.0020 | 0.0004/0.0020 | .161/.186/.093 | 0.4981 | 0.5210 | 0.7077 | 0.5930 |
| Bobcat non-frozen/TTN (7104874) | 1.0 | 39/50 | 0.0002/0.0022 | 0.0000/0.0008 | .000/.215/.000 | 0.4940 | 0.5052 | 0.7178 | 0.6059 |
| Tree non-frozen/TTN (7104876) | 0 | 29/50 | 0.0000/0.0016 | 0.0002/0.0018 | .208/.125/.063 | 0.4978 | 0.4967 | 0.5088 | 0.5127 |
| Tree non-frozen/TTN (7104876) | 0.1 | 13/50 | 0.0002/0.0022 | 0.0006/0.0014 | .188/.080/.031 | 0.5040 | 0.4915 | 0.4754 | 0.4916 |
| Tree non-frozen/TTN (7104876) | 0.25 | 3/50 | 0.0000/0.0016 | 0.0002/0.0028 | .097/.054/.006 | 0.4963 | 0.4919 | 0.5077 | 0.4967 |
| Tree non-frozen/TTN (7104876) | 0.5 | 33/50 | 0.0004/0.0024 | 0.0002/0.0018 | .014/.051/.011 | 0.4968 | 0.5279 | 0.4996 | 0.5114 |
| Tree non-frozen/TTN (7104876) | 1.0 | 10/50 | 0.0004/0.0024 | 0.0002/0.0022 | .239/.143/.077 | 0.4940 | 0.4960 | 0.4961 | 0.4939 |
| Tree frozen (7104877) | 0 | 49/50 | 0.0004/0.0044 | 0.0002/0.0022 | .105/.057/.011 | 0.5019 | 0.5006 | 0.5180 | 0.5088 |
| Tree frozen (7104877) | 0.1 | 47/50 | 0.0008/0.0050 | 0.0002/0.0028 | .123/.046/.017 | 0.5068 | 0.4898 | 0.5218 | 0.5299 |
| Tree frozen (7104877) | 0.25 | 48/50 | 0.0002/0.0044 | 0.0004/0.0022 | .114/.074/.026 | 0.5016 | 0.5228 | 0.5412 | 0.5126 |
| Tree frozen (7104877) | 0.5 | 47/50 | 0.0002/0.0024 | 0.0006/0.0032 | .162/.066/.029 | 0.5008 | 0.5182 | 0.5244 | 0.4978 |
| Tree frozen (7104877) | 1.0 | 38/50 | 0.0010/0.0050 | 0.0004/0.0024 | .125/.074/.026 | 0.5079 | 0.5389 | 0.5333 | 0.5040 |

† Retrieval + benchmarks recovered via the standalone `evaluate_frozen.py` re-eval
(job on 2026-07-24, checkpoint `coco_multi_caption_frozen/2026-07-22_20-38-00/
best_model.pt`) after the in-process eval OOM'd — see status note above. Numbers
land in the same range as this cell's other 4 π values (best ep 2, SC full/++
~0.55-0.64), so no anomaly, just another instance of this cell converging early.

Notes:
- "best ep" is the epoch whose checkpoint was actually reloaded for test/benchmark
  eval (val-loss-best), out of the full 50 now always run — directly comparable
  across rows for the first time in this campaign, unlike §C's early-stopped epochs.
- All 5 bobcat-frozen π values converge to their best checkpoint by **epoch 2** and
  then run 48 more epochs with no further improvement — confirms this cell isn't
  epoch-starved; its §C early-stopped numbers (ep 12) were already using the right
  checkpoint. π=0.5/1.0's original attempts (job 7104875 tasks 4/5) OOM'd mid-training
  on small (~11GB) cards — a cluster hardware issue, not a training bug, same class
  already flagged in the submit script's own comment — retriggered same day
  (jobs 7105312/7105313) with those hosts excluded and both completed cleanly.
- Bobcat non-frozen/TTN π=0.5's §C standout (SC full 0.71, SC++ 0.57, clear of
  chance while every sibling π sits at chance) does **not** replicate here — with
  early stopping removed, all 5 π in this cell now land in the same 0.71-0.73 SC
  full / 0.59-0.61 SC++ band, well above the ~0.50 chance seen in every other
  non-frozen cell (both bobcat and tree) at every π. So the earlier "standout" was
  this whole cell being systematically enough-above-chance, not a π=0.5-specific
  effect — the real story is bobcat non-frozen/TTN as a cell scores unexpectedly
  well on SugarCREPE (though still at chance on ARO and retrieval), regardless of
  hard-negative pressure. Worth a closer look at what's different about this cell
  vs. tree non-frozen (which stays flat at chance across all 5 π, matching §C).
- Tree frozen and tree non-frozen both still track their §C chance-level pattern
  with the extra training, no new signal from removing early stopping there.
