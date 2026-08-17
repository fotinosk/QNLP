# Thesis correction list — `main.tex`

Review date: 2026-08-15. Line numbers refer to `main.tex` at commit `a366773`.

Verified against `llm/research_log.md`, `qnlp/image_tower/classification/quantum/make_figures.py`,
`qnlp/image_tower/classification/quantum/results/*.json`, `qnlp/discoviz/models/image_model.py`,
`qnlp/discoviz/models/cp_node.py`, `qnlp/scripts/aro_contrastive/{config,run}.py`,
`qnlp/core/training/losses/contrastive.py`.

Type key: **[F]** factual error · **[C]** coherence/contradiction · **[O]** overclaim or missing
qualification · **[M]** mechanical.

Status key: `TODO` · `DONE` · `WONTFIX` (with reason).

---

## Abstract (L59–69)

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 1 | 67 | **F** | DONE | "exceeds its parameter-matched neural counterpart on the majority of heads" — false under every reading. Loses all four heads to `mlp_reference` (63.0/52.7/61.0/94.1 vs 93.1/74.1/84.0/98.6); loses 3 of 4 to `classical_full`. `research_log.md:52`: *"`mlp_reference` beats every TTN arm on every head."* | Delete; state the true `classical_bare` comparison and concede the MLP lead. |
| 2 | 67 | **F** | DONE | "exceeds its direct classical analogue … on shape **and material**" — material is 61.0 vs 56.5 = +4.5 against an MDE of ≈7.6. §5.2 itself calls material parity. The `+8.8` material win was the superseded 30-epoch result. | Shape only, parity on the other three. |
| 3 | 63 | **O** | DONE | Claims the topology family was discriminated by four properties: entanglement generation vs gate density, gradient variance, depolarizing-noise tolerance, classical simulability. The body delivers **gradient variance only**. The topology noise figure was retired (`research_log.md:676`) because the QTTN arm scored 42.2%, below the single-attribute ceiling, under a broken scalar readout. | Cut to what Ch. 4 actually shows. Do not reintroduce the retired noise comparison. |
| 4 | 63 / Ch. 3 | **C** | WONTFIX | MERA is surveyed in the abstract but is not a candidate architecture in §3.2. It appears only in Fig. 4.1. | Author's call: the drop is considered adequately explained in Ch. 2. |
| 5 | 67 | **O** | DONE | The binding sentence reports only the classical arms. The quantum binding result is the chapter's headline and §7.1.3's central claim. | Add the quantum arm and the shuffle control. |
| 6 | 59 | **M** | TODO | Abstract exceeds the 200-word cap noted in the source comment. Was ~250 before the #1–#5 fixes, **333 after** — those fixes each replaced a short false claim with a longer true one. Needs a compression pass, not a revert. | — |

---

## Chapter 3 — Theoretical Framework

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 7 | 219, 285 | **M** | DONE | "Section 3.4" does not exist. The Density Matrix Wall is §3.2.5. | Replaced all hardcoded section numbers with `\label`/`\ref` (`sec:density_wall`, `sec:readout_protocols`, `sec:ablations`, `sec:search_space`, `sec:hybrid_eval`, `sec:ansatz_design`, `ch:synthetic`, `ch:aro`), so they cannot drift again. |
| 8 | 157–179 | **C** | DONE | Three encodings defined (amplitude, scalar $R_y$, multi-axis); §4.3.1 sweeps **four**, including `zz_map`, which is never defined. | Added §3.1.1.4, the second-order Pauli-Z feature map, with its circuit and the product-vs-entangled-encoding contrast. |
| 9 | 167, 313, 336 | **M** | DONE | Same object named three ways: "Scalar Rotation Encoding (Scalar $R_y$)", `angle`, `scalar_ry`. | Canonicalised on `scalar_ry`; the `angle` code alias is noted once in §3.1.1.2. §4.3.1 updated. |
| 10 | 182–193, 345 | **C** | DONE | §3.1.2 defines HEA / ALT / IQP; §4.3.2 compares `strongly_entangling` vs `iqp`. HEA→`strongly_entangling` is never stated, and **ALT is never resolved anywhere**. | §3.1.2.1 now distinguishes the star-entangler `hea` from the ring-entangler `strongly_entangling` and states they are not interchangeable; §4.3.2 repeats it at the point of use. ALT resolved by the re-run (see #45). |
| 11 | 155 | **M** | DONE | "As a results" | "As a result" |

---

## Chapter 4 — Topological Optimization

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 12 | 409–434 | **O** | DONE | §4.4 lists `classical_full` (352p) and `mlp_reference` (999p) but reports neither result. `classical_full` **ties** the 287p quantum arm (`research_log.md:800`); the 999p MLP beats it. Reporting only the `classical_bare` win reads as selection. | Added Table 4.1 with all six arms and their accuracies. All four resolved comparisons now stated (`classical_bare` +9.7, `quantum_hybrid` +9.9, `mlp_param_matched` +18.8, `classical_full` tie), the `classical_full` tie framed as the stronger claim, and both qualifications added (the 999p MLP leads; the classical arms were tuned while the quantum arm was not). |
| 13 | 308 | **M** | DONE | Unresolved TODO in the source asking why HEA was abandoned. | Answered by the re-run (HEA and IQP are tied, not HEA-over-IQP) and deleted. See #45. |
| 14 | 442 | **M** | DONE | "Code for the 3 models can be found in the appendix" — Appendix B contains `CoherentQTTNClassifier`, `ClassicalTTNClassifier` (bare & full), and `CPQuadRankLayer`. The last is the **DisCoViz** node, not a Ch. 4 arm. | Added a preamble to Appendix B mapping each listing to its arm and stating that `CPQuadRankLayer` belongs to Ch. \ref{ch:aro}, included for comparison of the two CP implementations. |
| 15 | 247–256, 445–451 | **M** | DONE | Commented-out duplicate of §4.1; two consecutive "CHAPTER 5: CLEVR" banner comments. | Both deleted. |

---

## Chapter 5 — CLEVR

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 16 | 532–548 | **C** | DONE | Table 5.1 quotes the 90-epoch budget only; the figure on the same page plots **both** (solid/hatched). `make_figures.py:495`: *"NEITHER BUDGET IS PRIVILEGED — quote both or neither."* A reader sees numbers in the figure absent from the table. | Resolved the other way, per author decision: the FIGURE is now 90-epoch only, matching the table. Justified in text and in `fig_clevr_attributes` — 90 is the converged budget, and quoting it is conservative (at 30 the quantum arm wins two heads, not one). |
| 17 | 513–554 | **O** | DONE | Never states that no single epoch budget is fair to all four heads — they share one summed loss and converge at very different rates. `research_log.md` Finding 2 flags this as *"must be stated in the write-up."* | Existing head-trading paragraph tightened: degradation is stated as resolved rather than "slight", and the protocol limitation named. |
| 18 | 488, 502 | **M** | DONE | `as shown in \ref{fig:...}` — missing the word "Figure". | Both. |
| 19 | 556–562 | **O** | DONE | The relational quantum arms ran at **30 epochs** while §5.2 establishes the tower needs ~90 (`research_log.md:54, 1183`: *"a floor, not a ceiling"*). Unstated. | One clause added, framed as a floor not a ceiling. |
| 20 | 571–601 | **F** | DONE | The binding probe ran at **32×32**; every other CLEVR experiment ran at 16×16. Unstated, and it explains the parameter jump (462 → 725) that otherwise looks like an error — fewer classes but more parameters. | Stated in §5.1.3, with the note that the tree still uses 16 patches/qubits and only the input width changes. |
| 21 | 573 | **M** | DONE | Footnote cites Table 5.1 for "84.8%–99.1%", but 84.8 is `mlp_param_matched`, which has no row there (converged early; no 90-epoch run needed). | Footnote now quotes the 90-epoch range 91.8–98.6 against the 50.6 floor, matching Table 5.1, and states why a conjunction task needs an attribute all arms perceive. |
| 22 | 575 | **M** | DONE | `they do not bind *better* than` — Markdown asterisks render literally in LaTeX. | `\emph{better}` |
| 23 | 575 | **M** | DONE | "we introduce conduct a binding probe exepriment" | "we conduct a binding probe experiment" |
| 24 | 575 | **F** | DONE | "Every architecture was evaluated against a patch-shuffled control" — the quantum arms were not (Table 5.2 shows `---`). | "Every classical architecture…" |
| 25 | 600 | **F** | DONE | "$-0.6$ gap against an MDE of 5.5" is not derivable from Table 5.2, which reports best seeds (73.8 − 69.0 = **+4.8**). The −0.6 is the 15-seed mean difference. | Reworded to "on their full distributions rather than their best runs", and labelled a bounded null. |
| 26 | 600 | **M** | DONE | "This supports out previous tests" | "our" |
| 27 | 606–620 | **F** | DONE | "On the same images, a 683-parameter network tells the two arrangements apart 94.6%." Different builds: the CLIP probe uses 224px canvases (cell 96) from `clevr_objects_64_val.npz`; the binding probe used `clevr_binding_size_32_*.npz`. | Changed to "On composites built the same way", with a footnote giving the 224x224 vs 32x32 resolution difference and why a frozen encoder cannot be re-fit. |
| 28 | 611–616 | **M** | DONE | The quoted 0.0050 / 0.0654 / 7.6% come from `c7_clip_geometry_floorbg_results.json`, not the main run (0.0050 / 0.0690 / 7.2%). Both defensible; the reader can't tell which. | Same footnote as #27 names the jitter-referenced variant and states why it is the conservative choice. |

---

## Chapter 6 — ARO

Verified against the actual ARO training code. The node type checks out — `TTNImageModel` is built
from `CPQuadRankLayer`, so the CP continuity claim with Chs. 4–5 holds. Five things around it do not.

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 29 | 695 | **C** | DONE | *"replacing … CLIP ViT with the hierarchical CP-TTN yields vastly superior geometric and spatial binding"* contradicts §5.4: *"within the isolated vision tower, there is no compositional advantage for tensor networks over a plain MLP."* `thesis_outline.md:379` anticipates this: *"the advantage is not attributable to the tensor network binding better, because in a controlled vision-only test it doesn't."* | Rewritten. The gain is now attributed to recovering arrangement information that frozen CLIP discards (measured in §5.5), with §5.4's null stated explicitly: tensor networks do not bind better, they are a sufficient tower that preserves arrangement. |
| 30 | 633, 646 | **F** | DONE | "8×8 (64-patch) image grid", "64-qubit input layer". Actual config: `image_size=64`, `patch_size=4` → **16×16 = 256 patches**, `depth = log₄(256) = 4` (confirmed by `gains = [2.0, 1.5, 1.0, 1.0]`). | Corrected to a 16x16 grid of 256 patches, depth four, 256-qubit input layer. Framed as physically impossible rather than merely expensive. |
| 31 | 646 | **F** | DONE | "contracts … to a single root tensor … into a final 512-dimensional image embedding." `in_dim` runs 64→128→256→512→**1024**, then `Linear(1024, 512)` + L2. The 512 is the output embedding, not the root. | Corrected: four levels, bond dim 64->128->256->512->1024, 1024-dim root, linear projection to the 512-dim shared embedding. |
| 32 | 643–646 | **C/O** | DONE | **The ARO tower carries an explicit learned positional embedding** — `image_model.py:44-45`, a `[1, 256, 64]` parameter (16,384 values) gated by learned `pos_scale`, added at L88. Contradicts §4.3.4 (ancilla rejected), §5.3 (*"absolutely no explicit positional parameters"* as the critical finding) and §7.1.3 (*"captured by the architecture rather than supplied to it"*). | New §6.2.3 Positional Encoding discloses the gated learned embedding, and reconciles it with Chs 4-5: that was a bounded null on a depth-two 16-patch tree; at depth four over 256 patches the implicit signal is weaker. States the gating makes it safe and that it was not measured at this depth. |
| 33 | 635, 639 | **C** | DONE | The ARO tower uses `dropout=0.3` and residuals on layers 2–3 (`use_res = True if i > 1 else False`), making it the **`classical_full`** analogue, not `classical_bare` — which Ch. 4 defines as the CP arm stripped of exactly these, and §4.3.4 rejected them for the quantum node. | §6.1 now states the tower is the classical_full analogue (dropout + residuals on the upper two levels), and that this departs from the quantum node which §4.3.4 showed is harmed by both. |
| 34 | 662 | **F** | DONE | "where $\tau$ is a **learned** temperature parameter." `ExperimentConfig.temperature = 0.07`, passed as a plain float into `InfoNCE` (`contrastive.py:22-27`). Not learnable. `thesis_outline.md:380` lists learnable temperature as a **failed** experiment (collapsed to ~0.015). | Corrected to a fixed tau = 0.07, with one clause on why the learnable variant was abandoned. |
| 35 | 637–651 | **O** | DONE | The **bilinear patch embedding** (separate colour and pixel factors multiplied elementwise, `image_model.py:37-40, 83-85`) is the entire input stage and appears nowhere in §6.2. | New §6.2.1 Bilinear Patch Embedding describing the colour/pixel factorisation and its elementwise combination. |
| 36 | 667 | **O** | WONTFIX | $\lambda$ and $\alpha$ described but never valued. Config: `triplet_weight = 40000.0`, `triplet_margin = 0.2`. The literature review calls the $\lambda$ rebalancing the breakthrough. | Author's call: hyperparameter values left out of the prose. |
| 37 | 669 | **O** | DONE | "the *only* variable changing is the vision encoder." DisCoClip's CLIP tower is **frozen**; the CP-TTN is **trained**. Frozen-vs-trained is confounded with hierarchical-vs-attention. | §6.3 now names both differences (topology AND frozen-vs-trained), states the delta is attributable to the encoder as a whole, and names the missing control. |
| 38 | 697 | **O** | DONE | CLIP / OpenCLIP / BLIP rows are **zero-shot**; DisCoViz trains on the ARO train split with ARO's own hard negatives in the loss. *(No leakage — `run.py:161-166` evaluates on the held-out `test_parquet` and the code comment names and avoids the pooled-split hazard. The DisCoClip comparison is unaffected.)* | Stated in both the table caption and the results text; the zero-shot rows are separated from the trained rows explicitly. |
| 39 | 697 | **O** | DONE | "severely dominates", "completely outperforms", "utterly fails" — for 59.78 vs 52.90 against a 50% chance floor. | "severely dominates", "completely outperforms", "utterly fails", "vastly superior" all removed. |
| 40 | 691, 695 | **M** | DONE | "$+7.64\%$ and $+3.97\%$ performance gain" — these are percentage **points**. | Now "points" throughout. |
| 41 | §6.4 | **O** | TODO | The negative-results table planned in `thesis_outline.md:380` (alignment warmup → embedding collapse; learnable temperature → collapse; hard-negative mining → widened train/val gap; NLC+MLP head → conflicting gradients; frozen-CLIP text-only → cosine ≈ −0.0016) is absent. | Add it. Cheap credibility, and it pre-empts #34. |

---

## Chapter 7 — Conclusion

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 42 | 731–734 | **O** | DONE | "outperforms its unconstrained classical counterpart while using roughly a third fewer parameters, and this parameter efficiency **reproduces on real data**." On synthetic it is 33% fewer and +9.7 resolved; on CLEVR it is **18%** fewer and one resolved head of four. | Split into synthetic and CLEVR claims: a resolved win at a third fewer parameters on synthetic (plus the classical_full tie), against an 18%% deficit and one resolved head of four on CLEVR. What transfers is now stated as "not worse at matched or lower capacity" rather than "consistently better". |
| 43 | 747–751 | **C** | DONE | "It does so in the variant carrying no positional parameters whatsoever … adding an explicit positional mechanism changes nothing measurable." True of the quantum tower, false of the ARO tower (#32), two subsections apart. | Added a third boundary: implicit position was measured on a depth-two 16-patch tree, and the scaled tower of Ch. 6 contracts four levels over 256 patches and carries an explicit gated embedding. Framed as a deliberate departure at a different scale, with the depth at which implicit position stops sufficing named as unmeasured. |

---

## Appendix

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 44 | 976 | **M** | DONE | "used in Chapter 3 to investigate trainability" → the synthetic chapter is **Chapter 4**. | Now `\ref{ch:synthetic}`. Every hardcoded chapter number in the document converted to a label. |

---

## Found during correction work (2026-08-15)

| # | Line | Type | Status | Issue | Fix |
|---|---|---|---|---|---|
| 45 | 341 (old) | **F** | DONE | §4.3.1 cited the single-node survey's `multi_axis + IQP` at **95.3%**, and §4.3.2 explained the later 80.9% as that figure "corrected" for task difficulty. **The 95.3% is not reproducible from the code in the tree.** `benchmark_encodings_ansatze.py` and `synthetic_shapes.py` are both byte-identical to the commit that produced the figure (`29591e8`), with no uncommitted changes; re-running `harvest_sweeps()`'s exact protocol gives ~62–64%. It is also internally implausible: 95.3% on one 4-qubit node at 8×8 exceeds the 89.3% the full 16-qubit coherent tree reaches at 16×16. | Survey re-run at 30 seeds (`rerun_encoding_ansatz_survey.py`). §4.3.1 rewritten on the new data; the "inflated metric" explanation in §4.3.2 removed. Old figure retired to `results/superseded/`. |
| 46 | 337 (old) | **F** | DONE | Figure 4.2 was `encoding_ansatz_sweep.png`, whose own `results/superseded/README.md` says it "may not be cited for ranking configurations, or the ansatz decision" — while the caption claimed it demonstrated "the general superiority of the `multi_axis` encoding". | Replaced by `encoding_ansatz_survey.png`, generated from `encoding_ansatz_survey_rerun_results.json` by `make_figures.fig_encoding_ansatz_survey()`. |
| 47 | §4.3.1 | **O** | DONE | The survey's `amplitude` encoding was reported in the log at 78.1% as "maximum qubit compression … retaining model capacity". At 30 seeds all three amplitude arms sit at **51.5–52.3%**, i.e. exactly on the 50% single-attribute ceiling — it is not learning the colour–shape conjunction at all. | §4.3.1 now states this as a substantive elimination, and introduces the single-attribute ceiling as the reference line rather than the 25% chance floor. |

| 48 | 265, 1024 | **F** | DONE | Both §4.1 and Appendix A describe the dataset as "four geometric classes (circles, squares, triangles, and crosses)". That is true of the *generator*, but in `overlapping` mode --- the default, and the mode every reported experiment uses --- the four classes are a 2x2 design over {red, green} x {circle, square}. Triangles and crosses are drawn only in the two diagnostic modes. The mode actually used was never stated. | Both passages rewritten: the mode of record is named, the 2x2 design is given, and the 50%% single-attribute ceiling is derived from it in both places. This also supplies the reference line the Ch. 4 figures and the re-run survey depend on. |
| 49 | Appendix B | **M** | DONE | Two code listings have drifted from source: `_body` omits the `pos` argument and `forward` omits `p_noise`. Both are abridged versions predating later features. With §6.2.3 now discussing positional encoding, a reader comparing them finds it absent. | Added a note stating the listings are abridged to the structurally relevant forward path, and naming what is omitted (alternative readouts/encodings, density-matrix paths, the positional argument used by `quantum_on_wire` and the scaled tower). |

**Net effect on the argument: none, but #48 supplies a missing premise.** Several accuracy figures in Chs. 4-5 are only interpretable against the 50%% ceiling rather than the 25%% chance floor, and that ceiling follows from the 2x2 design the text had not stated.

**Net effect on the argument: none downstream.** The encoding conclusion (`multi_axis` leads)
survives and is independently resolved by R2 on the full tree, which is what every later
architecture decision rests on. What changed is that the ansatz was never resolvable on
accuracy at either scale — so IQP is now justified on parameter count (8 vs 12 per node),
gate depth and training stability, which is what the code comments recorded all along.

| 50 | 859, 893 | **C** | DONE | Ch. 7 was written against the pre-correction Ch. 6 and says "a single small image size at shallow depth is the only configuration trained end to end". Once Ch. 6 correctly describes a depth-four, 256-patch tower that *is* trained end to end, the claim is false as stated — it is true only of the **coherent quantum** configuration. | Both passages now scope the limit to the coherent tower and name the classical tower's four levels explicitly, turning a stale claim into the asymmetry that motivates the future work. |
| 51 | 823 | **O** | DONE | §7.1.4's "surpasses a far larger foundation model on relational reasoning" repeats the zero-shot-vs-trained issue corrected in Ch. 6 (#38). | Qualified in one clause: a small task-adapted model against a zero-shot general-purpose one, indicating what such a model can reach rather than ranking architectures. |

---

## Investigated and withdrawn

These were raised during review and do **not** require changes. Recorded so they are not re-opened.

- **Table 5.2's identical `49.3 ± 2.7` for `classical_full` and `mlp_param_matched`** — genuine, matches
  the source table at `research_log.md:1082-1087`. Not a transcription error.
- **Missing 90-epoch `mlp_param_matched` row in Table 5.1** — the MLPs are converged and gain nothing
  from 90 epochs (`research_log.md:1154-1161`). Only the footnote citation (#21) needs touching.
- **Suspected ARO evaluation leakage** — does not occur. `aro_contrastive/run.py:161-166` evaluates on
  the held-out test split and the code comment explicitly names and avoids the pooled-split hazard.
- **Seed-count asymmetries and n=3 power caveats throughout Ch. 5** — out of scope by author decision.
  None introduce a factual error, and Table 5.2's best-seed reporting is documented in its own caption.
  Note that the 5-of-15 result is statistically sound independent of this: under the null, P(seed > 3
  binomial sd) ≈ 0.13%, so 5/15 survives any multiple-comparison correction.
