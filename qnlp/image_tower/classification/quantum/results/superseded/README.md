# Superseded figures — do not use in the thesis

Retained for the audit trail, moved out of `results/` so they cannot be picked up
by accident. Full triage: `llm/quantum_investigation_roadmap.md` Section 8, Phase B.
Regenerated replacements: `results/figures/`.

**Common defect.** Every figure here was produced by a model that compresses the
whole image to a single scalar before an `nn.Linear(1, 4)` head, so all results sit
in a 50–57% band against the 50% single-attribute shortcut ceiling. A null result in
that regime cannot distinguish "the mechanism does not help" from "the bottleneck
dominates". They also use `scalar_ry` encoding, which R2 later measured as costing
~12.6 points against `multi_axis`.

| figure | file | additional problem |
|---|---|---|
| 8 | `topology_noise_resilience.png` | **RETIRED, not regenerated.** The QTTN arm scored 42.2% — *below* the single-attribute ceiling, i.e. it had not reliably learned even one attribute. It underpins two separate narrative threads (MERA-vs-QTTN expressivity, and Question F's founding observation), both now withdrawn. The QTTN-vs-MERA decision rests on contraction complexity, a scaling argument independent of accuracy. |
| 9, 10 | `quantum_residual_comparison_*.png` | 3 seeds. |
| 11, 12 | `ancilla_residual_comparison_*.png` | 3 seeds. **Figure 12's caption asserts the mixed-channel noise-robustness claim that was retracted the same day** — actively misleading if reused. |
| 13, 14 | `mixed_channel_extended_seeds_*.png` | 10 seeds. The methodological lesson survives in text; the numbers do not. |
| 15, 16 | `spatial_ancilla_comparison_*.png` | 5 seeds; the reported 2.2-pt difference is well inside a ~10-pt resolution limit. |
| 19 | `noise_scheduling_training.png` | Question F's premise is now attributed to a readout-calibration artifact. |

## Added 2026-08-08 — ONE ansatz figure retired (corrected 2026-08-08, see note)

> **CORRECTION.** Both ansatz figures were moved here on 2026-08-08 and that was wrong for
> one of them. `encoding_ansatz_sweep.png` has been **returned to `results/`**: it is
> superseded as *evidence for the ansatz decision*, but it is the project's only record of
> the *search space explored*, and nothing replaces that. R2 ran only the two survivors.
> The two roles were conflated; they are separated below.

| figure | file | status |
|---|---|---|
| — | `ansatz_comparison_noise.png` | **RETIRED — do not use.** |
| 4 | `encoding_ansatz_sweep.png` | **RETURNED to `results/`** — see "Retained as a survey figure" below. |

### `ansatz_comparison_noise.png` — retired

**Untriaged in the original audit.** Three defects, and the third is disqualifying:
(a) benchmarks HEA/IQP/ALT; (b) single seed at 8x8, i.e. a single 4-qubit node;
(c) **contradicts the Experiment & Metrics Record** — the table logs `MULTI_AXIS + IQP`
at `p_crit > 0.200` (never below 50% in range) while the figure shows IQP crossing 50%
near p=0.12 and reaching ~20% by p=0.2. Unresolved; do not cite either number until
reconciled.

⚠️ **Do not reuse its flat HEA curve as evidence of noise robustness.** Depolarizing noise
contracts expectation values multiplicatively, `<Z> -> (1-p)^d <Z>`, preserving sign and
order, so an argmax decision can be noise-invariant while the representation is destroyed.
A curve flat across the whole sweep is more likely measuring that than robustness.

Replacement: `results/figures/ansatz_comparison.png` (R2, 21 seeds, 16x16, noiseless).

### `encoding_ansatz_sweep.png` — retained as a survey figure

**Note that this figure does NOT share the p_crit defect above — it AGREES with the
Metrics Record.** Its `MULTI_AXIS + IQP` curve stays above 50% across the whole sweep
(~92% at p=0.15, ~65% at p=0.2), which is exactly the logged `p_crit > 0.200`. The
contradiction is specific to `ansatz_comparison_noise.png`.

Nor is "HEA and ALT are not in the codebase" an objection to a survey figure — surveying
options you then drop is the point of a survey.

**What it may be cited for**: documenting the space searched — 4 encodings x 3 ansatze,
including AMPLITUDE (which classified at two qubits and four parameters) and ZZ feature
maps. **What it may not be cited for**: ranking configurations, or the ansatz decision,
which R2 settled at 21 seeds.

Three scope facts belong in its caption, none of which invalidate it:
1. **One seed per configuration.** Adequate to show what was explored, not to separate
   configurations that land close together.
2. **8x8, four qubits — a single node, not the tower.** It surveys node-level circuits.
   The tower is 16 qubits at depth 2.
3. **The x-axis is depolarizing noise, which the project later closed as out of scope.**
   The survey content is the p=0 column; the rest characterises a single node.

⚠️ **Caption must reconcile the scale difference or it reads as a regression**: this figure
puts `MULTI_AXIS + IQP` at 95.3% (8x8, one node, one seed) while R2 puts it at 80.9%
(16x16, full tree, 21 seeds). Different tasks, not a decline.

## Retained in `results/` (unaffected — no classifier head involved)

`fidelity_distributions.png`, `barren_plateau_scaling.png`, `topology_barren_plateaus.png`,
`entropy_vs_tree_depth.png`, `entropy_propagation_vs_noise.png`.

Figures 6 and 7 (`barren_plateau_*`) need a **caption fix only**: "empirical BP immunity
proof" overstates 4 points at N≤20 against an asymptotic strawman. Data sound — reframe
as consistent with known theory (Task R5.2).

## Usable with a stated caveat (retained in `results/`)

| figure | file | caveat |
|---|---|---|
| 2 | `training_metrics.png` | 4-dim readout, so not bottlenecked. Valid as "gradients flow, it converges"; not valid for accuracy claims (256/15 protocol superseded). |
| 3 | `noise_tolerance_curve.png` | Computes no `p_crit`; the log's "≈0.05" was a narrative reading. Under the formal <50% criterion this model's value is 0.10 (Task R5.1). |
| 4 | `encoding_ansatz_sweep.png` | One seed per config across 12 configs. R2 supersedes the conclusion: encoding resolved, ansatz not. |
| 5 | `representation_bias_results.png` | Data fine; the "proves the QTTN learns spatial representations" claim needs the R4 classical control alongside. |
