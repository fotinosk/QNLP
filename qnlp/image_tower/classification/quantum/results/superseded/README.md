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
