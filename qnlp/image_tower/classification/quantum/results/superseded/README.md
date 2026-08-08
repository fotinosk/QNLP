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

## Added 2026-08-08 — the two ansatz figures (different defect: wrong architecture)

These do **not** share the scalar-readout defect above. They are superseded because they
benchmark **an ansatz family that is not in the codebase**. `qttn_core.ANSATZE` is
`("strongly_entangling", "iqp")`; HEA and ALT were dropped before the architecture of
record existed. Regenerated replacements: `results/figures/encoding_ansatz.png` and
`results/figures/ansatz_comparison.png`, both from `r2_ansatz_results.json` (21 seeds,
16x16, protocol of record).

| figure | file | why superseded |
|---|---|---|
| 4 | `encoding_ansatz_sweep.png` | Sweeps 4 encodings x **HEA/IQP/ALT** at 8x8 with **one seed per config**. Two of the three ansatze do not exist in the shipped model, and one seed cannot support a 12-way ranking. Its conclusion (multi_axis+IQP) happens to be right, but R2 is what established it. |
| — | `ansatz_comparison_noise.png` | **Untriaged in the original audit.** Three defects: (a) benchmarks HEA/IQP/ALT; (b) single seed at 8x8, i.e. a single 4-qubit node; (c) **contradicts the Experiment & Metrics Record** — the table logs `MULTI_AXIS + IQP` at `p_crit > 0.200` (never below 50% in range) while the figure shows IQP crossing 50% near p=0.12 and reaching ~20% by p=0.2. The discrepancy is unresolved; do not cite either number without reconciling them. |

⚠️ **Do not reuse the flat HEA curve as evidence of noise robustness.** Depolarizing noise
contracts expectation values multiplicatively, `<Z> -> (1-p)^d <Z>`, preserving sign and
order, so an argmax decision can be perfectly noise-invariant while the representation is
being destroyed. A curve that is flat across the whole sweep is more likely measuring that
than robustness.

**The ansatz decision itself is unaffected** — it was settled noiselessly by R2 (`iqp`
`80.9 +/- 4.9` vs `strongly_entangling` `79.3 +/- 9.7`, `+1.6` against a 4.8-pt limit ->
**not resolved**). IQP is adopted for shallower gate depth, not for accuracy and not for
noise tolerance.

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
