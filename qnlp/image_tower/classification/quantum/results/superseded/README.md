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

## Added 2026-08-08 — BOTH ansatz figures retired (2026-08-08, then 2026-08-15)

> **HISTORY.** Both ansatz figures were moved here on 2026-08-08.
> `encoding_ansatz_sweep.png` was then returned to `results/` the same day, on the grounds
> that it was superseded as *evidence for the ansatz decision* but was the project's only
> record of the *search space explored*. **That return is reversed as of 2026-08-15**: the
> survey was re-run at 30 seeds and the original figure's numbers turned out not to be
> reproducible from the code in this repository. The search-space role is now served by a
> figure generated from JSON. See below.

| figure | file | status |
|---|---|---|
| — | `ansatz_comparison_noise.png` | **RETIRED — do not use.** |
| 4 | `encoding_ansatz_sweep.png` | **RETIRED 2026-08-15 — do not use.** Replaced by `figures/encoding_ansatz_survey.png`. |

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

### `encoding_ansatz_sweep.png` — RETIRED 2026-08-15 (reverses the 2026-08-08 correction)

**This figure was returned to `results/` on 2026-08-08 as the project's only record of the
search space explored. It is now retired outright: its numbers are not reproducible from
the code in this repository.** The survey was re-run
(`rerun_encoding_ansatz_survey.py`, 30 seeds) and the search-space role it was retained for
is now served by `results/figures/encoding_ansatz_survey.png`, which is generated from JSON.

**The finding.** Running `harvest_sweeps()`'s exact protocol — fixed data seed 42, no torch
seeding, 512/128, 8x8, 8 epochs, AdamW lr=0.03 wd=1e-4 — now yields `61.7 / 63.3 / 64.1`
across repeats for `MULTI_AXIS + IQP`, against the logged **95.3%**. Everything that could
explain a gap that size was checked and eliminated:

- `benchmark_encodings_ansatze.py` — one commit (`29591e8`, 2026-07-18, the day after the
  log entry), never modified since, no uncommitted changes.
- `synthetic_shapes.py` — same single commit, unmodified. Dataset mode is `overlapping` in
  both (the shared default).
- Protocol identical, including the 128-sample eval set: `95.3% = 122/128` and
  `96.9% = 124/128` exactly, so the original used the same test set size.

The only uncontrolled variable is the environment (PennyLane is now 0.43.2; the July
version is unrecorded). **That is not why the numbers are retired.** They are retired
because they are internally implausible: **95.3% on ONE 4-qubit node at 8x8 exceeds the
89.3% the full 16-qubit coherent tree reaches at 16x16** (R7). A quarter of the qubits, a
quarter of the pixels, one node instead of a hierarchy, outscoring the whole tower. That
ordering cannot be right whatever produced it, and it holds independently of the re-run.

This also voids the reconciliation this section previously demanded — that 95.3% (8x8, one
node) vs 80.9% (16x16, tree) be explained as "different tasks, not a decline". At the *same*
8x8 single-node task the number is ~64%, so the gap was never task difficulty.

**Nothing downstream moves.** The ordering survives the re-run: `multi_axis` still leads at
every ansatz, which is what R2 independently confirms at 16x16 on the tree. Two things
change, both in the honest direction: `hea` and `iqp` are **tied** at every encoding
(63.3 vs 64.0 at multi_axis, 1.7-pt limit), so the old figure's HEA-over-IQP ordering was
never resolvable; and `amplitude` sits at the **single-attribute ceiling** (51.5–52.3), i.e.
it is not learning the colour–shape conjunction at all — which the single-seed sweep
obscured by reporting 78.1%.

Replacement: `results/figures/encoding_ansatz_survey.png`, from
`results/encoding_ansatz_survey_rerun_results.json`.

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
| 4 | ~~`encoding_ansatz_sweep.png`~~ | **RETIRED 2026-08-15 — moved here, not usable with a caveat.** Numbers not reproducible; see the retirement note above. |
| 5 | `representation_bias_results.png` | Data fine; the "proves the QTTN learns spatial representations" claim needs the R4 classical control alongside. |
