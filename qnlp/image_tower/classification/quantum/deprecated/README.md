# Deprecated scripts — retained, not deleted

These are **superseded**, not wrong-and-discarded. They are kept because they are
the reproduction record for results that appear in `llm/research_log.md`: deleting
them would leave logged numbers with no way to regenerate them.

**Do not build on anything in here.** Every one of these defines its own model
class, and that drift is precisely what allowed two silent regressions to survive
for a month (see `llm/research_log.md`, "Code Audit" 2026-07-27 and
"Code Audit #2" 2026-07-28). All current work routes through
`qttn_core.py` + `phase15_common.py`.

| file | what it produced | why superseded |
|---|---|---|
| `investigate_quantum_residuals.py` | Figures 9–10; residual methods 1–2 (2026-07-26) | Scalar-readout bottleneck (`nn.Linear(1,4)`) — the whole image compressed to one number. Re-run as R3. |
| `investigate_ancilla_residuals.py` | Figures 11–12; residual methods 3–5 | Same bottleneck. Figure 12's caption also asserts a noise-robustness claim retracted the same day. Re-run as R3. |
| `investigate_mixed_channel_seeds.py` | Figures 13–14; the 10-seed retraction | Same bottleneck. Superseded by R3 at 30 seeds. The *methodological* lesson (never trust single-seed noise sweeps) stands. |
| `investigate_spatial_ancilla.py` | Figures 15–16; Question C.1 ablation | Same bottleneck, and n=5 could only resolve ~10-pt effects while reporting a 2.2-pt difference. Re-run as R3. |
| `investigate_noise_regularization.py` | Figure 19; Questions E.2 and F | Same bottleneck. Question F's founding observation is now attributed to a readout-calibration effect. Noise track closed 2026-07-28. |
| `hybrid_trainer.py` | never completed (dimension bug, no training loop) | Rejected on principle 2026-07-26: `Linear(16,4)` before a 4-qubit VQC reduces the qubit count the architecture requires. See the checkable boundary in `llm/quantum_implementation_plan.md`. |

## Current equivalents

* Model: `qttn_core.CoherentQTTNClassifier` (coherent tree, the model of record) or
  `HierarchicalQTTNClassifier` (the hybrid, retained only to reproduce R1–R4).
* Harness: `phase15_common` — one training loop, the pinned architecture and
  protocol, and Welch comparisons that always report the resolution limit.
* Guards: `test_qttn_core.py` — each test targets a failure this project actually
  shipped, including the scalar readout and the loss of tree coherence.
