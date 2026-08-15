# Thesis Outline — Quantum-Inspired Vision Models and Their Integration into a Quantum-Inspired Vision-Language Model

**Format**: long-form report / paper-style, ~30–40 pages excluding appendices.
**Status**: draft structure, updated 2026-08-05 to include Task C7. Research phase closed except the C6 quantum seed array.
**Sources**: `llm/research_log.md`, `llm/quantum_investigation_roadmap.md`, `llm/quantum_implementation_plan.md`, `COCO_EXPERIMENTS.md`.

---

## Three-part spine

| Part | Question it answers | Pages |
|---|---|---|
| **I — Quantum-inspired vision models** | What is a tensor-network image tower, and what is its quantum counterpart? | 11 |
| **II — Investigation of models and properties** | Which architectural choices matter, which are settled, and what do they cost? | 18 |
| **III — Integration with the quantum-inspired language model** | Does the vision tower work as half of a two-tower VLM? | 6 |
| | Front and back matter (intro, conclusion) | 4 |
| | **Total** | **39** |

Part II is the bulk and the contribution. Part I builds the object; Part III places it in the system it was built for.

> **⚠️ SCOPE DECISION (2026-08-04, provisional): COCO retrieval is OUT; ARO is the only Part III evaluation.** The COCO runs are logged in `COCO_EXPERIMENTS.md` but never produced above-random retrieval, so leaning on them weakens the document. Part III evaluates the two-tower model **on compositionality alone**, which is also the property the whole thesis is about. This may be revisited; if COCO returns, it slots in as a §III.2 retrieval subsection and the negative-results table in §III.3.2 expands back (see Appendix C).

> **⚠️ STRUCTURAL UPDATE (2026-08-05): Task C7 changes how Part II ends.** C6 established that tensor networks bind but not better than an MLP, which left the project's central claim — that frozen CLIP fails on compositional tasks where our model succeeds — without a mechanism. **C7 supplies it from the other side**: frozen CLIP's embedding geometry is 13× less sensitive to a compositional swap than to a content change, on the same composites where a 683-parameter MLP scores 94.6%. So Part II now ends on a positive result, and §II.6 is new.

**The narrative Part III must be honest about**: the *classical* tensor-network vision tower is integrated with the DisCoCat language model and has ARO results. The *quantum* tower has never been integrated — Steps 4 and 5 of the build plan were never executed. So Part III is: one integration with results, one with a design and a costed blocker. Say which is which in its first paragraph.

---

## FIGURE INVENTORY

Assign figures from this table only. **A 2026-07-28 audit found 11 of 20 figures compromised** by a scalar-readout bottleneck that pinned every result into a 50–57% band against a 50% shortcut ceiling; those are in `results/superseded/` and must not be reused. The `superseded/README.md` explains each one individually.

### Safe as-is — no classifier head involved, unaffected by the audit

| # | file | shows | section |
|---|---|---|---|
| F1 | `results/fidelity_distributions.png` | Pairwise state fidelity vs CNOT count, against the Haar reference | §I.1.3 |
| F2 | `results/entropy_vs_tree_depth.png` | Root-qubit entropy at depth 1 vs depth 2, against the `ln 2` ceiling | §I.1.3 |
| F3 | `results/barren_plateau_scaling.png` | Gradient variance vs qubit count, semi-log and log-log | §II.1.1 |
| F4 | `results/topology_barren_plateaus.png` | Same, for QTTN / MPS / MERA | §II.1.2 |
| F5 | `results/entropy_propagation_vs_noise.png` | Entropy at level-1 vs root under depolarizing noise | §II.3.2 |

⚠️ **F3 and F4 need a caption fix, not a regeneration.** The original caption calls it an "empirical BP immunity proof", which overstates four points at N≤20 against an asymptotic strawman. Reframe as *consistent with the known result that hierarchical circuits with local observables avoid exponential gradient decay*.

### Regenerated after the audit — these are the results figures of record

| # | file | shows | section |
|---|---|---|---|
| F6 | `results/figures/model_comparison.png` | Accuracy **and accuracy-per-parameter** for all six models, with chance, the single-attribute ceiling and the MLP reference marked | §II.2.2 |
| F7 | `results/figures/readout_bottleneck.png` | 34.1 / 35.6 / 78.4 — the readout width finding, with the one-qubit-marginal explanation | §I.3.2 |
| F8 | `results/figures/ablations_and_noise.png` | Node-level ablations and noise sweeps. **Caption must state these are hybrid-tree data** — mixed-channel and the ancilla were deliberately not ported to the coherent tree | §II.3.1 |
| F19 | `results/figures/encoding_ansatz.png` | **NEW 2026-08-08.** Encoding × ansatz, 21 seeds per arm, with an effect-size-vs-MDE panel making "encoding resolved, ansatz not" visible in one look | §II.2.1 |
| F20 | `results/figures/ansatz_comparison.png` | **NEW 2026-08-08.** `iqp` vs `strongly_entangling` at `multi_axis`: mean ± 1 s.d. training curves plus per-seed spread. Noiseless, matching project scope | §II.2.1 |

All three come from `make_figures.py`, with provenance stated in every caption. **Non-negotiable house rule: every accuracy figure carries the MLP reference line.** Its absence is exactly what let a broken classical baseline sit at chance unnoticed.

### CLEVR data figures — verification artifacts, and they earn their place in the text

| # | file | shows | section |
|---|---|---|---|
| F9 | `figures/clevr_crop_calibration.png` | Crop geometry sweep; fill fraction is depth-invariant (0.250 small / 0.500 large) | §II.4.1 |
| F10 | `figures/clevr_relation_examples.png` | Relation montage by class. **The only real check that CLEVR's rotated direction vectors are applied correctly** — a sign error would still yield a balanced, plausible dataset and a plausible ~25% result | §II.4.1 |
| F11 | `figures/clevr_binding_size_examples.png` | Binding composites, both classes, showing identical marginals and jittered placement | §II.5.2 |
| F12 | `figures/clevr_c7_matched_quads.png` | The C7 matched quads: base / jitter / swap / content. The `swap` column must show the same two objects with sides exchanged | §II.6.1 |

⚠️ F10 was **overwritten rather than superseded** when the relation data was rebuilt; the pre-change version is in commit `e980b18` if the old dataset ever needs illustrating.

### The survey figure — main text, not appendix

| # | file | shows | section |
|---|---|---|---|
| **F15** | `results/encoding_ansatz_sweep.png` | **4 encodings × 3 ansätze under a noise sweep — the record of the space that was searched.** This is the evidence for the thesis's opening claim that a *family* of quantum image encoders was investigated, and **nothing replaces it**: R2 (F19/F20) ran only the two survivors. Includes AMPLITUDE, which classified at **two qubits and four parameters**, and ZZ feature maps. | §II.2.1 |

**Cite it for what was explored, never for ranking** — the ranking is F19/F20. Three scope facts belong in the caption, none of which invalidate it:
1. **One seed per configuration.** Enough to show what was tried; not enough to separate configurations that land close together.
2. **8×8, four qubits — a single node, not the tower.** It surveys node-level circuits; the shipped tower is 16 qubits at depth 2. *(This is a larger caveat than the seed count and the easier one to forget.)*
3. **The x-axis is depolarizing noise**, which the project later closed as out of scope, so the survey content is the p=0 column and everything right of it characterises a single node.

⚠️ **The caption must reconcile the scale gap or a reader will see a regression**: `MULTI_AXIS + IQP` is `95.3%` here (8×8, one node, one seed) and `80.9%` in F19 (16×16, full tree, 21 seeds). Different tasks, not a decline.

✅ **Note this figure agrees with the Metrics Record** — its `MULTI_AXIS + IQP` curve stays above 50% across the whole sweep, matching the logged `p_crit > 0.200`. The `p_crit` contradiction is specific to the retired F18.

### Usable with a stated caveat — prefer the appendix

| # | file | caveat |
|---|---|---|
| F13 | `results/training_metrics.png` | Valid as "gradients flow and it converges"; **not** valid for accuracy claims (superseded protocol). |
| F14 | `results/noise_tolerance_curve.png` | Computes no `p_crit`; under the formal <50% criterion this model's value is 0.10, not the 0.05 quoted in early text. |
| ~~F18~~ | ~~`results/ansatz_comparison_noise.png`~~ | ✅ **RESOLVED 2026-08-08 — superseded by F20.** Moved to `results/superseded/`. ⚠️ Its `p_crit` disagreement with the Metrics Record is **recorded but not reconciled** — see the note under R1 below. |
| F16 | `results/representation_bias_results.png` | Data sound, but the "proves the model learns spatial representations" claim needs the classical control alongside. |
| F17 | `figures/clevr_binding_examples.png` | The abandoned shape-binding build — use only in the §II.5.4 footnote. |

### Must be created — no figure exists yet

| # | shows | section | notes |
|---|---|---|---|
| **N1** | **C6 binding vs patch-shuffled, paired bars, four arms + chance line** | §II.5.3 | **The strongest construction result in the project and it has no figure.** Highest priority; deserves the cleanest figure in the document. |
| **N2** | **C7 cosine geometry: ceiling / effect / floor with 95% CIs, and the swap-invariance ratio** | §II.6.2 | The other headline. Only the input montage (F12) exists. |
| N3 | C4 per-seed training curves, showing both collapse modes | §II.4.4 | One panel of *learned-then-diverged*, one of *never-left-chance*. Makes the instability legible in a way the collapse count cannot. |
| N4 | CLEVR per-head accuracy at 30 vs 90 epochs, quantum and classical | §II.4.3 | The head-trading result. Grouped bars, four heads × two budgets. |
| N5 | Score vs parameter count scatter, CLEVR arms | §II.4.3 | The CLEVR analogue of F6. Optional if space is tight. |

### ✅ R1 — DONE 2026-08-08: the ansatz figures are regenerated (scope corrected the same day)

**F19 and F20 are new**, produced by `make_figures.py` from `r2_ansatz_results.json`.

> **⚠️ SCOPE CORRECTION.** R1 initially retired *both* originals. That was right for **F18** (`ansatz_comparison_noise.png`, archived in `results/superseded/`) and **wrong for F15**, which has been returned to `results/`. Two roles were conflated:
> * **As evidence for the ansatz decision** — F15 is superseded. One seed cannot rank twelve configurations, and R2 settled the decision at 21 seeds.
> * **As documentation of the search space** — F15 is the only such record, and **R2 does not replace it because R2 ran only the two survivors.** Retire it and the thesis's opening claim ("we investigate a family of quantum image encoders — four encodings, several ansätze") loses its evidence.
>
> Two of the original objections do not survive scrutiny: **"HEA and ALT are not in the codebase" is not an objection to a survey figure** — surveying options you then drop is the point — and **the `p_crit` contradiction is specific to F18**. F15 actually *agrees* with the Metrics Record: its `MULTI_AXIS + IQP` curve stays above 50% across the sweep, matching the logged `p_crit > 0.200`.
>
> What remains is scope, handled in the caption, not by retirement: one seed, 8×8 single node, and a noise axis the project later closed.

**No retraining was needed** — R2 had already run the correct comparison (`{scalar_ry, multi_axis} × {strongly_entangling, iqp}`, 21 seeds per arm, 16×16, protocol of record) and its results were on disk unplotted. The old figures were not measuring something newer; they were measuring something that no longer existed.

What the replacements fix:
* **Right ansätze.** `iqp` vs `strongly_entangling`, the two in `qttn_core.ANSATZE`. The originals benchmarked HEA/IQP/ALT, two of which are not in the codebase.
* **Adequate power.** 21 seeds per arm against one seed per config, so the effect sizes carry resolution limits.
* **Noiseless**, matching the project's scope decision. Noise resilience was never a selection criterion, and a noise sweep here could only ever be a single-node characterisation.
* **The MDE is now visible**, not just stated — F19's right panel plots each effect against its resolution band, so "encoding resolved, ansatz not" is legible at a glance.

⚠️ **One thread deliberately left open**: the `p_crit` disagreement between the retired `ansatz_comparison_noise.png` and the Metrics Record (`MULTI_AXIS + IQP` logged at `p_crit > 0.200`, figure showing it cross 50% near p ≈ 0.12) is **recorded but not reconciled**. It no longer blocks anything, since neither artifact is cited and the noise line is closed — but if any `p_crit` number is quoted in the thesis, resolve this first.

**Realistic figure count for 39 pages: 13–16.** Main text: **F1–F12** (theory, architecture, CLEVR data), **F15** (the survey), **F19–F20** (the ansatz decision), **N1–N2** (the two missing headliners) — 17 if all are used, so expect to cut two or three. Appendices: **N3–N5**, **F13–F14**, **F16–F17**, and the retired set.

---

## 0. Introduction (3 pp)

0.1 **Motivation** — compositionality is the standing failure of contrastive VLMs. ARO/SugarCrepe: the same bag of words arranged two ways, two different labels; frozen CLIP scores near chance. A model whose forward pass *is* a spatial or syntactic contraction ought to bind attributes to positions rather than pooling them.

0.2 **Why vision first** — the language side already exists (DisCoCat/lambeq). The open question is the *vision* tower: whether a tree tensor network is a viable image encoder, and whether replacing its classical CP nodes with quantum unitaries costs or buys anything.

0.3 **Three answers, and one of them is negative.** All three in the abstract:
   1. The quantum tree is a viable image tower. At **287 parameters** it reaches `89.3% ± 3.9` on 16×16 synthetic shapes, beating a matched classical CP tree by `+9.7` and a same-size MLP by `+18.8`; on CLEVR it beats `classical_full` on shape and material at fewer parameters.
   2. **Inside the vision tower, tensor networks show no compositional advantage over a plain MLP.** On a marginal-controlled binding probe a 683-parameter MLP beats the 930-parameter CP tree. Both branches were pre-registered; the negative one fired.
   3. **But the frozen contrastive baseline is nearly blind to the same structure.** A compositional swap moves CLIP's embedding **13.1× less** than a content change — 7.6% of it — on composites where that 683-parameter MLP scores `94.6%`. An 88M-parameter encoder discards binding information that is plainly present in the pixels.

   (2) and (3) together are the honest form of the thesis's claim: **the advantage over CLIP is real and measured, but it does not come from the tensor network's compositional inductive bias.** Say that in the introduction rather than letting it emerge.

0.4 **Contributions**, with numbers:
   - A coherent quantum TTN image tower, entangled across levels, 287 params (Part I).
   - Question A.3 answered: the unitarity-constrained node beats the unconstrained CP node by `+9.7` pts at 33% fewer parameters (§II.2).
   - Verification that the architecture is trainable at the widths used, consistent with known results for isometric tensor networks (§II.1). ⚠️ **Phrase it this way, not as "barren-plateau immunity."** MPS and MERA are equally well-behaved, so this is an *enabling* result, not a differentiator — and four points at N ≤ 20 cannot separate polynomial from exponential.
   - Question C.2(i) answered: implicit tree topology alone encodes spatial relations at classical parity (§II.4).
   - A compositional binding probe with identical class marginals and a passing manipulation check (§II.5).
   - A direct geometric measurement of frozen CLIP's binding invariance, in the metric CLIP uses at inference (§II.6).
   - A methodology contribution: two code audits and the standing rules they earned (§II.8).

0.5 Roadmap.

---

# PART I — QUANTUM-INSPIRED VISION MODELS (11 pp)

## I.1 Background: tensor networks as image encoders (4 pp)

1.1 CP decomposition, MPS, TTN, MERA. Bond dimension; the area law; why a tree's contraction order encodes a spatial hierarchy.
1.2 Variational quantum circuits: encodings, ansätze, barren plateaus.
1.3 **The TTN↔VQC correspondence — the theoretical spine.**
   - *CP-rank ↔ entangling gate density.* Local operations are LOCC and cannot raise Schmidt rank; CNOT/CZ can. Measured: 0→6 CNOTs takes root entropy `0.0000 → 0.5139` against the `ln 2 ≈ 0.693` ceiling, 3 CNOTs capturing 73% with clear diminishing returns. **This is why the ansatz uses 3 CNOTs per node** — the design follows the measurement.
     → **F1** (`fidelity_distributions.png`). Caption should make the reading explicit: as CNOT density rises the output qubit becomes mixed and the fidelity distribution shifts away from the flat Haar profile.
   - *Partial trace ↔ pooling.* One qubit per node ⇒ χ=2 ⇒ the entropy bound is **architecturally guaranteed**, not empirical. State this correction; it is the easiest place in the document to over-claim.
   - The non-guaranteed part: entropy climbs `72.7% → 89.7%` of the ceiling from depth 1 to depth 2.
     → **F2** (`entropy_vs_tree_depth.png`). Caption must mark the `ln 2` line as a fixed ceiling and note only two depths were measured.
1.4 Related work: quantum-inspired ML, QML for vision, the compositionality-benchmark literature.

## I.2 The classical tensor-network vision tower (2 pp)

2.1 `TTNImageModel` — 4×4 patches, bilinear colour×pixel embedding, hierarchical quadtree of CP nodes, explicit residual and dropout.
2.2 The four control architectures, **defined once here and used for every comparison in Part II**: `classical_bare`, `classical_full`, `mlp_reference`, `mlp_param_matched`.
2.3 State the standing requirement now: every comparison carries a matched-parameter classical control and a stated minimum detectable effect. Give the reason — a mis-specified baseline once produced a fake 57.8-point "quantum advantage."

## I.3 The quantum vision tower (5 pp)

3.1 **The architecture of record** — reproduce the block diagram: 16×16 → 16 patches → `Linear(48,3)`, tanh·π → **one 16-qubit device**, patch *p* on wire *p* → RX/RY/RZ → level-1 IQP blocks with per-block weights → survivors stay **quantum** → level-2 IQP → ⟨Z⟩ on four wires → `Linear(4, n_classes)`. **287 parameters.**
   → This diagram is effectively a figure; typeset it properly rather than as a code block.

3.2 **Two load-bearing properties, each established the hard way.** This subsection is the chapter's argument:
   - *Coherence.* Survivors pass between levels as qubits, never measured mid-tree. Enforced by a regression test: a survivor's Bloch vector must be < 1 (measured `0.38–0.73` after training, vs exactly 1 for a product state). Inter-level entanglement is worth **+9.9 pts** over an otherwise identical tree that measures between levels.
   - *Readout width.* Root-only readout scores `35.6%`; four top-layer wires score `78.4%` — **a 43-point gap**, because the root is one qubit's marginal of a 16-qubit state.
     → **F7** (`readout_bottleneck.png`). Caption must make the point that the gap is **not** "more numbers helps" but that a single qubit's marginal discards the state — otherwise a reader files it as a trivial capacity effect.

3.3 **The classical-boundary rule**, in checkable form: *the encoder may set state-preparation parameters but may not reduce the qubit count the architecture would otherwise require.* Three-row verdict table. Then dissolve the apparent tension: the `+12.6` pts from widening the encoder came from using all three rotation parameters of a qubit where single-axis encoding used one — a **quantum** resource used fully.

3.4 **Simulation strategy and its walls** — this fixes every experimental scale in Part II:
   - Qubit recycling is exact (16→7 qubits, output difference `0.0`), valid for hardware, but does not extend past depth 2 in simulation.
   - `default.tensor` runs depth-3 (64 qubits, 0.13 s) and depth-4 (256 qubits, 1.20 s) forward passes but has no backprop: 56.7 s per parameter-shift step. **Forward-feasible, training-infeasible.**
   - Statevector autograd walls at N≈25 (>100 GB).
   - Parameter broadcasting gave a ~30× speedup.
   → Optional: a small table of device × depth × wall-clock. Cheap to produce and it makes the 16×16 decision self-evident.

---

# PART II — INVESTIGATION OF MODELS AND PROPERTIES (18 pp)

Open with the protocol, once: 1024 train / 30 epochs (90 where stated), last-5-epoch mean, unpaired Welch, MDE with every comparison. An earlier 256/15 protocol was underpowered; results are never compared across that boundary.

## II.1 Trainability and topology (2 pp)

1.1 **Trainability.** Gradient variance `5.94e-2` to `1.51e-1` across N ∈ {4, 9, 16, 20} — >62,000× the `2^-N` prediction at N=20; log-log slope consistent with `O(1/poly(N))` over the range measured.
   **Method, stated so the claim can be sized**: 100 trials per width; each draws random weights *and* random inputs, runs the circuit forward, measures a **local** observable, and records the gradient of **one** leaf-level parameter. The variance is over those 100 values. It characterises the *initialisation landscape*, not training.
   ⚠️ **Write "consistent with", not "proof of immunity."** Four points at N ≤ 20 against an extrapolated strawman cannot separate polynomial from exponential; the `62,000×` figure is a statement about N=20 specifically. A 100-sample variance carries ~14% relative error and the figure has no error bars, so the curve's wiggles are noise.
   → **F3**, with the rescoped caption (§Figure Inventory).
1.2 **Topology comparison** — QTTN vs MPS vs MERA, all well-behaved at N=20.
   ⚠️ **This is an enabling result, not a differentiator — frame it that way explicitly.** Theory predicts all three should be fine: the relevant results cover shallow and hierarchical circuits with local observables, and all three qualify. **Finding the alternatives immune too is confirmation the measurement works**; the outcome that should have raised concern is the opposite one. The experiment's job is to license the rest of the project — had it failed there would be no CLEVR phase.
   Two further limits: the MPS tested is a **shallow** chain, so it says nothing about the deep chains theory actually warns about; and its accuracy half came from a retired figure whose QTTN arm scored below the single-attribute ceiling. **The QTTN-vs-MERA choice rests on contraction complexity, a scaling argument independent of both accuracy and trainability.**
   → **F4**, same caption fix.

## II.2 Encoding, ansatz, and the parameter-efficiency result (3 pp)

2.1 **The search, then the decision — in that order.**
   *First, what was surveyed*: 4 encodings × 3 ansätze, including AMPLITUDE (classifying at **two qubits and four parameters**) and ZZ feature maps. This is what licenses calling the investigation a survey rather than a single-architecture study.
   → **F15** (`encoding_ansatz_sweep.png`), with the three scope caveats from the inventory in its caption. Present it as *the space searched*, and say plainly that the ranking comes later and from a different run — otherwise a reader will try to read the decision off it.
   *Then the decision*: `multi_axis` beats single-axis by `+12.6` (limit 3.6, **resolved**). IQP vs strongly-entangling is `+1.6` against a 4.8 limit — write "**not resolved**", never "IQP is better". IQP is adopted for **shallower gate depth**; do not attach an accuracy or a noise-tolerance justification to it.
   → **F19** (`encoding_ansatz.png`) and **F20** (`ansatz_comparison.png`). F19's right panel plots each effect against its resolution band, so "resolved" and "not resolved" are visible rather than asserted — use it rather than restating the numbers in prose.
   ⚠️ **Do not repeat the July claim that IQP uses fewer parameters.** In `qttn_core` both ansätze allocate the same `[layers, qubits, 3]` tensor and `_iqp_block` leaves the third rotation column unread, so the counts are identical and roughly a third of the IQP arm's ansatz weights are **dead** — zero gradient, frozen at initialisation. The 287-parameter figure counts them, which makes the efficiency claim conservative rather than wrong. State the convention in §2.2's table caption; a reader who opens the code will find this.
2.2 **The headline table** (16×16 synthetic shapes, 4 seeds):

   | model | score | params |
   |---|---|---|
   | `mlp_reference` | `96.0 ± 1.5` | 999 |
   | **`quantum_coherent`** | **`89.3 ± 3.9`** | **287** |
   | `classical_full` | `88.4 ± 5.5` | 352 |
   | `classical_bare` | `79.6 ± 10.3` | 428 |
   | `quantum_hybrid` (superseded) | `79.4 ± 8.0` | 211 |
   | `mlp_param_matched` | `70.5 ± 18.1` | 257 |

   → **F6** (`model_comparison.png`) — **the single most important figure in Part II.** Its accuracy-per-parameter panel *is* the thesis's positive claim; lead with that panel, not the raw-accuracy one. Both the chance line and the MLP reference must be visible.
2.3 **Question A.3 answered.** Under matched constraints the unitarity-constrained node beats the unconstrained CP node at 33% fewer parameters.
   ⚠️ Caveats that travel with the table: the classical arms were tuned while the quantum arm inherited its learning rate, so the quantum figure is **conservative**; and `classical_bare` moved `33.9 → 56.7 → 79.6` across three revisions of its search space, which is what produced the standing rule **sweep, never derive, any capacity parameter**.

## II.3 Properties investigated and closed (3 pp)

3.1 **Mechanisms rejected** (30 seeds each).
   ⚠️ **These ran on the hybrid, one day before the port to the coherent tree** — "architecture of record" meant something different at the time. `mixed_channel` and the ancilla were never ported; `reupload` exists in `qttn_core`'s `MODES` but was not re-run. **Write it as: the rejections stand at node scale and were not re-validated at the final architecture** — not that these mechanisms were shown to fail on the coherent tree. "We tested and rejected five residual mechanisms" reads as if it happened on the shipped model, and it did not.

   | mechanism | result | verdict |
   |---|---|---|
   | data re-uploading | `+1.5`, limit 3.8 | **true null** — "no demonstrated benefit at effects ≥ 3.8 pts", never "harmful" |
   | near-identity init | none | affects early dynamics only |
   | mixed-unitary channel | `−10.9`, limit 4.7 | worse, resolved. **0/30 failures** — not unstable, just worse |
   | ancilla-controlled mixing | as above | rejected |
   | LCU / LCU-lite | as above, plus shot overhead | rejected |
   | explicit spatial ancilla | `−19.5`, limit 3.8 | worse — **for translation-invariant classification only** |

   → **F8** (`ablations_and_noise.png`). ⚠️ **Caption must state these are hybrid-tree measurements** — mixed-channel and the ancilla were never ported to the coherent tree, because each ancilla doubles the statevector and both were rejected decisively. Presenting them as coherent-tree results would be wrong.
   The classical tower *does* benefit from residuals (`+31.7`), so the asymmetry is itself a finding: the classical analogy does not transfer.

3.2 **Noise, and why it is closed.** Tolerance to p≈0.02, degrading past p≈0.05. Mechanism: depolarizing noise contracts expectation values multiplicatively, preserving sign and order, so an argmax boundary survives. Training under noise does **not** help (`55.0` vs `54.1` at both levels).
   → **F14** in an appendix, with the `p_crit` definition reconciled; **F5** (`entropy_propagation_vs_noise.png`) here if §II.3.2 runs long enough to earn it, captioned with its ceiling-proximity confound — the level-1-to-root entropy gap narrows under noise, but the root already sits near the `ln 2` ceiling, so this is a partial answer.
   **Standing limitation: every noise result characterises a single 4–5 qubit node, never the tree** (`default.mixed` is O(4^N); 16 qubits needs ~68 GB).

3.3 **Positional encoding, part 1** — the `−19.5` result is real but **scoped**: position barely affects the label in that task, and it tests a per-quadrant ancilla, not the per-patch design. The relational question is settled in §II.4.4.

## II.4 Scaling to CLEVR: perception and relations (4 pp)

4.1 **Data pipeline, and the two plan errors it corrected.** CLEVR has **no** 1- or 2-object scenes, so datasets are built by cropping objects out of full scenes; and relation labels must use CLEVR's **camera-rotated** basis (~49° off the world axes).
   → **F9** (`clevr_crop_calibration.png`) — crops are world-scaled by depth, so fill fraction is depth-invariant (0.250 / 0.500, matching the analytic prediction). This is what keeps `size` learnable at 16×16, so the figure carries a real argument.
   → **F10** (`clevr_relation_examples.png`) — **include this one and say why**: it is the only check that the rotated direction vectors are applied correctly, since a sign error would still produce a balanced dataset and a plausible ~25% result. A verification artifact that changes what you'd believe is worth a figure.

4.2 **The learnability gate, and a retracted prediction.** All four heads learnable at 16×16 (`95.2 / 69.9 / 82.5 / 99.0`). This **retracts** the project's own prediction that `material` would be unlearnable — it is the second-best head. Consequence: no scaling route needed.
   ⚠️ One sentence on the first run declaring 64×64 unlearnable from a single hardcoded learning rate: an optimisation artifact nearly became a data conclusion.

4.3 **Single-object attributes.** Readout of record adopted on **stability** (0/3 vs 1/3 collapses; std 5–35× tighter). Present 30- and 90-epoch results **side by side, never one alone**:
   - 30 epochs: quantum beats `classical_full` on shape `+12.7` and material `+8.8`, both resolved, at fewer parameters.
   - 90 epochs: colour `27.1 → 63.0` (resolved), material `−9.7` and size `−2.0` resolvably **worse**. Four heads share one summed loss and converge at different rates; longer training **trades** heads.
   - Classical at 90 epochs: `classical_bare` colour `52.5 → 73.4`, but variance explodes (`±26.2`), so quantum's `63.0` sits *inside* the classical spread.
   → **N4** (create) — grouped bars, four heads × two budgets, quantum and classical. The head-trading result is hard to see in two tables and obvious in one figure.
   → **N5** (create, optional) — CLEVR score-vs-parameters scatter, the analogue of F6.
   ⚠️ `mlp_reference` beats **every** TTN arm on **every** head. State this early and plainly.

4.4 **Relations, and Question C.2.** Structure as the correction it was:
   - The controls, once run: `mlp_reference` `65.4`, `classical_full` `51.3`, **`classical_bare` `49.6 ± 5.6` at 428 params against the quantum tower's 462**, clearing the 26.6% floor by ~23 pts.
   - Converged `quantum_none` `46.2 ± 2.5` and `quantum_on_wire` `48.0 ± 3.8`, both **statistically tied** with `classical_bare`. ⚠️ **12 of 20 seeds ended at chance**, in two distinct modes.
   → **N3** (create) — two panels of per-seed curves: *learned-then-diverged* (one seed peaked `50.2%` at epoch 12, above the classical mean, then fell to ~24%) and *never-left-chance*. The collapse count alone conflates two failures with different fixes; the figure separates them.
   - **Question C.2(i) is ANSWERED: yes.** With no positional parameters at all, the tower does a relational task at classical parity. This discharges §II.3.3's scope condition.
   - **C.2(ii) is not answered**: report as a **bounded null at its MDE** — a legitimate finding supporting omission of explicit positional encoding.
   ⚠️ Converged subgroup = diagnostic, not arm. ⚠️ Ran at 30 epochs where ~90 are needed, so these are a **floor, not a ceiling**.

## II.5 The compositional binding probe (3 pp)

5.1 **Why the earlier tasks cannot settle the premise.** Attribute classification is pure perception. The relation task is a single-referent readout. **Neither has the property that makes CLIP fail on ARO: the same bag of features arranged two ways, two labels.**

5.2 **Design.** One cube + one sphere per composite; label = which side carries the bound attribute. **Class-conditional marginals identical by construction**, so a bag-of-features model is at chance *provably*. Placement is **jittered, not aligned** to the 2×2 level-1 blocks — aligned placement would hand each block one object, which is the TTN's claimed inductive bias, i.e. rigging the test.
   → **F11** (`clevr_binding_size_examples.png`) — show both classes side by side so the identical-marginals property is visible rather than asserted. This figure does real work: a reader who has not seen the composites will not believe the control.

5.3 **The manipulation check.** Patch-shuffled, every arm sits at **exactly chance** (`49.3–51.0`); unshuffled, `91.5–96.0`.

   | arm | params | binding | shuffled | gap |
   |---|---|---|---|---|
   | `mlp_reference` | 1397 | `96.0 ± 1.3` | `50.2 ± 2.1` | **+45.8** |
   | **`mlp_param_matched`** | **683** | **`94.6 ± 2.9`** | `49.3 ± 2.7` | +45.3 |
   | `classical_full` | 942 | `93.2 ± 1.4` | `49.3 ± 2.7` | +43.9 |
   | `classical_bare` | 930 | `91.5 ± 3.1` | `51.0 ± 1.4` | +40.5 |
   | *best quantum seed* | 757 | *`73.8`* | — | — |

   → **N1 (create) — the highest-priority missing figure.** Paired bars, unshuffled vs shuffled, four arms, chance line marked. A +40-point drop to *exactly* chance is the most visually convincing result in the project and it currently exists only as numbers.

5.4 **The abandoned first build, kept as a methodological finding.** Shape-binding was run first and is **confounded**: two of four arms sit at the shape-perception floor, so their 50% measures perception, not binding. **A binding task is a conjunction of perception and binding, so bind only what every arm already sees.** The confound ran in the direction that would have flattered the thesis. Half a page; **F17** in the appendix if illustrated at all.

5.5 **The four statements the data supports:**
   1. Tensor networks — classical and quantum — perform compositional binding where a bag-of-features model is provably at chance.
   2. They do **not** do so better than a parameter-matched MLP; the MLP leads by 1.4–3.1 points at **fewer parameters than any TTN arm**.
   3. **The quantum tree binds using only its topology** — updated 2026-08-13, array complete 30/30. `quantum_none` (**725 params, no positional parameters at all**) puts **5 of 15 seeds >3 binomial sd above the floor**, peaking at `69.0` (8.1 sd). An *existence* claim, now resting on five seeds rather than one, so it no longer depends on post-hoc selection.
   4. **Explicit positional encoding adds nothing detectable**: `on_wire − none = −0.6` against an MDE of 5.5, at n=15 per arm with **both arms carrying signal**. A genuine bounded null — report it as support for omitting positional encoding.
   5. The task **saturates** (spread 4.5 pts against MDEs of 1.3–2.3): it shows *whether* an architecture binds, not *how well*.

   ⚠️ **Never write that the MLP cannot capture these relations.** The shuffle control proves a *bag-of-features* model is at chance; `MLPReference` carries position and can bind. The control validates the **task**, never an architecture's inability.
   ⚠️ **Capability, not reliability, and say both.** 10/15 and 12/15 seeds sit at or near floor, and on an all-seeds basis both quantum arms are **resolvably worse** than `classical_bare` (`−36.6`, limit 3.9; `−37.2`, limit 4.9). The existence claim is strong; there is no performance claim.
   ⚠️ `none` has *more* above-floor seeds than `on_wire` (5 vs 3) — **not resolvable**, do not report as a direction.
   → **F22** (`clevr_binding.png`). Quantum arms are plotted **per seed**, not as means: the distribution is bimodal, so a mean with an error bar would describe neither the capability nor the failure.

## II.6 Why the frozen baseline fails: CLIP's binding geometry (2 pp) — NEW

> This section is what makes Part II end on a positive result. C6 showed the tensor network's compositional advantage does not exist; C7 shows the *baseline's* deficit does, and measures it.

6.1 **The question C6 left open.** C6 could not explain why frozen CLIP fails on ARO/SugarCrepe, because it never tested CLIP. C7 does, on the same composites.
   **Why cosine similarity rather than a trained probe**: cosine similarity *is* what CLIP does at inference — retrieval, zero-shot classification and the contrastive objective all reduce to it. "Swapped images sit at near-identical cosine similarity" is a statement about what the model does, not about what could in principle be dug back out.
   **What makes the number interpretable**, since bare cosine similarity is meaningless on anisotropic embeddings: **matched quads** (the swap reuses the same two crops, so the only difference is which side each object is on), and **every item carries its own ceiling and floor**, making the statistic scale-free.
   → **F12** (`clevr_c7_matched_quads.png`) — show base / jitter / swap / content. The construction is the argument; a reader who doesn't grasp the quad won't accept the ratio.

6.2 **Result** — `openai/clip-vit-base-patch32`, 500 matched quads at 224×224:

   | comparison | cos sim (95% CI) | role |
   |---|---|---|
   | `jitter` — same objects, re-placed | `0.9777 ± 0.0009` | ceiling |
   | **`swap` — same objects, sides exchanged** | **`0.9727 ± 0.0011`** | **the effect** |
   | `content` — different objects, same arrangement | `0.9123 ± 0.0022` | floor |

   A compositional swap moves the embedding `0.0050`; a content change moves it `0.0654` — **13.1× further**. The swap accounts for **7.6%** of a content change. **`swap_invariance = 0.924`**.
   → **N2 (create)** — the three comparisons with CIs and the derived ratio. Second-highest figure priority after N1.

6.3 **Two controls, both passed.**
   - **Void guard**: ceiling − floor is `0.0654`, 3.3× the `0.02` threshold, so CLIP *is* discriminating these images. The invariance is specific to the swap, not a general failure to see anything. Without this the result would be indistinguishable from "the inputs are out of distribution."
   - **Padding robustness**: at canvas 224 / cell 96, ~60% of each image is padding, and with black padding every composite reads as "two small photos on black", compressing all similarities upward. Re-run with CLEVR floor-grey padding: `swap_invariance` `0.928 → 0.924`. Nothing changes, exactly as the ratio construction predicts.

6.4 **The contrast that makes it a result.** On the same construction a **683-parameter MLP classifies the swap at `94.6%`** (§II.5). **An 88M-parameter frozen CLIP encoder discards binding information that a 683-parameter model reads off the pixels.** That is the mechanism behind bag-of-words behaviour, stated in the metric CLIP actually uses.

6.5 **Limitations, declared.** This measures **salience in the geometry, not recoverability by a probe** — a linear probe might still extract binding from a low-variance direction; for a contrastive retrieval model the geometric claim is the relevant one, but they are different claims and must not be conflated. **Frozen image tower only**; the full image–text setting ARO probes is not tested here. The `content` floor changes *both* objects, so `0.924` is the conservative choice. Composites are synthetic and CLEVR renders are somewhat out of distribution for CLIP — which is what the void guard exists to catch, and it passed.

6.6 **CLIP is not literally blind — it is 13× less sensitive.** `swap` and `jitter` differ by `0.0050` against CIs of ~`0.001`, so the swap *is* detected. The honest phrasing is "a compositional swap produces 7.6% of the embedding movement of a content change", **not** "CLIP cannot see it".

## II.7 What the quantum node buys and costs (1 p)

7.1 **Buys** — parameter efficiency, replicated on two datasets. The unitarity constraint is not a handicap at these scales.
7.2 **Costs** — optimisation instability is the binding constraint, not capacity. Collapse rates 12/20 and 5/8, in two modes needing different fixes. **Collapse is characterised, not fixed** — report the rate beside every converged-seed figure.

## II.8 Methodological findings (1 p)

- Two code audits, each invalidating a body of results, each re-run rather than assumed: the **scalar-readout bottleneck** and the **non-coherent tree**.
- Standing rules earned from specific failures: matched-parameter controls from day one; a stated MDE with every comparison; **sweep, never derive, any capacity parameter**; **viability may only be judged by a classical arm** (a model arm may not certify its own task impossible).
- `default.mixed` silently returns wrong values under parameter broadcasting in PennyLane 0.43.2.
- **The figure audit belongs here**: 11 of 20 figures were compromised, and the response was to regenerate from a single `make_figures.py` with the MLP reference line enforced in code rather than by discipline.

---

# PART III — INTEGRATION WITH THE QUANTUM-INSPIRED LANGUAGE MODEL (6 pp)

> **First paragraph sets expectations**: the *classical* vision tower is integrated and evaluated on ARO. The *quantum* tower is not — the design is specified and the blocker is costed.
> **Scope note in the text**: this part evaluates compositionality, not retrieval. Say it as a choice, since compositionality is the property the thesis is about.

## III.1 The existing language model (1.5 pp)

1.1 DisCoCat and lambeq: a CCG parse becomes a contraction diagram, so **sentence structure determines topology, not sequence length**.
1.2 **The structural symmetry that motivates the thesis**: both towers are tensor-network contractions — one over syntax, one over space. Neither is a sequence model.
1.3 The sparsity problem: 68,922 symbols at ~5 occurrences each. Responses: CP-factored symbols, tied-noun embeddings.

## III.2 The two-tower VLM and its training setup (2 pp)

2.1 Per-modality alignment heads, no shared projection weights, symmetric InfoNCE, fixed temperature. ARO configuration: three forward passes per sample, InfoNCE + triplet with the triplet term dominant.
2.2 **Hard negatives are the whole design.** ARO negatives are syntactic perturbations — the same words reordered — so the negative is *only* separable by structure. **State the symmetry with §II.5 explicitly: the binding probe is this task's vision-only analogue**, with marginals controlled by construction instead of by perturbation. That parallel is why Parts II and III belong in one document.
2.3 **The strictly-linear constraint and where it was violated** — the image tower is purely multilinear; the text tower's NLC variant adds a gated GELU. The gate saturated, and stacking it with an MLP head produced conflicting gradients.
2.4 Training configuration stated once for reproducibility.

## III.3 Compositionality evaluation (2 pp)

3.1 **ARO against frozen CLIP.** This is where the headline claim lives. **§II.6 now supplies its mechanism from the CLIP side** — cite it directly here, since a measured 13× geometric insensitivity is a far better explanation than an appeal to architecture. And be exact about what §II.5 removed: the advantage is **not** attributable to the tensor network binding better, because in a controlled vision-only test it doesn't.
3.2 **Negative results, as a table** — restricted to those bearing on the ARO configuration: alignment warmup → embedding collapse; learnable temperature → collapsed to ~0.015; hard-negative mining → widened the train/val gap; NLC + MLP head → conflicting gradients; frozen-CLIP text-only → cosine ~−0.0016, the text model learned nothing. This table earns trust in §II.5's negative result.
   *(COCO-specific failures move to Appendix C under the current scope.)*

## III.4 Integrating the quantum vision tower (0.5 pp)

4.1 **The design**: the quantum tower's classification head becomes a `Linear(4, embedding_dim)` alignment head with L2 norm; everything upstream unchanged. No NaN/infeasibility machinery is needed — circuit topology is fixed and resource use bounded, unlike the classical einsum path.
4.2 **Why it has not been run, costed**: three forward passes per sample against ~62 min per quantum seed, on a task needing far more than 1024 samples, at a multi-seed budget forced by the collapse rate. An engineering blocker with a known shape.
4.3 **What would make it viable**, in priority order: fix the optimiser (collapse rate is worth more than seed count — the difference between ~40 and ~9 seeds per arm); SPSA for depth-3+; higher bond dimension, the principled fix for the readout bottleneck.

---

## 4. Conclusion and future work (1 p)

Restate the three answers from §0.3, in the order that makes the honest claim: the quantum tower is more parameter-efficient than its classical counterpart; the compositional advantage that motivated the architecture does not appear in the vision tower under a controlled test; but the frozen baseline is measurably, quantifiably worse at exactly the thing the benchmarks probe. **The advantage over CLIP is real; its source is not the tensor network's inductive bias.** Open threads: the C6 implicit-positional array; the quantum tower's integration; a rebuilt binding probe on `material`; and extending C7 from the frozen image tower to the full image–text setting.

---

## Appendices

- **A** — Rejected mechanisms in full: five residual methods, the spatial ancilla, the noise sweeps. Figures **F13–F16**, each with its caveat stated in the caption.
- **B** — Full Experiment & Metrics Record, with the three protocol boundaries marked. Do not compare across a boundary.
- **C** — Superseded and out-of-scope results: the hybrid tree, the underpowered protocol, **the 11 audited figures** (`results/superseded/`, with its README), the shape-binding build (**F17**), and the COCO retrieval line. Keeping these visible makes the scope decisions reversible.
- **D** — Reproducibility: conda env `qnlp`, `qttn_core.CoherentQTTNClassifier`, `phase15_common` / `clevr_common`, `make_figures.py`, `run_c7_clip_geometry`, cluster recipes, 65 regression tests, cost model. Additional figures **N3–N5**.

---

## Working notes for the drafting sessions

**Order to write in**: Part I §I.3 → Part II (§II.2, §II.5, §II.6 first — freshest and most settled) → Part I §I.1 → Part III → introduction and conclusion last.

**Figure work, in priority order:**
1. **N1** — C6 binding vs shuffled paired bars. The project's strongest construction result has no figure.
2. **N2** — C7 cosine geometry with CIs. The other headline; only the input montage exists.
3. ✅ ~~**R1** — regenerate the encoding × ansatz figures~~ **DONE 2026-08-08** (F19, F20).
4. Caption fixes to **F3/F4** (rescope the barren-plateau claim) and **F14** (`p_crit` definition). While fixing F14, reconcile the `p_crit` disagreement recorded under R1 — it is documented but still unresolved.
4. **N3/N4** if space allows; **N5** optional.
5. Verify **F6/F7/F8** are current — they were regenerated on 2026-07-29 and predate every CLEVR result, so they cover Phase 1 only. That is correct for where they are cited, but confirm rather than assume.

**Blocking on you**: §III.3.1 needs the ARO numbers — per-subtest scores for the tensor-network VLM and the frozen-CLIP comparison, and which run is of record.
**Non-blocking**: the C6 quantum seed array. §II.5 stands on current data.

**Standing style rules:**
- Every accuracy gets a parameter count beside it.
- Every comparison gets its MDE. "No significant difference" without the minimum detectable effect is not a finding.
- Report per head; never average across colour/shape/material/size.
- Say "not resolved" where it is not resolved.
- **Every accuracy figure carries the MLP reference line** — enforced in `make_figures.py`, not left to discipline.
