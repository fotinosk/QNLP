# Quantum Image Tower — Investigation Summary

**A settled account of the investigation, 2026-07-17 to 2026-08-04.**

This document states the investigation's findings at their final confidence, in the order that makes them comprehensible rather than the order they were discovered in. Superseded results, retractions, and the sequence of corrections are **not** reproduced here — they are in `research_log.md`, which remains the record of what happened and when.

What *is* reproduced here: every caveat that still qualifies a live claim. A summary that dropped those would be shorter and wrong.

**Companion documents**: `research_log.md` (chronological record, including everything superseded), `quantum_investigation_roadmap.md` (research questions and task plan), `quantum_implementation_plan.md` (architecture and design rules), `thesis_outline.md` (write-up structure).

---

## 1. The question

A classical tree tensor network contracts four child nodes into a parent through a CP-decomposed linear map of rank *R*. A quantum tree replaces that map with a parameterised **unitary** followed by a **partial trace**. The investigation asks what that substitution costs and what it buys — and, one level up, whether either kind of tree captures *compositional* structure in images the way the project's premise requires.

Three questions were answerable and got answers:

- **A.3 — expressibility.** Does the unitarity constraint reduce representational capacity relative to an unconstrained CP node at matched parameter count? **No. It improves it.**
- **C.2(i) — implicit position.** Does a tree's fixed topology encode spatial relations without any explicit positional encoding? **Yes.**
- **The compositional premise.** Do tensor networks bind attributes to positions better than a bag-of-features model? **They bind. They do not bind better.**

One question was closed on principle rather than answered: whether classical components inside the pipeline could substitute for quantum circuit capacity. The project's direction is a purely quantum implementation, so the boundary was made checkable instead (§4.3).

---

## 2. The architecture

### 2.1 The model of record

```
Image (16×16)  →  16 patches of 4×4×3
       │
       ▼   Linear(48, 3) per patch, tanh·π          [classical I/O boundary]
  3 rotation angles per patch
       │
       ▼   ONE 16-qubit device; patch p on wire p
  RX, RY, RZ per wire                               [multi-axis encoding]
       │
       ▼   Level-1: IQP block unitaries on
           [0-3] [4-7] [8-11] [12-15], per-block weights
  survivors 0, 4, 8, 12 stay QUANTUM (no measurement)
       │
       ▼   Level-2: IQP unitary on [0, 4, 8, 12]
       │
       ▼   ⟨Z⟩ on wires 0, 4, 8, 12                 [readout = top_layer_qubits]
  Linear(4, n_classes)                              [classical I/O boundary]
```

**287 parameters.** Runs on `lightning.qubit` with adjoint differentiation, ~17.5 min per 30-epoch run on synthetic shapes.

Code: `qttn_core.CoherentQTTNClassifier`, config `phase15_common.COHERENT_ARCH`, regression-tested.

### 2.2 Two properties that are load-bearing

**The tree is coherent.** Survivors pass between levels as *qubits*, never measured and re-encoded. This is enforced by a regression test rather than by convention: a survivor's Bloch vector must be shorter than 1, measured at `0.38–0.73` after training against exactly 1 for a product state. Inter-level entanglement is worth **+9.9 points** over an otherwise identical tree that measures between levels.

**The readout is four wires, not one.** Reading only the root qubit scores `35.6%`; reading the four top-layer wires scores `78.4%` — **a 43-point gap**, because the root is a single qubit's marginal of a 16-qubit state. The tower does not contract to a single root. This is a documented workaround for a χ=1 simulation limit, not a design preference; higher bond dimension is the principled fix and is affordable on hardware but not under statevector training.

### 2.3 Why these components

Each element of the ansatz follows from a measurement, not a preference:

| choice | evidence |
|---|---|
| 3 CNOTs per node ("all children to parent") | Entropy transfer saturates: 0→6 CNOTs moves root entropy `0.0000 → 0.5139` against the `ln 2 ≈ 0.693` ceiling; 3 CNOTs capture 73%, with clear diminishing returns after. |
| `multi_axis` encoding (RX/RY/RZ) | `+12.6` pts over single-axis `scalar_ry`, against a 3.6-pt resolution limit — resolved. |
| IQP ansatz | `+1.6` over strongly-entangling against a 4.8-pt limit — **not resolved**. IQP is chosen for being cheaper, not better. |
| `top_layer_multi_pauli` readout on CLEVR (12 values) | Chosen on **stability**: 0/3 seed collapses vs the narrow readout's 1/3, and per-head standard deviation 5–35× tighter. Costs ~3× under adjoint differentiation (one backward pass per observable) — free on hardware, not in simulation. |

---

## 3. Theory and simulation

### 3.1 The TTN↔VQC correspondence

**CP-rank maps to entangling gate density.** Local operations (rotations, Hadamards) are LOCC: they cannot change the Schmidt rank across a register partition, and are the quantum analogue of a classical CP factor matrix. Non-local operations (CNOT, CZ) raise it. So sweeping CP rank corresponds to sweeping CNOT count, and the entropy measurement above fixes the operating point.

**Partial trace maps to pooling.** One qubit per node gives χ=2, so the entanglement bound at every level is `ln 2`. This match to the classical area-law bound is **architecturally guaranteed by the single-qubit bottleneck, not an empirical finding** — a single qubit's entropy cannot exceed `ln 2` regardless of tree depth. The non-guaranteed result is that root entropy climbs from `72.7%` of that ceiling at depth 1 to `89.7%` at depth 2: deeper trees generate entanglement that saturates the fixed bottleneck more fully. Two depths were measured.

### 3.2 Trainability

Gradient variance across N ∈ {4, 9, 16, 20} qubits stays between `5.94e-2` and `1.51e-1`. Under an exponential barren plateau, variance at N=20 would be ≈`9.5e-7` — the measured value is **more than 62,000× larger**, and the log-log trend is consistent with `O(1/poly(N))`. Hierarchical topology with local observables is trainable at these widths.

MPS and MERA show the same immunity at N=20. MERA reaches higher accuracy (`51.6` vs QTTN `42.2` and MPS `43.0`) because its disentanglers capture correlations across block boundaries that a tree misses; QTTN is the most noise-tolerant of the three and MPS the least, consistent with MPS's linear depth accumulation. *(Measured under an early, underpowered protocol — see §7.1.)*

### 3.3 Simulation limits, and what they fix

These determine every experimental scale in the project:

- **Qubit recycling is exact** — a 16-qubit tree reduces to 7 physical qubits with an output difference of `0.0`. It remains valid as a *hardware* qubit-budget technique, but does not extend past depth 2 in simulation: deferred measurement allocates an ancilla wire per measurement, and tree-traversal blows up as ~2^resets (depth 3 needs ~50 resets; 24 already hangs).
- **Tensor-network simulation runs forward passes far past statevector limits** — `default.tensor(method='tn')` does depth-3 (64 qubits, 32×32) in 0.13 s and depth-4 (256 qubits, 64×64) in 1.20 s. But it has no backprop, and parameter-shift costs 56.7 s per gradient step at depth 3. **Forward-feasible, training-infeasible.**
- **Statevector autograd walls at N≈25** — PyTorch caches a 2^N × 16-byte statevector at every gate, so a 25-qubit circuit with 200 gates needs >100 GB.
- **Parameter broadcasting gave a ~30× speedup** (>30 s per batch down to 33 s per epoch) and is what makes the project affordable at all.

**Net effect: 16×16 images, depth-2 trees, χ=1 is the only configuration with validated end-to-end training**, and every result in this document is at that scale.

---

## 4. Design rules, settled

### 4.1 No residual or skip connections

Five quantum-native mechanisms were tested at the architecture of record over 30 seeds:

| mechanism | result | verdict |
|---|---|---|
| data re-uploading | `+1.5` vs baseline, limit 3.8 | **true null** — no demonstrated benefit at effects ≥ 3.8 pts |
| near-identity initialisation | no measurable effect | affects early dynamics only; washes out by convergence |
| mixed-unitary channel | `−10.9`, limit 4.7 | worse, resolved. 0/30 training failures — not unstable, just worse |
| ancilla-controlled soft mixing | inherits the same behaviour | rejected |
| LCU / LCU-lite | same, plus real shot overhead (root discards ~half of trials) | rejected |

Baseline for these comparisons: `79.4 ± 8.0`, 0/30 failures.

The classical CP tower *does* use an explicit residual, and it is strongly load-bearing there (`+31.7` pts). The asymmetry is a genuine finding: the classical analogy does not transfer.

### 4.2 No explicit spatial ancilla, with a scope condition

On translation-invariant single-object classification the explicit positional ancilla costs **`−19.5` pts against a 3.8-pt limit** — resolved, and decisive for that task class.

**This does not generalise to "positional encoding is harmful."** The task is one where position barely affects the label, so a negative result is close to structurally guaranteed, and the tested ancilla is per-quadrant (4 positions) rather than per-patch (16). The relational question is settled separately and came out differently in one half (§6.2).

Consequence for qubit budget: the ancilla's removal takes a 16-patch model from 80 qubits to 64, or to 16 under the multi-axis encoding actually used.

### 4.3 Classical components: a checkable boundary

**The classical encoder may set the parameters of state preparation, but may not reduce the number of qubits the architecture would otherwise require.**

| case | verdict |
|---|---|
| `Linear(48,3)` → RX, RY, RZ on one qubit | **Permitted** — the architecture assigns one qubit per patch; the encoder fills that qubit's three rotation parameters and cannot express more than the circuit consumes. |
| `Linear(16,4)` → 4-qubit VQC for data needing 16 qubits | **Prohibited** — reduces qubit count. |
| Classical scalars carried between tree levels | **Prohibited** — replaces a quantum bond outright. |

This resolves what looks like a tension: widening the patch encoder from `Linear(48,1)` to `Linear(48,3)` bought `+12.6` pts, a larger effect than any quantum architectural change measured in the project. That gain was **not** added classical capacity — single-axis encoding left two of each qubit's three rotation parameters unused. The improvement came from using the *quantum* resource fully.

### 4.4 Noiseless only

The project proceeds without noisy emulation, on evidence rather than convenience:

- Tolerance is characterised: stable to p≈0.02, degrading past p≈0.05.
- The mechanism is understood: depolarizing noise contracts expectation values multiplicatively, `⟨Z⟩_noisy ≈ (1−p)^d ⟨Z⟩_clean`, which shrinks values toward zero but **preserves sign and relative order** — so an argmax decision boundary survives.
- Training under noise does **not** help: `55.0 ± 3.5` noiseless against `54.1 ± 5.0` at both p=0.02 and p=0.05, evaluated clean throughout. The classical-dropout analogy does not hold. (An eval-time effect where noise *improves* under-trained models is real and repeatedly observed, but is a different phenomenon.)

**Standing limitation: every noise result characterises a single 4–5 qubit node, never the full tree.** `default.mixed` is O(4^N) and a 16-qubit noisy circuit needs ~68 GB. This is stated in the write-up, not fixed.

---

## 5. Phase 1 — synthetic shapes

**Protocol**: 16×16 overlapping-mode shapes, 1024 train / 64 test / 30 epochs, scored on the last-5-epoch mean, unpaired Welch comparisons, minimum detectable effect reported alongside every comparison.

**Result** (4 seeds):

| model | score | params |
|---|---|---|
| `mlp_reference` | `96.0 ± 1.5` | 999 |
| **`quantum_coherent`** | **`89.3 ± 3.9`** | **287** |
| `classical_full` (CP + residual + dropout) | `88.4 ± 5.5` | 352 |
| `classical_bare` (CP) | `79.6 ± 10.3` | 428 |
| `mlp_param_matched` | `70.5 ± 18.1` | 257 |

Resolved: beats the bare classical CP tree by `+9.7` (limit 6.3) and a same-size MLP by `+18.8` (limit 9.0); **ties** `classical_full`. Best score-per-parameter in the project.

**Question A.3 is answered.** Under matched constraints — no residual, no dropout on either side — the unitarity-constrained quantum node beats the unconstrained classical CP node while using **33% fewer parameters**. The unitarity constraint is not a handicap at this scale.

**Two caveats that must travel with this table:**
1. The classical arms were hyperparameter-tuned while the quantum arm inherited `lr=0.03` from an earlier task. The quantum figure is therefore **conservative**.
2. `classical_bare` moved `33.9 → 56.7 → 79.6` across three revisions of its search space. Deriving a capacity parameter (CP rank) from a parameter budget silently forced rank=1 and produced three different "measurements" of the same baseline. The `mlp_reference` arm is what caught this, and the rule that came out of it — **sweep, never derive, any capacity parameter** — is now enforced in the figure code.

---

## 6. Phase 2 — CLEVR

**Data**: individual objects cropped out of full CLEVR scenes (the dataset has no 1- or 2-object scenes; every scene has 3–10). Crops are world-scaled by depth so the fill fraction is depth-invariant — 0.250 for small objects, 0.500 for large, matching the analytic prediction exactly. This is what keeps `size` learnable at 16×16. Relation labels use CLEVR's **camera-rotated** basis (~49° off the world axes), not world x/y.

**Resolution**: 16×16 is the resolution of record. All four attribute heads are learnable there (classical reference `95.2 / 69.9 / 82.5 / 99.0` on colour/shape/material/size), and 32×32 is statistically tied on every head, so the smaller size wins on cost. No scaling route — bigger patches, SPSA, or higher bond dimension — is required.

**Protocol**: 1024 train / 512 val, 30 epochs (90 where stated), last-5-epoch mean, unpaired Welch, MDE reported. Four attribute heads share one summed loss.

### 6.1 Single-object attributes

At 30 epochs the quantum tower (462 params) beats `classical_full` (551 params) on **shape `+12.7`** and **material `+8.8`**, both resolved — Phase 1's parameter-efficiency result reproducing on real data.

At 90 epochs the picture **trades rather than improves**: colour rises `27.1 → 63.0` (resolved), while material falls `−9.7` and size `−2.0`, both resolved. Four heads sharing one summed loss converge at very different rates. This is a finding about multi-head training, not a nuisance.

The classical arms were also run at 90 epochs: `classical_bare`'s colour moves `52.5 → 73.4`, confirming it had not converged at 30 either. But variance explodes (`±26.2`, `±20.3`), so the quantum arm's 90-epoch colour sits *inside* the classical spread rather than below it.

**Neither budget is privileged. Quote both tables or neither.**

⚠️ **`mlp_reference` beats every TTN arm — quantum and classical — on every head.** The parameter-efficiency claim is about the quantum node versus the classical CP node. It is not a claim about beating classical vision, and must not be written as one.

### 6.2 Relations, and Question C.2

The relational task (left/right/front/behind, 4-way, 26.6% majority floor) is learnable by the tensor-network family: `classical_bare` reaches `49.6 ± 5.6` at **428 parameters against the quantum tower's 462** — the direct structural counterpart, clearing the floor by ~23 points. `mlp_reference` reaches `65.4 ± 14.7`.

The quantum tower reaches parity **on the seeds that train**:

| arm | all seeds | collapsed | converged only |
|---|---|---|---|
| `quantum_none` (462 params) | `31.2 ± 10.5` | 7/10 | **`46.2 ± 2.5`** |
| `quantum_on_wire` (494 params) | `36.4 ± 12.4` | 5/10 | **`48.0 ± 3.8`** |
| `classical_bare` (428 params) | `49.6 ± 5.6` | 0/10 | — |

Converged `quantum_none` vs `classical_bare` is `−3.4` against a 5.2-pt limit; `quantum_on_wire` is `−1.6` against 5.2. **Both unresolved — statistical parity.**

**Question C.2(i) is answered: yes.** `quantum_none` has *no* positional parameters at all — position enters only through the fixed patch→wire assignment — and it does a relational task at classical parity. The implicit tree topology carries enough spatial information. This discharges the scope condition attached to §4.2.

**Question C.2(ii) is not answered.** Explicit position adds `+5.3` against a 10.7-pt limit across all seeds, `+1.8` against 5.4 on converged seeds — resolving that would need 27–38 seeds per arm. Report it as a **bounded null quoted at its MDE**, which is a legitimate finding supporting the omission of explicit positional encoding.

⚠️ **Read the converged subgroup as a diagnostic, not an arm** — it is post-hoc conditioning. ⚠️ These runs used 30 epochs where §6.1 shows ~90 are needed, and several converged seeds were still climbing at the cutoff, so **these numbers are a floor on capability, not a ceiling**.

### 6.3 The compositional binding probe

**Why it was necessary.** Attribute classification is pure perception — one object, one label, nothing to bind. The relational task is "locate the off-centre object, report its direction" — spatial, but still a single-referent readout. Neither has the property that makes contrastive models fail on ARO: *the same bag of features arranged two ways, yielding two different labels*. So neither can discriminate the project's premise, and a task that could had to be built.

**Design.** Synthetic composites, one object per side, label = which side carries the bound attribute. **Class-conditional marginals are identical by construction**, so global feature content carries zero information and only the binding of an attribute to a position separates the classes — a bag-of-features model is at chance *provably*. Object placement is **jittered, not aligned** to the tree's 2×2 level-1 blocks; aligned placement would hand each block exactly one object, which is the TTN's claimed inductive bias, and would rig the test.

**The bound attribute must be one every arm can perceive.** A binding task is a conjunction of perception and binding, so an arm that cannot see the attribute scores at floor for reasons that have nothing to do with composition. `size` satisfies this (single-object accuracy spans 84.8–99.1 across all arms); it was chosen for that reason.

**The manipulation check — the strongest construction result in the project.** Under patch shuffling every arm falls to *exactly* chance; unshuffled, every arm reaches 91–96%.

| arm | params | binding | shuffled | gap |
|---|---|---|---|---|
| `mlp_reference` | 1397 | `96.0 ± 1.3` | `50.2 ± 2.1` | **+45.8** |
| **`mlp_param_matched`** | **683** | **`94.6 ± 2.9`** | `49.3 ± 2.7` | +45.3 |
| `classical_full` | 942 | `93.2 ± 1.4` | `49.3 ± 2.7` | +43.9 |
| `classical_bare` | 930 | `91.5 ± 3.1` | `51.0 ± 1.4` | +40.5 |
| *best quantum seed (`on_wire`, 8 seeds)* | 757 | *`73.8`* | — | — |

A **+40.5 to +45.8** point gap that cannot come from unbound features, since the marginals are identical by construction. **Any score above chance on this task requires binding an attribute to a position.**

**The four statements the data supports:**

1. **Tensor networks perform compositional binding** — classical and quantum — on a task where a bag-of-features model is provably at chance. `classical_bare` `91.5 ± 3.1`, `classical_full` `93.2 ± 1.4`, over 10 seeds.
2. **They do not do so better than a parameter-matched MLP.** A **683-parameter** MLP — fewer parameters than any tensor-network arm — beats `classical_bare` by `+3.1` (limit 2.8, resolved) and ties `classical_full`.
3. **The quantum tree demonstrates the capability but reaches it unreliably.** One seed reaches `73.8` against a `51.0` floor — about **10 binomial standard deviations** on 512 validation samples, so not luck, and on a marginal-controlled task it cannot be anything but binding. This is an *existence* claim about architectural capacity, which is the kind that survives post-hoc seed selection. But 5 of 8 seeds sat at floor: **capability, not reliability.**
4. **The task saturates.** Total spread across arms is 4.5 points against MDEs of 1.3–2.3, so it can show *whether* an architecture binds, not *how well*. It cannot rank architectures finely in either direction.

⚠️ **Never write that the MLP cannot capture these relations.** The shuffle control proves that a *bag-of-features* model is at chance. `MLPReference` is not one — it flattens **positioned** patch embeddings into a fully-connected trunk, so it carries position and can bind. The control validates the **task**; it never validates an architecture's inability.

⚠️ The demonstrated quantum seed is the `on_wire` variant, *with* explicit positional parameters and **+32 parameters** over the implicit-only arm, so any positional advantage is confounded with capacity. The implicit-only arm has **no data** — its full seed array was lost to a cluster failure and is being re-run. **This is the only conclusion the re-run can change**: one clearly-above-floor implicit seed would extend the capability claim to the architecturally interesting variant.

### 6.4 What this says about the premise

The project's premise is that a frozen contrastive baseline fails on compositional tasks where the tensor network succeeds. **The binding probe neither supports nor contradicts that headline** — the frozen baseline was never run in the vision tower, and the headline rests entirely on the language-side ARO evaluation.

The probe was designed to supply the headline's *mechanism* — "the tensor network's advantage is binding, not perception" — and it **does not**. Within the vision tower there is no compositional advantage for tensor networks over a plain MLP. Both outcomes were pre-registered before the numbers came in; this is the branch that fired.

---

## 7. Method and protocol

### 7.1 Standing requirements

Each was adopted after a specific failure, and each is now enforced in code or in the figure pipeline:

1. **A matched-parameter classical control alongside every quantum result.** Without one, "our model scores X%" is uninterpretable — and a mis-specified baseline once produced a fake 57.8-point advantage.
2. **A stated resolution limit (MDE) with every comparison.** "No significant difference" without the minimum detectable effect is not a finding. Three results in this project are true nulls and are reported as bounded nulls at their MDE.
3. **Sweep, never derive, any capacity parameter.** Deriving CP rank from a parameter budget silently forced rank=1.
4. **Viability may only be judged by a classical arm.** A model arm may not certify its own task impossible; unassessed is not the same as failed.
5. **Report per head; never average** across colour/shape/material/size.
6. **Every accuracy carries a parameter count.**

**Protocol boundaries.** Results are never compared across these:
- An early 256-sample / 15-epoch protocol was underpowered for this task regardless of architecture. The topology and early ansatz benchmarks sit here.
- The synthetic-shapes protocol of record: 1024 train / 64 test / 30 epochs.
- The CLEVR protocol: 1024 train / 512 val, four simultaneous heads, 30 or 90 epochs.

### 7.2 Cost model

| run | cost |
|---|---|
| quantum seed, synthetic shapes, 30 epochs | ~17.5 min |
| quantum seed, CLEVR, 30 epochs | ~62 min |
| quantum seed, CLEVR, 90 epochs (cluster, `default.qubit` fallback) | 14–27 h |
| classical arm with tuning grid | ~5 h per invocation (~36 configs × 3 seeds), **not free** |

Long and multi-seed runs go to the SGE cluster as array jobs; seed sharding maps directly onto the array layout that the merge tooling expects. Local runs are capped at 4–5 concurrent workers, and contention has corrupted measurements more than once — trust timings only when nothing else is running.

### 7.3 Tooling findings worth reporting

- **PennyLane 0.43.2 `default.mixed` silently returns wrong values under parameter broadcasting.** No error, no warning.
- The `default.tensor` device supports exact parameter-shift gradients (matching backprop to 1e-11) but no backprop, which is the binding constraint on deeper trees.
- Two guard failures in opposite directions: one permitted a model arm to certify its own task impossible; one was blunt enough to discard an entire valid run over a single weak arm. Both were caught by reading output, not by tests.
- A power-analysis helper over-reported required seeds by 5× before being fixed.

---

## 8. Status, limitations, and what would come next

### 8.1 Status

**Experimentation is closed** except one quantum seed array for the binding probe's implicit-positional arm, which is being re-run after a cluster wall-clock failure. Everything else is complete; the remaining gaps below are declared limitations, not open threads.

### 8.2 Limitations, all declared rather than discovered

- **Scale.** 16×16 images, depth-2 trees, χ=1 is the only end-to-end trained configuration. Deeper trees are forward-simulable but not trainable.
- **Noise.** Every noise result characterises a single 4–5 qubit node, never the tree.
- **Optimisation stability is the binding constraint, not capacity.** Collapse rates reach 12/20 on relations and 5/8 on binding, in two modes needing different fixes: *learned-then-diverged* (a learning-rate/schedule failure — one seed peaked above the classical mean at epoch 12 before falling to chance) and *never-left-chance* (an initialisation failure). **Collapse is characterised, not fixed**; the rate is reported beside every converged-seed figure so the conditioning stays visible.
- **The binding probe saturates**, and its composites are synthetic. The natural relational task is the ecological-validity companion.
- **The `on_wire` arm carries +32 parameters**, so any positional advantage is confounded with capacity.
- **Question C.2(ii) will likely end without a verdict**, since a comparison between two arms that both sit at floor measures nothing.
- **Two epoch budgets, neither privileged.** Both tables must be quoted.

### 8.3 Next steps, in priority order

1. **Fix the optimiser.** Collapse rate is worth far more than seed count: a pooled standard deviation of `11.5` (contaminated) versus `~3` (converged) is the difference between ~40 and ~9 seeds per arm, i.e. between ~260 h and ~60 h of cluster time. Sweep learning rate and epochs **together** — a longer budget gives an unstable run more time to diverge.
2. **SPSA on the tensor-network device**, which is what enables genuinely deeper trees.
3. **Higher bond dimension**, the principled fix for the readout bottleneck — affordable on hardware and under tensor-network simulation, not under statevector training.
4. **Rebuild the binding probe on `material`**, the one attribute with both adequate perception coverage across arms and remaining headroom. `size` equalised perception at the cost of most of the dynamic range.
5. **Integrate the quantum tower with the language model.** Never attempted; the design is specified and the blocker is cost — three forward passes per sample against ~62 min per seed, on a task needing far more than 1024 samples, at a multi-seed budget forced by the collapse rate.

---

## 9. Reproducibility

- **Environment**: conda env `qnlp`. Cluster: `/SAN/intelsys/discoviz/envs/qnlp311/bin/python`, project at `/SAN/intelsys/discoviz/fotinos/QNLP`.
- **Model**: `qttn_core.CoherentQTTNClassifier` — do not write a new model class; per-script model drift is what both code audits found.
- **Config**: `phase15_common.COHERENT_ARCH`.
- **Harnesses**: `phase15_common` (Phase 1), `clevr/clevr_common.py` (Phase 2).
- **Data**: `qnlp/utils/data/clevr_objects.py`, `qnlp/utils/data/synthetic_shapes.py`.
- **Merging sharded cluster runs**: `clevr/combine_clevr.py`.
- **Tests**: `test_qttn_core.py` + `clevr/test_clevr.py`, 65 tests.
- **Cluster setup, job scripts, merge recipes**: `clevr/CLUSTER.md`. Phase-2 orientation: `qnlp/image_tower/classification/clevr/README.md`.

⚠️ **Figures**: a 2026-07-28 audit found 11 of 20 figures compromised by the readout bottleneck. Use only the regenerated set; superseded figures are retained under `results/superseded/` with a README explaining each.
