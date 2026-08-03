# Phase 2 — CLEVR

Implementation of `llm/quantum_investigation_roadmap.md` Section 3, tasks C0–C5.
Read that section and the `research_log.md` Current Status block first.

## Two corrections to the written plan

1. **CLEVR has no 1- or 2-object scenes** (3–10 per scene, always). The plan's
   "filter to scenes with exactly one object" returns zero rows. Datasets are
   built by **cropping objects out of full scenes** instead.
2. **The plan's relation helper uses world axes.** CLEVR's left/right/front/behind
   are defined against the **camera-rotated** basis, ~49° away. Use
   `clevr_objects.relation_label`.

## Standing requirements (each from a specific Phase-1 failure)

1. A matched-parameter classical control beside **every** quantum number.
2. The resolution limit reported with every comparison (`pc.compare` does this).
3. Noiseless only.
4. **Sweep, never derive, any capacity parameter.** Deriving CP rank from a
   parameter budget silently forced rank=1 and produced three different
   "measurements" of the same baseline.

## Status

| task | state |
|---|---|
| **C0** data | ✅ done — crops cached at 16/32/64, both montages inspected |
| **C1** learnability gate | ✅ **passed** — all four heads learnable, none dropped |
| **C5** resolution route | ✅ **closed, no work needed** — 16×16 suffices |
| **C2** readout width | ✅ done — **adopt `top_layer_multi_pauli`** (12 values), on stability |
| **C3** attributes | ✅ done — quantum beats `classical_full` on shape/material at fewer params |
| **C3b** quantum lr sweep | 🖥️ **ready to submit** — `qsub scripts/submit_c3b_lr_sweep.sh` (4 lrs × 3 seeds). Un-dropped: it was cut for *local* cost only |
| **C3c** epoch budget | ✅ done — **the budget was binding**: colour `27.1 → 63.0` at 90 epochs, but it trades heads |
| **C4** relational | ← next, and required (Question C.2) |

### 📌 C3c — train one promising seed for longer

```bash
$PY -u -m qnlp.image_tower.classification.clevr.run_c3_attributes \
    --only-quantum --seeds 0 --epochs 90 --out c3long_s0 --out-suffix _long90
```

**Seed 0 only.** It is the one seed still improving when the budget ran out
(`+0.750` pts/epoch on colour over the last 10 epochs, best overall mean 64.7);
seeds 1 and 2 had plateaued at −0.245 and −0.182. ~9 h unattended (60 epochs
≈ 6 h is also informative).

**Settles**: whether colour's 27% is the epoch budget or the architecture. The
30-epoch budget came from a **4-class** synthetic task and was never re-examined
for an **8-way** head.

**Diagnostic only** — one seed cannot be quoted as an arm. If it shows the budget
binds, the whole C3 comparison has to be re-run at the longer budget so the
classical arms get the same extension.

### C3 headline (16×16, readout `top_layer_multi_pauli`)

| arm | params | color | shape | material | size |
|---|---|---|---|---|---|
| **`quantum_coherent`** | **462** | 27.0 | **58.9** | **70.0** | 96.1 |
| `classical_full` | 551 | 76.6 | 46.3 | 61.2 | 97.7 |
| `mlp_reference` | 1186 | 90.4 | 64.5 | 81.8 | 99.1 |

> ⚠️ **THE TABLE ABOVE IS AT 30 EPOCHS AND IS BUDGET-SUPERSEDED.** C3c showed the quantum
> colour head had not converged at 30. At **90 epochs** colour goes `27.1 → 63.0` and the
> verdicts change: quantum beats `classical_full` on shape, **ties** on colour and material,
> loses on size. The classical arms were **not** re-run at 90, so neither budget is
> privileged — **never quote one table without the other.**

**Two things must accompany any quotation of these numbers:**

1. **Colour is under-trained, not capacity-limited — and separately confounded.**
   Two free checks rule out the capacity reading: `classical_full` wins colour
   through a **narrower** head (bond_dim 8 vs the quantum model's 12 values), and
   the quantum colour curve **peaks at epoch 29 of 30** while `size` plateaus by
   epoch 3. The 30-epoch budget came from a 4-class task and is too short for the
   8-way head. On top of that, the classical arms were tuned over 60 configs while
   the quantum arm ran at one inherited `lr=0.03` (sweep dropped by decision). So
   say "untuned and under-trained quantum vs tuned classical" — never "capacity
   limit". Shape and material are unaffected: wins achieved *despite* the handicap.
2. **`mlp_reference` beats every TTN arm on every head.** The claim is about the
   quantum-vs-classical *node* within the tensor-network family, not about beating
   classical vision.

**Resolution of record: 16×16** (`--img-size 16`, patch 4). C1's table:

| resolution | color | shape | material | size |
|---|---|---|---|---|
| **16×16** | 95.2 | 69.9 | 82.5 | 99.0 |
| 32×32 | 93.6 | 72.2 | 84.0 | 98.2 |
| *floor* | 15.2 | 35.4 | 50.2 | 50.6 |

16×16 and 32×32 are statistically tied on every head; the smaller wins on cost
and needs no C5 caveat. This retires the old prediction that `material` would be
unlearnable at 16×16 — that assumed downsampled *full scenes*, and C0 crops
objects.

## Order

```
C0 build  →  C1 gate  →  C2 readout  →  C3 attributes
                      ↘  [C5 — closed by C1]
                         C4 relational (+ ancilla2 only if C4 resolves)
```

C1 is **blocking**: if an MLP cannot learn an attribute at a resolution, no
quantum model will, and a quantum null there measures the data, not the
architecture.

## Two hyperparameter traps, both already hit

1. **Never run a CLEVR arm at a single inherited learning rate.** C1's first run
   declared 64×64 unlearnable on all four heads; it was one hardcoded `lr=0.01`,
   and `lr=0.003` scores 85–96%. `run_c1_learnability.py` now sweeps.
2. **The quantum arms inherit `lr=0.03` from Phase 1, where the loss had ONE
   head.** CLEVR sums four cross-entropies, so the effective gradient is ~4×
   larger. C2's narrow arm collapsed on 1 of 3 seeds. `combine_clevr.py` reports
   per-seed collapses precisely because an arm *mean* hides them.

## Commands

Use the env python directly — `conda run` buffers all output until exit.

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/envs/qnlp/bin/python

# C0 — once. Calibrate, build, then LOOK AT both montages.
$PY -m qnlp.image_tower.classification.clevr.build_clevr_crops --calibrate
$PY -m qnlp.image_tower.classification.clevr.build_clevr_crops
$PY -m qnlp.image_tower.classification.clevr.build_clevr_crops --relation-montage

# Guards
$PY -m pytest qnlp/image_tower/classification/quantum/test_qttn_core.py \
              qnlp/image_tower/classification/clevr/test_clevr.py -q

# C1 — classical only, minutes. GATES EVERYTHING BELOW.
$PY -m qnlp.image_tower.classification.clevr.run_c1_learnability --seeds 0 1 2

# C2 — readout width, at the C1-chosen resolution
$PY -m qnlp.image_tower.classification.clevr.run_c2_readout --img-size 16 --seeds 0 1 2 3 4

# C3 — classical arms first (seconds), then shard the quantum arm
$PY -m qnlp.image_tower.classification.clevr.run_c3_attributes --skip-quantum
$PY -m qnlp.image_tower.classification.clevr.run_c3_attributes --only-quantum --seeds 0 1 --out-suffix _w0

# C4 — Question C.2, required
$PY -m qnlp.image_tower.classification.clevr.run_c4_relational --seeds 0 1 2 3 4
```

### Sharding a quantum task across workers

A quantum seed costs ~62 min, so C2/C3/C4 are run one worker per seed and merged
afterwards. Distinct `--out` prefixes stop workers clobbering each other; each
checkpoints after every seed.

```bash
for S in 0 1 2; do
  nohup $PY -u -m qnlp.image_tower.classification.clevr.run_c2_readout \
      --img-size 16 --seeds $S --out c2_s$S > /tmp/c2_s$S.log 2>&1 &
done

# then merge -- this is where the actual result is produced, since a single-seed
# worker has no variance and so can resolve nothing on its own
$PY -m qnlp.image_tower.classification.clevr.combine_clevr \
    --prefix c2 --seeds 0 1 2 \
    --arms top_layer_qubits top_layer_multi_pauli --params 342 462
```

## 🖥️ Use the cluster for long, multi-seed runs — it costs nothing locally

**The local machine is the wrong place for anything long.** A quantum seed is
~62 min (~3 h at the 12-value readout), the laptop caps at 4–5 concurrent
workers, and contention there has already corrupted results twice this phase (a
timing measurement that reported 64 test samples costing *more* than 512, and a
C2 run whose wall time roughly doubled because diagnostics were running
alongside).

The project has an SGE cluster already set up — see any of `scripts/submit_*.sh`:

| | |
|---|---|
| submit | `qsub scripts/submit_<name>.sh` |
| project dir | `/SAN/intelsys/discoviz/fotinos/QNLP` |
| python | `/SAN/intelsys/discoviz/envs/qnlp311/bin/python` |
| job output | `/SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/` |
| resources | `#$ -l tmem=32G`, `#$ -l h_rt=12:0:0` (raise `h_rt` for long runs) |

**➡️ Full setup checklist, job scripts and merge recipes: [`CLUSTER.md`](CLUSTER.md).**

**Seed sharding is a natural SGE array job** — one task per seed, which is
exactly how `combine_clevr.py` expects the checkpoints to be laid out. No array
job exists in `scripts/` yet; sketch (untested):

```bash
#$ -t 1-10                      # one task per seed
SEED=$((SGE_TASK_ID - 1))
$PYTHON -m qnlp.image_tower.classification.clevr.run_c3_attributes \
    --only-quantum --seeds $SEED --out c3_s$SEED
```

then merge locally (or on the cluster) with
`combine_clevr --prefix c3 --seeds 0 1 2 ...`.

**What this unblocks.** Several decisions in the log were made *on local-cost
grounds* and should be revisited now, since none of them are expensive on the
cluster:

* **10 seeds instead of 3** for every quantum arm — C2 and C3 both ran at 3
  purely to bound wall time, and 3 seeds is why several C2 comparisons came back
  unresolved.
* **C3b, the quantum lr sweep** — dropped for cost. It is the one thing that
  would close the tuning-asymmetry confound on colour, which is currently a
  permanent stated limitation of the results.
* **C3c, the 90-epoch long run** — framed as "~9 h, expensive"; trivial as one
  cluster job.
* **C4's `ancilla2` escalation** — ~4× cost, currently gated behind "only if the
  free arm resolves".

**Two gotchas before submitting:**

* The `.npz` crop caches under `data/datasets/` (~140 MB) must exist on the
  cluster — rsync them or re-run `build_clevr_crops` there.
* Node `arbuckle` lacks AVX and crashes polars with SIGILL; the fix is
  `polars[rtcompat]`. The Phase-2 data path uses pyarrow rather than polars, so
  it should be unaffected — but `load_clevr_to_atlas` does use polars.

## Operational notes

* **At most 4–5 concurrent quantum workers, locally.** Ten concurrent 16-qubit
  `lightning.qubit` processes exhausted 18 GB in Phase 1 and six were killed
  silently. Concurrency also badly distorts timing measurements — a contended
  run measured 8× slower than the same config run alone. **On the cluster this
  limit does not apply; use one job (or array task) per seed.**
* Every runner checkpoints after each seed, so a killed worker costs one seed.
* **The Phase-1 cost model does not carry over.** See `research_log.md`
  (2026-07-30) for the measured CLEVR figure. If the budget binds, the cheapest
  lever is `test_samples`, not seeds.
* Smoke-run anything new at `--epochs 2 --seeds 0` before committing a grid.

## Layout

| file | role |
|---|---|
| `qnlp/utils/data/clevr_objects.py` | crop geometry, relation labels, cache, loaders |
| `build_clevr_crops.py` | C0 CLI: `--calibrate`, `--relation-montage`, build |
| `clevr_common.py` | multi-head training loop; **all statistics imported from `phase15_common`** |
| `run_c1_learnability.py` … `run_c4_relational.py` | the experiments |
| `combine_clevr.py` | merge sharded workers; per-seed collapse detection; the comparisons |
| `test_clevr.py` | data + harness guards |

Models come from `qttn_core` and `run_r4_classical_control`. **Do not write a new
model class** — per-script model drift caused both Phase-1 audits.
