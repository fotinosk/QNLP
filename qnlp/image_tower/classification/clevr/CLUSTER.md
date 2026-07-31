# Running Phase 2 (CLEVR) on the cluster

Long, multi-seed runs belong here, not on the laptop. A quantum seed is ~62 min
(~3 h at the 12-value readout), the laptop caps at 4–5 concurrent workers, and
local contention has already corrupted two measurements this phase.

**Everything below assumes SGE** (`qsub`, `#$` directives), matching the existing
`scripts/submit_*.sh`.

---

## Setup checklist

Work top to bottom. Steps 1–4 are one-off; step 5 gates everything after it.

### ☐ 1. Code is on the cluster and current

```bash
# from the laptop
rsync -av --delete \
  --exclude '.git' --exclude '__pycache__' --exclude 'data/' \
  --exclude 'runs/' --exclude 'mlartifacts/' --exclude '*.db' \
  ./ ucapfky@<login-node>:/SAN/intelsys/discoviz/fotinos/QNLP/
```

Phase 2 needs these, all new — confirm they arrived:

| path | why |
|---|---|
| `qnlp/utils/data/clevr_objects.py` | crop geometry, relation labels, loaders |
| `qnlp/image_tower/classification/clevr/` | harness, runners, combiner, verifier |
| `qnlp/image_tower/classification/quantum/qttn_core.py` | **modified** — 12-value readout, multi-head, `positional` |
| `qnlp/image_tower/classification/quantum/run_r4_classical_control.py` | **modified** — patch-size + multi-head generalisation |
| `qnlp/image_tower/classification/quantum/phase15_common.py` | **modified** — `seeds_needed` convergence fix |
| `scripts/submit_clevr_build_data.sh`, `submit_c3b_lr_sweep.sh`, `submit_c4_relational.sh`, `submit_c4_classical.sh` | the jobs |

> `data/` is deliberately excluded — rebuild it on the cluster (step 3) rather
> than shipping 140 MB of `.npz`, so the cluster copy is reproducible from source
> and cannot silently carry a differently-calibrated `CROP_K`.

### ☐ 2. Environment can actually run the tower

The **single biggest risk**, and the one that wastes a whole array job if left
until later. `/SAN/intelsys/discoviz/envs/qnlp311` predates this work, so
`pennylane` / `pennylane-lightning` may not be installed.

```bash
/SAN/intelsys/discoviz/envs/qnlp311/bin/python -c \
  "import torch, pennylane as qml; print(torch.__version__, qml.__version__); \
   qml.device('lightning.qubit', wires=4); print('lightning OK')"
```

If it fails:

```bash
/SAN/intelsys/discoviz/envs/qnlp311/bin/pip install pennylane pennylane-lightning
```

`lightning.qubit` is not strictly required — `default.qubit` gives identical
results — but it is **~4× faster**, and the cost model assumes it.

### ☐ 3. Build the data

```bash
qsub scripts/submit_clevr_build_data.sh
```

Downloads ~0.9 GB (one train + one test parquet shard) into `HF_HOME`, then
writes `data/datasets/clevr_{objects,relations}_{16,32,64}_{train,val}.npz` plus
manifests. Runs its own dependency check first and fails fast.

### ☐ 4. Verify readiness — **do not skip**

```bash
qsub scripts/submit_clevr_build_data.sh -v VERIFY_ONLY=1
# or directly on a login node:
cd /SAN/intelsys/discoviz/fotinos/QNLP && PYTHONPATH=. \
  /SAN/intelsys/discoviz/envs/qnlp311/bin/python \
  -m qnlp.image_tower.classification.clevr.verify_cluster_data
```

Must print **READY**. It checks imports, `lightning.qubit`, every cache file's
shape and label set, that each split holds enough samples for the protocol
(1024 train / 512 val), that no head is degenerate, and — critically — that the
cached `CROP_K` matches the code. A mismatched `CROP_K` trains fine and produces
numbers that cannot be compared with anything in `research_log.md`, which is
worse than a crash.

### ☐ 5. Smoke-test one task before submitting an array

```bash
PYTHONPATH=. /SAN/intelsys/discoviz/envs/qnlp311/bin/python \
  -m qnlp.image_tower.classification.clevr.run_c4_relational \
  --positional none --seeds 0 --epochs 2 --skip-classical --out smoke_s0
```

Two epochs, minutes. Catches a broken environment for the price of one task
instead of twenty.

### ☐ 6. Submit

```bash
qsub scripts/submit_c4_classical.sh    # seconds-per-seed arms, single job
qsub scripts/submit_c4_relational.sh   # array 1-20, ~3 h per task
qsub scripts/submit_c3b_lr_sweep.sh    # array 1-12, ~3 h per task
qstat -u ucapfky
```

### ☐ 7. Merge — **this is where the result is produced**

A single-seed task has no variance and can resolve nothing on its own.

```bash
# C4
python -m qnlp.image_tower.classification.clevr.combine_clevr \
    --prefix c4 --task relations --seeds 0 1 2 3 4 5 6 7 8 9 \
    --arms quantum_none quantum_on_wire

# C3b -- one merge per lr, then compare the four
for LR in 0.003 0.01 0.03 0.1; do
  python -m qnlp.image_tower.classification.clevr.combine_clevr \
      --prefix c3b_lr$LR --seeds 0 1 2 --arms quantum_coherent --params 462
done
```

---

## The jobs

| script | shape | tasks | per task | what it settles |
|---|---|---|---|---|
| `submit_clevr_build_data.sh` | single | 1 | ~1 h | data + readiness |
| `submit_c4_classical.sh` | single | 1 | ~1 h | C4's classical controls, 10 seeds |
| `submit_c4_relational.sh` | array | 20 | ~3 h | **Question C.2** — 2 positional arms × 10 seeds |
| `submit_c3b_lr_sweep.sh` | array | 12 | ~3 h | the colour confound — 4 lrs × 3 seeds |

**10 seeds on C4, not 3.** Locally we were held to 3 purely by wall time, and
that is exactly why several C2 comparisons came back unresolved.

**C3b includes lr=0.03 deliberately** so C3's existing quantum arm is reproduced
*inside* the sweep, making the comparison like-for-like rather than against a
number measured on a different day.

---

## Reading the results

**C4 — check the viability guard first.** If `quantum_none` does not clear the
majority floor, both arms are at chance and their comparison measures nothing;
the runner suppresses the Question C.2 verdict in that case by design. Escalate
to `positional='ancilla2'` (18 qubits, ~4× cost) **only** if `on_wire` shows a
*resolved* effect.

**C3b — read colour specifically.** That is the head the sweep exists to settle.
**If a better lr is found, the whole C3 comparison must be re-run at it** —
quoting a tuned quantum arm against separately-tuned classical arms would invert
the asymmetry rather than remove it.

---

## Gotchas

* **Node `arbuckle` lacks AVX** and crashes polars with SIGILL; the fix is
  `polars[rtcompat]`. The Phase-2 data path uses **pyarrow**, so it should be
  unaffected — but `load_clevr_to_atlas` does use polars.
* **`OMP_NUM_THREADS=1` is set in the array scripts.** One slot per task; letting
  torch spawn a thread per core makes concurrent tasks fight each other.
* **Write to `/SAN/...`, never `$HOME`** — every cache env var is redirected in
  the scripts for this reason.
* **`h_rt` is a hard kill.** 12 h is set for ~3 h tasks; raise it before running
  anything longer (e.g. C3c's 90-epoch run at ~9 h).
* Results land in `qnlp/image_tower/classification/quantum/results/` — rsync that
  directory back to the laptop before merging locally.
