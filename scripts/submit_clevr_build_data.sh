#!/bin/bash
# Task C0 on the cluster: build the CLEVR object-crop and relation-crop caches.
#
# RUN THIS FIRST. Every C1-C4 job reads data/datasets/clevr_*.npz and will fail
# immediately without it (get_clevr_loaders raises FileNotFoundError by design
# rather than silently training on nothing).
#
# Rebuilds from the HuggingFace parquet rather than rsyncing the ~140 MB of .npz
# from the laptop, so the cluster copy is reproducible from source and cannot
# silently drift from a differently-calibrated CROP_K.
#
# Downloads ~0.9 GB (one train shard + one test shard) into HF_HOME on first run.
#
# Submit:  qsub scripts/submit_clevr_build_data.sh
# Verify:  qsub scripts/submit_clevr_build_data.sh -v VERIFY_ONLY=1
#$ -l tmem=32G
#$ -l h_rt=8:0:0
#$ -S /bin/bash
#$ -j y
#$ -N clevr_build_data
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/.pip_cache

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

export PYTHONPATH=$PROJECT_DIR
export HF_HOME=$CACHE_DIR/huggingface_cache
export TORCH_HOME=$CACHE_DIR/torch_cache
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache
export MPLCONFIGDIR=$CACHE_DIR/.matplotlib_cache
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

PYTHON=$ENV_DIR/bin/python
cd $PROJECT_DIR

echo "========================================="
echo "CLEVR data build | started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

# Fail fast and loudly if the environment cannot run the tower at all. Finding
# this out after a 10-hour array job has burned its slots is the expensive path.
echo "--- dependency check ---"
$PYTHON - <<'PYEOF' || exit 1
import importlib, sys

# Split by what each dependency actually blocks. pyarrow is needed ONLY to build
# the crop caches from parquet; the C1-C4 runners read .npz through numpy and
# never import it. Reporting them together once cost a run: a broken pyarrow
# aborted the check before it ever tested lightning.qubit, which is the thing
# that governs whether the quantum arms are affordable.
TRAIN_DEPS = ("torch", "pennylane", "numpy")
BUILD_DEPS = ("pyarrow", "PIL", "huggingface_hub", "matplotlib")

def probe(mods):
    bad = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append(f"{m}: {e}")
    return bad

train_bad, build_bad = probe(TRAIN_DEPS), probe(BUILD_DEPS)

for label, bad in (("TRAINING (C1-C4)", train_bad), ("DATA BUILD only", build_bad)):
    print(f"  {label}: {'OK' if not bad else 'BROKEN'}")
    for b in bad:
        print(f"    - {b}")

# Always report lightning, even if the build deps are broken -- it is what the
# cost model assumes and it is independent of the parquet path.
if not train_bad:
    import pennylane as qml
    try:
        qml.device("lightning.qubit", wires=4)
        print("  lightning.qubit: OK (quantum arms run at the costed ~3 h/seed)")
    except Exception as e:
        print(f"  lightning.qubit: UNAVAILABLE ({e})")
        print("    default.qubit gives identical results but is ~4x slower -- re-budget h_rt.")

if train_bad:
    print("\nFATAL: training dependencies are broken. C3b/C4 cannot run.")
    sys.exit(1)
if build_bad:
    print("\nCannot BUILD data here, but training deps are fine. Two options:")
    print("  (a) rsync the .npz caches from the laptop (recommended -- no shared env changes):")
    print("      rsync -av data/datasets/clevr_*.npz <cluster>:$PWD/data/datasets/")
    print("      then re-run this script with -v VERIFY_ONLY=1")
    print("  (b) repair the env, e.g. NumPy 2 vs a NumPy-1-compiled pyarrow:")
    print("      pip install -U 'pyarrow>=17'   # NOT into a shared env without checking")
    sys.exit(2)
print("all imports OK")
PYEOF

if [ "$VERIFY_ONLY" = "1" ]; then
    echo "--- VERIFY_ONLY: checking existing caches ---"
    $PYTHON -m qnlp.image_tower.classification.clevr.verify_cluster_data
    echo "Job finished at $(date)"
    exit 0
fi

# TASKS selects what to rebuild; default both.
#
# ⚠️ REBUILD ONLY WHAT YOU NEED IF ANOTHER JOB IS RUNNING. These commands
# OVERWRITE data/datasets/clevr_<task>_*.npz in place, and every C1-C4 task opens
# those files when it starts. Rewriting the `objects` cache underneath a live C3/
# C3b/C3c array job can hand a task a half-written .npz -- so when C3c is on the
# queue and only the relation data changed, submit with:
#
#     qsub scripts/submit_clevr_build_data.sh -v TASKS=relations
#
TASKS=${TASKS:-both}
echo "--- building crops at 16/32/64 px (TASKS=$TASKS) ---"

if [ "$TASKS" = "both" ] || [ "$TASKS" = "objects" ]; then
    $PYTHON -m qnlp.image_tower.classification.clevr.build_clevr_crops \
        --tasks objects --train-scenes 3000 --val-scenes 1500 || exit 1
fi

# Relations need more source scenes: only ~1 pair per scene survives the
# occlusion, ambiguity-margin and crop-size filters, and the set is then
# subsampled to exactly balance the four classes.
if [ "$TASKS" = "both" ] || [ "$TASKS" = "relations" ]; then
    $PYTHON -m qnlp.image_tower.classification.clevr.build_clevr_crops \
        --tasks relations --train-scenes 5000 --val-scenes 2500 || exit 1
fi

echo "--- verifying ---"
$PYTHON -m qnlp.image_tower.classification.clevr.verify_cluster_data || exit 1

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
