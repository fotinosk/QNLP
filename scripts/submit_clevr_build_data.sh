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
missing = []
for m in ("torch", "pennylane", "numpy", "pyarrow", "PIL", "huggingface_hub", "matplotlib"):
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m}: {e}")
if missing:
    print("MISSING DEPENDENCIES:\n  " + "\n  ".join(missing))
    sys.exit(1)
import pennylane as qml
try:
    qml.device("lightning.qubit", wires=4)
    print("lightning.qubit: OK")
except Exception as e:
    print(f"lightning.qubit UNAVAILABLE ({e}) -- the quantum arms need it; "
          f"default.qubit works but is ~4x slower.")
print("all imports OK")
PYEOF

if [ "$VERIFY_ONLY" = "1" ]; then
    echo "--- VERIFY_ONLY: checking existing caches ---"
    $PYTHON -m qnlp.image_tower.classification.clevr.verify_cluster_data
    echo "Job finished at $(date)"
    exit 0
fi

echo "--- building object + relation crops at 16/32/64 px ---"
$PYTHON -m qnlp.image_tower.classification.clevr.build_clevr_crops \
    --tasks objects --train-scenes 3000 --val-scenes 1500 || exit 1

# Relations need more source scenes: only ~1 pair per scene survives the
# occlusion, ambiguity-margin and crop-size filters, and the set is then
# subsampled to exactly balance the four classes.
$PYTHON -m qnlp.image_tower.classification.clevr.build_clevr_crops \
    --tasks relations --train-scenes 5000 --val-scenes 2500 || exit 1

echo "--- verifying ---"
$PYTHON -m qnlp.image_tower.classification.clevr.verify_cluster_data || exit 1

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
