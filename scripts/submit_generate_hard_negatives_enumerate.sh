#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=72:0:0
# CPU-only bobcat parsing (tagging + CCG search), no GPU needed. 5 slots for memory
# headroom, matching submit_coco_create_dataset*.sh (same parser, similar cost shape).
# tmem history: 16G (undersized guess, ~90G peaks from SYNCHRONIZED worker reloads,
# fixed via jitter not memory) -> 24G (untested overcorrection, caused long queue wait
# at 120G total) -> back to 16G (~80G total), now justified by job 7085501's actual
# maxvmem=37.950G at these settings — ~2x real headroom, not a guess. See
# HARD_NEG_PI_SWEEP_PLAN.md for the full diagnosis.
#$ -pe smp 5
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -N gen_hard_negs_enum
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/nltk_data
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/.pip_cache

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

export PYTHONPATH=$PROJECT_DIR
export HF_HOME=$CACHE_DIR/huggingface_cache
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers_cache
export TORCH_HOME=$CACHE_DIR/torch_cache
export NLTK_DATA=$CACHE_DIR/nltk_data
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache
export MPLCONFIGDIR=$CACHE_DIR/.matplotlib_cache
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "[enumerate]: bobcat-parse + lemmatize + enumerate swaps + dual-compile (bobcat + tree_no_type)"
echo "========================================="

cd $PROJECT_DIR

# BOBCAT_TRAIN/TREE_TRAIN let you point this at the Phase 0 matched datasets
# instead of the originals — see submit_generate_hard_negatives_enumerate_sharded.sh
# for the exact qsub -v example. Unset (default) falls back to
# generate_hard_negatives.py's own BOBCAT_TRAIN/TREE_TRAIN constants (originals).
TRAIN_PATH_ARGS=()
[ -n "$BOBCAT_TRAIN" ] && TRAIN_PATH_ARGS+=(--bobcat-train "$BOBCAT_TRAIN")
[ -n "$TREE_TRAIN" ] && TRAIN_PATH_ARGS+=(--tree-train "$TREE_TRAIN")

# Smoke-test first: qsub -v SMOKE=2000 scripts/submit_generate_hard_negatives_enumerate.sh
if [ -n "$SMOKE" ]; then
    echo "SMOKE RUN: limiting to $SMOKE captions"
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate \
        --limit "$SMOKE" --parts-dir data/datasets/coco_hard_negs_compiled_smoke_parts \
        --max-workers "${MAX_WORKERS:-4}" --worker-batch-size "${WORKER_BATCH_SIZE:-100}" \
        --max-tasks-per-child "${MAX_TASKS_PER_CHILD:-10}" \
        "${TRAIN_PATH_ARGS[@]}"
else
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate \
        --max-workers "${MAX_WORKERS:-4}" --worker-batch-size "${WORKER_BATCH_SIZE:-100}" \
        --max-tasks-per-child "${MAX_TASKS_PER_CHILD:-10}" \
        "${TRAIN_PATH_ARGS[@]}"
fi

echo "========================================="
echo "Job finished successfully at $(date)"
echo "If killed/timed out: just resubmit this same script — completed parts under"
echo "data/datasets/coco_hard_negs_compiled*_parts/ are skipped automatically."
echo "========================================="
