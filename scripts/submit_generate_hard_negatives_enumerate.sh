#!/bin/bash
#$ -l tmem=24G
#$ -l h_rt=72:0:0
# CPU-only bobcat parsing (tagging + CCG search), no GPU needed. 5 slots for memory
# headroom, matching submit_coco_create_dataset*.sh (same parser, similar cost shape).
# tmem bumped 16G->24G (5 slots => ~120G total): observed maxvmem~90G with 5 fully-
# loaded workers (parser+tagger+ansatz each) at tmem=16G (~80G total) — was undersized,
# not a leak (see HARD_NEG_PI_SWEEP_PLAN.md).
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

# Smoke-test first: qsub -v SMOKE=2000 scripts/submit_generate_hard_negatives_enumerate.sh
if [ -n "$SMOKE" ]; then
    echo "SMOKE RUN: limiting to $SMOKE captions"
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate \
        --limit "$SMOKE" --parts-dir data/datasets/coco_hard_negs_compiled_smoke_parts \
        --max-workers "${MAX_WORKERS:-5}" --worker-batch-size "${WORKER_BATCH_SIZE:-100}" \
        --max-tasks-per-child "${MAX_TASKS_PER_CHILD:-10}"
else
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate \
        --max-workers "${MAX_WORKERS:-5}" --worker-batch-size "${WORKER_BATCH_SIZE:-100}" \
        --max-tasks-per-child "${MAX_TASKS_PER_CHILD:-10}"
fi

echo "========================================="
echo "Job finished successfully at $(date)"
echo "If killed/timed out: just resubmit this same script — completed parts under"
echo "data/datasets/coco_hard_negs_compiled*_parts/ are skipped automatically."
echo "========================================="
