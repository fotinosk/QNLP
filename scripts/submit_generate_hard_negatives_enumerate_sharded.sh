#!/bin/bash
#$ -l tmem=12G
#$ -l h_rt=72:0:0
# 20-way sharded version of submit_generate_hard_negatives_enumerate.sh — 20
# independent SGE array tasks instead of one big job (bumped from 10 -> 20,
# 2026-07-21, to increase parallelism; see HARD_NEG_PI_SWEEP_PLAN.md — changing
# the shard count requires killing and resubmitting the whole array, since
# bi % 10 and bi % 20 assign batches differently; already-completed batches are
# unaffected and get skipped instantly by the new shard set regardless of which
# shard count produced them). Batch indices are GLOBAL (computed over the full
# sorted caption list) and each task only touches batches where
# bi % NUM_SHARDS == SHARD_INDEX, so all tasks can safely write into the SAME
# --parts-dir concurrently with zero file-collision risk. Do NOT run this
# alongside submit_generate_hard_negatives_enumerate.sh (non-sharded) at the same
# time — that script claims ALL batches (num_shards=1), so running both would have
# every shard racing the non-sharded job over the same not-yet-done batches
# (wasteful duplicate compute, not corruption, but pointless) — kill one before
# starting the other.
#
# 2 workers/task x 12G/slot = 24G/task requested (20 tasks x 24G = 480G total
# ACROSS the whole array, but each task is scheduled independently — a much
# smaller PER-TASK reservation than the single-job version's 80G, which should
# schedule far more easily). max_tasks_per_child kept at 5 here (NOT 10, unlike
# the non-sharded script's current default) — deliberately the validated-safe
# value from job 7085501, not the untested-at-scale 10, since sharding is new
# and not worth compounding with an unvalidated setting at the same time.
#$ -pe smp 2
#$ -t 1-20
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -N gen_hard_negs_enum_sh
#$ -M ucapfky@ucl.ac.uk
#$ -m a
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

NUM_SHARDS=20
SHARD_INDEX=$((SGE_TASK_ID - 1))
# Separate cache dir per shard — avoids the diskcache write-race risk from
# multiple concurrent processes sharing one cache dir (see plan doc). No real
# cost: cache-hit benefit within this job is already near-zero since every
# caption is unique and parsed exactly once regardless.
SHARD_CACHE_PATH=$CACHE_DIR/lambeq_bobcat_shard_${SHARD_INDEX}/diskcache

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID, Task ID: $SGE_TASK_ID (shard $SHARD_INDEX/$NUM_SHARDS)"
echo "Running on: $(hostname)"
echo "[enumerate, sharded]: bobcat-parse + lemmatize + enumerate swaps + dual-compile"
echo "========================================="

cd $PROJECT_DIR

# BOBCAT_TRAIN/TREE_TRAIN let you point this at the Phase 0 matched datasets
# instead of the originals, e.g.:
#   qsub -v BOBCAT_TRAIN=data/datasets/coco_single_caption_nlc_matched_train.parquet,\
#TREE_TRAIN=data/datasets/coco_single_caption_nlc_tree_no_type_matched_train.parquet \
#   scripts/submit_generate_hard_negatives_enumerate_sharded.sh
# Unset (default) falls back to generate_hard_negatives.py's own BOBCAT_TRAIN/TREE_TRAIN
# constants (the original, unmatched datasets).
TRAIN_PATH_ARGS=()
[ -n "$BOBCAT_TRAIN" ] && TRAIN_PATH_ARGS+=(--bobcat-train "$BOBCAT_TRAIN")
[ -n "$TREE_TRAIN" ] && TRAIN_PATH_ARGS+=(--tree-train "$TREE_TRAIN")

# Smoke-test first: qsub -v SMOKE=2000 scripts/submit_generate_hard_negatives_enumerate_sharded.sh
if [ -n "$SMOKE" ]; then
    echo "SMOKE RUN: limiting to $SMOKE captions"
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate \
        --limit "$SMOKE" --parts-dir data/datasets/coco_hard_negs_compiled_smoke_parts \
        --cache-path "$SHARD_CACHE_PATH" \
        --max-workers "${MAX_WORKERS:-2}" --worker-batch-size "${WORKER_BATCH_SIZE:-100}" \
        --max-tasks-per-child "${MAX_TASKS_PER_CHILD:-5}" \
        --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" \
        "${TRAIN_PATH_ARGS[@]}"
else
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives enumerate \
        --cache-path "$SHARD_CACHE_PATH" \
        --max-workers "${MAX_WORKERS:-2}" --worker-batch-size "${WORKER_BATCH_SIZE:-100}" \
        --max-tasks-per-child "${MAX_TASKS_PER_CHILD:-5}" \
        --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" \
        "${TRAIN_PATH_ARGS[@]}"
fi

echo "========================================="
echo "Task $SGE_TASK_ID (shard $SHARD_INDEX) finished successfully at $(date)"
echo "If a task failed/timed out: resubmit just that task, e.g."
echo "  qsub -t $SGE_TASK_ID scripts/submit_generate_hard_negatives_enumerate_sharded.sh"
echo "or resubmit the whole array (qsub -t 1-20 ...) — already-done batches from"
echo "ANY shard are skipped automatically, so re-running everything is also safe,"
echo "just less targeted."
echo "========================================="
