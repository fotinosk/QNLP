#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=2:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N eval_ckpt_frz
#$ -M ucapfky@ucl.ac.uk
#$ -m a
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Standalone retrieval+benchmark eval for an already-trained coco_multi_caption
# FROZEN-image-tower checkpoint (run_frozen.py), for when training succeeded but
# the training script's own in-process final eval crashed (e.g. the checkpoint-
# reload CUDA OOM inside EinsumModel.load_state_dict — see HARD_NEG_PI_SWEEP_PLAN.md
# / RESULTS.md), in a FRESH process/GPU context here instead. Non-frozen checkpoints
# use submit_evaluate_checkpoint.sh — the two checkpoint formats are incompatible
# (text_model_state_dict/text_head_state_dict here vs. model_state_dict there).
# Usage: qsub -v CHECKPOINT=runs/checkpoints/coco_multi_caption_frozen/<ts>/best_model.pt \
#   scripts/submit_evaluate_checkpoint_frozen.sh
# DATASET (optional): override the retrieval test-set dataset name — REQUIRED if
# the original training run set ML_DATASET_NAME explicitly (e.g. every bobcat cell
# in the pi-sweep pins coco_single_caption_nlc). Without it, the built-in
# linear/non-linear heuristic can silently grab the wrong test set.
#   qsub -v CHECKPOINT=...,DATASET=coco_single_caption_nlc scripts/submit_evaluate_checkpoint_frozen.sh

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=$ENV_DIR/bin/python

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: set CHECKPOINT, e.g."
    echo "  qsub -v CHECKPOINT=runs/checkpoints/coco_multi_caption_frozen/<ts>/best_model.pt scripts/submit_evaluate_checkpoint_frozen.sh"
    exit 1
fi

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "[evaluate_frozen]: standalone retrieval+benchmark eval"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset override: ${DATASET:-(none — using built-in heuristic)}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

DATASET_ARGS=()
[ -n "$DATASET" ] && DATASET_ARGS+=(--dataset "$DATASET")

$PYTHON -m qnlp.scripts.coco_multi_caption.evaluate_frozen "$CHECKPOINT" "${DATASET_ARGS[@]}" || {
    echo "EVAL FAILED (exit $?)"
    exit 1
}

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
