#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=2:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N eval_ckpt
#$ -M ucapfky@ucl.ac.uk
#$ -m a
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Standalone retrieval+benchmark eval for an already-trained coco_multi_caption
# checkpoint, for when training succeeded but the training script's own
# in-process final eval crashed (e.g. CUDA OOM from fragmented memory left
# over after training, in a FRESH process/GPU context here instead).
# Usage: qsub -v CHECKPOINT=runs/checkpoints/coco_multi_caption/<ts>/best_model.pt \
#   scripts/submit_evaluate_checkpoint.sh

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
    echo "  qsub -v CHECKPOINT=runs/checkpoints/coco_multi_caption/<ts>/best_model.pt scripts/submit_evaluate_checkpoint.sh"
    exit 1
fi

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "[evaluate]: standalone retrieval+benchmark eval"
echo "Checkpoint: $CHECKPOINT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.evaluate "$CHECKPOINT" || {
    echo "EVAL FAILED (exit $?)"
    exit 1
}

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
