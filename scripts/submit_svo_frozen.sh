#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N svo_frozen
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Diagnostic control run: trains SVO-Probes/SVO-Swap with a FROZEN CLIP
# ViT-B/32 image tower in place of the from-scratch TTNImageModel (see
# qnlp/scripts/svo/run_frozen.py's docstring, and the "image tower is the
# bottleneck" section of SVO_EXPERIMENTS.md). Requires the same
# data/datasets/svo_{train,val,test}_probes.parquet and svo_swap_eval.parquet
# as submit_svo.sh.

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

# --- Experiment config (SVO_ML_ prefix — see qnlp/scripts/svo/config.py) ---
# embedding_dim must stay 512 (CLIP ViT-B/32's output dim) — config.py's
# default already matches, no override needed.
# `:=` (not `export ...=128`) so a qsub -v override survives — an
# unconditional export here silently clobbered every `-v
# SVO_ML_BATCH_SIZE=...` passed at submit time (R3/R4/R5 all actually ran
# at 128 despite intending 64 — see TTN_CIFAR_EXPERIMENTS.md).
: "${SVO_ML_BATCH_SIZE:=128}"
export SVO_ML_BATCH_SIZE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-svo_frozen_clip}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Batch size: $SVO_ML_BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.svo.run_frozen
STATUS=$?

echo "========================================="
if [ $STATUS -eq 0 ]; then
    echo "Job finished successfully at $(date)"
else
    echo "Job FAILED (exit code $STATUS) at $(date)"
fi
echo "========================================="
exit $STATUS
