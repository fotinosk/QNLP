#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N coco_sc_linear_frozen
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
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

# --- Bobcat parser (default datasets: coco_single_caption_{split}.parquet) ---
# No PARSER_VERSION / ML_DATASET_NAME override: linear + frozen defaults to the
# Bobcat-parser coco_single_caption dataset in both run_frozen.py scripts.

# --- Experiment config: FROZEN CLIP image tower, LINEAR text contractions ---
export ML_USE_NON_LINEAR_CONTRACTIONS=false
export ML_EMBEDDING_DIM=512  # must match the frozen CLIP ViT-B/32 output dim
export ML_BOND_DIM=10
export ML_BATCH_SIZE=256
# Linear-mode weight-norm layer keeps the contraction bounded, so a healthy LR is safe.
export ML_TEXT_LR=0.002
export ML_TEXT_WEIGHT_DECAY=0.001

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_sc_linear_frozen}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Parser version: $PARSER_VERSION"
echo "Dataset: $ML_DATASET_NAME"
echo "Embedding dim: $ML_EMBEDDING_DIM   Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS   Frozen image tower: true"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_single_caption.run_frozen

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
