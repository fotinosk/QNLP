#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
# NOTE: the linear text model (~1.07B params) + AdamW needs ~17 GB, so it OOMs on
# small (~11 GB) cards. Add the cluster's GPU-memory resource here once known.
#$ -S /bin/bash
#$ -j y
#$ -N coco_mc_tree_lin
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

# --- Tree-no-type parser: versions all dataset/eval paths (derived + parquet names) ---
export PARSER_VERSION=tree_no_type
# The tree parser only builds the single non-linear dataset (path column is
# ignored in linear mode), so linear training reads the same nlc build.
export ML_DATASET_NAME=coco_single_caption_nlc_tree_no_type

# --- Experiment config: LINEAR contractions, TTN image tower ---
export ML_EMBEDDING_DIM=512
export ML_BOND_DIM=10
export ML_USE_NON_LINEAR_CONTRACTIONS=false
# Overflow is prevented structurally by the linear-mode weight-norm layer in
# EinsumModel, so a healthy LR is safe.
export ML_TEXT_LR=0.002
export ML_TEXT_WEIGHT_DECAY=0.001
export ML_BATCH_SIZE=256

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_mc_tree_lin}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Parser version: $PARSER_VERSION"
echo "Dataset: $ML_DATASET_NAME"
echo "Embedding dim: $ML_EMBEDDING_DIM"
echo "Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS"
echo "Text LR: $ML_TEXT_LR  Text weight decay: $ML_TEXT_WEIGHT_DECAY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.run

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
