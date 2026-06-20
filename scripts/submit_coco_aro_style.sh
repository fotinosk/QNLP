#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N coco_aro_style
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

# --- Experiment config (mirrors ARO defaults, override as needed) ---
export ML_EMBEDDING_DIM=512
export ML_BOND_DIM=10
export ML_USE_NON_LINEAR_CONTRACTIONS=true
export ML_BATCH_SIZE=128

export ML_TEXT_LR=0.001
export ML_IMAGE_LR=0.00005
export ML_HEAD_LR=0.001
export ML_TEXT_WEIGHT_DECAY=0.001
export ML_IMAGE_WEIGHT_DECAY=0.05
export ML_HEAD_WEIGHT_DECAY=0.001

export ML_TEMPERATURE=0.07
export ML_TRIPLET_WEIGHT=40000.0
export ML_TRIPLET_MARGIN=0.2

export ML_MAX_EPOCHS=100
export ML_PATIENCE=10
export ML_MAX_GRAD_NORM=1.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_aro_style_nlc}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "embedding_dim: $ML_EMBEDDING_DIM  bond_dim: $ML_BOND_DIM"
echo "triplet_weight: $ML_TRIPLET_WEIGHT  temperature: $ML_TEMPERATURE"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_aro_style.run

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
