#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N aro_frozen
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

# --- Experiment config (matches ARO ExperimentConfig defaults) ---
export ML_BOND_DIM=10
export ML_USE_NON_LINEAR_CONTRACTIONS=true
export ML_BATCH_SIZE=128
# Frozen image tower is ViT-B/32 (512-dim) — text embedding dim must match.
export ML_EMBEDDING_DIM=512

# --- Contraction-path ablation: select dataset variant via -v ML_DATASET_SUFFIX=_random ---
#   ""        optimal-path datasets (default)
#   "_rtl"    right-to-left path datasets
#   "_random" random connected-path datasets
export ML_DATASET_SUFFIX=${ML_DATASET_SUFFIX:-""}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-aro_frozen}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Bond dim: $ML_BOND_DIM"
echo "Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS"
echo "Dataset suffix: '$ML_DATASET_SUFFIX'"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.aro_contrastive.run_frozen

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
