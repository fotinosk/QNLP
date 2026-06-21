#!/bin/bash
#$ -l tmem=64G
#$ -l h_rt=24:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N coco_train_frozen
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# --- Create all directories BEFORE any file operations ---
mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/nltk_data
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/.pip_cache

# --- Set up all paths to project space (NOT home!) ---
PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

# --- Redirect ALL environment variables to project space ---
export PYTHONPATH=$PROJECT_DIR

export HF_HOME=$CACHE_DIR/huggingface_cache
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers_cache
export TORCH_HOME=$CACHE_DIR/torch_cache
export NLTK_DATA=$CACHE_DIR/nltk_data
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache
export MPLCONFIGDIR=$CACHE_DIR/.matplotlib_cache
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

# --- Experiment config ---
# embedding_dim=512 must match CLIP ViT-B/32 output dim.
export ML_EMBEDDING_DIM=512
export ML_BOND_DIM=${BOND_DIM:-10}
export ML_USE_NON_LINEAR_CONTRACTIONS=${NON_LINEAR:-false}
export ML_MAX_GRAD_NORM=0.1
export ML_BATCH_SIZE=256

# --- MLflow: disabled on cluster, metrics go to job output log ---
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_sc_frozen_bond${BOND_DIM:-10}}"

# --- Reduce fragmentation from CUDA allocations ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Use full path to Python (no activation needed) ---
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Embedding dim: $ML_EMBEDDING_DIM"
echo "Bond dim: $ML_BOND_DIM"
echo "Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS"
echo "Max grad norm: $ML_MAX_GRAD_NORM"
echo "Batch size: $ML_BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_single_caption.run_frozen

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
