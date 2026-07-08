#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N coco_mc_tree
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

# --- Tree-no-type parser: versions all dataset/eval paths (derived + parquet names) ---
export PARSER_VERSION=tree_no_type
# The tree parser only builds the single non-linear dataset (path column is
# ignored in linear mode), so both modes read the nlc build.
export ML_DATASET_NAME=coco_single_caption_nlc_tree_no_type

# --- Experiment config ---
export ML_USE_NON_LINEAR_CONTRACTIONS=true
export ML_BATCH_SIZE=256

# --- Reduce fragmentation from CUDA allocations ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- MLflow: disabled on cluster, metrics go to job output log ---
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_mc_tree}"

# --- Use full path to Python (no activation needed) ---
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Parser version: $PARSER_VERSION"
echo "Dataset: $ML_DATASET_NAME"
echo "Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.run

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
