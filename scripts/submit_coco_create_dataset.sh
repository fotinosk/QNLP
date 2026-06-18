#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=12:0:0
#$ -S /bin/bash
#$ -j y
#$ -N coco_create_dataset
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
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

# Hugging Face / Transformers cache
export HF_HOME=$CACHE_DIR/huggingface_cache
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers_cache
export TORCH_HOME=$CACHE_DIR/torch_cache

# NLTK data
export NLTK_DATA=$CACHE_DIR/nltk_data

# Pip cache
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache

# Matplotlib cache
export MPLCONFIGDIR=$CACHE_DIR/.matplotlib_cache

# Python bytecode (prevent writing to home)
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache

# Override any default cache locations
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

# --- Use full path to Python (no activation needed) ---
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Using Python: $PYTHON"
echo "Non-linear: ${NON_LINEAR:-0}"
echo "All cache directories point to: $CACHE_DIR"
echo "========================================="

cd $PROJECT_DIR

# Pass --paths if NON_LINEAR=1 is set
EXTRA_ARGS=""
if [ "${NON_LINEAR:-0}" = "1" ]; then
    EXTRA_ARGS="--paths"
fi

echo "Starting dataset creation at $(date)"
$PYTHON -m qnlp.scripts.coco_single_caption.create_dataset $EXTRA_ARGS

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
