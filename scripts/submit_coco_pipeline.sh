#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=48:0:0
#$ -S /bin/bash
#$ -j y
#$ -N coco_preprocessing
#$ -pe smp 10
#$ -R y
#$ -cwd

# EXPLICITLY set output directories to project space (NOT home!)
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# --- Create all directories BEFORE any file operations ---
mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/bobcat/diskcache
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

# Bobcat cache
export GLOBAL_CONSTANTS_BOBCAT_CACHE_PATH=$CACHE_DIR/bobcat/diskcache

# --- Override any default cache locations ---
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

# --- Use full path to Python (no activation needed) ---
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Using Python: $PYTHON"
echo ""
echo "Resource allocation:"
echo "  Memory: 16GB tmem, 16GB h_vmem"
echo "  Time: 48 hours"
echo "  CPU cores: 10 (SMP)"
echo ""
echo "All cache directories point to: $CACHE_DIR"
echo "Output files go to: /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/"
echo "========================================="

# Verify no home directory paths are used
echo "Checking environment variables:"
echo "NLTK_DATA: $NLTK_DATA"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "HF_HOME: $HF_HOME"
echo "MPLCONFIGDIR: $MPLCONFIGDIR"
echo "PYTHONPYCACHEPREFIX: $PYTHONPYCACHEPREFIX"
echo "========================================="

# Show system info
echo "System information:"
echo "CPU cores available: $(nproc)"
echo "Memory available: $(free -h | grep Mem | awk '{print $2}')"
echo "========================================="

cd $PROJECT_DIR

# Run the pipeline with beefed-up parameters
$PYTHON -m qnlp.preprocessing_pipelines.coco.pipeline \
    --chunk-size 1000 \
    --max-workers 8 \
    --worker-batch-size 2000

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
