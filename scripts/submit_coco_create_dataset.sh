#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=12:0:0
#$ -S /bin/bash
#$ -j y
#$ -N coco_create_dataset
#$ -M ucapfky@ucl.ac.uk
#$ -m abe

# Exit on errors
set -e
# Enable debugging
set -x

# Set up directories
PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
CONDA_ENV=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

# Create cache directories
mkdir -p $CACHE_DIR/nltk_data
mkdir -p $CACHE_DIR/.pip_cache

# Set environment variables
export PYTHONPATH=$PROJECT_DIR
export NLTK_DATA=$CACHE_DIR/nltk_data
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache

# Log job details
echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Working directory: $(pwd)"
echo "Non-linear: ${NON_LINEAR:-false}"
echo "========================================="

# Set up micromamba
export MAMBA_ROOT_PREFIX=/home/kinianlo/micromamba
eval "$(~/.local/bin/micromamba shell hook --shell bash)"
micromamba activate $CONDA_ENV

# Verify environment
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

cd $PROJECT_DIR

# Pass --paths if NON_LINEAR=1 is set
EXTRA_ARGS=""
if [ "${NON_LINEAR:-0}" = "1" ]; then
    EXTRA_ARGS="--paths"
fi

echo "Starting dataset creation at $(date)"
python -m qnlp.scripts.coco_single_caption.create_dataset $EXTRA_ARGS

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
