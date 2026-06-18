#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N coco_train_nlc
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
mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/mlflow_db

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
export ML_BOND_DIM=20
export ML_USE_NON_LINEAR_CONTRACTIONS=true

# --- MLflow run name ---
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_nlc_bond30}"

# --- Use full path to Python (no activation needed) ---
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Bond dim: $ML_BOND_DIM"
echo "Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MLflow run name: $MLFLOW_RUN_NAME"
echo "========================================="

cd $PROJECT_DIR

# Start mlflow server in the background
$PYTHON -m mlflow server \
    --backend-store-uri sqlite:///$PROJECT_DIR/mlflow_db/mlflow.db \
    --default-artifact-root $PROJECT_DIR/mlflow_db/artifacts \
    --port 8080 &
MLFLOW_PID=$!
echo "MLflow server started (PID $MLFLOW_PID)"

sleep 5

$PYTHON -m qnlp.scripts.coco_single_caption.run

kill $MLFLOW_PID

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
