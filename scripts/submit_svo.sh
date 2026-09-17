#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N svo_final
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Trains + evaluates the ContrastiveVLM on SVO-Probes, fully independent of
# the COCO pipeline. Requires scripts/submit_svo_pipeline.sh to have already
# produced data/datasets/svo_{train,val,test}.parquet, svo_{val,test}_probes.parquet,
# and svo_swap_eval.parquet.

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

# --- Experiment config (SVO_ML_ prefix — see qnlp/scripts/svo/config.py) ---
# No overrides here: config.py's defaults ARE the current baseline. Pass
# -v SVO_ML_<FIELD>=<value> to qsub for one-off deviations instead of
# hardcoding them here, so this script can't silently shadow config.py
# again the way SVO_ML_USE_NON_LINEAR_CONTRACTIONS=true did.
export SVO_ML_BATCH_SIZE=128

# --- Reduce fragmentation from CUDA allocations ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- MLflow: disabled on cluster, metrics go to job output log ---
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-svo_final}"

# --- Use full path to Python (no activation needed) ---
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Batch size: $SVO_ML_BATCH_SIZE"
echo "Non-linear: $SVO_ML_USE_NON_LINEAR_CONTRACTIONS"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.svo.run
STATUS=$?

echo "========================================="
if [ $STATUS -eq 0 ]; then
    echo "Job finished successfully at $(date)"
else
    echo "Job FAILED (exit code $STATUS) at $(date)"
fi
echo "========================================="
exit $STATUS
