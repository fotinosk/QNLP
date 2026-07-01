#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=4:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N aro_eval_frozen
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs

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

# Pass via -v:
#   qsub -v ML_CHECKPOINT=/path/best_text_model.pt,ML_DATASET_SUFFIX=_random scripts/submit_aro_evaluate_frozen.sh
CHECKPOINT=${ML_CHECKPOINT:-""}
SUFFIX=${ML_DATASET_SUFFIX:-""}
SPLIT=${ML_SPLIT:-test}
BATCH_SIZE=${ML_BATCH_SIZE:-128}

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Checkpoint: $CHECKPOINT"
echo "Suffix: '$SUFFIX'  Split: $SPLIT  Batch: $BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: set ML_CHECKPOINT to the path of best_text_model.pt before submitting."
    echo "  qsub -v ML_CHECKPOINT=/path/best_text_model.pt,ML_DATASET_SUFFIX=_random scripts/submit_aro_evaluate_frozen.sh"
    exit 1
fi

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.aro_contrastive.evaluate_frozen \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --suffix "$SUFFIX" \
    --batch_size "$BATCH_SIZE"

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
