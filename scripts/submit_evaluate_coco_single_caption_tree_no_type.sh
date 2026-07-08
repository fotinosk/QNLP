#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=4:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N coco_sc_eval_tree
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

# REQUIRED: the evaluate module does NOT set this internally, so it must be
# exported here to route benchmark/retrieval parquets to the *_tree_no_type sets.
export PARSER_VERSION=tree_no_type

# Set this to the checkpoint from the finished tree training run:
#   ML_CHECKPOINT=/path/to/best_model.pt qsub scripts/submit_evaluate_coco_single_caption_tree_no_type.sh
CHECKPOINT=${ML_CHECKPOINT:-""}

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Parser version: $PARSER_VERSION"
echo "Checkpoint: $CHECKPOINT"
echo "========================================="

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: set ML_CHECKPOINT to the path of best_model.pt before submitting."
    exit 1
fi

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.evaluate "$CHECKPOINT" --batch_size 128

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
