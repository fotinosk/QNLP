#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=4:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N eval_sc_frozen_tree
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

# Routes benchmark/retrieval parquets to the tree_no_type dataset.
export PARSER_VERSION=tree_no_type
export ML_DATASET_NAME=coco_single_caption_nlc_tree_no_type

# Pass checkpoint via -v:
#   qsub -v ML_CHECKPOINT=/path/to/best_model.pt scripts/submit_evaluate_coco_single_caption_tree_no_type_frozen.sh
#   qsub -v ML_CHECKPOINT=/path/to/best_model.pt,ML_BATCH_SIZE=128 scripts/submit_evaluate_coco_single_caption_tree_no_type_frozen.sh
CHECKPOINT=${ML_CHECKPOINT:-""}
BATCH_SIZE=${ML_BATCH_SIZE:-256}

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Parser version: $PARSER_VERSION"
echo "Dataset: $ML_DATASET_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Batch size: $BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: set ML_CHECKPOINT to the path of best_model.pt before submitting."
    echo "  qsub -v ML_CHECKPOINT=/path/to/best_model.pt scripts/submit_evaluate_coco_single_caption_tree_no_type_frozen.sh"
    exit 1
fi

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_single_caption.evaluate_frozen \
    --checkpoint "$CHECKPOINT" \
    --batch_size $BATCH_SIZE

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
