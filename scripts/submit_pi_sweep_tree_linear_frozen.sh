#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N pi_tree_frz
#$ -M ucapfky@ucl.ac.uk
#$ -m a
#$ -R y
#$ -cwd
#$ -t 1-5
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Phase C pi-sweep — cell: TREE_NO_TYPE parser, LINEAR contractions, FROZEN
# CLIP ViT-B/32 image tower. SGE array task ID -> pi: 1->0, 2->0.1, 3->0.25,
# 4->0.5, 5->1.0. Rerun a single failed pi with e.g.
#   qsub -t 3 scripts/submit_pi_sweep_tree_linear_frozen.sh
# Based on submit_coco_multi_caption_tree_no_type_linear_frozen.sh with
# hard-negative env added.

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/nltk_data
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/.pip_cache

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

# --- pi from array task ID ---
PI_VALUES=(0 0.1 0.25 0.5 1.0)
PI=${PI_VALUES[$((SGE_TASK_ID - 1))]}

# --- Tree-no-type parser: versions all dataset/eval paths ---
export PARSER_VERSION=tree_no_type
export ML_DATASET_NAME=coco_single_caption_nlc_tree_no_type

# --- Hard negatives (Phase B): the TREE-scoped companion file. pi=0 never
# loads the bank. ---
export ML_HARD_NEG_PI=$PI
export ML_HARD_NEGS_DATASET=coco_hard_negs_tree_no_type

# --- Experiment config: FROZEN CLIP image tower, LINEAR text contractions ---
export ML_USE_NON_LINEAR_CONTRACTIONS=false
export ML_EMBEDDING_DIM=512  # must match the frozen CLIP ViT-B/32 output dim
export ML_BOND_DIM=10
export ML_BATCH_SIZE=256
export ML_TEXT_LR=0.002
export ML_TEXT_WEIGHT_DECAY=0.001

# --- No early stopping this rerun: patience > max_epochs (50) so every cell
# trains the full schedule, removing epoch count as a confound between cells. ---
export ML_PATIENCE=1000

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-pi_sweep_tree_linear_frozen}_pi${PI}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID, Task ID: $SGE_TASK_ID (pi=$PI)"
echo "Running on: $(hostname)"
echo "Cell: tree_no_type / linear / frozen"
echo "Parser version: $PARSER_VERSION"
echo "Dataset: $ML_DATASET_NAME   Hard negs: $ML_HARD_NEGS_DATASET   pi: $ML_HARD_NEG_PI"
echo "Embedding dim: $ML_EMBEDDING_DIM   Bond dim: $ML_BOND_DIM   Batch: $ML_BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.run_frozen || {
    echo "Task $SGE_TASK_ID (pi=$PI) FAILED (exit $?)"
    exit 1
}

echo "========================================="
echo "Task $SGE_TASK_ID (pi=$PI) finished successfully at $(date)"
echo "========================================="
