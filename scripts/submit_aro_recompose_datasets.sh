#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=4:0:0
#$ -S /bin/bash
#$ -j y
#$ -N aro_recompose
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Re-runs only the dataset-composition stage (ContrastivePairStrategy), not
# the CCG-compile stage -- ARO's derived_v1 chunks already carry
# processed_text (Pipeline._process_chunk's required_cols always included
# it), so no recompile is needed to pick up PAPER_EXPERIMENTS_PLAN.md's M3
# true_processed_text/false_processed_text columns, added to
# ContrastivePairStrategy this session.

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

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "========================================="

cd $PROJECT_DIR

echo "--- aro_train/val/test.parquet ---"
$PYTHON -m qnlp.scripts.aro_contrastive.create_dataset

echo "--- aro_eval.parquet ---"
$PYTHON -m qnlp.scripts.aro_contrastive.create_eval_dataset

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
