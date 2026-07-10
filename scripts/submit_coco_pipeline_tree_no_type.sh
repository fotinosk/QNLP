#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=48:0:0
#$ -S /bin/bash
#$ -j y
#$ -N coco_preprocess_tree
#$ -pe smp 10
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/bobcat/diskcache
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

# Tree-no-type parser. The module also sets this internally; exported here for visibility.
export PARSER_VERSION=tree_no_type

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "Parser version: $PARSER_VERSION"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.preprocessing_pipelines.coco.pipeline_tree_no_type \
    --chunk-size 1000 \
    --max-workers 8 \
    --worker-batch-size 500

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
