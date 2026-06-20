#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=4:0:0
#$ -l gpu=false
#$ -S /bin/bash
#$ -j y
#$ -N coco_aro_style_create_dataset
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/.pip_cache

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

export PYTHONPATH=$PROJECT_DIR
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache

# Dataset strategy: bm25_hard (top-10), bm25_medium (top-50), random
export DATASET_STRATEGY=${DATASET_STRATEGY:-bm25_hard}
export DATASET_TOP_K=${DATASET_TOP_K:-10}

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Strategy: $DATASET_STRATEGY  top_k: $DATASET_TOP_K"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_aro_style.create_dataset \
    --strategy $DATASET_STRATEGY \
    --top_k $DATASET_TOP_K

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
