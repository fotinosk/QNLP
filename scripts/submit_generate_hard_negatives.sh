#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=12:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N gen_hard_negs
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

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
echo "Phase A0: generate hard-negative swap candidates (both parsers, one pass)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

# Smoke-test first (set SMOKE=5000 in qsub -v to only process 5k captions).
if [ -n "$SMOKE" ]; then
    echo "SMOKE RUN: limiting to $SMOKE captions"
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives --limit "$SMOKE" \
        --output data/datasets/coco_hard_neg_specs_smoke.parquet
else
    $PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives

    echo "--- Validating port against colleague's negs (exact-match subset) ---"
    $PYTHON -m qnlp.scripts.coco_multi_caption.validate_hard_negatives
fi

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
