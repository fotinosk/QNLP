#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=12:0:0
# h_rt bumped 2h->12h (2026-07-21) as headroom — CLIP encoding itself should be
# fast (low thousands of unique swap words, batched 512/pass), but score_stage
# also has to read every enumerate part file (~10k+ small parquet files at
# worker_batch_size=100) off /SAN before it can even start scoring, and /SAN
# I/O has been a repeated source of surprises this session. Not raised to 72h
# like the CPU enumerate job since this reserves a GPU, which is more
# inconsiderate to hold for a long time on a shared cluster if unused.
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N gen_hard_negs_score
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
export HF_HOME=$CACHE_DIR/huggingface_cache
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers_cache
export TORCH_HOME=$CACHE_DIR/torch_cache
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "[score]: CLIP-score swap-word hardness, write final coco_hard_negs*_train.parquet"
echo "Requires: the 'enumerate' stage already completed (all parts present)."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.generate_hard_negatives score || {
    echo "SCORE STAGE FAILED (exit $?) — skipping validation, job unsuccessful."
    exit 1
}

echo "--- Validating port against colleague's negs (exact-match subset) ---"
$PYTHON -m qnlp.scripts.coco_multi_caption.validate_hard_negatives

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
