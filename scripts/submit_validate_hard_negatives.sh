#!/bin/bash
#$ -l tmem=24G
#$ -l h_rt=2:0:0
#$ -S /bin/bash
#$ -j y
#$ -N validate_hard_negs
#$ -M ucapfky@ucl.ac.uk
#$ -m a
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Standalone port-validation of the generated hard negatives against the
# colleague's released negs (Jaccard over (t, {w1,w2}) sets on the exact-match
# caption subset). Pure CPU (polars + NLTK) — no GPU needed; killed on the
# login node by its memory limits (loads ~1.2M candidate rows + the 600k-line
# negs jsonl into python dicts), hence a proper compute-node job.

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/nltk_data

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

export PYTHONPATH=$PROJECT_DIR
export NLTK_DATA=$CACHE_DIR/nltk_data
export MPLCONFIGDIR=$CACHE_DIR/.matplotlib_cache
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "[validate]: hard-negative port validation vs colleague's negs (lemma-normalized)"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.validate_hard_negatives

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
