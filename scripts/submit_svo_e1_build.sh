#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=8:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_e1_build
#$ -pe smp 4
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# DISCOCLIP_REPRODUCTION_PLAN.md's E1: compile discoclip's own released
# train/val/test CSVs (data/svo/discoclip_reference/) through OUR pipeline,
# with NO word-frequency filtering, into a dedicated isolated LMDB (not
# the shared or lemmafix store). Also builds the matching SVO-Swap set
# from their svo_probes_swapped.csv. See
# qnlp/scripts/svo/build_reference_dataset.py's docstring for full detail.

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

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.svo.build_reference_dataset
STATUS=$?

echo "========================================="
if [ $STATUS -eq 0 ]; then
    echo "Job finished successfully at $(date)"
else
    echo "Job FAILED (exit code $STATUS) at $(date)"
fi
echo "========================================="
exit $STATUS
