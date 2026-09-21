#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=4:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_rowsplit
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# DISCOCLIP_REPRODUCTION_PLAN.md's Phase 5.2 finding: the reference's SVO
# split has no positive-image dedup (39.4% of its test rows share their
# exact caption/positive-image pair with a train row), while ours does --
# making 0.8355 and our numbers not comparable tasks. This regenerates the
# already-clean lemmafix compiled data (PARSER_VERSION=lemmafix, no CCG
# recompile needed -- everything is already in that LMDB/derived dir) with
# a plain per-row 60/20/20 split (SVO_PREP_SPLIT_MODE=row) instead, then
# rebuilds the SVO-Swap set against the new test split.
#
# Does NOT run the CCG-compile stage (qnlp.preprocessing_pipelines.svo.pipeline)
# -- nothing about the compiled diagrams changes, only the split assignment.

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

export PARSER_VERSION=lemmafix
export SVO_PREP_WORD_FREQ_THRESHOLD=50
export SVO_PREP_SPLIT_MODE=row
export SVO_PREP_OUTPUT_SUFFIX=_thresh50_lemmafix_rowsplit

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "SPLIT_MODE=$SVO_PREP_SPLIT_MODE  OUTPUT_SUFFIX=$SVO_PREP_OUTPUT_SUFFIX"
echo "========================================="

cd $PROJECT_DIR

echo "--- Stage 1/2: row-split + 60/20/20 (no CCG recompile) ---"
$PYTHON -m qnlp.scripts.svo.prepare_datasets

echo "--- Stage 2/2: SVO-Swap construction against the new test split ---"
$PYTHON -m qnlp.scripts.svo.build_svo_swap

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
