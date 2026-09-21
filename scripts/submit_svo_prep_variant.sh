#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=4:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_prep_variant
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Generic re-prep of SVO datasets from the already-clean lemmafix compiled
# data (PARSER_VERSION=lemmafix, no CCG recompile -- everything is already
# in that LMDB/derived dir): word-frequency threshold and/or split mode are
# supplied via `qsub -v`, defaulting to the original row-split variant
# (DISCOCLIP_REPRODUCTION_PLAN.md's Phase 5.2: the reference's SVO split has
# no positive-image dedup, ours does, making 0.8355 and our numbers not
# comparable -- SVO_PREP_SPLIT_MODE=row replicates their looser protocol).
# Also rebuilds the SVO-Swap set against whatever test split results.
#
# `:=` (not unconditional `export ...=`) so a qsub -v override survives --
# see submit_svo.sh/submit_svo_frozen.sh's history of this exact bug.
#
# Does NOT run the CCG-compile stage (qnlp.preprocessing_pipelines.svo.pipeline)
# -- nothing about the compiled diagrams changes, only the filter/split.

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
: "${SVO_PREP_WORD_FREQ_THRESHOLD:=50}"
: "${SVO_PREP_SPLIT_MODE:=row}"
: "${SVO_PREP_OUTPUT_SUFFIX:=_thresh50_lemmafix_rowsplit}"
export SVO_PREP_WORD_FREQ_THRESHOLD SVO_PREP_SPLIT_MODE SVO_PREP_OUTPUT_SUFFIX

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "THRESHOLD=$SVO_PREP_WORD_FREQ_THRESHOLD  SPLIT_MODE=$SVO_PREP_SPLIT_MODE  OUTPUT_SUFFIX=$SVO_PREP_OUTPUT_SUFFIX"
echo "========================================="

cd $PROJECT_DIR

echo "--- Stage 1/2: filter + split (no CCG recompile) ---"
$PYTHON -m qnlp.scripts.svo.prepare_datasets

echo "--- Stage 2/2: SVO-Swap construction against the new test split ---"
$PYTHON -m qnlp.scripts.svo.build_svo_swap

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
