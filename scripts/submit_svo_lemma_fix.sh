#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=48:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_lemma_fix
#$ -pe smp 2
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# DISCOCLIP_REPRODUCTION_PLAN.md's "corrected fix": recompile the SVO atlas
# with the new post-parse SymbolLemmatizeStep, under a fresh PARSER_VERSION
# tag so this writes to a brand-new LMDB (data/sentence_mapping_lemmafix/)
# and derived dir (derived_lemmafix/) instead of mutating the shared,
# text-hash-keyed bobcat cache every other project (ARO/COCO/CLEVR) also
# reads from — a fresh LMDB also forces every sentence to be genuinely
# recompiled (not skipped as a cache hit), which is required since
# CCGCompilerStep only ever relabels symbols for rows it freshly compiles
# this run.
#
# Combined with R1's word-freq threshold=50 (the only Phase-2 fix that
# individually cleared the +0.03 noise floor) via a new output suffix so
# this doesn't collide with any existing dataset variant. Does NOT rerun
# load_svo_to_atlas.py — the raw ingested atlas (data_manifest.parquet) is
# unversioned/shared and already exists; only the CCG-compile stage onward
# needs the new parser version.

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

export PARSER_VERSION=lemmafix
export SVO_PREP_WORD_FREQ_THRESHOLD=50
export SVO_PREP_OUTPUT_SUFFIX=_thresh50_lemmafix

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "PARSER_VERSION=$PARSER_VERSION"
echo "OUTPUT_SUFFIX=$SVO_PREP_OUTPUT_SUFFIX"
echo "========================================="

cd $PROJECT_DIR

echo "--- Stage 1/3: CCG compile (fresh LMDB, SymbolLemmatizeStep applied) ---"
$PYTHON -m qnlp.preprocessing_pipelines.svo.pipeline

echo "--- Stage 2/3: word-frequency filter + 60/20/20 split ---"
$PYTHON -m qnlp.scripts.svo.prepare_datasets

echo "--- Stage 3/3: SVO-Swap construction (relabeled symbols) ---"
$PYTHON -m qnlp.scripts.svo.build_svo_swap

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
