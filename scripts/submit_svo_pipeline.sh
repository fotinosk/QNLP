#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=48:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_preprocess
#$ -pe smp 8
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# CPU-only data prep for SVO-Probes: atlas ingest -> CCG compile -> word-freq
# filter + 60/20/20 split -> SVO-Swap construction. Run once before
# submit_svo.sh. Requires data/svo/raw/{svo_probes_corrected.csv,images,images_old}
# to already be rsynced to this project dir.

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

echo "--- Stage 1/4: atlas ingest ---"
$PYTHON -m qnlp.scripts.load_svo_to_atlas

echo "--- Stage 2/4: CCG compile (corrected_sentence) ---"
$PYTHON -m qnlp.preprocessing_pipelines.svo.pipeline

echo "--- Stage 3/4: word-frequency filter + 60/20/20 split ---"
$PYTHON -m qnlp.scripts.svo.prepare_datasets

echo "--- Stage 4/4: SVO-Swap construction ---"
$PYTHON -m qnlp.scripts.svo.build_svo_swap

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
