#!/bin/bash
#$ -l tmem=8G
#$ -l h_rt=12:0:0
# tmem is PER SLOT; 5 slots x 8G = 40G for the 4 CCG workers (+ main). Without
# -pe smp the job gets one slot's memory and OOM-kills a worker, hanging the pool.
# Winoground is tiny (~800 sentences) so 4 workers is ample.
#$ -pe smp 5
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -N wino_process_tree
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
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

echo "Compiling Winoground captions into tree-no-type CCG diagrams..."
$PYTHON -m qnlp.preprocessing_pipelines.winoground.pipeline_tree_no_type

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
