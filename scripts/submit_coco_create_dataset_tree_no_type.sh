#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=72:0:0
# Path planning (dp optimizer) is single-threaded, so the 5 slots are for memory
# headroom (5 x 16G) to hold the ~566k-atom frame + path cache, not parallelism.
#$ -pe smp 5
#$ -R y
#$ -S /bin/bash
#$ -j y
#$ -N coco_create_ds_tree
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

# Always non-linear: the single nlc dataset serves linear training too
# (-> coco_single_caption_nlc_tree_no_type_{train,val,test}).
echo "Starting tree-no-type COCO dataset creation at $(date)"
$PYTHON -m qnlp.scripts.coco_single_caption.create_dataset_tree_no_type

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
