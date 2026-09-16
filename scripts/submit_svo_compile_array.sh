#!/bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=4:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_compile_array
#$ -t 1-16
#$ -pe smp 2
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Pre-warms the LMDB compile cache in parallel across many small/medium nodes
# (this cluster is mostly 4-8 core nodes; a job array of small tasks schedules
# far faster than one big multi-core request waiting for a large node).
#
# Each of the 16 tasks compiles a disjoint 1/16 slice of the ~8.4k unique SVO
# captions into its own LMDB (data/sentence_mapping_svo_shards/shard_<i>) to
# avoid concurrent-write risk on the shared store over NFS.
#
# Run AFTER submit_svo_pipeline.sh's atlas-ingest stage would run (needs
# data/atlases/svo/data_manifest.parquet to exist) — either run atlas ingest
# manually first, or just run this before submit_svo_pipeline.sh with atlas
# ingest done as a quick separate step (see scripts/submit_svo_pipeline.sh).
#
# After ALL array tasks finish, run scripts/submit_svo_merge_shards.sh once,
# THEN submit_svo_pipeline.sh (its CCG-compile stage becomes a fast cache hit).

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
echo "Job ID: $JOB_ID  Task: $SGE_TASK_ID/$SGE_TASK_LAST"
echo "Running on: $(hostname)"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.svo.compile_shard --max-workers 2

echo "========================================="
echo "Task $SGE_TASK_ID finished at $(date)"
echo "========================================="
