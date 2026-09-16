#!/bin/bash
#$ -l tmem=8G
#$ -l h_rt=1:0:0
#$ -S /bin/bash
#$ -j y
#$ -N svo_merge_shards
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Run once, after every task of submit_svo_compile_array.sh has finished.
# Single-writer merge of all per-shard LMDBs into the real LMDB store.

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311

export PYTHONPATH=$PROJECT_DIR
PYTHON=$ENV_DIR/bin/python

cd $PROJECT_DIR

echo "Job started: $(date)"
$PYTHON -m qnlp.scripts.svo.merge_shard_lmdbs
echo "Job finished: $(date)"
