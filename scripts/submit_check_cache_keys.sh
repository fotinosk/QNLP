#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=1:0:0
#$ -S /bin/bash
#$ -j y
#$ -N check_cache_keys
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311

export PYTHONPATH=$PROJECT_DIR
PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "========================================="

cd $PROJECT_DIR
$PYTHON scripts/check_cache_keys.py

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
