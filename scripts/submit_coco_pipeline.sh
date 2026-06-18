#!/bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=24:0:0
#$ -S /bin/bash
#$ -j y
#$ -N coco_preprocessing
#$ -pe smp 5
#$ -R y

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
CONDA_ENV=/SAN/intelsys/discoviz/envs/qnlp311

export PYTHONPATH=$PROJECT_DIR
export NLTK_DATA=/SAN/intelsys/discoviz/fotinos/nltk_data
export GLOBAL_CONSTANTS_BOBCAT_CACHE_PATH=/SAN/intelsys/discoviz/fotinos/cache/bobcat/diskcache

echo "Job started: $(date)"
echo "Running on: $(hostname)"

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

cd $PROJECT_DIR
python -m qnlp.preprocessing_pipelines.coco.pipeline \
    --chunk-size 500 \
    --max-workers 4 \
    --worker-batch-size 1000

echo "Job finished: $(date)"
