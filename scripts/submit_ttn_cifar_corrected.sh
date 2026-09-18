#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=6:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N ttn_cifar_corrected
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# TTN_CIFAR_EXPERIMENTS.md — corrected full CIFAR-10 run, post Stage 0.2 fix
# (TTNClassifier now reads TTNImageModel's pre-L2-norm output), at 32x32
# with patch_size=2 (this document's target config), including the
# raw-pixel logistic regression baseline (~0.40 is the document's explicit
# minimum bar for TTN, not just the 0.10 majority baseline).

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export IMAGE_MODEL_IMAGE_SIZE=32
export IMAGE_MODEL_PATCH_SIZE=2

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "IMAGE_MODEL_IMAGE_SIZE=$IMAGE_MODEL_IMAGE_SIZE  IMAGE_MODEL_PATCH_SIZE=$IMAGE_MODEL_PATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.discoviz.diagnostic.ttn_supervised_probe \
    --dataset cifar10 --arch all --data-root $PROJECT_DIR/data/cifar10 \
    --epochs 40 --batch-size 128 --lr 1e-3 --patience 8
STATUS=$?

echo "========================================="
if [ $STATUS -eq 0 ]; then
    echo "Job finished successfully at $(date)"
else
    echo "Job FAILED (exit code $STATUS) at $(date)"
fi
echo "========================================="
exit $STATUS
