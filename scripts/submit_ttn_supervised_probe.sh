#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=8:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N ttn_probe
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Supervised capacity probe: TTNImageModel + linear classifier vs. a small
# CNN and a from-scratch ResNet-18, on CIFAR-10 (64x64) and on SVO's own
# images labeled with their top-20 `obj` classes. No text tower, no
# contrastive loss, no CCG — isolates whether the image tower itself has
# the capacity to ground images, independent of everything else the SVO
# campaign has been tuning. See
# qnlp/discoviz/diagnostic/ttn_supervised_probe.py's docstring.

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

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Using Python: $PYTHON"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.discoviz.diagnostic.ttn_supervised_probe \
    --dataset both --arch all --label-col obj --top-k 20 \
    --epochs 30 --batch-size 128 --lr 1e-3 --patience 6 \
    --data-root $PROJECT_DIR/data/cifar10
STATUS=$?

echo "========================================="
if [ $STATUS -eq 0 ]; then
    echo "Job finished successfully at $(date)"
else
    echo "Job FAILED (exit code $STATUS) at $(date)"
fi
echo "========================================="
exit $STATUS
