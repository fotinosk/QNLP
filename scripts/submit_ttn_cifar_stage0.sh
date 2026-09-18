#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=4:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N ttn_cifar_stage0
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# TTN_CIFAR_EXPERIMENTS.md Stage 0 — harness and gates. Runs, in order:
#   0.1  CIFAR-10 @ 32x32 with patch_size=2 (16x16=256 leaves, depth 4)
#   0.4  spread trace at random init, at this same 32x32 config
#   0.3  overfit-500 gate for ttn/cnn/resnet18 (dropout off, no augmentation)
# 0.2 (verify the probe reads pre-L2-norm output) is a code fix already
# applied in TTNClassifier, not a separate run.

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

# Stage 0.1: CIFAR-10 @ 32x32, patch_size=2 -> 256 leaves, depth 4.
export IMAGE_MODEL_IMAGE_SIZE=32
export IMAGE_MODEL_PATCH_SIZE=2
# Stage 0.3 explicitly asks for regularisation off for the overfit gate.
export IMAGE_MODEL_DROPOUT=0

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "IMAGE_MODEL_IMAGE_SIZE=$IMAGE_MODEL_IMAGE_SIZE  IMAGE_MODEL_PATCH_SIZE=$IMAGE_MODEL_PATCH_SIZE  IMAGE_MODEL_DROPOUT=$IMAGE_MODEL_DROPOUT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

echo "--- Stage 0.4: spread trace, random init, CIFAR-10 @ 32x32 ---"
$PYTHON -m qnlp.discoviz.diagnostic.tower_spread_trace \
    --cifar-root $PROJECT_DIR/data/cifar10 -n 128 --embedding_dim 128

echo "--- Stage 0.3: overfit-500 gate (ttn, cnn, resnet18) ---"
$PYTHON -m qnlp.discoviz.diagnostic.ttn_supervised_probe \
    --dataset cifar10 --arch all --data-root $PROJECT_DIR/data/cifar10 \
    --overfit-n 500 --overfit-epochs 200 --batch-size 64 --lr 1e-3
STATUS=$?

echo "========================================="
if [ $STATUS -eq 0 ]; then
    echo "Job finished successfully at $(date)"
else
    echo "Job FAILED (exit code $STATUS) at $(date)"
fi
echo "========================================="
exit $STATUS
