#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=8:0:0
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -N ttn_cifar_ablation
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

# Generic runner for TTN_CIFAR_EXPERIMENTS.md's "parallel batch plan" --
# every row is exactly this script with a different set of `qsub -v`
# overrides (IMAGE_MODEL_* config vars, PROBE_* CLI args below). One
# variable changes per job; nothing is hardcoded here so no job can
# silently clobber another's config the way earlier submit scripts did.

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

# Defaults match the document's standard CIFAR grid config; every
# IMAGE_MODEL_* var can be overridden via -v without touching this file.
export IMAGE_MODEL_IMAGE_SIZE=${IMAGE_MODEL_IMAGE_SIZE:-32}
export IMAGE_MODEL_PATCH_SIZE=${IMAGE_MODEL_PATCH_SIZE:-2}

PROBE_ARCH=${PROBE_ARCH:-ttn}
PROBE_EPOCHS=${PROBE_EPOCHS:-100}
PROBE_PATIENCE=${PROBE_PATIENCE:-20}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-128}
PROBE_LR=${PROBE_LR:-1e-3}

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID  Job name: $JOB_NAME"
echo "Running on: $(hostname)"
echo "--- IMAGE_MODEL_* env ---"
env | grep '^IMAGE_MODEL_' | sort
echo "--- probe args ---"
echo "arch=$PROBE_ARCH epochs=$PROBE_EPOCHS patience=$PROBE_PATIENCE batch_size=$PROBE_BATCH_SIZE lr=$PROBE_LR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.discoviz.diagnostic.ttn_supervised_probe \
    --dataset cifar10 --arch $PROBE_ARCH \
    --epochs $PROBE_EPOCHS --patience $PROBE_PATIENCE --batch-size $PROBE_BATCH_SIZE --lr $PROBE_LR \
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
