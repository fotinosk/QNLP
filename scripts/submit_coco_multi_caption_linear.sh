#!/bin/bash
#$ -l tmem=32G
#$ -l h_rt=48:0:0
#$ -l gpu=true
# The linear text model (~1.07B params) + AdamW state needs ~17 GB; a small
# ~11 GB card OOMs. Request ~20 GB so it lands on a 24 GB-class GPU. If your
# cluster uses a different resource name than gpu_mem, adjust this line.
#$ -l gpu_mem=20G
#$ -S /bin/bash
#$ -j y
#$ -N coco_mc_linear
#$ -M ucapfky@ucl.ac.uk
#$ -m abe
#$ -R y
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

# --- Experiment config: LINEAR contractions, TTN image tower ---
# Linear mode reads the coco_single_caption_{split}.parquet datasets (no path column).
export ML_EMBEDDING_DIM=512
export ML_BOND_DIM=10
export ML_USE_NON_LINEAR_CONTRACTIONS=false
# Linear multilinear contraction is prone to weight-norm drift → overflow. Use a
# lower text LR and a higher text weight decay (the restoring force on the norms)
# to keep the model in its well-conditioned regime and avoid non-finite embeddings.
export ML_TEXT_LR=0.0005
export ML_TEXT_WEIGHT_DECAY=0.01
export ML_BATCH_SIZE=256

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MLFLOW_DISABLED=true
export MLFLOW_RUN_NAME="${RUN_NAME:-coco_mc_linear}"

PYTHON=$ENV_DIR/bin/python

echo "========================================="
echo "Job started: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on: $(hostname)"
echo "Embedding dim: $ML_EMBEDDING_DIM"
echo "Bond dim: $ML_BOND_DIM"
echo "Non-linear: $ML_USE_NON_LINEAR_CONTRACTIONS"
echo "Text LR: $ML_TEXT_LR  Text weight decay: $ML_TEXT_WEIGHT_DECAY"
echo "Batch size: $ML_BATCH_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "========================================="

cd $PROJECT_DIR

$PYTHON -m qnlp.scripts.coco_multi_caption.run

echo "========================================="
echo "Job finished successfully at $(date)"
echo "========================================="
