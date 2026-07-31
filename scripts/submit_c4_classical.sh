#!/bin/bash
# Task C4, classical controls. Single job, NOT an array: these arms take seconds
# per seed, and the 60-config tuning sweep must run exactly once rather than
# being repeated by every array task.
#
# Runs classical_bare (CP, rank swept freely), classical_full (CP + residual +
# dropout -- the structural counterpart to the quantum tower), mlp_reference and
# mlp_param_matched. Standing requirement 1: never report a quantum accuracy
# without a size-matched classical reference beside it.
#
# 10 seeds: cheap, and tighter classical estimates shrink the resolution limit on
# every comparison. Welch handles the unequal n against the quantum arms.
#
# PREREQUISITE: qsub scripts/submit_clevr_build_data.sh
# Submit:       qsub scripts/submit_c4_classical.sh
#$ -l tmem=16G
#$ -l h_rt=6:0:0
#$ -S /bin/bash
#$ -j y
#$ -N c4_classical
#$ -M ucapfky@ucl.ac.uk
#$ -m ae
#$ -R y
#$ -cwd
#$ -o /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/
#$ -e /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs/

mkdir -p /SAN/intelsys/discoviz/fotinos/QNLP/job_outputs
mkdir -p /SAN/intelsys/discoviz/fotinos/cache/.pip_cache

PROJECT_DIR=/SAN/intelsys/discoviz/fotinos/QNLP
ENV_DIR=/SAN/intelsys/discoviz/envs/qnlp311
CACHE_DIR=/SAN/intelsys/discoviz/fotinos/cache

export PYTHONPATH=$PROJECT_DIR
export HF_HOME=$CACHE_DIR/huggingface_cache
export TORCH_HOME=$CACHE_DIR/torch_cache
export PIP_CACHE_DIR=$CACHE_DIR/.pip_cache
export MPLCONFIGDIR=$CACHE_DIR/.matplotlib_cache
export PYTHONPYCACHEPREFIX=$CACHE_DIR/pycache
export XDG_CACHE_HOME=$CACHE_DIR/.xdg_cache

PYTHON=$ENV_DIR/bin/python
cd $PROJECT_DIR

echo "========================================="
echo "C4 classical controls | started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

# --skip-quantum: the quantum model is still CONSTRUCTED once, to size the tuning
# budget at 1.5x its parameter count, but never trained here. Without this flag
# the job would also train 10 quantum seeds (~30 h) that the array job is already
# covering.
$PYTHON -m qnlp.image_tower.classification.clevr.run_c4_relational \
    --img-size 16 \
    --readout top_layer_multi_pauli \
    --positional none \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --skip-quantum \
    --out c4_cls \
    --out-suffix _classical || exit 1

echo "Job finished at $(date)"
