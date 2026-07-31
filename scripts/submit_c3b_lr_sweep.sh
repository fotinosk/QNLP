#!/bin/bash
# Task C3b: quantum learning-rate sweep. SGE ARRAY JOB -- one task per (lr, seed).
#
# WHY THIS EXISTS. C3 compared a quantum arm at a SINGLE inherited lr=0.03
# against classical arms that each got a 60-config search (4 lrs x 3 bond dims x
# 5 ranks). That asymmetry is why the C3 colour result (27.0% vs classical 76.6%)
# currently cannot be reported as anything but confounded -- it cannot separate
# architecture from optimisation. This job closes it.
#
# The lr was inherited from Phase 1, where the loss had ONE head. CLEVR sums FOUR
# cross-entropies, so the effective gradient scale is ~4x larger and 0.03 was
# never re-examined for that. C1 already showed this session that a single
# untuned lr can drive every head to its floor.
#
# Layout: 4 lrs x 3 seeds = 12 tasks.
#   lr index = (SGE_TASK_ID-1) / 3 -> {0.003, 0.01, 0.03, 0.1}
#   seed     = (SGE_TASK_ID-1) % 3 -> {0, 1, 2}
# lr=0.03 is included deliberately: it reproduces C3's quantum arm inside the
# same sweep, so the comparison is like-for-like rather than against a number
# measured on a different day.
#
# PREREQUISITE: qsub scripts/submit_clevr_build_data.sh
# Submit:       qsub scripts/submit_c3b_lr_sweep.sh
#$ -l tmem=16G
#$ -l h_rt=12:0:0
#$ -S /bin/bash
#$ -j y
#$ -N c3b_lr_sweep
#$ -t 1-12
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=$ENV_DIR/bin/python
cd $PROJECT_DIR

LRS=(0.003 0.01 0.03 0.1)
IDX=$((SGE_TASK_ID - 1))
LR=${LRS[$((IDX / 3))]}
SEED=$((IDX % 3))
TAG=lr${LR}

echo "========================================="
echo "C3b | task $SGE_TASK_ID | lr=$LR | seed=$SEED"
echo "started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

$PYTHON -m qnlp.image_tower.classification.clevr.run_c3_attributes \
    --img-size 16 \
    --readout top_layer_multi_pauli \
    --only-quantum \
    --lr $LR \
    --seeds $SEED \
    --out c3b_${TAG}_s${SEED} \
    --out-suffix _${TAG}_s${SEED} || exit 1

echo "Task $SGE_TASK_ID finished at $(date)"

# ---------------------------------------------------------------------
# Merging: each lr is its own arm. combine_clevr globs
# {prefix}_s{seed}_{arm}_partial.json, and these are written as
# c3b_lr<LR>_s<SEED>_quantum_coherent_partial.json -- so merge PER LR:
#
#   for LR in 0.003 0.01 0.03 0.1; do
#     $PYTHON -m qnlp.image_tower.classification.clevr.combine_clevr \
#         --prefix c3b_lr$LR --seeds 0 1 2 --arms quantum_coherent --params 462
#   done
#
# Then compare the four lrs head-by-head, and read COLOUR specifically -- that is
# the head the sweep exists to settle.
#
# IF A BETTER lr IS FOUND: the whole C3 comparison must be re-run at it, because
# quoting a tuned quantum arm against classical arms tuned separately would just
# invert the asymmetry rather than remove it.
# ---------------------------------------------------------------------
