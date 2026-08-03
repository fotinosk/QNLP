#!/bin/bash
# Task C4d: fix the OPTIMISER on the relational task before spending seeds on
# Question C.2. SGE ARRAY JOB -- one task per (lr, seed).
#
# WHY THIS RUNS BEFORE MORE SEEDS
# -------------------------------
# The 2026-08-02 re-run collapsed 12 of 20 seeds to chance. On the 8 that
# trained, the tower is at PARITY with its classical counterpart:
#
#     quantum_none      converged 46.2 +/- 2.5 (3/10)   462 params
#     quantum_on_wire   converged 48.0 +/- 3.8 (5/10)   494 params
#     classical_bare              49.6 +/- 5.6 (0/10)   428 params   <- both tied
#
# So capability is not the issue; survival is. And the collapses come in TWO
# kinds, which is what makes this an lr sweep rather than an init sweep:
#
#   * 3 of quantum_none's 7 collapsed AFTER LEARNING -- seed 2 reached 50.2% at
#     epoch 12, above the classical mean, held ~45-50% for seven epochs, then
#     fell to ~24% for the last ten. That is a learning-rate/schedule failure.
#   * the other 4 never left chance at all.
#
# The payoff is measured, not hoped for: pooled std is 11.5 with the collapses in
# and ~3 without, which is the difference between ~40 and ~9 seeds per arm to
# resolve the observed +5.3 C.2 effect -- roughly 260 h against 60 h of cluster
# time. Fixing the optimiser IS the cheap route to Question C.2.
#
# PRIMARY METRIC IS THE COLLAPSE RATE, NOT ACCURACY. Pick the config that trains
# 10/10 seeds, not the one with the best contaminated mean. `combine_clevr` now
# splits "diverged after learning" from "never left chance" -- read that split.
#
# WHY 90 EPOCHS, AND WHY lr AND EPOCHS TOGETHER
# ---------------------------------------------
# C3c (2026-08-02) showed the tower needs ~3x the 30-epoch budget on its hardest
# head, and several CONVERGED seeds here were still climbing at the cutoff
# (quantum_none seed 6 ends 52.3, seed 9 ends 49.8; on_wire seed 1 ends 54.7).
# So 30 epochs under-trains this task too. But a longer budget also gives an
# unstable run more time to diverge, so the two cannot be swept sequentially --
# a good lr at 30 epochs may not survive to 90. Every arm here runs at 90.
#
# Layout: 3 lrs x 5 seeds = 15 tasks, quantum_none only (the C.2 baseline arm).
# Do NOT sweep on_wire too -- that doubles the cost to answer a question about
# the optimiser, which is arm-independent.
#
# PREREQUISITE: the nearest-partner relation cache (manifest must show
#               "relation_partner_rule": "nearest").
#
# Submit:  qsub scripts/submit_c4_lr_sweep.sh
#$ -l tmem=16G
#$ -l h_rt=24:0:0
#$ -S /bin/bash
#$ -j y
#$ -N c4_lr_sweep
#$ -t 1-15
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

# One BLAS thread per task: the scheduler gives us one slot, and letting torch
# spawn a thread per core makes concurrent array tasks fight each other.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=$ENV_DIR/bin/python
cd $PROJECT_DIR

# lr=0.1 is excluded deliberately: C3b measured it BROKEN (2/3 collapsed, every
# head at chance). Sweeping it again would burn 5 slots to reconfirm a known
# failure. The sweep goes DOWN from the current 0.03, which is the direction the
# divergence signature points.
LRS=(0.003 0.01 0.03)
N_SEEDS=5
EPOCHS=90

IDX=$((SGE_TASK_ID - 1))
LR=${LRS[$((IDX / N_SEEDS))]}
SEED=$((IDX % N_SEEDS))

echo "========================================="
echo "C4d | task $SGE_TASK_ID | lr=$LR | seed=$SEED | epochs=$EPOCHS | arm=quantum_none"
echo "started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

# --skip-classical: the classical arms are already measured at this data and do
# not depend on the quantum lr. Including them would repeat the 60-config tuning
# sweep in all 15 tasks.
$PYTHON -m qnlp.image_tower.classification.clevr.run_c4_relational \
    --img-size 16 \
    --readout top_layer_multi_pauli \
    --positional none \
    --seeds $SEED \
    --lr $LR \
    --epochs $EPOCHS \
    --skip-classical \
    --out c4d_lr${LR}_s$SEED \
    --out-suffix _lr${LR}_s${SEED} || exit 1

echo "Task $SGE_TASK_ID finished at $(date)"

# ---------------------------------------------------------------------
# Merge, one call PER LR (each is its own arm):
#
#   for LR in 0.003 0.01 0.03; do
#     $PYTHON -m qnlp.image_tower.classification.clevr.combine_clevr \
#         --prefix c4d_lr$LR --task relations --seeds 0 1 2 3 4 \
#         --arms quantum_none
#   done
#
# READ THE COLLAPSE SPLIT FIRST, then the converged-seed mean. The config to
# carry forward is the one that stops seeds collapsing -- a better mean with the
# same collapse rate has not fixed anything, it has just reshuffled which seeds
# survived.
#
# THEN, and only then, re-run the full C.2 comparison (both positional arms) at
# the winning lr and 90 epochs. At a converged-only std of ~3, that needs ~9
# seeds/arm rather than the ~40 the contaminated variance demands.
# ---------------------------------------------------------------------
