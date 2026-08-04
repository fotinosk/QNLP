#!/bin/bash
# Task C6: the shape-binding compositional probe -- QUANTUM ARMS.
# SGE ARRAY JOB -- one task per (positional arm, seed).
#
# THE CLOSING EXPERIMENT OF THE INVESTIGATION.
#
# It is the first task in the phase that can discriminate the project's premise.
# `mlp_reference` beats every TTN arm on C3 and on C4's relations -- but neither
# of those tasks is compositional (one is pure perception, the other a
# single-referent directional readout), so those wins say nothing about whether
# the TTN captures compositional structure. C6 builds a task where the
# class-conditional marginals are IDENTICAL BY CONSTRUCTION -- one large and one
# small object in every image, label = which is on the left -- so a bag-of-features
# model is at chance provably.
#
# THE HEADLINE IS AN INTERACTION, AND BOTH DIRECTIONS ARE PRE-REGISTERED:
# on perception the MLP beats the TTNs; if that advantage narrows or reverses on
# binding the compositional claim is supported, and if it holds then the
# compositional advantage does not hold in the vision tower and THAT is what gets
# written up. Do not decide which reading applies after seeing the numbers.
#
# It also gives Question C.2(ii) its sharpest test: binding requires position to
# be CONJOINED WITH CONTENT ("large AND left"), not merely read out ("something is
# left"), so C4's +1.8 (limit 5.4) may have reflected the low positional demand
# of a directional readout rather than the encoding.
#
# Layout: tasks 1-15  -> positional=none,    seeds 0-14
#         tasks 16-30 -> positional=on_wire, seeds 0-14
#
# 15 seeds because ONLY THE QUANTUM ARMS COLLAPSE (~45% in C4; the classical and
# MLP arms were 0/10). 15 leaves ~8 healthy. The cheap arms run separately at 10
# via submit_c6_classical.sh -- 10 rather than fewer because a tighter classical
# estimate shrinks the MDE on every quantum-vs-classical comparison, and they
# cost minutes.
#
# 90 epochs: C3c showed the tower needs ~3x the 30-epoch budget, and several
# CONVERGED C4 seeds were still climbing at the cutoff.
#
# Cosine decay is ON by default (disable with --no-cosine). It targets a measured
# failure mode: 3 of quantum_none's 7 C4 collapses happened AFTER the seed
# reached 35-50%, one peaking at 50.2% before falling back to chance. It cannot
# help the 4 that never left chance, so expect at best half the collapses to go.
#
# PREREQUISITES:
#   1. data/datasets/clevr_binding_size_32_{train,val}.npz  (build_clevr_binding --attribute size)
#   2. THE MONTAGE HAS BEEN LOOKED AT. A left/right swap would still produce a
#      perfectly balanced dataset and a plausible 50% result.
#   3. The manipulation check in submit_c6_cheap.sh has PASSED. If any arm beats chance on shuffled
#      input the composites leak a non-positional cue and every number from this
#      job is uninterpretable.
#
# Submit:  qsub scripts/submit_c6_binding.sh
#$ -l tmem=16G
# 48h, not 24. MEASURED, not guessed: the tasks that completed took 22-23 h EACH
# on a node carrying only four of them. 24 h was set from a ~9 h estimate that
# was wrong by 2.5x -- the 32x32 canvas pushes 4x the pixels through the
# classical patch encoder, on top of 90 epochs and 12 observables.
#$ -l h_rt=48:0:0
# -tc 8: CAP CONCURRENT TASKS. Without it SGE packed 22 of the 30 tasks onto a
# single node (saunders-608-9); all 22 were killed mid-epoch with no traceback,
# while the 8 spread across two other nodes finished. That is the failure mode
# already in the cost model -- "run at most 4-5 concurrent workers; ten
# concurrent 16-qubit processes exhausted 18 GB and six were killed silently" --
# and it cost a full day and the entire quantum_none arm.
#$ -tc 8
#$ -S /bin/bash
#$ -j y
#$ -N c6_binding
#$ -t 1-30
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

N_SEEDS=15
IDX=$((SGE_TASK_ID - 1))
if [ $IDX -lt $N_SEEDS ]; then
    ARM=none
    SEED=$IDX
else
    ARM=on_wire
    SEED=$((IDX - N_SEEDS))
fi

echo "========================================="
echo "C6 | task $SGE_TASK_ID | positional=$ARM | seed=$SEED | 90 epochs | cosine decay"
echo "started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

$PYTHON -m qnlp.image_tower.classification.clevr.run_c6_binding \
    --img-size 32 \
    --readout top_layer_multi_pauli \
    --positional $ARM \
    --attribute size \
    --seeds $SEED \
    --epochs 90 \
    --only-quantum \
    --out c6_s$SEED \
    --out-suffix _${ARM}_s${SEED} || exit 1

echo "Task $SGE_TASK_ID finished at $(date)"

# ---------------------------------------------------------------------
# After this array AND submit_c6_classical.sh have finished:
#
#   $PYTHON -m qnlp.image_tower.classification.clevr.combine_clevr \
#       --prefix c6 --task binding --img-size 32 \
#       --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
#       --arms quantum_none quantum_on_wire
#
# READ THE COLLAPSE SPLIT FIRST -- combine_clevr now separates "diverged after
# learning" from "never left chance", and they call for different fixes. Quote
# the converged subgroup as agreed, with the collapse rate printed beside it so
# the post-hoc conditioning stays visible.
#
# ESCALATION: ancilla2 only if on_wire shows a RESOLVED effect. Note it would be
# confounded anyway -- on_wire already carries +32 parameters over none.
# ---------------------------------------------------------------------
