#!/bin/bash
# Task C4 (Question C.2): CLEVR spatial relations, with and without explicit
# per-patch position. SGE ARRAY JOB -- one task per (positional arm, seed).
#
# This is the REQUIRED experiment of Phase 2. R3 rejected the spatial ancilla at
# -19.5 pts, but on a translation-invariant single-object task where position
# barely affects the label, so the negative result was near-structural. Here
# POSITION IS THE LABEL, and this is the only place the question can be settled.
#
# Layout: tasks 1-10  -> positional=none,    seeds 0-9
#         tasks 11-20 -> positional=on_wire, seeds 0-9
# 10 seeds because cluster time is free; locally we were held to 3, which is why
# several C2 comparisons came back unresolved.
#
# Checkpoints land as c4_s<seed>_<arm>_partial.json, which is exactly the layout
# combine_clevr.py globs. Classical controls run separately (seconds, not hours)
# via submit_c4_classical.sh -- do not fold them in here or every array task
# would redundantly repeat the 60-config tuning sweep.
#
# PREREQUISITE: qsub scripts/submit_clevr_build_data.sh   (and let it finish)
#
# Submit:  qsub scripts/submit_c4_relational.sh
# Merge:   see the tail of this file
#$ -l tmem=16G
#$ -l h_rt=12:0:0
#$ -S /bin/bash
#$ -j y
#$ -N c4_relational
#$ -t 1-20
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
# spawn a thread per core makes concurrent array tasks fight each other. Local
# contention already corrupted two measurements this phase.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON=$ENV_DIR/bin/python
cd $PROJECT_DIR

N_SEEDS=10
IDX=$((SGE_TASK_ID - 1))
if [ $IDX -lt $N_SEEDS ]; then
    ARM=none
    SEED=$IDX
else
    ARM=on_wire
    SEED=$((IDX - N_SEEDS))
fi

echo "========================================="
echo "C4 | task $SGE_TASK_ID | positional=$ARM | seed=$SEED"
echo "started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

$PYTHON -m qnlp.image_tower.classification.clevr.run_c4_relational \
    --img-size 16 \
    --readout top_layer_multi_pauli \
    --positional $ARM \
    --seeds $SEED \
    --skip-classical \
    --out c4_s$SEED \
    --out-suffix _${ARM}_s${SEED} || exit 1

echo "Task $SGE_TASK_ID finished at $(date)"

# ---------------------------------------------------------------------
# After the array AND submit_c4_classical.sh have both finished:
#
#   $PYTHON -m qnlp.image_tower.classification.clevr.combine_clevr \
#       --prefix c4 --task relations --seeds 0 1 2 3 4 5 6 7 8 9 \
#       --arms quantum_none quantum_on_wire
#
# READ THE VIABILITY GUARD FIRST. If quantum_none does not clear the majority
# floor, the two arms are both at chance and their comparison measures nothing --
# the runner suppresses the Question C.2 verdict in that case by design.
#
# ESCALATION: only if on_wire shows a RESOLVED effect is positional='ancilla2'
# (18 qubits, ~4x cost) worth building, to confirm the mechanism rather than the
# encoding. On a null, stop.
# ---------------------------------------------------------------------
