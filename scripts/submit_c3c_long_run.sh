#!/bin/bash
# Task C3c: does a longer epoch budget rescue the colour head?
# SGE ARRAY JOB -- one task per (lr, seed) at 90 epochs instead of 30.
#
# WHY THIS IS THE LAST OPEN EXPLANATION FOR COLOUR.
#   * readout width  -- EXCLUDED: classical_full wins colour through a NARROWER
#                       head (bond_dim 8) than the quantum model's 12 values.
#   * learning rate  -- EXCLUDED by C3b: swept over a 30x range, colour never got
#                       within 40 pts of classical_full.
#   * epoch budget   -- UNTESTED. Colour peaked at epoch 29 OF 30 in C3, gaining
#                       +6.1 pts between the first and second halves of training,
#                       while size plateaued by epoch 3. The 30-epoch budget was
#                       inherited from a 4-CLASS synthetic task and never
#                       re-examined for an 8-WAY head.
#
# WHY THREE LEARNING RATES AND NOT JUST 0.03. Lower lr converges more slowly, so
# the low-lr arms are the MOST likely to be under-trained at 30 epochs -- testing
# only lr=0.03 would probe the least under-trained config and would be the
# weakest possible version of this experiment. C3b's colour ordering
# (0.01 > 0.003 > 0.03) is itself unresolved at 3 seeds, and more budget could
# reorder it.
#
# Layout: 3 lrs x 3 seeds = 9 tasks.
#   lr index = (SGE_TASK_ID-1) / 3 -> {0.003, 0.01, 0.03}
#   seed     = (SGE_TASK_ID-1) % 3 -> {0, 1, 2}
# Same seeds as C3/C3b, so 30-epoch and 90-epoch runs are directly comparable.
#
# READ AS A DIAGNOSTIC. If colour climbs, the 30-epoch protocol is the binding
# constraint and the WHOLE C3 comparison must be re-run at the longer budget --
# the classical arms would need the same extension, or a training-budget
# asymmetry simply replaces the tuning one C3b just removed.
#
# h_rt: 90 epochs is 3x a C3 task. ~9 h with lightning.qubit, ~36 h without
# (default.qubit is ~4x slower). 48 h is set to cover the bad case; lower it to
# 16 h if lightning is working, so the scheduler queues these sooner.
#
# PREREQUISITE: data present + verify_cluster_data prints READY.
# Submit: qsub scripts/submit_c3c_long_run.sh
#$ -l tmem=16G
#$ -l h_rt=48:0:0
#$ -S /bin/bash
#$ -j y
#$ -N c3c_long_run
#$ -t 1-9
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

EPOCHS=90
LRS=(0.003 0.01 0.03)
IDX=$((SGE_TASK_ID - 1))
LR=${LRS[$((IDX / 3))]}
SEED=$((IDX % 3))

echo "========================================="
echo "C3c | task $SGE_TASK_ID | lr=$LR | seed=$SEED | epochs=$EPOCHS"
echo "started $(date) | host $(hostname) | job $JOB_ID"
echo "========================================="

$PYTHON -m qnlp.image_tower.classification.clevr.run_c3_attributes \
    --img-size 16 \
    --readout top_layer_multi_pauli \
    --only-quantum \
    --lr $LR \
    --epochs $EPOCHS \
    --seeds $SEED \
    --out c3c_lr${LR}_s${SEED} \
    --out-suffix _long${EPOCHS}_lr${LR}_s${SEED} || exit 1

echo "Task $SGE_TASK_ID finished at $(date)"

# ---------------------------------------------------------------------
# Merge per lr, then compare against the SAME lr at 30 epochs (C3b):
#
#   for LR in 0.003 0.01 0.03; do
#     $PYTHON -m qnlp.image_tower.classification.clevr.combine_clevr \
#         --prefix c3c_lr$LR --seeds 0 1 2 --arms quantum_coherent --params 462
#   done
#
# The decisive question is not "is 90 > 30" on the mean, but whether the COLOUR
# curve is still rising at epoch 90. Check the per-epoch curves in the partial
# JSONs, exactly as the epoch-29-of-30 finding was read off C3.
# ---------------------------------------------------------------------
