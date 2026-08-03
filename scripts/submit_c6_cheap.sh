#!/bin/bash
# Task C6 + C3d: everything that is NOT a quantum arm. Single job, minutes.
#
# Three things, all cheap, bundled because they share the classical tuning path:
#
#   1. C6 CLASSICAL ARMS -- classical_bare, classical_full, mlp_reference,
#      mlp_param_matched on the binding composites, 10 seeds, 90 epochs.
#      BINDING ON `size`, NOT `shape`. The first run bound on shape and is
#      confounded: C3 single-object shape accuracy is classical_bare 36.5 and
#      mlp_param_matched 36.9 against a 35.4 floor -- two of four arms cannot
#      perceive the attribute at all, so their binding scores measured
#      perception. Worse, quantum_coherent is the BEST TTN at shape (58.9 vs
#      classical_full's 46.3), so a quantum win there would have looked
#      compositional while being perceptual. On `size` every arm scores
#      84.8-99.1, so perception is equalised and a failure is a BINDING failure.
#      These carry the experiment's actual comparison: the MLP beats every TTN
#      arm on PERCEPTION (C3, C4), and the question is whether that advantage
#      narrows or reverses on BINDING.
#
#   2. C6 MANIPULATION CHECK -- the same arms on patch-SHUFFLED input.
#      NOT a per-architecture score: because the marginals are controlled,
#      shuffling destroys the only signal present, so every architecture MUST
#      land at exactly chance. It tests the DATA. If any arm comes back
#      resolvably above 50%, the composites leak a non-positional cue (lighting,
#      an intensity gradient, a pasting artifact correlated with class) and
#      EVERY C6 number is uninterpretable until the task is rebuilt.
#      RUN THIS BEFORE THE QUANTUM ARRAY -- it costs minutes and gates ~270 h.
#
#   3. C3D -- classical arms at 90 epochs on objects and relations.
#      C3c quoted a 90-epoch quantum arm against 30-epoch classical arms: the
#      C3b tuning asymmetry reappearing on the epoch axis. It was closed once
#      with "classical converges faster", which is only half right. Measured
#      epoch at which each arm reaches 98% of its own final-5 score:
#
#          classical_bare  colour 23/30 (objects), 24/30 (relations)
#          classical_full  colour 26/30 (objects), 20/30 (relations)
#          mlp_reference   12          mlp_param_matched  6
#
#      The MLPs are converged and gain nothing. The CP TREES' COLOUR HEAD IS
#      STILL CLIMBING AT THE CUTOFF -- and colour is exactly where C3c's headline
#      sits (27.1 -> 63.0). Note --tune-epochs is raised to match: tuning at 10
#      and running at 90 selects configs that are good at 10, which moves the
#      asymmetry to the hyperparameter axis instead of removing it.
#
# Submit:  qsub scripts/submit_c6_cheap.sh
#$ -l tmem=16G
#$ -l h_rt=8:0:0
#$ -S /bin/bash
#$ -j y
#$ -N c6_cheap
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

SEEDS="0 1 2 3 4 5 6 7 8 9"

echo "===== 1/3  C6 classical arms, binding, 90 epochs ====="
$PYTHON -m qnlp.image_tower.classification.clevr.run_c6_binding \
    --img-size 32 --seeds $SEEDS --epochs 90 --tune-epochs 30 \
    --attribute size --skip-quantum --out c6_cheap --out-suffix _classical || exit 1

echo "===== 2/3  C6 MANIPULATION CHECK (patch-shuffled) -- GATES THE QUANTUM ARRAY ====="
# --reuse-tuning: the shuffled data is at chance by construction, so re-running
# the ~36-config grid on it would rank configs on noise. Reusing the unshuffled
# tuning is both cheaper and more correct. Repeating it is most of why the first
# run took 5 h rather than the "minutes" estimated.
$PYTHON -m qnlp.image_tower.classification.clevr.run_c6_binding \
    --img-size 32 --seeds $SEEDS --epochs 90 --attribute size \
    --reuse-tuning qnlp/image_tower/classification/quantum/results/c6_binding_classical_results.json \
    --skip-quantum --shuffled --out c6_shuf --out-suffix _shuffled || exit 1

echo "===== 3/3  C3d classical arms at 90 epochs ====="
$PYTHON -m qnlp.image_tower.classification.clevr.run_c3_attributes \
    --img-size 16 --seeds $SEEDS --epochs 90 --tune-epochs 30 \
    --skip-quantum --out c3d_obj --out-suffix _c3d_90ep || exit 1

$PYTHON -m qnlp.image_tower.classification.clevr.run_c4_relational \
    --img-size 16 --seeds $SEEDS --epochs 90 --tune-epochs 30 \
    --skip-quantum --out c3d_rel --out-suffix _c3d_90ep || exit 1

echo "Job finished at $(date)"

# ---------------------------------------------------------------------
# CHECK STEP 2 BEFORE SUBMITTING submit_c6_binding.sh.
# The runner prints an explicit pass/fail line and records
# `manipulation_check_passed` in c6_binding_shuffled_results.json. On a failure,
# rebuild the composites -- do not start the 30-task quantum array.
# ---------------------------------------------------------------------
