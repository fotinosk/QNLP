#!/bin/bash
# Two COCO single-caption runs as a SLURM job array:
#   task 0 -> linear text model, bond_dim 25
#   task 1 -> non-linear contraction text model, bond_dim 10
#
# Submit with:  sbatch scripts/cluster/coco_experiments.sbatch
#
# Lines marked EDIT must be adapted to your cluster (partition/account/modules).

#SBATCH --job-name=coco_qnlp
#SBATCH --array=0-1
#SBATCH --gres=gpu:1                 # EDIT: your GPU request
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/logs/slurm-%x-%A_%a.out
# EDIT: #SBATCH --partition=<your_partition>
# EDIT: #SBATCH --account=<your_account>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs/logs

# --- environment ------------------------------------------------------------
# EDIT: match how your cluster exposes conda (module load anaconda, etc.)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qnlp

# Non-interactive MLflow: file store in-repo, no server/stdin needed.
# EDIT: point at your tracking server instead if you have one.
export MLFLOW_TRACKING_URI="file:./mlruns"

# --- per-task configuration -------------------------------------------------
case "$SLURM_ARRAY_TASK_ID" in
  0)
    export ML_USE_NON_LINEAR_CONTRACTIONS=false
    export ML_BOND_DIM=25
    export ML_RUN_NAME="coco_linear_bond25"
    ;;
  1)
    export ML_USE_NON_LINEAR_CONTRACTIONS=true
    export ML_BOND_DIM=10
    export ML_RUN_NAME="coco_nlc_bond10"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID | nlc=$ML_USE_NON_LINEAR_CONTRACTIONS bond=$ML_BOND_DIM run=$ML_RUN_NAME"
srun python -m qnlp.scripts.coco_single_caption.run
