#!/usr/bin/env bash
# Generic one-GPU job: runs "uv run python <args>". Account, partition and time go on the sbatch line.
#SBATCH --job-name=mamba2
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-none} on ${SLURM_NODELIST} at $(date): $*"
"$HOME/.local/bin/uv" run python "$@"
echo "done at $(date)"
