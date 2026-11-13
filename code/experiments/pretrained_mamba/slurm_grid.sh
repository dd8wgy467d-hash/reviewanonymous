#!/usr/bin/env bash
# Array job over the grid: task k = (layout, tau, haystack) with k = 60*layout + 10*tau + haystack,
# layouts A B C, tau inf 4096 2048 1024 512 128, haystacks 0..9 (180 tasks). MAMBA2_SIZE picks the model.
# Optional: D_ARGS (e.g. "--d 16400 32775") runs other d values, TAG names their part files.
#SBATCH --job-name=niah
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
k=$SLURM_ARRAY_TASK_ID
layouts=(A B C); taus=(inf 4096 2048 1024 512 128)
layout=${layouts[$((k / 60))]}; tau=${taus[$((k / 10 % 6))]}; h=$((k % 10))
name=$([ "$tau" = inf ] && echo asis || echo "tau$tau")
dir=results/mamba2_${MAMBA2_SIZE:-130m}/parts
mkdir -p "$dir"
echo "job ${SLURM_JOB_ID} task $k on ${SLURM_NODELIST} at $(date): layout $layout tau $tau haystack $h"
"$HOME/.local/bin/uv" run python eval_niah.py --layout "$layout" --tau "$tau" --haystacks "$h" ${D_ARGS:-} \
  --out "$dir/niah_${layout}_${name}${TAG:-}_h${h}.json"
echo "done at $(date)"
