#!/usr/bin/env bash
# The scan drift (experiments/appendix/parallelism/drift.py): one job per model, dtype and seed,
# each writing results/parallelism/drift/<dtype>/<model>_seed<seed>.json.
#
#   bash scripts/dispatch_drift.sh [--dry-run] [--models mamba aussm pdssm] [--dtypes float32 float16]
#                                  [--seeds 0 1 2 3 4]
#
# The template is scripts/slurm_gpu.sh (SLURM_TEMPLATE overrides it). TIME sets the wall time
# (default 01:00:00). GPU scatter-adds are made
# deterministic, so the two scans differ only by the order of their operations.
# Figures, from the JSONs: python -m experiments.appendix.parallelism.drift_plots
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
MOD=experiments.appendix.parallelism.drift
DRY=0; ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
SLURM_TEMPLATE="${SLURM_TEMPLATE:-$BASE/scripts/slurm_gpu.sh}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_deterministic_ops=true"
mkdir -p "$BASE/logs"
source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE" >/dev/null

mapfile -t TOKENS < <(cd "$BASE" && PYTHONPATH="$BASE" JAX_PLATFORMS=cpu "$NFSM_PY" -m "$MOD" --list-jobs "${ARGS[@]}")
for tok in "${TOKENS[@]}"; do
  name="drift_${tok//,/_}"
  if [[ $DRY == 1 ]]; then echo "[dry-run] $name"; continue; fi
  sbatch --parsable --job-name="$name" --time="${TIME:-01:00:00}" \
         --output="$BASE/logs/slurm_%j_$name.log" \
         "$SLURM_TEMPLATE" "$BASE" -m "$MOD" --job "$tok"
done
echo "drift: ${#TOKENS[@]} job(s) $([[ $DRY == 1 ]] && echo listed || echo submitted) with $(basename "$SLURM_TEMPLATE")"
