#!/usr/bin/env bash
# The finite-precision SSM simulations (experiments/appendix/ssms_fp): one CPU job per script.
#
#   bash scripts/dispatch_ssms_fp.sh [--dry-run] [--plot-only] [SCRIPT...]
#
# SCRIPT: finite_precision_counting (the default, and currently the only one).
# Outputs go to results/ssms_fp/. TIME sets the wall time (default 24:00:00).
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
DRY=0; EXTRA=""; SCRIPTS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --plot-only) EXTRA="--plot-only" ;;
    *) SCRIPTS+=("$a") ;;
  esac
done
[[ ${#SCRIPTS[@]} -eq 0 ]] && SCRIPTS=(finite_precision_counting)
mkdir -p "$BASE/logs"
for s in "${SCRIPTS[@]}"; do
  # --wrap runs the template as a plain script, so its GPU request is not applied.
  cmd=(sbatch --parsable --job-name="ssms_$s" --cpus-per-task=2 --mem-per-cpu=4G
       --time="${TIME:-24:00:00}" --output="$BASE/logs/slurm_%j_ssms_$s.log"
       --wrap "bash $BASE/scripts/slurm_gpu.sh $BASE -m experiments.appendix.ssms_fp.$s $EXTRA")
  if [[ $DRY == 1 ]]; then echo "[dry-run] ${cmd[*]}"; else echo "$s: job $("${cmd[@]}")"; fi
done
