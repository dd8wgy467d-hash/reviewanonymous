#!/usr/bin/env bash
# The wall-clock benchmark (experiments/appendix/walltime), one GPU node per row: writes
# results/walltime/configs.csv and submits one array task per row, each in a fresh process.
#
#   bash scripts/dispatch_walltime.sh [--dry-run] [--max-parallel N] [--ids 0-319 | 3,17,42]
#                                     [--raw-dir DIR] [--chunked]
#
# --raw-dir writes the JSONs elsewhere than results/walltime/raw (e.g. a rerun under another
# setting). The submitting shell's environment is exported to the jobs, so
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 bash scripts/dispatch_walltime.sh ... applies to them and is
# recorded in each JSON. --chunked uses each model's default time-chunking budget.
# TIME overrides the wall time per task (default 02:00:00). Logs: logs/walltime/slurm_%A_%a.log.
# Then, from the raw JSONs alone (no JAX, no measurement):
#   python -m experiments.appendix.walltime.aggregate   # results/walltime.csv, results/walltime_summary.md
#   python -m experiments.appendix.walltime.plots       # figures/walltime.pdf
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
DRY=0; MAXP=8; IDS=""; RUN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --max-parallel) MAXP="$2"; shift 2 ;;
    --ids) IDS="$2"; shift 2 ;;
    --raw-dir) RUN_ARGS+=(--raw-dir "$2"); shift 2 ;;
    --chunked) RUN_ARGS+=(--chunked); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$BASE/logs/walltime"
source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE" >/dev/null
N=$(cd "$BASE" && PYTHONPATH="$BASE" "$NFSM_PY" -m experiments.appendix.walltime.configs | tail -1)
IDS="${IDS:-0-$((N - 1))}"
cmd=(sbatch --parsable --array="${IDS}%${MAXP}" --time="${TIME:-02:00:00}"
     --output="$BASE/logs/walltime/slurm_%A_%a.log" "$BASE/scripts/slurm_walltime.sh" "$BASE" "${RUN_ARGS[@]}")
if [[ $DRY == 1 ]]; then echo "[dry-run] ${cmd[*]}"; echo "$N configurations"; exit 0; fi
jid=$("${cmd[@]}")
echo "walltime: array job $jid, tasks $IDS (at most $MAXP at once) of $N configurations"
