#!/usr/bin/env bash
# Submit one SLURM job per configuration (all seeds) to measure the failing length L_m by bisection;
# configurations already on disk or in the queue are skipped.
#
#   [TIME=hh:mm:ss] [CAP=N] bash scripts/dispatch_faillen.sh fsa|tso [--dry-run] [FILTERS...]
#
# FILTERS are passed to `faillen --list-jobs`, e.g. --cells nfsm mamba --tasks S4 A5.
# TIME overrides the wall time (default 8 h for nfsm configs, 4 h for the baselines).
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
SUITE="${1:?usage: dispatch_faillen.sh fsa|tso [--dry-run] [FILTERS...]}"; shift
DRY=0; FILTERS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY=1; else FILTERS+=("$a"); fi
done
OUT="${NFSM_RESULTS_DIR:-$BASE/results}/$SUITE/faillen"
JOB_ARGS=(); [[ -n "${CAP:-}" ]] && JOB_ARGS=(--cap "$CAP")
mkdir -p "$BASE/logs"

source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE" >/dev/null
mapfile -t TOKENS < <(cd "$BASE" && PYTHONPATH="$BASE" "$NFSM_PY" -m experiments.faillen \
                        "$SUITE" --list-jobs "${FILTERS[@]}")
declare -A QUEUED=()
while read -r j; do [[ -n "$j" ]] && QUEUED["$j"]=1; done < <(nfsm_queued_names)

n=0; skipped=0
for tok in "${TOKENS[@]}"; do
  [[ -z "$tok" ]] && continue
  name="${tok//,/_}"
  if [[ -f "$OUT/$name.json" || -n "${QUEUED[fl_${SUITE}_$name]:-}" ]]; then
    skipped=$((skipped + 1)); continue
  fi
  n=$((n + 1))
  case "$tok" in *,nfsm,*) t="${TIME:-08:00:00}" ;; *) t="${TIME:-04:00:00}" ;; esac
  if [[ $DRY == 1 ]]; then echo "$tok $t"; continue; fi
  sbatch --job-name="fl_${SUITE}_$name" --time="$t" \
         --output="$BASE/logs/slurm_%j_fl_${SUITE}_$name.log" --parsable \
         "$BASE/scripts/slurm_gpu.sh" "$BASE" -m experiments.faillen "$SUITE" --job "$tok" \
         "${JOB_ARGS[@]}" >/dev/null
done
echo "$SUITE faillen: $n job(s) $([[ $DRY == 1 ]] && echo listed || echo submitted), $skipped skipped"
