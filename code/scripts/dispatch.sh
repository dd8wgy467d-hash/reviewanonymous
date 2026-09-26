#!/usr/bin/env bash
# Submit one SLURM job per configuration of the fsa or tso suite; configurations already on disk or
# in the queue are skipped.
#
#   [STEPS=N] [TIME=hh:mm:ss] bash scripts/dispatch.sh fsa|tso [--dry-run] [FILTERS...]
#
# FILTERS are passed to `run.py --list-jobs`, e.g. --tasks S4 A5 --arms free --seeds 42 43.
# STEPS overrides the step budget; TIME the wall time (default per cell: nfsm 4 h, mamba 10 h,
# pdssm 36 h, aussm 48 h, sized for the slowest card); NFSM_RESULTS_DIR the results root
# (default results/).
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
SUITE="${1:?usage: dispatch.sh fsa|tso [--dry-run] [FILTERS...]}"; shift
DRY=0; FILTERS=()
for a in "$@"; do
  if [[ "$a" == "--dry-run" ]]; then DRY=1; else FILTERS+=("$a"); fi
done
JOB_ARGS=(); [[ -n "${STEPS:-}" ]] && JOB_ARGS=(--steps "$STEPS")
PARTS="${NFSM_RESULTS_DIR:-$BASE/results}/$SUITE/parts"
mkdir -p "$BASE/logs"

wall_time() {
  if [[ -n "${TIME:-}" ]]; then echo "$TIME"; return; fi
  case "$1" in
    *,nfsm,*) echo 04:00:00 ;; *,mamba,*) echo 10:00:00 ;; *,pdssm,*) echo 36:00:00 ;; *) echo 48:00:00 ;;
  esac
}

source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE" >/dev/null
mapfile -t TOKENS < <(cd "$BASE" && PYTHONPATH="$BASE" "$NFSM_PY" -m "experiments.$SUITE.run" \
                        --list-jobs "${FILTERS[@]}")
declare -A QUEUED=()
while read -r j; do [[ -n "$j" ]] && QUEUED["$j"]=1; done < <(nfsm_queued_names)

n=0; skipped=0
for tok in "${TOKENS[@]}"; do
  name="${tok//,/_}"
  if [[ -f "$PARTS/$name.json" || -n "${QUEUED[${SUITE}_$name]:-}" ]]; then
    skipped=$((skipped + 1)); continue
  fi
  n=$((n + 1))
  if [[ $DRY == 1 ]]; then echo "$tok $(wall_time "$tok")"; continue; fi
  sbatch --job-name="${SUITE}_$name" --time="$(wall_time "$tok")" \
         --output="$BASE/logs/slurm_%j_${SUITE}_$name.log" --parsable \
         "$BASE/scripts/slurm_gpu.sh" "$BASE" -m "experiments.$SUITE.run" --job "$tok" \
         "${JOB_ARGS[@]}" >/dev/null
done
echo "$SUITE: $n job(s) $([[ $DRY == 1 ]] && echo listed || echo submitted), $skipped skipped"
