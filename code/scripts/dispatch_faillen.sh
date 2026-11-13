#!/usr/bin/env bash
# Submit one SLURM job per configuration (all seeds) to measure the failing length L_m by bisection;
# configurations already on disk or in the queue are skipped. With --per-seed, one job per
# configuration and seed instead (only seeds with a checkpoint, and not yet in the configuration's
# JSON); faillen merges the seeds of parallel jobs into one JSON under a lock.
#
#   [TIME=hh:mm:ss] [CAP=N] bash scripts/dispatch_faillen.sh fsa|tso [--dry-run] [--per-seed] [FILTERS...]
#
# FILTERS are passed to `faillen --list-jobs`, e.g. --cells nfsm mamba --tasks S4 A5.
# TIME overrides the wall time (default 8 h for nfsm configs, 4 h for the baselines).
set -euo pipefail
BASE="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")")" && pwd)"
SUITE="${1:?usage: dispatch_faillen.sh fsa|tso [--dry-run] [FILTERS...]}"; shift
DRY=0; PER_SEED=0; FILTERS=()
for a in "$@"; do
  case "$a" in --dry-run) DRY=1 ;; --per-seed) PER_SEED=1 ;; *) FILTERS+=("$a") ;; esac
done
CKPTS="${NFSM_RESULTS_DIR:-$BASE/results}/$SUITE/checkpoints"
OUT="${NFSM_RESULTS_DIR:-$BASE/results}/$SUITE/faillen"
JOB_ARGS=(); [[ -n "${CAP:-}" ]] && JOB_ARGS=(--cap "$CAP")
mkdir -p "$BASE/logs"

source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE" >/dev/null
mapfile -t TOKENS < <(cd "$BASE" && PYTHONPATH="$BASE" "$NFSM_PY" -m experiments.faillen \
                        "$SUITE" --list-jobs "${FILTERS[@]}")
declare -A QUEUED=()
while read -r j; do [[ -n "$j" ]] && QUEUED["$j"]=1; done < <(nfsm_queued_names)

has_seed() {   # has_seed JSON SEED: the configuration's JSON already holds this seed
  [[ -f "$1" ]] && "$NFSM_PY" -c 'import json,sys; sys.exit(sys.argv[2] not in json.load(open(sys.argv[1]))["seeds"])' "$1" "$2"
}

submit() {     # submit TOKEN NAME [--seeds SEED]
  local tok="$1" name="$2"; shift 2
  local t
  case "$tok" in *,nfsm,*) t="${TIME:-08:00:00}" ;; *) t="${TIME:-04:00:00}" ;; esac
  n=$((n + 1))
  if [[ $DRY == 1 ]]; then echo "$tok $* $t"; return; fi
  sbatch --job-name="fl_${SUITE}_$name" --time="$t" \
         --output="$BASE/logs/slurm_%j_fl_${SUITE}_$name.log" --parsable \
         "$BASE/scripts/slurm_gpu.sh" "$BASE" -m experiments.faillen "$SUITE" --job "$tok" \
         "$@" "${JOB_ARGS[@]}" >/dev/null
}

n=0; skipped=0
for tok in "${TOKENS[@]}"; do
  [[ -z "$tok" ]] && continue
  name="${tok//,/_}"
  if [[ $PER_SEED == 0 ]]; then
    if [[ -f "$OUT/$name.json" || -n "${QUEUED[fl_${SUITE}_$name]:-}" ]]; then
      skipped=$((skipped + 1)); continue
    fi
    submit "$tok" "$name"
    continue
  fi
  for ckpt in "$CKPTS/${name}"_*.msgpack; do
    [[ -f "$ckpt" ]] || continue
    seed="${ckpt##*_}"; seed="${seed%.msgpack}"
    if has_seed "$OUT/$name.json" "$seed" || [[ -n "${QUEUED[fl_${SUITE}_${name}_$seed]:-}" ]]; then
      skipped=$((skipped + 1)); continue
    fi
    submit "$tok" "${name}_$seed" --seeds "$seed"
  done
done
echo "$SUITE faillen: $n job(s) $([[ $DRY == 1 ]] && echo listed || echo submitted), $skipped skipped"
