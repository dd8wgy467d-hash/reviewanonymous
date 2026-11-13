#!/usr/bin/env bash
# =============================================================================
#  nfsm_env.sh -- shared helpers, sourced by every experiment script. Needs bash 4+.
#
#  Exports NFSM_PY, the Python interpreter that runs the experiments, so that it is chosen in
#  exactly one place. Resolution order (first hit wins):
#    1. $NFSM_PY if already exported          -- explicit override
#    2. $BASE/.venv/bin/python if present     -- uv: `uv sync --extra mps|cuda|cpu`
#    3. `python3` on PATH                     -- an already-activated environment; warns
#
#  The JAX backend follows the extra the venv was synced with. To force a fallback for a single
#  launch, set NFSM_PLATFORM (mps|cuda|gpu|cpu|tpu); it is inherited by the Python process:
#    NFSM_PLATFORM=cpu bash scripts/dispatch.sh fsa
# =============================================================================

# Resolve and export $NFSM_PY for a given project root.  Usage: nfsm_python "$BASE"
nfsm_python() {
    local base="$1"
    if [[ -n "${NFSM_PY:-}" ]]; then
        export NFSM_PY
    elif [[ -x "$base/.venv/bin/python" ]]; then
        export NFSM_PY="$base/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        export NFSM_PY="python3"
        echo "[nfsm-env] WARNING: no $base/.venv found; using '$(command -v python3)'." >&2
        echo "[nfsm-env]   Mac:     uv sync --extra mps" >&2
        echo "[nfsm-env]   Cluster: uv sync --extra cuda   (or activate your env / pip install -e '.[cuda]')" >&2
    else
        echo "[nfsm-env] ERROR: no python interpreter found." >&2
        return 1
    fi
    echo "[nfsm-env] NFSM_PY=$NFSM_PY  NFSM_PLATFORM=${NFSM_PLATFORM:-<auto>}"
}


# Print the names of this user's queued jobs, one per line, for de-duplicating submissions.
# Prints nothing when squeue is unavailable, so --dry-run works off-cluster.
nfsm_queued_names() {
    command -v squeue >/dev/null 2>&1 || return 0
    squeue -u "${USER:-$(id -un)}" -h -o "%j" 2>/dev/null || true
}
