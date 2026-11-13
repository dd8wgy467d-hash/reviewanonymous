#!/usr/bin/env bash
#SBATCH --job-name=nfsm
#SBATCH --export=ALL
#SBATCH --output=logs/slurm_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# Run one Python command on a GPU node:  sbatch scripts/slurm_gpu.sh BASE ARGS...
# e.g. ARGS = -m experiments.fsa.run --job S4,factored,nfsm,1,free,42
#
# Site-specific scheduler settings are passed through the environment rather than hard-coded here,
# so that the same template works on any SLURM cluster. Before submitting, export as needed:
#   SBATCH_ACCOUNT    the accounting group to charge
#   SBATCH_PARTITION  the partition (queue) to submit to
#   SBATCH_QOS        the quality of service
# sbatch reads these itself; the dispatchers pass them through.
#
# The interpreter is $BASE/.venv (uv sync --extra cuda), resolved by nfsm_env.sh. If the site
# instead provides a conda environment, export NFSM_CONDA_ENV=<name> (and NFSM_CONDA_SH to the
# conda profile script if it is not under $HOME/miniconda3 or $HOME/anaconda3).
BASE="$1"; shift
cd "$BASE" || exit 1
if [[ -n "${NFSM_CONDA_ENV:-}" ]]; then
    for sh in "${NFSM_CONDA_SH:-}" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        [[ -n "$sh" && -f "$sh" ]] && { source "$sh"; break; }
    done
    conda activate "$NFSM_CONDA_ENV"
fi
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE"
export WANDB_START_METHOD=thread WANDB_CONSOLE=off TF_CPP_MIN_LOG_LEVEL=3 GRPC_VERBOSITY=NONE
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export MPLCONFIGDIR="${MPLCONFIGDIR:-$BASE/logs/.mplcache/${SLURM_JOB_ID:-local}}"
mkdir -p "$MPLCONFIGDIR"
echo "job ${SLURM_JOB_ID:-local} on ${SLURM_NODELIST:-local} at $(date): $*"
"$NFSM_PY" "$@"
echo "done at $(date)"
