#!/usr/bin/env bash
#SBATCH --job-name=walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
# One array task of the wall-clock benchmark on a GPU node (the reported runs used an A100 80 GB):
#   sbatch --array=0-N scripts/slurm_walltime.sh BASE
# Row $SLURM_ARRAY_TASK_ID of results/walltime/configs.csv, in a fresh process. No module is
# loaded: the interpreter is $BASE/.venv (uv sync --extra cuda), resolved by nfsm_env.sh.
#
# Site-specific scheduler settings (account, partition, QOS) are read by sbatch from the
# environment: export SBATCH_ACCOUNT / SBATCH_PARTITION / SBATCH_QOS before submitting.
BASE="$1"; shift
cd "$BASE" || exit 1
export PYTHONPATH="$BASE${PYTHONPATH:+:$PYTHONPATH}"
source "$BASE/scripts/nfsm_env.sh"; nfsm_python "$BASE"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_LOG_COMPILES=1 TF_CPP_MIN_LOG_LEVEL=3
echo "task ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} (job ${SLURM_JOB_ID:-local}) on ${SLURM_NODELIST:-local} at $(date)"
echo "modules: ${LOADEDMODULES:-<none>}  python: $NFSM_PY ($("$NFSM_PY" --version 2>&1))"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader
"$NFSM_PY" -m experiments.appendix.walltime.run "$@"
echo "done at $(date)"
