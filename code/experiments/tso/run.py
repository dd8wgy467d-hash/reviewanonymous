"""TSO suite entry point: python -m experiments.tso.run --list-jobs | --job TOKEN | --aggregate."""

from experiments import runner
from experiments.tso import tasks

if __name__ == "__main__":
    runner.main(tasks)
