"""FSA suite entry point: python -m experiments.fsa.run --list-jobs | --job TOKEN | --aggregate."""

from experiments import runner
from experiments.fsa import tasks


def _extract(model, job):
    from experiments.fsa.extract import extract_machine
    return extract_machine(model, job)


if __name__ == "__main__":
    runner.main(tasks, extract=_extract)
