"""Recurrent cells and the name -> class registry."""

from nfsm.cells.nfsm import NFSM
from nfsm.cells.ssm import AUSSM, Mamba, PDSSM

CELL_REGISTRY = {"nfsm": NFSM, "mamba": Mamba, "aussm": AUSSM, "pdssm": PDSSM}

__all__ = ["NFSM", "Mamba", "AUSSM", "PDSSM", "CELL_REGISTRY"]
