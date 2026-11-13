"""Recurrent cells and the name -> class registry."""

from nfsm.cells.delta import DeltaProduct
from nfsm.cells.mingru import MinGRU
from nfsm.cells.nfsm import NFSM
from nfsm.cells.ssm import AUSSM, Mamba, PDSSM

CELL_REGISTRY = {"nfsm": NFSM, "mamba": Mamba, "aussm": AUSSM, "pdssm": PDSSM,
                 "mingru": MinGRU, "deltaproduct": DeltaProduct}

__all__ = ["NFSM", "Mamba", "AUSSM", "PDSSM", "MinGRU", "DeltaProduct", "CELL_REGISTRY"]
