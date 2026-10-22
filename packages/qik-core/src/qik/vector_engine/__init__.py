"""Vector engine: Nautilus-compatible bar execution, vectorized across configs.

Contract and assumptions: ``docs/vector-engine.md``. Entry point: ``simulate``.
"""

from .api import Bars, Costs, Instrument, Result, Rules, Signals, Sizing, simulate
from .grid import grid, stack

__all__ = [
    "Bars",
    "Costs",
    "Instrument",
    "Result",
    "Rules",
    "Signals",
    "Sizing",
    "grid",
    "simulate",
    "stack",
]
