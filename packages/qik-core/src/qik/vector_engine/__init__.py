"""Vector engine: Nautilus-compatible bar execution, vectorized across configs.

Contract and assumptions: ``docs/vector-engine.md``. Entry point: ``simulate``.
"""

from .api import Bars, Costs, Instrument, Result, Rules, Signals, Sizing, simulate
from .core import BarView, Core, Decision, check_causal, signals_from_core
from .grid import grid, stack

__all__ = [
    "BarView",
    "Bars",
    "Core",
    "Costs",
    "Decision",
    "Instrument",
    "Result",
    "Rules",
    "Signals",
    "Sizing",
    "check_causal",
    "grid",
    "signals_from_core",
    "simulate",
    "stack",
]
