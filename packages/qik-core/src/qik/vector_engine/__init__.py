"""Vector engine: Nautilus-compatible bar execution, vectorized across configs.

Contract and assumptions: ``docs/vector-engine.md``. Entry point: ``simulate``.
"""

from .api import (
    Bars,
    Compliance,
    Costs,
    FillModel,
    Instrument,
    MarketFill,
    Profile,
    Result,
    Rules,
    Signals,
    Sizing,
    require_nautilus,
    simulate,
)
from .core import BarView, Core, Decision, check_causal, signals_from_core
from .grid import grid, stack

__all__ = [
    "BarView",
    "Bars",
    "Compliance",
    "Core",
    "Costs",
    "Decision",
    "Instrument",
    "FillModel",
    "MarketFill",
    "Profile",
    "Result",
    "Rules",
    "Signals",
    "Sizing",
    "check_causal",
    "grid",
    "require_nautilus",
    "signals_from_core",
    "simulate",
    "stack",
]
