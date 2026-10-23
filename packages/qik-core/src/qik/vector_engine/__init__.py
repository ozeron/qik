"""Vector engine: Nautilus-compatible bar execution, vectorized across configs.

Contract and assumptions: ``docs/vector-engine.md``. Entry point: ``simulate``.
"""

from .api import (
    Bars,
    BracketAnchor,
    BracketPolicy,
    Compliance,
    Costs,
    FillModel,
    FillTiming,
    Flatten,
    Instrument,
    MarketFill,
    Profile,
    Result,
    Rules,
    SessionBoundary,
    Signals,
    Sizing,
    TargetGap,
    TickRound,
    require_nautilus,
    simulate,
)
from .core import BarView, Core, Decision, check_causal, signals_from_core
from .grid import grid, stack

__all__ = [
    "BarView",
    "Bars",
    "BracketAnchor",
    "BracketPolicy",
    "Compliance",
    "Core",
    "Costs",
    "Decision",
    "Instrument",
    "FillModel",
    "FillTiming",
    "Flatten",
    "SessionBoundary",
    "MarketFill",
    "Profile",
    "Result",
    "Rules",
    "Signals",
    "TargetGap",
    "TickRound",
    "Sizing",
    "check_causal",
    "grid",
    "require_nautilus",
    "signals_from_core",
    "simulate",
    "stack",
]
