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
from .compiled import CompiledBar, Position, run
from .core import BarView, Core, Decision, check_causal, signals_from_core
from .grid import grid, stack
from .kernel import (
    V_ENTRY_I,
    V_ENTRY_PX,
    V_EQUITY,
    V_LAST_EXIT_I,
    V_LAST_EXIT_PX,
    V_LAST_EXIT_REASON,
    V_QTY,
    V_SIDE,
    V_STOP,
    V_TARGET,
    V_WORK_LEVEL,
    V_WORK_SIDE,
)

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
    "run",
    "CompiledBar",
    "Position",
    "V_SIDE",
    "V_QTY",
    "V_ENTRY_I",
    "V_ENTRY_PX",
    "V_STOP",
    "V_TARGET",
    "V_WORK_SIDE",
    "V_WORK_LEVEL",
    "V_LAST_EXIT_I",
    "V_LAST_EXIT_PX",
    "V_LAST_EXIT_REASON",
    "V_EQUITY",
    "stack",
]

# The existing Core Decision remains public; typed strategies use TypedDecision
# (also available as qik.vector_engine.types.Decision).
from .htf import htf, htf_align
from .schema import BoolArray, Features, FloatArray, IntArray, Params, State, StrategySpec, strategy
from .types import HOLD, Bar, ExitReason, Position, Side, enter, exit, flatten, move_target, trail
from .types import Decision as TypedDecision

__all__ += [
    "State",
    "Params",
    "Features",
    "FloatArray",
    "IntArray",
    "BoolArray",
    "StrategySpec",
    "strategy",
    "Bar",
    "Position",
    "TypedDecision",
    "Side",
    "ExitReason",
    "HOLD",
    "enter",
    "exit",
    "trail",
    "move_target",
    "flatten",
    "htf",
    "htf_align",
]
