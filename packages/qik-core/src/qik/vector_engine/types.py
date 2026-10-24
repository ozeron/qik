"""Immutable numeric author records and numba-compatible decision helpers."""

from __future__ import annotations

import inspect
import math
from enum import IntEnum
from typing import NamedTuple

import numba as nb
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    infer_getattr,
    signature,
)
from numba.extending import overload, register_jitable


class Bar(NamedTuple):
    """Completed bar; ts is int64 UTC nanoseconds in compiled callers."""

    i: int
    ts: int  # timestamp nanoseconds; clock/calendar features precomputed outside JIT
    open: float
    high: float
    low: float
    close: float
    volume: float


class Position(NamedTuple):
    """Execution position and working-entry snapshot."""

    side: int
    qty: float
    entry_i: int
    entry_px: float
    stop: float
    target: float
    working_side: int
    working_level: float
    last_exit_i: int
    last_exit_px: float
    last_exit_reason: int
    equity: float
    working_long: float = math.nan
    working_short: float = math.nan


class Decision(NamedTuple):
    """Numeric signals in engine order; side corresponds to entry_side."""

    side: int = 0
    exit: bool = False
    stop: float = math.nan
    target: float = math.nan
    entry_limit: float = math.nan
    stop_points: float = math.nan
    target_points: float = math.nan
    entry_stop: float = math.nan
    stop_update: float = math.nan
    target_update: float = math.nan
    breakeven_after_points: float = math.nan
    breakeven_offset_points: float = math.nan
    flatten: bool = False
    exit_long: bool = False
    exit_short: bool = False
    entry_limit_ioc: float = math.nan
    entry_stop_long: float = math.nan
    entry_limit_ioc_long: float = math.nan
    stop_long: float = math.nan
    target_long: float = math.nan
    stop_points_long: float = math.nan
    target_points_long: float = math.nan
    entry_stop_short: float = math.nan
    entry_limit_ioc_short: float = math.nan
    stop_short: float = math.nan
    target_short: float = math.nan
    stop_points_short: float = math.nan
    target_points_short: float = math.nan


@infer_getattr
class _DecisionConstructor(AttributeTemplate):
    # Numba's namedtuple typer binds arguments without applying defaults. A
    # partial constructor then has a shorter type than HOLD, even though Python
    # constructs the full record. Specialize only our constructor, before the
    # strategy's return types are unified; padding in the adapter is too late.
    key = nb.types.NamedTupleClass

    def resolve___call__(self, classty):
        if classty.instance_class is not Decision:
            return None
        return nb.types.Function(_DecisionCall)


class _DecisionCall(AbstractTemplate):
    key = nb.types.NamedTupleClass

    def generic(self, args, kwargs):
        pysig = inspect.signature(Decision)
        bound = pysig.bind(*args, **kwargs)
        bound.apply_defaults()
        fields = tuple(
            value if isinstance(value, nb.types.Type) else nb.typeof(value) for value in bound.args
        )
        # Keep the full Python signature as well as the full return type:
        # lowering must materialize omitted defaults, including keyword gaps.
        return signature(nb.types.BaseTuple.from_types(fields, Decision), *fields).replace(
            pysig=pysig
        )


HOLD = Decision()


@register_jitable
def _enter(
    side: int,
    stop: float = math.nan,
    target: float = math.nan,
    stop_points: float = math.nan,
    target_points: float = math.nan,
    limit: float = math.nan,
    entry_stop: float = math.nan,
) -> Decision:
    """Request an entry with optional protection and working-entry prices."""
    # Arithmetic promotes IntEnum before casting; Numba cannot int(enum) directly.
    return Decision(
        int(side + 0),
        False,
        float(stop),
        float(target),
        float(limit),
        float(stop_points),
        float(target_points),
        float(entry_stop),
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        False,
        False,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )


def enter(
    side: int,
    *,
    stop: float = math.nan,
    target: float = math.nan,
    stop_points: float = math.nan,
    target_points: float = math.nan,
    limit: float = math.nan,
    entry_stop: float = math.nan,
) -> Decision:
    """Request an entry with optional protection and working-entry prices."""
    return _enter(side, stop, target, stop_points, target_points, limit, entry_stop)


@overload(enter)
def _overload_enter(
    side,
    stop=math.nan,
    target=math.nan,
    stop_points=math.nan,
    target_points=math.nan,
    limit=math.nan,
    entry_stop=math.nan,
):
    # Numba's dispatcher cannot fold keyword-only defaults. Keep the public
    # Python signature strict and lower compiled calls to a positional helper.
    def impl(
        side,
        stop=math.nan,
        target=math.nan,
        stop_points=math.nan,
        target_points=math.nan,
        limit=math.nan,
        entry_stop=math.nan,
    ):
        return _enter(side, stop, target, stop_points, target_points, limit, entry_stop)

    return impl


@register_jitable
def enter_limit_ioc(
    side: int,
    limit: float,
    stop: float = math.nan,
    target: float = math.nan,
    stop_points: float = math.nan,
    target_points: float = math.nan,
) -> Decision:
    """Attempt once at the next open, filling at that open within the inclusive cap."""
    return Decision(
        int(side + 0),
        False,
        float(stop),
        float(target),
        math.nan,
        float(stop_points),
        float(target_points),
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        False,
        False,
        False,
        float(limit),
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )


@register_jitable
def exit() -> Decision:
    """Exit a position and cancel working entry using ordinary exit timing."""
    return Decision(
        0,
        True,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        False,
        False,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )


@register_jitable
def trail(level: float) -> Decision:
    """Request an absolute stop amendment."""
    return Decision(
        0,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        float(level),
        math.nan,
        math.nan,
        math.nan,
        False,
        False,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )


@register_jitable
def move_target(level: float) -> Decision:
    """Request an absolute target amendment."""
    return Decision(
        0,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        float(level),
        math.nan,
        math.nan,
        False,
        False,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )


@register_jitable
def flatten() -> Decision:
    """Request the configured engine flatten policy, including working entries."""
    return Decision(
        0,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        True,
        False,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
    )


class Side(IntEnum):
    """Signed entry and position direction."""

    SHORT = -1
    FLAT = 0
    LONG = 1


class ExitReason(IntEnum):
    """Kernel exit codes; NONE means no prior exit."""

    NONE = -1
    SIGNAL = 1
    STOP = 2
    TARGET = 3
    STOP_GAP = 4
    PROTECT_FAIL = 5


@register_jitable
def enter_straddle(
    entry_stop_long: float = math.nan,
    entry_limit_ioc_long: float = math.nan,
    stop_long: float = math.nan,
    target_long: float = math.nan,
    stop_points_long: float = math.nan,
    target_points_long: float = math.nan,
    entry_stop_short: float = math.nan,
    entry_limit_ioc_short: float = math.nan,
    stop_short: float = math.nan,
    target_short: float = math.nan,
    stop_points_short: float = math.nan,
    target_points_short: float = math.nan,
) -> Decision:
    """Submit mutually exclusive long/short legs from this completed close."""
    return Decision(
        0,
        False,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        math.nan,
        False,
        False,
        False,
        math.nan,
        float(entry_stop_long),
        float(entry_limit_ioc_long),
        float(stop_long),
        float(target_long),
        float(stop_points_long),
        float(target_points_long),
        float(entry_stop_short),
        float(entry_limit_ioc_short),
        float(stop_short),
        float(target_short),
        float(stop_points_short),
        float(target_points_short),
    )
