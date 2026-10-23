"""Stop specifications only; the engine owns fills, ratchets and rejected amendments."""

from __future__ import annotations

import math

from qik.vector_engine.core import Decision


def _side(side: int) -> None:
    if side not in (-1, 1):
        raise ValueError("stop/target side must be -1 or +1")


def atr_stop(anchor: float, side: int, atr: float, multiple: float) -> float:
    """Raw ATR stop; caller selects anchor, ATR seed/freeze and tick conversion.

    Admitted: P059 BreakoutCore and C001 vector.signals.
    NaN ATR propagates during indicator warmup, as in both original cores.
    """
    _side(side)
    if multiple <= 0 or not math.isfinite(multiple):
        raise ValueError("ATR multiple must be finite and positive")
    if atr < 0 or math.isinf(atr):
        raise ValueError("ATR must be nonnegative or NaN during warmup")
    return anchor - side * (atr * multiple)


def trailing_stop(anchor: float, side: int, distance: float, *, tick: float) -> Decision:
    """Outward tick-rounded candidate from a caller-owned anchor and distance.

    Admitted: P059 high-water/frozen-ATR and C001 close/refreshing-ATR trails.
    Select ``Rules(stop_update_mode="ratchet")`` to ratchet in execution. This
    helper retains no position state and does not filter candidates against fills.
    """
    _side(side)
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError("tick must be finite and positive")
    if not math.isfinite(distance) or distance < 0 or not math.isfinite(anchor):
        raise ValueError("trail anchor/distance must be finite; distance must be nonnegative")
    raw = anchor - side * distance
    level = (math.floor(raw / tick) if side == 1 else math.ceil(raw / tick)) * tick
    return Decision(stop_update=level)
