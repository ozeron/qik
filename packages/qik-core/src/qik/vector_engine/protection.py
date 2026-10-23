"""Shared absolute/point protection policy; schedulers own order lifecycle."""

from __future__ import annotations

import numba as nb
import numpy as np

ERR_NONE = 0
ERR_PROTECT_REJECTED = 4
REASON_TARGET = 3
REASON_PROTECT_FAIL = 5


@nb.njit(cache=True, nogil=True)
def _level(
    side: int,
    anchor: float,
    level: float,
    pts: float,
    sign: float,
    tick: float = 0.0,
    rounding: int = 0,
) -> float:
    """A level as given, or ``pts`` points from ``anchor`` (``sign`` +1 target side, -1 stop side)."""

    value = level if np.isnan(pts) else anchor + sign * side * pts
    return _round_price(value, tick, rounding, sign * side)


@nb.njit(cache=True, nogil=True)
def _round_price(value: float, tick: float, rounding: int, direction: float) -> float:
    if rounding == 0 or np.isnan(value):
        return value
    steps = value / tick
    if rounding == 1:
        return np.floor(steps + 0.5) * tick
    return (np.ceil(steps - 1e-10) if direction > 0 else np.floor(steps + 1e-10)) * tick


@nb.njit(cache=True, nogil=True)
def _anchor(mode: int, decision: float, fill: float, legacy: float) -> float:
    if mode == 1:
        return decision
    return fill if mode == 2 else legacy


@nb.njit(cache=True, nogil=True)
def _in_market(side: int, level: float, close: float, is_stop: bool) -> bool:
    """A stop at or through ``close``, or a limit target at or through it (A10 / A11)."""

    if np.isnan(level):
        return False
    if is_stop:
        return level >= close if side > 0 else level <= close
    return level <= close if side > 0 else level >= close


@nb.njit(cache=True, nogil=True)
def _fill_then_protect(
    s: int,
    i: int,
    fill_px: float,
    st: float,
    tg: float,
    ci: float,
    protect_flatten: bool,
    count_only: bool,
    out_entry_i: np.ndarray,
    out_exit_i: np.ndarray,
    out_side: np.ndarray,
    out_entry_px: np.ndarray,
    out_exit_px: np.ndarray,
    out_reason: np.ndarray,
    out_stop: np.ndarray,
    k: int,
) -> tuple[int, int]:
    """V2 pattern (A15): an entry filled at ``fill_px`` whose protection meets the close ``ci``.

    A stop at or through the close is rejected (refused, or V2's flatten at the close); a target at
    or through it fills at the close. Returns ``(err, reason)``: reason 0 = position opened, else
    the reason of the same-bar trade written at row ``k``.
    """

    reason = 0
    if _in_market(s, st, ci, True):
        if not protect_flatten:
            return ERR_PROTECT_REJECTED, 0
        reason = REASON_PROTECT_FAIL
    elif _in_market(s, tg, ci, False):
        reason = REASON_TARGET
    if reason != 0 and not count_only:
        out_entry_i[k] = i
        out_exit_i[k] = i
        out_side[k] = s
        out_entry_px[k] = fill_px
        out_exit_px[k] = ci
        out_reason[k] = reason
        out_stop[k] = st
    return ERR_NONE, reason


@nb.njit(cache=True, nogil=True)
def levels(
    side: int,
    anchor: float,
    stop: float,
    target: float,
    stop_points: float,
    target_points: float,
    tick: float = 0.0,
    rounding: int = 0,
) -> tuple[float, float]:
    """Resolve a bracket from its selected anchor, preserving stop-before-target order."""
    return (
        _level(side, anchor, stop, stop_points, -1.0, tick, rounding),
        _level(side, anchor, target, target_points, 1.0, tick, rounding),
    )
