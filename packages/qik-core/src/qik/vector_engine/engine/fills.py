"""Fill policy helpers: direction masks, exit masks, path order and entry expiry."""

from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np


@nb.njit(cache=True, nogil=True)
def _no_reversal(o, e1, e2, c, m, pos):
    """True when no price update after path point ``m`` (0=O, 1=e1, 2=e2) moves against ``pos``.

    A resting limit fill pins one side of Nautilus' L1 book at the limit price; a later update
    moving the other way refreshes it (A13). A long's SELL limit pins the bid and is refreshed by a
    falling update; a short's BUY limit pins the ask and is refreshed by a rising one.
    """

    prev = o
    for k in range(1, 4):
        cur = e1 if k == 1 else (e2 if k == 2 else c)
        if k > m and ((pos > 0 and cur < prev) or (pos < 0 and cur > prev)):
            return False
        prev = cur
    return True


@nb.njit(cache=True, nogil=True)
def _force_for(mask, side):
    return bool(mask & 1) or (side > 0 and bool(mask & 2)) or (side < 0 and bool(mask & 4))


@nb.njit(cache=True, nogil=True)
def _exit_for(global_exit, long_exit, short_exit, side):
    return global_exit or (side > 0 and long_exit) or (side < 0 and short_exit)


@nb.njit(cache=True, nogil=True, inline="always")
def _high_first(adaptive: bool, open_px: float, high: float, low: float) -> bool:
    """Adaptive paths visit the nearer extreme first; ties visit low first."""
    return not adaptive or abs(open_px - high) < abs(open_px - low)


@nb.njit(cache=True, nogil=True, inline="always")
def _day_entry_expires(day_cancel: bool, expiry: bool) -> bool:
    """Only the cancel policy expires a pending DAY entry at its boundary."""
    return expiry and day_cancel


@nb.njit(cache=True, nogil=True, inline="always")
def _cancel_entry_commands(commands: Any, pending: int) -> None:
    """Remove queued commands belonging to the expired entry generation."""
    for q in range(len(commands) - 1, -1, -1):
        if commands[q][2] == pending:
            commands.pop(q)


@nb.njit(cache=True, nogil=True, inline="always")
def _target_fill_price(
    point: float, target: float, at_open: bool, gap_to_open: bool, side: int, slip: float
) -> float:
    """Apply the caller's target execution/slippage policy at a matched path point."""
    return point - side * slip if slip else (point if gap_to_open and at_open else target)


@nb.njit(cache=True, nogil=True)
def _venue_record(
    count_only,
    k,
    ei,
    i,
    side,
    ep,
    px,
    why,
    stop,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
):
    # The bound: a compiled run's buffer may be smaller than the count; it then reruns.
    if not count_only and k < len(out_entry_i):
        out_entry_i[k], out_exit_i[k], out_side[k] = ei, i, side
        out_entry_px[k], out_exit_px[k] = ep, px
        out_reason[k], out_stop[k] = why, stop


@nb.njit(cache=True, nogil=True)
def _venue_peek(
    count_only,
    k,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
):
    """Row ``k`` as ``_venue_record`` would overwrite it (zeros where it would skip the write)."""
    if not count_only and k < len(out_entry_i):
        return (
            np.int64(out_entry_i[k]),
            np.int64(out_exit_i[k]),
            np.int64(out_side[k]),
            np.float64(out_entry_px[k]),
            np.float64(out_exit_px[k]),
            np.int64(out_reason[k]),
            np.float64(out_stop[k]),
        )
    return (np.int64(0), np.int64(0), np.int64(0), 0.0, 0.0, np.int64(0), 0.0)


@nb.njit(cache=True, nogil=True)
def _venue_unrecord(
    count_only,
    k,
    prior,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
):
    """Put back the row ``_venue_peek`` read, so a refused bar leaves nothing past the count.

    Both helpers are plain calls: with ``inline="always"`` into the venue's inlined bar step,
    numba compiles this restore to a no-op.
    """
    if not count_only and k < len(out_entry_i):
        out_entry_i[k], out_exit_i[k], out_side[k] = prior[0], prior[1], prior[2]
        out_entry_px[k], out_exit_px[k] = prior[3], prior[4]
        out_reason[k], out_stop[k] = prior[5], prior[6]


__all__ = [
    "_cancel_entry_commands",
    "_day_entry_expires",
    "_exit_for",
    "_force_for",
    "_high_first",
    "_no_reversal",
    "_target_fill_price",
    "_venue_peek",
    "_venue_record",
    "_venue_unrecord",
]
