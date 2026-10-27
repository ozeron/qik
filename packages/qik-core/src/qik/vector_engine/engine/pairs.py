"""Straddle pair legs: pair reads, resting/level lookups and lifetime triggers."""

from __future__ import annotations

import numba as nb
import numpy as np

from qik.vector_engine.protection import _in_market

from .codes import _SLOT_REPLACE, _SLOT_SUBMIT, PAIR_BITS, PAIR_TRIGGER_BITS
from .events import _level_at, _pair_value, _side_at


@nb.njit(cache=True, inline="never")
def _straddle_column(pair, j):
    if pair is None or pair.shape[2] == 0:
        return np.empty((12, 0))
    return pair[j]


@nb.njit(cache=True)
def _pair_at(pair, i, core=None):
    if i < 0 or pair is None:
        return False
    if not (core is not None and core.event_mode) and pair.shape[1] <= i:
        return False
    # Any finite leg, without allocating a column per call (it runs several times a bar).
    for leg in range(pair.shape[0]):
        if not np.isnan(_pair_value(core, pair, leg, i)):
            return True
    return False


@nb.njit(cache=True)
def _has_pairs(pair):
    return pair is not None and pair.shape[1] > 0


@nb.njit(cache=True)
def _pair_rests(pair, i, core=None):
    """A DAY stop leg rests; an IOC-only pair attempts once and never crosses a session."""
    return _pair_at(pair, i, core) and not (
        np.isnan(_pair_value(core, pair, 0, i)) and np.isnan(_pair_value(core, pair, 6, i))
    )


@nb.njit(cache=True)
def _pair_level(pair, i, offset, bar, latency, core=None):
    if i < 0 or pair is None:
        return np.nan
    if not (core is not None and core.event_mode) and pair.shape[1] <= i:
        return np.nan
    if not np.isnan(_pair_value(core, pair, offset, i)):
        return _pair_value(core, pair, offset, i)
    return _pair_value(core, pair, offset + 1, i) if bar < i + latency + 1 else np.nan


@nb.njit(cache=True)
def _stop_pair(pair, i, core=None):
    """A pair whose legs are all resting stops (no IOC leg attempts at an open)."""
    return (
        _pair_rests(pair, i, core)
        and np.isnan(_pair_value(core, pair, 1, i))
        and np.isnan(_pair_value(core, pair, 7, i))
    )


@nb.njit(cache=True)
def _lifetime_crosses(next_bar, kind, d, triggers, pair, core=None):
    """order_lifetime=next_bar: a stop submission may reach the next session's bar."""
    return (
        next_bar
        and (kind == _SLOT_SUBMIT or kind == _SLOT_REPLACE)
        and (not np.isnan(_level_at(core, triggers, d, 5)) or _stop_pair(pair, d, core))
    )


@nb.njit(cache=True)
def _open_in_market(triggers, pair, sides, d, opening, core=None):
    """An open-relative trigger at or through its own activation open."""
    if _pair_at(pair, d, core):
        return (
            not np.isnan(_pair_value(core, pair, 0, d)) and opening >= _pair_value(core, pair, 0, d)
        ) or (
            not np.isnan(_pair_value(core, pair, 6, d)) and opening <= _pair_value(core, pair, 6, d)
        )
    return _in_market(-_side_at(core, sides, d), _level_at(core, triggers, d, 5), opening, True)


@nb.njit(cache=True)
def _event_pair_at(masks, values, rows, row):
    if row < 0:
        return False
    mask = masks[row] & PAIR_BITS
    # The mask preserves raw decisions for captures/declaration checks. Only the
    # two open-relative stop triggers can become NaN after publication (no next
    # open on the last bar); those bits alone do not prove a live transformed pair.
    return bool(mask) and (
        bool(mask & ~PAIR_TRIGGER_BITS)
        or not np.isnan(values[row, rows[11]])
        or not np.isnan(values[row, rows[17]])
    )


@nb.njit(cache=True)
def _event_pair_rests(masks, values, rows, row):
    return _event_pair_at(masks, values, rows, row) and not (
        np.isnan(values[row, rows[11]]) and np.isnan(values[row, rows[17]])
    )


@nb.njit(cache=True)
def _event_pair_level(masks, values, rows, row, offset, active):
    if not _event_pair_at(masks, values, rows, row):
        return np.nan
    stop = values[row, rows[11 + offset]]
    if not np.isnan(stop):
        return stop
    return values[row, rows[12 + offset]] if active else np.nan


@nb.njit(cache=True)
def _event_stop_pair(masks, values, rows, row):
    return (
        _event_pair_rests(masks, values, rows, row)
        and np.isnan(values[row, rows[12]])
        and np.isnan(values[row, rows[18]])
    )


@nb.njit(cache=True)
def _event_lifetime_crosses(next_bar, kind, trigger, masks, values, rows, row):
    return (
        next_bar
        and (kind == _SLOT_SUBMIT or kind == _SLOT_REPLACE)
        and (not np.isnan(trigger) or _event_stop_pair(masks, values, rows, row))
    )


@nb.njit(cache=True)
def _event_open_in_market(trigger, side, masks, values, rows, row, opening):
    if _event_pair_at(masks, values, rows, row):
        long_stop, short_stop = values[row, rows[11]], values[row, rows[17]]
        return (not np.isnan(long_stop) and opening >= long_stop) or (
            not np.isnan(short_stop) and opening <= short_stop
        )
    return _in_market(-side, trigger, opening, True)


__all__ = [
    "_has_pairs",
    "_lifetime_crosses",
    "_open_in_market",
    "_pair_at",
    "_pair_level",
    "_pair_rests",
    "_stop_pair",
    "_straddle_column",
    "_event_pair_at",
    "_event_pair_rests",
    "_event_pair_level",
    "_event_stop_pair",
    "_event_lifetime_crosses",
    "_event_open_in_market",
]
