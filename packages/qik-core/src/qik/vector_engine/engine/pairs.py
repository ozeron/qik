"""Straddle pair legs: pair reads, resting/level lookups and lifetime triggers."""

from __future__ import annotations

import numba as nb
import numpy as np

from qik.vector_engine.protection import _in_market

from .codes import _SLOT_REPLACE, _SLOT_SUBMIT


@nb.njit(cache=True, inline="never")
def _straddle_column(pair, j):
    if pair is None or pair.shape[2] == 0:
        return np.empty((12, 0))
    return pair[j]


@nb.njit(cache=True)
def _pair_at(pair, i):
    if pair is None or i < 0 or pair.shape[1] <= i:
        return False
    # Any finite leg, without allocating a column per call (it runs several times a bar).
    for leg in range(pair.shape[0]):
        if not np.isnan(pair[leg, i]):
            return True
    return False


@nb.njit(cache=True)
def _has_pairs(pair):
    return pair is not None and pair.shape[1] > 0


@nb.njit(cache=True)
def _pair_rests(pair, i):
    """A DAY stop leg rests; an IOC-only pair attempts once and never crosses a session."""
    return _pair_at(pair, i) and not (np.isnan(pair[0, i]) and np.isnan(pair[6, i]))


@nb.njit(cache=True)
def _pair_level(pair, i, offset, bar, latency):
    if pair is None or i < 0 or pair.shape[1] <= i:
        return np.nan
    if not np.isnan(pair[offset, i]):
        return pair[offset, i]
    return pair[offset + 1, i] if bar < i + latency + 1 else np.nan


@nb.njit(cache=True)
def _stop_pair(pair, i):
    """A pair whose legs are all resting stops (no IOC leg attempts at an open)."""
    return _pair_rests(pair, i) and np.isnan(pair[1, i]) and np.isnan(pair[7, i])


@nb.njit(cache=True)
def _lifetime_crosses(next_bar, kind, d, triggers, pair):
    """order_lifetime=next_bar: a stop submission may reach the next session's bar."""
    return (
        next_bar
        and (kind == _SLOT_SUBMIT or kind == _SLOT_REPLACE)
        and (not np.isnan(triggers[d]) or _stop_pair(pair, d))
    )


@nb.njit(cache=True)
def _open_in_market(triggers, pair, sides, d, opening):
    """An open-relative trigger at or through its own activation open."""
    if _pair_at(pair, d):
        return (not np.isnan(pair[0, d]) and opening >= pair[0, d]) or (
            not np.isnan(pair[6, d]) and opening <= pair[6, d]
        )
    return _in_market(-sides[d], triggers[d], opening, True)


__all__ = [
    "_has_pairs",
    "_lifetime_crosses",
    "_open_in_market",
    "_pair_at",
    "_pair_level",
    "_pair_rests",
    "_stop_pair",
    "_straddle_column",
]
