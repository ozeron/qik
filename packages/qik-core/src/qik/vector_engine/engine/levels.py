"""Level-store layout, level/risk writes and optional-column accessors."""

from __future__ import annotations

import numba as nb


@nb.njit(cache=True)
def _any_in(arr, lo, hi):
    """Any true value in a nonnegative half-open interval, clipped at the end."""
    for i in range(lo, min(hi, len(arr))):
        if arr[i]:
            return True
    return False


# (stored fields, declared fields, transformed legs, own levels 2/5, block row, own row).


@nb.njit(cache=True, inline="always")
def _layout_context(codes):
    return codes[0], codes[1], codes[2], codes[3], codes[4], codes[5]


@nb.njit(cache=True, inline="always")
def _level_row(store, blank, row):
    """One matcher input column: a stored row, or the read-only all-NaN row."""
    return store[row] if row >= 0 else blank


@nb.njit(cache=True, inline="always")
def _put_level(levels, layout, row, i, value):
    """Write a level the core transforms; other levels are read from their field's row."""
    legs, own, block, extra = layout[2], layout[3], layout[4], layout[5]
    if 11 <= row <= 22:
        if legs >> (row - 11) & 1:
            levels[block + row - 11, i] = value
    elif own >> row & 1:
        levels[extra + (1 if row == 5 and own & 4 else 0), i] = value


@nb.njit(cache=True)
def _store_risk(buffer, i, risk):
    if buffer is not None and len(buffer) > 0:
        buffer[i] = risk


@nb.njit(cache=True)
def _optional_at(a, i, default):
    if a is None or len(a) == 0:
        return default
    return a[i]


@nb.njit(cache=True)
def _optional_column(array, j):
    return array[0] if array.shape[1] == 0 else array[j]


__all__ = [
    "_any_in",
    "_layout_context",
    "_level_row",
    "_optional_at",
    "_optional_column",
    "_put_level",
    "_store_risk",
]
