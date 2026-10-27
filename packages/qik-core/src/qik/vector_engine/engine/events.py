"""Sparse decision row access and current-bar scratch adapters for the matchers."""

from __future__ import annotations

import numba as nb
import numpy as np
from numba.extending import overload

from .levels import _optional_at


@nb.njit(cache=True, inline="never")
def _past_event_row(state, bars, count, bar, account):
    # Only forward queries reach this cursor. Orders retain their own rows.
    cursor = state[6]
    if cursor and bars[cursor - 1] >= bar:
        account[2] = 14
        return -1
    while cursor < count[0] and bars[cursor] < bar:
        cursor += 1
    state[6] = cursor
    return cursor if cursor < count[0] and bars[cursor] == bar else -1


@nb.njit(cache=True, inline="always")
def _event_row(core, bar):
    state = core.event_state
    if bar == state[0]:
        return state[1]
    if bar > state[0]:
        return -1
    if bar < 0:
        return -1
    if bar == state[2]:
        return state[3]
    if bar == state[4]:
        return state[5]
    if bar == state[7]:
        return state[8]
    return _past_event_row(state, core.ev_bar, core.ev_count, bar, core.account)


@nb.njit(cache=True, inline="always")
def _remember_event(core, bar):
    if bar != core.event_state[2]:
        row = _event_row(core, bar) if bar >= 0 else -1
        core.event_state[2], core.event_state[3] = bar, row


@nb.njit(cache=True, inline="always")
def _remember_generation(core, bar):
    if bar != core.event_state[7]:
        row = _event_row(core, bar) if bar >= 0 else -1
        core.event_state[7], core.event_state[8] = bar, row


@nb.njit(cache=True, inline="always")
def _event_get(core, row, bar, default):
    if row < 0:
        return default
    event = _event_row(core, bar)
    return core.ev_val[event, row] if event >= 0 else default


def _side_at(core, column, bar):
    return int(column[bar])


@overload(_side_at, inline="always")
def _side_at_type(core, column, bar):
    if isinstance(core, nb.types.NoneType):

        def dense(core, column, bar):
            return int(column[bar])

        return dense

    def event(core, column, bar):
        if bar == core.event_state[0]:
            return int(column[0 if core.event_mode == 2 else bar])
        return int(_event_get(core, core.field_rows[0], bar, 0.0))

    return event


def _flag_at(core, column, bar, field):
    return bool(_optional_at(column, bar, False))


@overload(_flag_at, inline="always")
def _flag_at_type(core, column, bar, field):
    if isinstance(core, nb.types.NoneType):

        def dense(core, column, bar, field):
            return bool(_optional_at(column, bar, False))

        return dense

    def event(core, column, bar, field):
        if core.boundary == 1 and core.ends[bar] and core.exit_next_open:
            return False
        if bar == core.event_state[0]:
            return bool(column[0 if core.event_mode == 2 else bar])
        return bool(_event_get(core, core.field_rows[field], bar, 0.0))

    return event


def _level_at(core, column, bar, level):
    default = 0.0 if level == 23 else np.nan
    return _optional_at(column, bar, default)


@overload(_level_at, inline="always")
def _level_at_type(core, column, bar, level):
    if isinstance(core, nb.types.NoneType):

        def dense(core, column, bar, level):
            default = 0.0 if level == 23 else np.nan
            return _optional_at(column, bar, default)

        return dense

    def event(core, column, bar, level):
        default = 0.0 if level == 23 else np.nan
        if bar == core.event_state[0]:
            return (
                column[0 if core.event_mode == 2 else bar]
                if core.level_rows[level] >= 0
                else default
            )
        return _event_get(core, core.level_rows[level], bar, default)

    return event


def _pair_value(core, pair, leg, bar):
    return pair[leg, bar]


@overload(_pair_value, inline="always")
def _pair_value_type(core, pair, leg, bar):
    if isinstance(core, nb.types.NoneType):

        def dense(core, pair, leg, bar):
            return pair[leg, bar]

        return dense

    def event(core, pair, leg, bar):
        if bar == core.event_state[0]:
            return pair[leg, 0 if core.event_mode == 2 else bar] if pair.shape[1] else np.nan
        return _event_get(core, core.level_rows[11 + leg], bar, np.nan)

    return event


@nb.njit(cache=True, inline="always")
def _close_at(core, column, bar):
    if core is not None and core.event_mode == 2:
        current = int(column[0]) if core.event_state[0] == bar else 0
        return current | (int(core.masks[0, bar]) if bar < core.masks.shape[1] else 0)
    return int(_optional_at(column, bar, 0))


@nb.njit(cache=True, inline="always")
def _open_at(core, column, bar):
    if core is not None and core.event_mode == 2:
        return (
            column[0] == bar
            or column[1] == bar
            or (bar < core.masks.shape[1] and bool(core.masks[1, bar]))
        )
    return bool(_optional_at(column, bar, False))


@nb.njit(cache=True, inline="always")
def _current_flag(core, column, bar):
    return column[0 if core.event_mode == 2 else bar]


__all__ = [
    "_past_event_row",
    "_event_row",
    "_remember_event",
    "_remember_generation",
    "_event_get",
    "_side_at",
    "_side_at_type",
    "_flag_at",
    "_flag_at_type",
    "_level_at",
    "_level_at_type",
    "_pair_value",
    "_pair_value_type",
    "_close_at",
    "_open_at",
    "_current_flag",
]
