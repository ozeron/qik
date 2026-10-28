"""The slots arm's command ring and its plain-call helpers (single-matcher plan, stage 4).

The slots matcher kept its commands in a reflected list of ``(due, kind, decision bar, value)``
tuples. Here that list is a fixed ring: ``SCAP`` slots, a used-slot bitmask and an insertion
number per entry. Entries settle in insertion order whatever slot they sit in, so a removal from
the middle (an expired entry's commands, a cancelled working order) never disturbs the order.
Every field lives in one float64 array (``RING_ROWS`` x ``SCAP``): each is a bar index, a small
code or a price, and float64 holds them exactly. One array argument keeps a call's reference
counting small.

A command carries what the decision it belongs to needs later: the decision bar ``d`` and its
held identity (``ident``: the bar in dense storage, the retained event row in event storage, see
``book._held_ident``), the IOC and straddle flags of that decision, and the price a trigger or
amendment will take.

The bound: a command settles once its due bar has come, and every command is ready by the
close of its due bar (``phase 1`` accepts any due). Dues are at most ``i + latency + 1`` with
latency 0 or 1, so after a bar's settle only commands pushed on that bar and the one before can
remain. A bar pushes at most 8 commands and its settle pushes at most 3 more, so the live count
stays under 24; ``SCAP`` is 32. A full ring is a refusal (``ERR_RING_OVERFLOW``), never a drop;
``tests/vector_engine/matcher_diff/slot_ring_bound.py`` measures the live count over the corpus.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba.extending import overload

from qik.vector_engine.protection import _in_market

from .book import _held_level
from .codes import (
    _SLOT_AMEND_STOP,
    _SLOT_ARM_STOP,
    _SLOT_PROTECT,
    _SLOT_REPLACE,
    _SLOT_SUBMIT,
)

SCAP = 32
SLOT_FULL = (1 << SCAP) - 1
# Ring rows.
RW_DUE, RW_KIND, RW_D, RW_IDENT, RW_SEQ, RW_IOC, RW_PAIR, RW_VALUE = range(8)
RING_ROWS = 8


@nb.njit(cache=True, nogil=True)
def _spush(ring, qmask, qseq, due, kind, d, ident, value, ioc, pair):
    """Enqueue at the next insertion number. Returns ``(qmask, qseq, ok)``; ``ok`` is False, and
    the ring untouched, only when every slot is alive (the overflow refusal)."""
    if qmask == SLOT_FULL:
        return qmask, qseq, False
    slot = 0
    while (qmask >> slot) & 1:
        slot += 1
    ring[RW_DUE, slot] = due
    ring[RW_KIND, slot] = kind
    ring[RW_D, slot] = d
    ring[RW_IDENT, slot] = ident
    ring[RW_SEQ, slot] = qseq
    ring[RW_IOC, slot] = ioc
    ring[RW_PAIR, slot] = pair
    ring[RW_VALUE, slot] = value
    return qmask | (1 << slot), qseq + 1, True


@nb.njit(cache=True, nogil=True, inline="always")
def _spop(ring, qmask, now, open_ok, phase1, gen, ei):
    """The first entry, in insertion order, that may settle at ``now``: its due has come and the
    phase allows it (the close settles anything; the open only next-open and IOC commands and the
    protection of the entry that just filled). Returns the slot, or -1 when nothing is ready."""
    ready = -1
    best = np.inf
    m = qmask
    slot = 0
    while m != 0:
        if m & 1 and ring[RW_DUE, slot] <= now and ring[RW_SEQ, slot] < best:
            kind = int(ring[RW_KIND, slot])
            if (
                phase1
                or open_ok
                or ring[RW_IOC, slot] != 0.0
                or (kind == _SLOT_PROTECT and gen == int(ring[RW_D, slot]) and ei == now)
            ):
                ready = slot
                best = ring[RW_SEQ, slot]
        m >>= 1
        slot += 1
    return ready


@nb.njit(cache=True, nogil=True)
def _sdrop(ring, qmask, d):
    """Remove every entry of the decision at bar ``d`` (an expired or replaced entry's commands)."""
    m = qmask
    slot = 0
    while m != 0:
        if m & 1 and int(ring[RW_D, slot]) == d:
            qmask &= ~(1 << slot)
        m >>= 1
        slot += 1
    return qmask


@nb.njit(cache=True, nogil=True, inline="always")
def _sbound(ring, qmask, gen, stop, pos):
    """The best stop a queued amendment of generation ``gen`` already carries: ``(has, bound)``.

    ``has`` is whether a stop exists (the live one, or a queued amend or arm); ``bound`` starts at
    ``stop`` and is replaced by a queued value only when strictly better for ``pos``.
    """
    has = not np.isnan(stop)
    bound = stop
    m = qmask
    slot = 0
    while m != 0:
        if m & 1:
            kind = int(ring[RW_KIND, slot])
            if (kind == _SLOT_AMEND_STOP or kind == _SLOT_ARM_STOP) and int(
                ring[RW_D, slot]
            ) == gen:
                has = True
                value = ring[RW_VALUE, slot]
                if np.isnan(bound) or pos * (value - bound) > 0:
                    bound = value
        m >>= 1
        slot += 1
    return has, bound


def _pair_cols(pair):
    """Columns of a straddle block: zero for an absent (None or ``(12, 0)``) one."""
    return 0 if pair is None else pair.shape[1]


@overload(_pair_cols, inline="always")
def _pair_cols_type(pair):
    if isinstance(pair, nb.types.NoneType):
        return lambda pair: 0
    return lambda pair: pair.shape[1]


def _held_pair(core, pair, leg, ident):
    """Leg ``leg`` of the straddle block of the held decision ``ident``; NaN when it has none."""
    if pair is None:
        return np.nan
    if core is None or core.event_mode != 2:
        return pair[leg, ident] if 0 <= ident < pair.shape[1] else np.nan
    row = core.level_rows[11 + leg]
    return core.ev_val[ident, row] if ident >= 0 and row >= 0 else np.nan


@overload(_held_pair, inline="always")
def _held_pair_type(core, pair, leg, ident):
    if isinstance(pair, nb.types.NoneType):
        return lambda core, pair, leg, ident: np.nan
    if isinstance(core, nb.types.NoneType):
        return lambda core, pair, leg, ident: pair[leg, ident] if ident < pair.shape[1] else np.nan

    def event(core, pair, leg, ident):
        if core.event_mode != 2:
            return pair[leg, ident] if 0 <= ident < pair.shape[1] else np.nan
        row = core.level_rows[11 + leg]
        return core.ev_val[ident, row] if ident >= 0 and row >= 0 else np.nan

    return event


@nb.njit(cache=True, nogil=True)
def _held_pair_any(core, pair, ident):
    """The held decision carries a straddle: any leg present (a plain call; only declared pairs)."""
    for leg in range(12):
        if not np.isnan(_held_pair(core, pair, leg, ident)):
            return True
    return False


@nb.njit(cache=True, nogil=True)
def _held_pair_rests(core, pair, ident):
    """A DAY stop leg rests; an IOC-only pair attempts once and never crosses a session."""
    return _held_pair_any(core, pair, ident) and not (
        np.isnan(_held_pair(core, pair, 0, ident)) and np.isnan(_held_pair(core, pair, 6, ident))
    )


@nb.njit(cache=True, nogil=True)
def _held_stop_pair(core, pair, ident):
    """A pair whose legs are all resting stops (no IOC leg attempts at an open)."""
    return (
        _held_pair_rests(core, pair, ident)
        and np.isnan(_held_pair(core, pair, 1, ident))
        and np.isnan(_held_pair(core, pair, 7, ident))
    )


@nb.njit(cache=True, nogil=True)
def _held_pair_level(core, pair, ident, offset, d, bar, latency):
    """The working level the strategy sees for one pair side: the stop leg, or the IOC cap while
    its attempt is still ahead (decision ``d``, observed at ``bar``)."""
    value = _held_pair(core, pair, offset, ident)
    if not np.isnan(value):
        return value
    return _held_pair(core, pair, offset + 1, ident) if bar < d + latency + 1 else np.nan


@nb.njit(cache=True, nogil=True)
def _held_open_in_market(core, triggers, has_trigger, pair, side, ident, opening):
    """An open-relative trigger at or through its own activation open."""
    if _held_pair_any(core, pair, ident):
        long_stop = _held_pair(core, pair, 0, ident)
        short_stop = _held_pair(core, pair, 6, ident)
        return (not np.isnan(long_stop) and opening >= long_stop) or (
            not np.isnan(short_stop) and opening <= short_stop
        )
    trigger = _held_level(core, triggers, ident, 5, np.nan) if has_trigger else np.nan
    return _in_market(-side, trigger, opening, True)


@nb.njit(cache=True, nogil=True)
def _held_lifetime_crosses(core, next_bar, kind, ident, triggers, has_trigger, pair):
    """order_lifetime=next_bar: a stop submission may reach the next session's bar."""
    if not (next_bar and (kind == _SLOT_SUBMIT or kind == _SLOT_REPLACE)):
        return False
    trigger = _held_level(core, triggers, ident, 5, np.nan) if has_trigger else np.nan
    return not np.isnan(trigger) or _held_stop_pair(core, pair, ident)


__all__ = [
    "RING_ROWS",
    "RW_D",
    "RW_DUE",
    "RW_IDENT",
    "RW_IOC",
    "RW_KIND",
    "RW_PAIR",
    "RW_SEQ",
    "RW_VALUE",
    "SCAP",
    "SLOT_FULL",
    "_held_pair",
    "_pair_cols",
    "_held_lifetime_crosses",
    "_held_open_in_market",
    "_held_pair_any",
    "_held_pair_level",
    "_held_pair_rests",
    "_sbound",
    "_sdrop",
    "_spop",
    "_spush",
    "_held_stop_pair",
]
