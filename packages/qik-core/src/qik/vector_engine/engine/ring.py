"""The transition ring and the venue arm's plain-call helpers (single-matcher plan, stage 3).

The ring is the one queue behind both routes: single stages a next-open fill as a ring entry
``(due, kind, decision bar, held ident)``, venue keeps its command list in it. Entries settle in
insertion order once their due bar has come (dues need not be monotone); a full window with a hole
in it is compacted, and a window with every slot alive is an overflow refusal, never a drop.

The venue helpers (path sweep, limit slippage search, exit-row save and restore) are plain calls on
purpose: the loop reaches them from cold sites and one compiled copy keeps it small.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from qik.vector_engine.protection import REASON_TARGET, _in_market

from .codes import REASON_STOP, REASON_STOP_GAP
from .fills import _target_fill_price

# Transition ring. Single: next-open staging; a close enqueues, the next open settles in FIFO
# order. Venue: the command list, FIFO in insertion order, each entry settling once its due bar
# has come (dues need not be monotone).
K_OPEN_EXIT, K_OPEN_ENTRY = 1, 2
# Venue command kinds (the legacy codes): market entry, limit entry, protection, cancel entry,
# cancel protection, close, IOC entry (settles at its due open, before everything else).
K_MARKET, K_LIMIT, K_PROTECT, K_CANCEL, K_UNPROTECT, K_CLOSE, K_IOC = 1, 2, 3, 4, 5, 6, 7
CAP = 8  # a power of two; see the live-entry bound below
RING_FULL = (1 << CAP) - 1
ERR_RING_OVERFLOW = 15  # a hard refusal, never a drop; unreachable on both routes (bound below)


@nb.njit(cache=True, nogil=True)
def _ring_compact(ring, qmask, qhead, qtail):
    """Pack the live entries to the front in insertion order. Returns ``(qmask, qhead, qtail)``.

    A settled entry in the middle of the window leaves a hole that ``qhead`` cannot pass, so the
    window ``qtail - qhead`` can reach ``CAP`` with fewer than ``CAP`` entries alive.
    """
    packed = np.empty((ring.shape[0], CAP), np.int64)
    live = 0
    for t in range(qhead, qtail):
        slot = t & (CAP - 1)
        if (qmask >> slot) & 1:
            packed[:, live] = ring[:, slot]
            live += 1
    ring[:, :live] = packed[:, :live]
    return (1 << live) - 1, 0, live


@nb.njit(cache=True, nogil=True, inline="always")
def _ring_push(ring, qmask, qhead, qtail, due, kind, bar, ident):
    """Enqueue at the tail. Returns ``(qmask, qhead, qtail, ok)``; ``ok`` is False, and the ring
    untouched, only when every slot is alive: that is the overflow refusal, never a drop. A full
    window with a hole in it is compacted first, which keeps the insertion order."""
    if qtail - qhead >= CAP:
        if qmask == RING_FULL:
            return qmask, qhead, qtail, False
        qmask, qhead, qtail = _ring_compact(ring, qmask, qhead, qtail)
    slot = qtail & (CAP - 1)
    ring[0, slot], ring[1, slot], ring[2, slot], ring[3, slot] = due, kind, bar, ident
    return qmask | (1 << slot), qhead, qtail + 1, True


@nb.njit(cache=True, nogil=True)
def _cmd_push(ring, qmask, qhead, qtail, due, kind, bar, ident):
    """``_ring_push`` as a plain call: the venue arm pushes from many cold sites, and one compiled
    copy keeps the loop small."""
    return _ring_push(ring, qmask, qhead, qtail, due, kind, bar, ident)


@nb.njit(cache=True, nogil=True, inline="always")
def _ring_pop(ring, qmask, qhead, qtail, now):
    """Settle the first entry, in insertion order, whose due bar has come. Returns
    ``(slot, qmask, qhead)``; the slot is -1 when nothing is due. Dues need not be monotone: an
    entry behind a not-yet-due one is passed over, never reordered."""
    for t in range(qhead, qtail):
        slot = t & (CAP - 1)
        if (qmask >> slot) & 1 and ring[0, slot] <= now:
            qmask &= ~(1 << slot)
            while qhead < qtail and not (qmask >> (qhead & (CAP - 1))) & 1:
                qhead += 1
            return slot, qmask, qhead
    return -1, qmask, qhead


@nb.njit(cache=True, nogil=True, inline="always")
def _ring_pop_cmd(ring, qmask, qhead, qtail, now, ioc):
    """Venue: the first entry in insertion order whose due has come, among the IOC entries
    (``ioc``) or among every other kind. Returns ``(slot, qmask, qhead)``; the slot is -1 when
    nothing is ready. An entry behind a not-yet-due one is passed over, never reordered."""
    for t in range(qhead, qtail):
        slot = t & (CAP - 1)
        if (qmask >> slot) & 1 and ring[0, slot] <= now and (ring[1, slot] == K_IOC) == ioc:
            qmask &= ~(1 << slot)
            while qhead < qtail and not (qmask >> (qhead & (CAP - 1))) & 1:
                qhead += 1
            return slot, qmask, qhead
    return -1, qmask, qhead


@nb.njit(cache=True, nogil=True)
def _ring_drop(ring, qmask, qhead, qtail, bar):
    """Venue: remove every entry of the decision at ``bar`` (an expired entry's commands)."""
    for t in range(qhead, qtail):
        slot = t & (CAP - 1)
        if (qmask >> slot) & 1 and ring[2, slot] == bar:
            qmask &= ~(1 << slot)
    while qhead < qtail and not (qmask >> (qhead & (CAP - 1))) & 1:
        qhead += 1
    return qmask, qhead, qtail


@nb.njit(cache=True, nogil=True)
def _vsweep(oi, hi, li, ci, high_first, pos, stop, target, slip, rejected, adverse_first, gap):
    """Venue: the first path point (O, e1, e2, C) at which the live stop or target fills.

    Returns ``(reason, px)``, reason 0 for no fill. A stop is tested before the target at a
    point, and not at all once the venue rejected it. Slippage applies to the fill price of both
    legs, the target is also tested at the slipped point. Adverse-first suppresses the target
    on a bar whose extreme reaches the stop.
    """
    adverse_hit = (
        adverse_first
        and not rejected
        and not np.isnan(stop)
        and (li <= stop if pos > 0 else hi >= stop)
    )
    for m in range(4):
        pt = oi if m == 0 else (ci if m == 3 else (hi if (m == 1) == high_first else li))
        if not rejected and _in_market(pos, stop, pt, True):
            reason = REASON_STOP_GAP if m == 0 else REASON_STOP
            return reason, (pt - pos * slip if slip else (pt if m == 0 else stop))
        candidate = pt - pos * slip
        if not adverse_hit and _in_market(pos, target, candidate, False):
            return REASON_TARGET, _target_fill_price(pt, target, m == 0, gap, pos, slip)
    return 0, 0.0


@nb.njit(cache=True, nogil=True)
def _limit_fill_px(oi, hi, li, ci, high_first, side, slip, lim):
    """Venue: where a resting limit fills. Its own level, or with slippage the first path point
    whose adverse tick is still inside the limit."""
    px = lim
    if slip:
        for m in range(4):
            pt = oi if m == 0 else (ci if m == 3 else (hi if (m == 1) == high_first else li))
            candidate = pt + side * slip
            if (side > 0 and candidate <= lim) or (side < 0 and candidate >= lim):
                px = candidate
                break
    return px


@nb.njit(cache=True, nogil=True)
def _row_save(
    count_only, k, prior_i, prior_f, out_entry_i, out_exit_i, out_side, out_entry_px,
    out_exit_px, out_reason, out_stop,
):  # fmt: skip
    """Keep the row a trade is about to overwrite, so a refused bar can put it back."""
    if not count_only and k < len(out_entry_i):
        prior_i[0], prior_i[1], prior_i[2], prior_i[3] = (
            out_entry_i[k],
            out_exit_i[k],
            out_side[k],
            out_reason[k],
        )
        prior_f[0], prior_f[1], prior_f[2] = out_entry_px[k], out_exit_px[k], out_stop[k]


@nb.njit(cache=True, nogil=True)
def _row_restore(
    count_only, k, prior_i, prior_f, out_entry_i, out_exit_i, out_side, out_entry_px,
    out_exit_px, out_reason, out_stop,
):  # fmt: skip
    """Put back the row ``_row_save`` kept: a refused bar leaves nothing past the count."""
    if not count_only and k < len(out_entry_i):
        out_entry_i[k], out_exit_i[k], out_side[k], out_reason[k] = (
            prior_i[0],
            prior_i[1],
            prior_i[2],
            prior_i[3],
        )
        out_entry_px[k], out_exit_px[k], out_stop[k] = prior_f[0], prior_f[1], prior_f[2]


__all__ = [
    "CAP",
    "ERR_RING_OVERFLOW",
    "K_CANCEL",
    "K_CLOSE",
    "K_IOC",
    "K_LIMIT",
    "K_MARKET",
    "K_OPEN_ENTRY",
    "K_OPEN_EXIT",
    "K_PROTECT",
    "K_UNPROTECT",
    "RING_FULL",
    "_cmd_push",
    "_limit_fill_px",
    "_ring_compact",
    "_ring_drop",
    "_ring_pop",
    "_ring_pop_cmd",
    "_ring_push",
    "_row_restore",
    "_row_save",
    "_vsweep",
]
