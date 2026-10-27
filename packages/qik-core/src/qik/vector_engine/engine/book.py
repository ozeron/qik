"""The unified matcher (single-matcher plan, stage 2): one flat, event-gated loop.

Single-route semantics, and the production single path: ``single._column`` routes a column here
unless latency, or ONE_TICK without next-open entries, sends it to the venue matcher (R15). The
loop takes the numeric ABI of the old matcher (``legacy.single._column``, the test oracle) and
returns the same seven-field result, with or without a strategy callback.

Shape (docs/reviews/single-matcher-design/REPORT.md section 3):

* hot state is scalars the loop carries in registers: ``pos``, ``cnt``, the order record
  ``(ei, ep, stop, target)`` and ``sx``/``tx``, sentinel copies of the live stop and target
  (+-inf when absent, so an absent order never touches). There is no state tuple, no phase
  function and no return per branch: a refusal sets ``err`` and breaks;
* a position bar costs one range test against the sentinels. Only a touching bar enters
  ``_sweep``, which takes scalars and returns scalars; a bar with nothing resting, nothing in
  flight and no entry signal is one load and one test;
* a working limit entry keeps its side, level and chase bar in scalars and its protection
  levels in a four-cell array; next-open staging is a transition ring: a close enqueues
  ``(due, kind, decision bar, decision identity)``, the next open settles the ready ones in FIFO
  order, and an overflow is a refusal, never a drop;
* rule bits are decoded once, before the loop, from one ``int64``. The route-semantics
  differences between the single, venue and slots matchers are named compat fields in that word
  (``C_*``), set by the route entry, never a second copy of the code;
* a strategy callback (``core`` is a ``CoreContext``) publishes the close decision at three
  sites, as the old matcher did: before a chase cancels, after a resting entry fills and before
  its protection is accepted, and after the ordinary sweeps. The callback runs on every bar,
  before the bar can be called quiet; a callback or account refusal stops the column at once with
  ``account[2]`` set. Decisions published into the callback's one-bar scratch are read back at
  the same close; a queued entry keeps the *identity* of its decision (``_held_ident``): the
  retained event row in event storage, the bar in dense storage.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba.extending import overload

from qik.vector_engine.protection import (
    ERR_NONE,
    REASON_TARGET,
    _anchor,
    _fill_then_protect,
    _in_market,
    levels,
)

from .codes import (
    ERR_NEXT_OPEN_SESSION_END,
    ERR_PENDING_NEW_DAY,
    ERR_STALE_BOOK,
    ERR_STOP_IN_MARKET,
    ERR_TARGET_IN_MARKET,
    R_ADAPTIVE,
    R_ADVERSE_FIRST,
    R_BRACKET_ANCHOR,
    R_CHASE,
    R_COMBINE,
    R_CROSS_SESSION,
    R_DAY_CANCEL,
    R_EXIT_NEXT_OPEN,
    R_LATENCY_BARS,
    R_NEXT_OPEN,
    R_ONE_TICK,
    R_PROTECT_FLATTEN,
    R_REENTER,
    R_STALE_DROP,
    R_TARGET_GAP,
    R_TICK,
    R_TICK_ROUND,
    REASON_SIGNAL,
    REASON_STOP,
    REASON_STOP_GAP,
)
from .decision import _core_decision, _core_decision_rare
from .events import _close_at, _open_at
from .fills import _no_reversal

# Rule bits, decoded once from the settings record: the low word of the flags.
(
    F_ADAPTIVE,
    F_ADVERSE_FIRST,
    F_REENTER,
    F_STALE_DROP,
    F_TARGET_GAP,
    F_PROTECT_FLATTEN,
    F_NEXT_OPEN,
    F_EXIT_NEXT_OPEN,
    F_ONE_TICK,
    F_CROSS_SESSION,
    F_DAY_CANCEL,
) = (1 << k for k in range(11))
_RULE_BITS = (1 << 11) - 1

# Compat fields: what the three routes disagree on for identical rules (REPORT section 2), in the
# high word of the flags. Each difference owns one disjoint field, and a field with more than two
# policies names its values. A field is declared when the loop has a branch for it; the venue and
# slots arms land with their stages.
#
# In-market protection: the single route refuses an entry whose stop or target is at or through
# the entry price (ERR 1/2); venue and slots accept it and apply A10/A11 (ERR 4, or an
# immediate target trade).
C_IN_MARKET_SHIFT = 16
C_IN_MARKET_MASK = 1 << C_IN_MARKET_SHIFT
IN_MARKET_REFUSE = 1
# A working limit entry across a New York date change: the single route refuses (ERR 5).
# TODO(stage 3): V2 drops the chase and leaves a DAY limit resting (the venue arm). Clearing this
# field does not model that: the chase would stay armed. The venue arm needs its own named value,
# and a route that sets it needs the drop in the loop; until then 0 is not a valid policy.
C_NEW_DAY_SHIFT = 17
C_NEW_DAY_MASK = 3 << C_NEW_DAY_SHIFT
NEW_DAY_REFUSE = 1
COMPAT_SINGLE = (IN_MARKET_REFUSE << C_IN_MARKET_SHIFT) | (NEW_DAY_REFUSE << C_NEW_DAY_SHIFT)
if _RULE_BITS & (C_IN_MARKET_MASK | C_NEW_DAY_MASK) or C_IN_MARKET_MASK & C_NEW_DAY_MASK:
    raise RuntimeError("rule bits and compat fields must be disjoint masks")

# Transition ring: next-open staging. A close enqueues, the next open settles in FIFO order.
K_OPEN_EXIT, K_OPEN_ENTRY = 1, 2
CAP = 8  # a power of two; single holds at most a staged exit and a staged entry
RING_FULL = (1 << CAP) - 1
ERR_RING_OVERFLOW = 15  # a hard refusal; unreachable on this route (two staged at most)


def _has_lane(a):
    """A lane carries values: an absent (None or zero-length) column reads as its default."""
    return a is not None and len(a) > 0


@overload(_has_lane, inline="always")
def _has_lane_type(a):
    if isinstance(a, nb.types.NoneType):
        return lambda a: False
    return lambda a: len(a) > 0


def _lane_at(a, i, default):
    """``a[i]`` for a lane that ``_has_lane`` accepted; the default for an absent (None) one."""
    return default if a is None else a[i]


@overload(_lane_at, inline="always")
def _lane_at_type(a, i, default):
    if isinstance(a, nb.types.NoneType):
        return lambda a, i, default: default
    return lambda a, i, default: a[i]


def _scratch_at(core, i):
    """Where the decision published at bar ``i``'s close is read: index 0 of the callback's
    one-bar scratch in event storage, the bar itself otherwise (dense columns, or no callback)."""
    return i if core is None or core.event_mode != 2 else 0


@overload(_scratch_at, inline="always")
def _scratch_at_type(core, i):
    if isinstance(core, nb.types.NoneType):
        return lambda core, i: i
    return lambda core, i: 0 if core.event_mode == 2 else i


def _held_ident(core, i):
    """What a queued entry keeps of the decision made at bar ``i``'s close, read right after it
    was published: the bar in dense storage, the retained event row in event storage (``-1``: the
    decision was empty, every field at its default). A bar index cannot find a decision once the
    scratch is reused; a retained row never moves."""
    return i if core is None or core.event_mode != 2 else core.event_state[1]


@overload(_held_ident, inline="always")
def _held_ident_type(core, i):
    if isinstance(core, nb.types.NoneType):
        return lambda core, i: i
    return lambda core, i: core.event_state[1] if core.event_mode == 2 else i


def _held_side(core, side_sig, ident):
    """The side of a held decision (``ident`` from ``_held_ident``)."""
    if core is None or core.event_mode != 2:
        return int(side_sig[ident])
    row = core.field_rows[0]
    return int(core.ev_val[ident, row]) if ident >= 0 and row >= 0 else 0


@overload(_held_side, inline="always")
def _held_side_type(core, side_sig, ident):
    if isinstance(core, nb.types.NoneType):
        return lambda core, side_sig, ident: int(side_sig[ident])

    def event(core, side_sig, ident):
        if core.event_mode != 2:
            return int(side_sig[ident])
        row = core.field_rows[0]
        return int(core.ev_val[ident, row]) if ident >= 0 and row >= 0 else 0

    return event


def _held_level(core, lane, ident, level, default):
    """A level of a held decision; ``default`` when the decision or the level is absent. A dense
    lane is gated by the caller with ``_has_lane``."""
    if core is None or core.event_mode != 2:
        return default if lane is None else lane[ident]
    row = core.level_rows[level]
    return core.ev_val[ident, row] if ident >= 0 and row >= 0 else default


@overload(_held_level, inline="always")
def _held_level_type(core, lane, ident, level, default):
    if isinstance(core, nb.types.NoneType):
        return lambda core, lane, ident, level, default: _lane_at(lane, ident, default)

    def event(core, lane, ident, level, default):
        if core.event_mode != 2:
            return _lane_at(lane, ident, default)
        row = core.level_rows[level]
        return core.ev_val[ident, row] if ident >= 0 and row >= 0 else default

    return event


@nb.njit(cache=True, nogil=True, inline="always")
def _exit_flags(exit_sig, long_exits, short_exits, at, has_exit, has_long, has_short):
    """The exit signals published at ``at``: any side, long only, short only."""
    return (
        _lane_at(exit_sig, at, False) if has_exit else False,
        _lane_at(long_exits, at, False) if has_long else False,
        _lane_at(short_exits, at, False) if has_short else False,
    )


@nb.njit(cache=True, nogil=True)
def _rule_flags(rules, compat):
    """The rule bits and the route's compat fields as one word, read once per column."""
    flags = compat
    for bit, index in (
        (F_ADAPTIVE, R_ADAPTIVE),
        (F_ADVERSE_FIRST, R_ADVERSE_FIRST),
        (F_REENTER, R_REENTER),
        (F_STALE_DROP, R_STALE_DROP),
        (F_TARGET_GAP, R_TARGET_GAP),
        (F_PROTECT_FLATTEN, R_PROTECT_FLATTEN),
        (F_NEXT_OPEN, R_NEXT_OPEN),
        (F_EXIT_NEXT_OPEN, R_EXIT_NEXT_OPEN),
        (F_ONE_TICK, R_ONE_TICK),
        (F_CROSS_SESSION, R_CROSS_SESSION),
        (F_DAY_CANCEL, R_DAY_CANCEL),
    ):
        if rules[index] != 0.0:
            flags |= bit
    return flags


@nb.njit(cache=True, nogil=True)
def _routes_here(rules):
    """Whether single semantics own this column. Latency, and ONE_TICK without next-open
    entries, run on the venue matcher (R15), as in the old matcher."""
    return not (
        int(rules[R_LATENCY_BARS]) != 0 or (rules[R_ONE_TICK] != 0.0 and rules[R_NEXT_OPEN] == 0.0)
    )


@nb.njit(cache=True, nogil=True, inline="always")
def _put_trade(
    count_only, k, ei, xi, side, ep, xp, why, stop, o_ei, o_xi, o_sd, o_ep, o_xp, o_why, o_st
):
    """One trade row. Counted, and written only when the buffers are live and large enough."""
    if not count_only and k < len(o_ei):
        o_ei[k], o_xi[k], o_sd[k] = ei, xi, side
        o_ep[k], o_xp[k], o_why[k], o_st[k] = ep, xp, why, stop


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


@nb.njit(cache=True, nogil=True)
def _sweep(oi, hi, li, pos, stop, target, adaptive, adverse_first, target_gap):
    """First touch of a resting stop or target on one bar, along the path O, e1, e2, C.

    Returns ``(reason, px, point, high_first)``: reason 0 is no fill; ``point`` is the path point
    (0 the open, 1 the first extreme, 2 the second) that produced the fill. A stop already
    through at the open fills at the open; a target already through fills at its own level, or
    at the open under target-gap. Adverse-first suppresses a target on a bar whose extreme
    reaches the stop. The extremes: high then low, or the one nearer the open first when
    adaptive (strictly nearer high first; ties low first).

    Exact on the levels, not on the loop's sentinels: an absent level (NaN) never fills, whatever
    the bar holds, infinities included.
    """
    high_first = (not adaptive) or abs(oi - hi) < abs(oi - li)
    has_stop = not np.isnan(stop)
    has_target = not np.isnan(target)
    if pos > 0:
        adverse_hit = adverse_first and has_stop and li <= stop
        if has_stop and oi <= stop:
            return REASON_STOP_GAP, oi, 0, high_first
        if not adverse_hit and has_target and oi >= target:
            return REASON_TARGET, (oi if target_gap else target), 0, high_first
        if high_first:
            if not adverse_hit and has_target and hi >= target:
                return REASON_TARGET, target, 1, high_first
            if has_stop and li <= stop:
                return REASON_STOP, stop, 2, high_first
        else:
            if has_stop and li <= stop:
                return REASON_STOP, stop, 1, high_first
            if not adverse_hit and has_target and hi >= target:
                return REASON_TARGET, target, 2, high_first
    else:
        adverse_hit = adverse_first and has_stop and hi >= stop
        if has_stop and oi >= stop:
            return REASON_STOP_GAP, oi, 0, high_first
        if not adverse_hit and has_target and oi <= target:
            return REASON_TARGET, (oi if target_gap else target), 0, high_first
        if high_first:
            if has_stop and hi >= stop:
                return REASON_STOP, stop, 1, high_first
            if not adverse_hit and has_target and li <= target:
                return REASON_TARGET, target, 2, high_first
        else:
            if not adverse_hit and has_target and li <= target:
                return REASON_TARGET, target, 1, high_first
            if has_stop and hi >= stop:
                return REASON_STOP, stop, 2, high_first
    return 0, 0.0, 0, high_first


@nb.njit(cache=True, nogil=True, inline="always")
def _stop_sentinel(side, level):
    """A stop that never touches when absent: -inf for a long, +inf for a short."""
    return -side * np.inf if np.isnan(level) else level


@nb.njit(cache=True, nogil=True, inline="always")
def _target_sentinel(side, level):
    return side * np.inf if np.isnan(level) else level


@nb.njit(cache=True, nogil=True)
def _book_loop(
    o,
    h,
    lo,
    c,
    side_sig,
    exit_sig,
    entry_stop,
    entry_target,
    entry_limit,
    stop_pts,
    target_pts,
    new_day,
    rules,
    long_exits,
    short_exits,
    force_close,
    force_open,
    count_only,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
    base,
    core=None,
    ioc_limits=None,
    expiry_days=None,
):
    """The single-route loop, for a column that ``_routes_here`` accepts: numeric ABI, optional
    strategy callback.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    A callback or account refusal returns ``err == ERR_NONE`` with ``core.account[2]`` set.
    """
    n = o.shape[0]
    flags = _rule_flags(rules, COMPAT_SINGLE)
    adaptive = flags & F_ADAPTIVE != 0
    adverse_first = flags & F_ADVERSE_FIRST != 0
    reenter = flags & F_REENTER != 0
    stale_drop = flags & F_STALE_DROP != 0
    target_gap = flags & F_TARGET_GAP != 0
    protect_flatten = flags & F_PROTECT_FLATTEN != 0
    next_open = flags & F_NEXT_OPEN != 0
    exit_next_open = flags & F_EXIT_NEXT_OPEN != 0
    cross_session = flags & F_CROSS_SESSION != 0
    day_cancel = flags & F_DAY_CANCEL != 0
    refuse_in_market = (flags & C_IN_MARKET_MASK) >> C_IN_MARKET_SHIFT == IN_MARKET_REFUSE
    refuse_new_day = (flags & C_NEW_DAY_MASK) >> C_NEW_DAY_SHIFT == NEW_DAY_REFUSE
    tick = rules[R_TICK]
    slip = (
        tick if flags & F_ONE_TICK != 0 else 0.0
    )  # R15: one adverse tick on market and stop fills
    tick_round = int(rules[R_TICK_ROUND])
    combine = int(rules[R_COMBINE])
    bracket_anchor = int(rules[R_BRACKET_ANCHOR])
    chase = int(rules[R_CHASE])
    # Absent columns (None or zero length) read as their default; decided once, not per bar.
    has_exit = _has_lane(exit_sig)
    has_long = _has_lane(long_exits)
    has_short = _has_lane(short_exits)
    has_es = _has_lane(entry_stop)
    has_et = _has_lane(entry_target)
    has_sp = _has_lane(stop_pts)
    has_tp = _has_lane(target_pts)
    has_limit = _has_lane(entry_limit)
    has_ioc = _has_lane(ioc_limits)
    has_close = _has_lane(force_close)
    has_open = _has_lane(force_open)
    has_expiry = _has_lane(expiry_days)

    # Loop-carried scalars start as int64, not as literals: a literal initial value makes Numba
    # compile a second copy of every outlined callee they are passed to.
    zero, unset = 0, (-1)
    pos, cnt, ei, ep = zero, zero, unset, 0.0
    stop, target = np.nan, np.nan  # the published levels
    sx, tx = -np.inf, np.inf  # sentinel copies the sweep gate reads
    # A working limit entry: its side (0 none), level and chase bar are hot; the protection
    # levels and distances it will trade with are cold and live in ``rec``.
    pend, p_lim, p_chase = zero, 0.0, unset
    rec = np.full(4, np.nan)  # stop, target, stop points, target points of the working entry
    # Transition ring: next-open staging. A close enqueues, the next open settles in FIFO order.
    ring = np.empty((4, CAP), np.int64)  # due bar, kind, decision bar, decision identity
    qmask, qhead, qtail = zero, zero, zero
    err, eb = ERR_NONE, -1
    for i in range(n):
        at = _scratch_at(core, i)  # where this bar's published decision is read
        s = 0
        if core is None:
            # Dense columns hold every decision already: a bar with nothing to do is skipped.
            # A callback column cannot be: the strategy observes every bar (decided below).
            s = int(side_sig[i])
            if pos == 0 and pend == 0 and qmask == 0 and s == 0:
                continue  # nothing rests, nothing is in flight, nothing to enter
            xg, xl, xs = _exit_flags(
                exit_sig, long_exits, short_exits, i, has_exit, has_long, has_short
            )
            cmask = _lane_at(force_close, i, 0) if has_close else 0
            oflat = _lane_at(force_open, i, False) if has_open else False
        else:
            # Nothing is published yet at this bar's open: the strategy decides at the close.
            xg, xl, xs = False, False, False
            cmask = 0
            oflat = _open_at(core, force_open, i)
        exited = False
        busy = False  # a working entry resolved on this bar: the sleeve could not act (A15)
        pin_side = 0  # side of a position a target filled this bar at a stale quote (A13)
        pin = 0.0
        oi = o[i]
        ci = c[i]
        if pend != 0:
            if day_cancel and (_lane_at(expiry_days, i, False) if has_expiry else new_day[i]):
                pend, p_chase = zero, unset
            if pend != 0 and new_day[i] and not day_cancel and refuse_new_day:
                err, eb = ERR_PENDING_NEW_DAY, i
                break
        # --- the open: a forced flatten, then the transitions staged at the last close ---
        if oflat:
            if pos != 0:
                _put_trade(
                    count_only, base + cnt, ei, i, pos, ep, oi - pos * slip, REASON_SIGNAL, stop,
                    out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                    out_reason, out_stop,
                )  # fmt: skip
                cnt += 1
            pos, pend, qmask, qhead, qtail, exited = zero, zero, zero, zero, zero, True
        while qmask != 0:
            ready, qmask, qhead = _ring_pop(ring, qmask, qhead, qtail, i)
            if ready < 0:
                break
            d = ring[2, ready]
            if ring[1, ready] == K_OPEN_EXIT:
                _put_trade(
                    count_only, base + cnt, ei, i, pos, ep, oi - pos * slip, REASON_SIGNAL, stop,
                    out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                    out_reason, out_stop,
                )  # fmt: skip
                cnt += 1
                pos = zero
                exited = True
                continue
            held = ring[3, ready]
            sd = _held_side(core, side_sig, held)
            cap = _held_level(core, ioc_limits, held, 10, np.nan) if has_ioc else np.nan
            if not np.isnan(cap) and sd * (oi - cap) > 0:
                continue  # the IOC cap is through at the open: the entry is dropped
            fill = oi + sd * slip
            st, tg = levels(
                sd,
                _anchor(bracket_anchor, c[d], fill, fill),
                _held_level(core, entry_stop, held, 0, np.nan) if has_es else np.nan,
                _held_level(core, entry_target, held, 1, np.nan) if has_et else np.nan,
                _held_level(core, stop_pts, held, 3, np.nan) if has_sp else np.nan,
                _held_level(core, target_pts, held, 4, np.nan) if has_tp else np.nan,
                tick,
                tick_round,
                combine=combine,
                fill=fill,
            )
            # Protection placed now is validated at the actual entry open (R1).
            if refuse_in_market:
                if _in_market(sd, st, oi, True):
                    err, eb = ERR_STOP_IN_MARKET, i
                    break
                if _in_market(sd, tg, oi, False):
                    err, eb = ERR_TARGET_IN_MARKET, i
                    break
            pos, ei, ep, stop, target = sd, i, fill, st, tg
            sx, tx = _stop_sentinel(sd, st), _target_sentinel(sd, tg)
        if err != ERR_NONE:
            break
        called = False  # the strategy has published this bar's close decision
        if pend != 0:
            # A resting limit fills at its own level once any path point reaches it (A5); its
            # protection meets this bar's close, not the rest of the bar (A15).
            filled = (pend > 0 and lo[i] <= p_lim) or (pend < 0 and h[i] >= p_lim)
            busy = filled or i == p_chase
            if core is not None and not filled and i == p_chase:
                # A chase is a close command, so the cancellation decision precedes that fill.
                _core_decision_rare(
                    core, i, oi, h[i], lo[i], ci, pos, ei, ep, stop, target, pend, p_lim, stop,
                    cnt, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                    out_reason, out_stop,
                )  # fmt: skip
                if core.account[2] != 0:
                    break
                called = True
                xg, xl, xs = _exit_flags(
                    exit_sig, long_exits, short_exits, at, has_exit, has_long, has_short
                )
            # A chase is a close command: an exit on this bar flattens instead of chasing.
            if filled or (i == p_chase and not (xg or (pend > 0 and xl) or (pend < 0 and xs))):
                side = pend
                pend = zero
                px = p_lim if filled else ci
                st, tg = levels(
                    side,
                    _anchor(bracket_anchor, c[p_chase - chase], px, px),
                    rec[0],
                    rec[1],
                    rec[2],
                    rec[3],
                    tick,
                    tick_round,
                    combine=combine,
                    fill=px,
                )
                if core is not None and filled:
                    # The resting entry has printed, but its callback protection is a close
                    # command: the strategy observes the position before the legs are accepted.
                    _core_decision_rare(
                        core, i, oi, h[i], lo[i], ci, side, i, px, np.nan, np.nan, pend,
                        np.nan, st, cnt, out_entry_i, out_exit_i, out_side, out_entry_px,
                        out_exit_px, out_reason, out_stop,
                    )  # fmt: skip
                    if core.account[2] != 0:
                        break
                    called = True
                e, why = _fill_then_protect(
                    side, i, px, st, tg, ci, protect_flatten, count_only, out_entry_i,
                    out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop,
                    base + cnt,
                )  # fmt: skip
                if e != ERR_NONE:
                    err, eb = e, i
                    break
                if why != 0:
                    cnt += 1
                    exited = True
                else:
                    pos, ei, ep, stop, target = side, i, px, st, tg
                    sx, tx = _stop_sentinel(side, st), _target_sentinel(side, tg)
        elif pos != 0:
            if pos > 0:
                touched = (oi <= sx) | (lo[i] <= sx) | (oi >= tx) | (h[i] >= tx)
            else:
                touched = (oi >= sx) | (h[i] >= sx) | (oi <= tx) | (lo[i] <= tx)
            if touched:
                reason, px, point, high_first = _sweep(
                    oi, h[i], lo[i], pos, stop, target, adaptive, adverse_first, target_gap
                )
                if reason != 0:
                    if reason != REASON_TARGET:
                        px -= pos * slip
                    _put_trade(
                        count_only, base + cnt, ei, i, pos, ep, px, reason, stop,
                        out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                        out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                    if reason == REASON_TARGET and point > 0:  # a gap fill pins nothing
                        e1 = h[i] if high_first else lo[i]
                        e2 = lo[i] if high_first else h[i]
                        if _no_reversal(oi, e1, e2, ci, point, pos):
                            pin_side = pos
                            pin = px
                    pos = zero
                    exited = True
        # --- decisions at bar i's close ---
        if core is not None:
            if not called:
                _core_decision(
                    core, i, oi, h[i], lo[i], ci, pos, ei, ep, stop, target, pend,
                    p_lim if pend else np.nan, stop, cnt, out_entry_i, out_exit_i, out_side,
                    out_entry_px, out_exit_px, out_reason, out_stop,
                )  # fmt: skip
                if core.account[2] != 0:
                    break
            # Whatever site published it, the close decision is read back from the scratch.
            s = int(side_sig[at])
            xg, xl, xs = _exit_flags(
                exit_sig, long_exits, short_exits, at, has_exit, has_long, has_short
            )
            cmask = _close_at(core, force_close, i)
            if pos == 0 and pend == 0 and qmask == 0 and s == 0:
                continue  # the strategy has observed the bar; there is nothing to act on
        if cmask & 8:
            if pos != 0 or pend != 0:
                err, eb = ERR_NEXT_OPEN_SESSION_END, i
                break
            continue
        if cmask != 0:
            fside = pos if pos != 0 else pend
            if (cmask & 1) or (fside > 0 and cmask & 2) or (fside < 0 and cmask & 4):
                if pos != 0:
                    _put_trade(
                        count_only, base + cnt, ei, i, pos, ep, ci - pos * slip, REASON_SIGNAL,
                        stop, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                        out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                pos, pend = zero, zero
                continue
        if oflat:
            continue
        xside = pos if pos != 0 else pend
        exit_now = xg or (xside > 0 and xl) or (xside < 0 and xs)
        stage_exit = exit_now and pos != 0 and exit_next_open
        if stage_exit:
            if i + 1 == n or (new_day[i + 1] and not cross_session):
                err, eb = ERR_NEXT_OPEN_SESSION_END, i
                break
            qmask, qhead, qtail, ok = _ring_push(
                ring, qmask, qhead, qtail, i + 1, K_OPEN_EXIT, i, -1
            )
            if not ok:
                err, eb = ERR_RING_OVERFLOW, i
                break
        if exit_now and not exit_next_open and pos != 0:
            _put_trade(
                count_only, base + cnt, ei, i, pos, ep, ci - pos * slip, REASON_SIGNAL, stop,
                out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                out_reason, out_stop,
            )  # fmt: skip
            cnt += 1
            pos = zero
            exited = True
            pend = zero
        if next_open or not np.isnan(_lane_at(ioc_limits, at, np.nan) if has_ioc else np.nan):
            # R1 decisions remain at the close; stage only actionable orders. At the following
            # open a market exit executes before a replacement entry and before the price sweep.
            enter = (
                (pos == 0 or (stage_exit and reenter))
                and (next_open or (pend == 0 and not busy))
                and s != 0
                and (reenter or not exited)
            )
            if enter:
                if i + 1 == n or (new_day[i + 1] and not cross_session):
                    err, eb = ERR_NEXT_OPEN_SESSION_END, i
                    break
                qmask, qhead, qtail, ok = _ring_push(
                    ring, qmask, qhead, qtail, i + 1, K_OPEN_ENTRY, i, _held_ident(core, i)
                )  # fmt: skip
                if not ok:
                    err, eb = ERR_RING_OVERFLOW, i
                    break
            continue
        if exit_now and not exit_next_open:
            pend = zero  # an exit also cancels a working entry
        if pos == 0 and pend == 0 and not busy and s != 0 and (reenter or not exited):
            lim = _lane_at(entry_limit, at, np.nan) if has_limit else np.nan
            es = _lane_at(entry_stop, at, np.nan) if has_es else np.nan
            et = _lane_at(entry_target, at, np.nan) if has_et else np.nan
            sp = _lane_at(stop_pts, at, np.nan) if has_sp else np.nan
            tp = _lane_at(target_pts, at, np.nan) if has_tp else np.nan
            if not np.isnan(lim):
                # A same-bar target fill's stale quote does not touch a limit entry (A13).
                if (s > 0 and lim >= ci) or (s < 0 and lim <= ci):
                    # marketable: fills at the close (A11); V2 anchors point levels to the limit
                    st, tg = levels(
                        s, _anchor(bracket_anchor, ci, ci, lim), es, et, sp, tp, tick,
                        tick_round, combine=combine, fill=ci,
                    )  # fmt: skip
                    e, why = _fill_then_protect(
                        s, i, ci, st, tg, ci, protect_flatten, count_only, out_entry_i,
                        out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop,
                        base + cnt,
                    )  # fmt: skip
                    if e != ERR_NONE:
                        err, eb = e, i
                        break
                    if why != 0:
                        cnt += 1
                    else:
                        pos, ei, ep, stop, target = s, i, ci, st, tg
                        sx, tx = _stop_sentinel(s, st), _target_sentinel(s, tg)
                else:
                    pend, p_lim, p_chase = s, lim, i + chase
                    rec[0], rec[1], rec[2], rec[3] = es, et, sp, tp
                continue
            st, tg = levels(
                s, ci, es, et, sp, tp, tick, tick_round, combine=combine, fill=ci
            )  # fmt: skip
            if refuse_in_market:
                if _in_market(s, st, ci, True):
                    err, eb = ERR_STOP_IN_MARKET, i
                    break
                if _in_market(s, tg, ci, False):
                    err, eb = ERR_TARGET_IN_MARKET, i
                    break
            if (
                pin_side == s
                and not np.isnan(st)
                and ((s > 0 and st >= pin) or (s < 0 and st <= pin))
            ):
                if not stale_drop:
                    err, eb = ERR_STALE_BOOK, i
                    break
                st = np.nan  # Nautilus rejects the stop and cancels the linked target (A13)
                tg = np.nan
            pos, ei, ep, stop, target = s, i, ci, st, tg
            sx, tx = _stop_sentinel(s, st), _target_sentinel(s, tg)
    return cnt, err, eb, pos, ei, ep, stop


__all__ = [
    "CAP",
    "COMPAT_SINGLE",
    "C_IN_MARKET_MASK",
    "C_IN_MARKET_SHIFT",
    "C_NEW_DAY_MASK",
    "C_NEW_DAY_SHIFT",
    "ERR_RING_OVERFLOW",
    "F_ADAPTIVE",
    "F_ADVERSE_FIRST",
    "F_CROSS_SESSION",
    "F_DAY_CANCEL",
    "F_EXIT_NEXT_OPEN",
    "F_NEXT_OPEN",
    "F_ONE_TICK",
    "F_PROTECT_FLATTEN",
    "F_REENTER",
    "F_STALE_DROP",
    "F_TARGET_GAP",
    "IN_MARKET_REFUSE",
    "K_OPEN_ENTRY",
    "K_OPEN_EXIT",
    "NEW_DAY_REFUSE",
    "RING_FULL",
    "_book_loop",
    "_exit_flags",
    "_has_lane",
    "_held_ident",
    "_held_level",
    "_held_side",
    "_lane_at",
    "_put_trade",
    "_ring_compact",
    "_ring_pop",
    "_ring_push",
    "_routes_here",
    "_rule_flags",
    "_scratch_at",
    "_stop_sentinel",
    "_sweep",
    "_target_sentinel",
]
