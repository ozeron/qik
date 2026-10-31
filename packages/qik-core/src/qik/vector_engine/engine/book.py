"""The unified matcher (single-matcher plan, stages 2 and 3): one flat, event-gated loop.

Two arms, picked by the route word ``compat`` (``COMPAT_SINGLE``, ``COMPAT_VENUE``). The single
arm is the single-order matcher and the production single path: ``entry._column`` routes a column
to it unless latency, or ONE_TICK without next-open entries, sends it to the venue arm (R15). The
venue arm is the per-command settlement matcher: commands are ring entries settling once their due
bar has come, with the protection 0..3 machine, ``rejected_stop`` and protect-flatten, the limit
slippage search and the IOC command. The loop takes the numeric ABI of the matchers it replaced
and returns the same seven-field result, with or without a strategy callback. Both arms share the callback site
and the refusal/ring helpers; they differ in everything between the bar's open and its close
decision, which the route word selects once per column.

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

from qik.vector_engine.account import MODE_QTY_FEATURE
from qik.vector_engine.economic import E_MODE
from qik.vector_engine.protection import (
    ERR_NONE,
    ERR_PROTECT_REJECTED,
    REASON_PROTECT_FAIL,
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
    EXIT_CLOCK_NATIVE_OPEN,
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
from .decision import _core_decision, _core_decision_rare, _entry_quantity
from .events import _close_at, _open_at
from .fills import _high_first, _no_reversal, _venue_record
from .levels import _any_in
from .ring import (
    CAP,
    ERR_RING_OVERFLOW,
    K_CANCEL,
    K_CLOSE,
    K_IOC,
    K_LIMIT,
    K_MARKET,
    K_OPEN_ENTRY,
    K_OPEN_EXIT,
    K_PROTECT,
    K_UNPROTECT,
    _cmd_push,
    _limit_fill_px,
    _ring_drop,
    _ring_pop,
    _ring_pop_cmd,
    _ring_push,
    _row_restore,
    _row_save,
    _vsweep,
)

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
# A working limit entry across a New York date change: every route refuses (ERR 5), the venue
# route only for a pending entry that carries a limit. V2 itself drops a staged chase there and
# leaves the DAY limit resting, but no route models that (``reference.py`` refuses the same
# input), so the field has one policy. A value that drops the chase would change refusals into
# trades, which is a digest-changing decision for the owner, not a compat bit.
C_NEW_DAY_SHIFT = 17
C_NEW_DAY_MASK = 3 << C_NEW_DAY_SHIFT
NEW_DAY_REFUSE = 1
# Which matcher's semantics own the column. The venue route (latency, or ONE_TICK without
# next-open entries) runs a per-command settlement state machine: the protection 0..3 machine,
# the limit slippage search, the IOC command. It never reads next-open fills and does not model
# the stale quote (A13), so those rule bits are masked off for it in ``_rule_flags``. The slots
# route (``book_slots._slots_loop``) reads every rule; it differs from the other two in what it
# accepts and refuses, and it is its own loop: see the slots values below.
C_ROUTE_SHIFT = 19
C_ROUTE_MASK = 3 << C_ROUTE_SHIFT
ROUTE_VENUE = 1
ROUTE_SLOTS = 2
C_ROUTE_VENUE_BIT = ROUTE_VENUE << C_ROUTE_SHIFT  # the venue route's bit; slots has its own
_VENUE_IGNORES = F_NEXT_OPEN | F_EXIT_NEXT_OPEN | F_STALE_DROP
# The slots route accepts protection at or through the entry price and applies A10/A11 (ERR 4, ERR 8
# for a MIT target, or an immediate target trade), like the venue; it refuses a working order
# across a New York date change only while a trigger, a limit or a resting straddle leg is
# working, and not on the bar after a GTD stop was submitted (``order_lifetime=next_bar``).
IN_MARKET_ACCEPT = 0
NEW_DAY_REFUSE_RESTING = 2
COMPAT_SINGLE = (IN_MARKET_REFUSE << C_IN_MARKET_SHIFT) | (NEW_DAY_REFUSE << C_NEW_DAY_SHIFT)
COMPAT_VENUE = (NEW_DAY_REFUSE << C_NEW_DAY_SHIFT) | (ROUTE_VENUE << C_ROUTE_SHIFT)
COMPAT_SLOTS = (
    (IN_MARKET_ACCEPT << C_IN_MARKET_SHIFT)
    | (NEW_DAY_REFUSE_RESTING << C_NEW_DAY_SHIFT)
    | (ROUTE_SLOTS << C_ROUTE_SHIFT)
)
if (
    _RULE_BITS & (C_IN_MARKET_MASK | C_NEW_DAY_MASK | C_ROUTE_MASK)
    or C_IN_MARKET_MASK & (C_NEW_DAY_MASK | C_ROUTE_MASK)
    or C_NEW_DAY_MASK & C_ROUTE_MASK
):
    raise RuntimeError("rule bits and compat fields must be disjoint masks")


# Venue scalars that are cold (loaded on events): indices into one int64 array.
# decision bar and retained identity of the pending entry, the position's decision bar and side,
# protection 0 none / 1 submitted / 2 working / 3 cancel in flight, then flags.
VS_PD, VS_PIDENT, VS_GEN, VS_GSIDE, VS_PROT = 0, 1, 2, 3, 4
VS_CLOSING, VS_REASON, VS_REJECTED, VS_RESTING, VS_CHASING = 5, 6, 7, 8, 9
VS_SIZE = 10


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


@nb.njit(cache=True, nogil=True)
def _entry_levels(
    core, side, anchor, fill, held, entry_stop, entry_target, stop_pts, target_pts,
    has_es, has_et, has_sp, has_tp, tick, tick_round, combine,
):  # fmt: skip
    """The protection levels of the held decision ``held`` for an entry of ``side`` that filled
    at ``fill``. A plain call: the loop resolves them at several entry sites, and the retained-row
    reads expand large in the callback build, so one compiled copy keeps the loop small."""
    return levels(
        side,
        anchor,
        _held_level(core, entry_stop, held, 0, np.nan) if has_es else np.nan,
        _held_level(core, entry_target, held, 1, np.nan) if has_et else np.nan,
        _held_level(core, stop_pts, held, 3, np.nan) if has_sp else np.nan,
        _held_level(core, target_pts, held, 4, np.nan) if has_tp else np.nan,
        tick,
        tick_round,
        combine=combine,
        fill=fill,
    )


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
    if compat & C_ROUTE_VENUE_BIT:
        flags &= ~_VENUE_IGNORES  # the venue never reads these (R15 keeps them off its route)
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


@nb.njit(cache=True, nogil=True)
def _native_open_close(side, open_px, stop, target):
    """Resting protection sees the open quote before a scheduled market close.

    Native qualification witnesses this on a real Nautilus quote stream. Repro
    keeps its historical command-first convention. The settings width is fixed.
    """
    if not np.isnan(stop) and side * (open_px - stop) <= 0:
        return REASON_STOP_GAP, open_px
    if not np.isnan(target) and side * (open_px - target) >= 0:
        return REASON_TARGET, target
    return REASON_SIGNAL, open_px


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
    compat=COMPAT_SINGLE,
):
    """The matcher loop: numeric ABI, optional strategy callback, the route in ``compat``.

    ``COMPAT_SINGLE`` is the single route (``_routes_here``); ``COMPAT_VENUE`` the venue route
    (latency, or ONE_TICK without next-open entries): same arguments, same seven-field result.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    A callback or account refusal returns ``err == ERR_NONE`` with ``core.account[2]`` set.
    """
    n = o.shape[0]
    flags = _rule_flags(rules, compat)
    venue = flags & C_ROUTE_VENUE_BIT != 0
    lat = int(rules[R_LATENCY_BARS])
    adaptive = flags & F_ADAPTIVE != 0
    adverse_first = flags & F_ADVERSE_FIRST != 0
    reenter = flags & F_REENTER != 0
    stale_drop = flags & F_STALE_DROP != 0
    target_gap = flags & F_TARGET_GAP != 0
    protect_flatten = flags & F_PROTECT_FLATTEN != 0
    next_open = flags & F_NEXT_OPEN != 0
    exit_next_open = flags & F_EXIT_NEXT_OPEN != 0
    native_open_priority = rules[R_EXIT_NEXT_OPEN] == EXIT_CLOCK_NATIVE_OPEN
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
    # compile a second copy of every outlined callee they are passed to. ``n - n`` is a typed zero.
    zero = n - n
    unset = zero - 1
    pos, cnt, ei = zero, zero, unset
    ep = np.nan if venue else 0.0  # the venue's initial entry price is NaN
    stop, target = np.nan, np.nan  # the published levels
    sx, tx = -np.inf, np.inf  # sentinel copies the sweep gate reads
    # A working limit entry: its side (0 none), level and chase bar are hot; the protection
    # levels and distances it will trade with are cold and live in ``rec``.
    pend, p_lim, p_chase = zero, 0.0, unset
    rec = np.full(4, np.nan)  # stop, target, stop points, target points of the working entry
    # Venue only: the cold state, and the exit row a refused bar puts back (see ``VS_*``).
    vs = np.zeros(VS_SIZE, np.int64)
    prior_i, prior_f = np.zeros(4, np.int64), np.zeros(3)
    vs[VS_GEN], vs[VS_REASON] = -1, REASON_SIGNAL
    ring = np.empty((4, CAP), np.int64)  # due bar, kind, decision bar, decision identity
    qmask, qhead, qtail = zero, zero, zero
    err, eb = ERR_NONE, -1
    for i in range(n):
        at = _scratch_at(core, i)
        if venue:
            # --- the venue route, up to the strategy's close decision ---
            # A per-command settlement machine (the legacy venue matcher): commands are ring
            # entries that settle once their due bar has come, in insertion order.
            exited = False
            busy = pend != 0  # an entry in flight or working: the sleeve cannot place another
            swept = zero  # the side of the position a stop/target exit closed on this bar
            oi = o[i]
            ci = c[i]
            if core is None:
                s = int(side_sig[i])
                xg, xl, xs = _exit_flags(
                    exit_sig, long_exits, short_exits, i, has_exit, has_long, has_short
                )
                cmask = _lane_at(force_close, i, 0) if has_close else 0
                oflat = _lane_at(force_open, i, False) if has_open else False
                if (
                    pos == 0
                    and pend == 0
                    and qmask == 0
                    and s == 0
                    and vs[VS_PROT] == 0
                    and vs[VS_CLOSING] == 0
                ):
                    continue  # nothing rests, nothing is in flight, nothing to enter
            else:
                s = 0
                xg, xl, xs = False, False, False
                cmask = 0
                oflat = _open_at(core, force_open, i)
            pin_side = 0
            pin = 0.0
            if pend != 0 and not np.isnan(p_lim):
                if day_cancel and (_lane_at(expiry_days, i, False) if has_expiry else new_day[i]):
                    qmask, qhead, qtail = _ring_drop(ring, qmask, qhead, qtail, vs[VS_PD])
                    pend = zero
                    vs[VS_RESTING], vs[VS_CHASING], vs[VS_CLOSING] = 0, 0, 0
                    busy = False
                elif refuse_new_day and new_day[i] and not day_cancel:
                    err, eb = ERR_PENDING_NEW_DAY, i
                    break
            if oflat:
                if pos != 0:
                    _put_trade(
                        count_only, base + cnt, ei, i, pos, ep, oi - pos * slip, REASON_SIGNAL,
                        stop, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                        out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                pos, pend, qmask, qhead, qtail = zero, zero, zero, zero, zero
                vs[VS_PROT], vs[VS_CLOSING], vs[VS_RESTING], vs[VS_CHASING] = 0, 0, 0, 0
            # IOC entries settle at their due open, before anything else rests or fills.
            while qmask != 0:
                ready, qmask, qhead = _ring_pop_cmd(ring, qmask, qhead, qtail, i, True)
                if ready < 0:
                    break
                vd = ring[2, ready]
                if pend == 0 or vs[VS_PD] != vd:
                    continue
                held = ring[3, ready]
                cap = _held_level(core, ioc_limits, held, 10, np.nan) if has_ioc else np.nan
                if pend * (oi - cap) > 0:
                    pend = zero
                    vs[VS_RESTING], vs[VS_CHASING], vs[VS_CLOSING] = 0, 0, 0
                    continue
                st, tg = _entry_levels(
                    core,
                    pend,
                    _anchor(bracket_anchor, c[vd], oi, oi),
                    oi,
                    held,
                    entry_stop,
                    entry_target,
                    stop_pts,
                    target_pts,
                    has_es,
                    has_et,
                    has_sp,
                    has_tp,
                    tick,
                    tick_round,
                    combine,
                )
                if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                    _entry_quantity(core, vd, cnt)
                pos, ei, ep, stop, target = pend, i, oi, st, tg
                sx, tx = _stop_sentinel(pos, st), _target_sentinel(pos, tg)
                vs[VS_GEN], vs[VS_GSIDE] = vd, pos
                pend = zero
                vs[VS_RESTING], vs[VS_CHASING], vs[VS_REJECTED] = 0, 0, 0
                vs[VS_PROT] = 0 if vs[VS_CLOSING] != 0 else 1
                qmask, qhead, qtail, ok = _ring_push(
                    ring, qmask, qhead, qtail, i + lat,
                    K_CLOSE if vs[VS_CLOSING] != 0 else K_PROTECT, vd, -1,
                )  # fmt: skip
                if not ok:
                    err, eb = ERR_RING_OVERFLOW, i
                    break
            if err != ERR_NONE:
                break
            if (
                vs[VS_CLOSING] != 0
                and pend == 0
                and pos == 0
                and (vs[VS_PROT] == 0 or vs[VS_PROT] == 3)
            ):
                vs[VS_CLOSING] = 0
            hf = _high_first(adaptive, oi, h[i], lo[i])
            # The first matching point. A newly filled entry has no resting protection yet.
            if pend != 0 and vs[VS_RESTING] != 0:
                ext = lo[i] if pend > 0 else h[i]
                if (pend > 0 and ext + slip <= p_lim) or (pend < 0 and ext - slip >= p_lim):
                    vd = vs[VS_PD]
                    held = vs[VS_PIDENT]
                    px = _limit_fill_px(oi, h[i], lo[i], ci, hf, pend, slip, p_lim)
                    st, tg = _entry_levels(
                        core,
                        pend,
                        _anchor(bracket_anchor, c[vd], px, p_lim),
                        px,
                        held,
                        entry_stop,
                        entry_target,
                        stop_pts,
                        target_pts,
                        has_es,
                        has_et,
                        has_sp,
                        has_tp,
                        tick,
                        tick_round,
                        combine,
                    )
                    if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                        _entry_quantity(core, vd, cnt)
                    pos, ei, ep, stop, target = pend, i, px, st, tg
                    sx, tx = _stop_sentinel(pos, st), _target_sentinel(pos, tg)
                    vs[VS_GEN], vs[VS_GSIDE] = vd, pos
                    pend = zero
                    vs[VS_RESTING], vs[VS_CHASING], vs[VS_REJECTED] = 0, 0, 0
                    vs[VS_PROT] = 0 if vs[VS_CLOSING] != 0 else 1
                    qmask, qhead, qtail, ok = _ring_push(
                        ring, qmask, qhead, qtail, i + lat,
                        K_CLOSE if vs[VS_CLOSING] != 0 else K_PROTECT, vd, -1,
                    )  # fmt: skip
                    if not ok:
                        err, eb = ERR_RING_OVERFLOW, i
                        break
            elif pos == 0 and vs[VS_PROT] == 2 and vs[VS_REJECTED] != 0 and not np.isnan(target):
                # A flat reduce-only target is cancelled before its liquidity is checked.
                ext = h[i] if vs[VS_GSIDE] > 0 else lo[i]
                if _in_market(vs[VS_GSIDE], target, ext, False):
                    vs[VS_PROT] = 0
            elif pos != 0 and vs[VS_PROT] >= 2:
                if pos > 0:
                    touched = (
                        (oi <= sx) | (h[i] <= sx) | (lo[i] <= sx) | (ci <= sx)
                        | (oi - slip >= tx) | (h[i] - slip >= tx)
                        | (lo[i] - slip >= tx) | (ci - slip >= tx)
                    )  # fmt: skip
                else:
                    touched = (
                        (oi >= sx) | (h[i] >= sx) | (lo[i] >= sx) | (ci >= sx)
                        | (oi + slip <= tx) | (h[i] + slip <= tx)
                        | (lo[i] + slip <= tx) | (ci + slip <= tx)
                    )  # fmt: skip
                if touched:
                    reason, px = _vsweep(
                        oi, h[i], lo[i], ci, hf, pos, stop, target, slip, vs[VS_REJECTED] != 0,
                        adverse_first, target_gap,
                    )  # fmt: skip
                    if reason != 0:
                        _row_save(
                            count_only, base + cnt, prior_i, prior_f, out_entry_i, out_exit_i,
                            out_side, out_entry_px, out_exit_px, out_reason, out_stop,
                        )  # fmt: skip
                        _put_trade(
                            count_only, base + cnt, ei, i, pos, ep, px, reason, stop,
                            out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                            out_reason, out_stop,
                        )  # fmt: skip
                        cnt += 1
                        swept = pos
                        pos = zero
                        exited = True
                        qmask, qhead, qtail, ok = _ring_push(
                            ring, qmask, qhead, qtail, i + lat, K_UNPROTECT, vs[VS_GEN], -1
                        )
                        if not ok:
                            err, eb = ERR_RING_OVERFLOW, i
                            break
                        vs[VS_PROT] = 3
            called = False
        else:
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
                    why, close_px = REASON_SIGNAL, oi
                    if native_open_priority:
                        why, close_px = _native_open_close(pos, oi, stop, target)
                    _put_trade(
                        count_only, base + cnt, ei, i, pos, ep, close_px - pos * slip, why, stop,
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
                st, tg = _entry_levels(
                    core,
                    sd,
                    _anchor(bracket_anchor, c[d], fill, fill),
                    fill,
                    held,
                    entry_stop,
                    entry_target,
                    stop_pts,
                    target_pts,
                    has_es,
                    has_et,
                    has_sp,
                    has_tp,
                    tick,
                    tick_round,
                    combine,
                )
                # Protection placed now is validated at the actual entry open (R1).
                if refuse_in_market:
                    if _in_market(sd, st, oi, True):
                        err, eb = ERR_STOP_IN_MARKET, i
                        break
                    if _in_market(sd, tg, oi, False):
                        err, eb = ERR_TARGET_IN_MARKET, i
                        break
                if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                    _entry_quantity(core, d, cnt)
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
                    if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                        _entry_quantity(core, p_chase - chase, cnt)
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
                # The venue publishes the protection that is working, not the one in flight.
                _core_decision(
                    core, i, oi, h[i], lo[i], ci, pos, ei, ep,
                    stop if not venue or (vs[VS_PROT] >= 2 and vs[VS_REJECTED] == 0) else np.nan,
                    target if not venue or vs[VS_PROT] >= 2 else np.nan,
                    pend, p_lim if pend else np.nan, stop, cnt, out_entry_i, out_exit_i, out_side,
                    out_entry_px, out_exit_px, out_reason, out_stop,
                )  # fmt: skip
                if core.account[2] != 0:
                    break
            s = int(side_sig[at])
            xg, xl, xs = _exit_flags(
                exit_sig, long_exits, short_exits, at, has_exit, has_long, has_short
            )
            cmask = _close_at(core, force_close, i)
            if venue:
                oflat = _open_at(core, force_open, i)  # the venue re-reads it after the callback
            if (
                pos == 0
                and pend == 0
                and qmask == 0
                and s == 0
                and (not venue or (vs[VS_PROT] == 0 and vs[VS_CLOSING] == 0))
            ):
                continue
        if cmask & 8:
            if pos != 0 or pend != 0:
                err, eb = ERR_NEXT_OPEN_SESSION_END, i
                break
            continue
        if venue:
            # --- the venue route, from the strategy's close decision: commands, then settlement ---
            fside = pos if pos != 0 else pend
            if ((cmask & 1) or (fside > 0 and cmask & 2) or (fside < 0 and cmask & 4)) or oflat:
                if pos != 0:
                    _venue_record(
                        count_only, base + cnt, ei, i, pos, ep, ci - pos * slip, REASON_SIGNAL,
                        stop, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                        out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                pos, pend, qmask, qhead, qtail = zero, zero, zero, zero, zero
                vs[VS_PROT], vs[VS_CLOSING], vs[VS_RESTING], vs[VS_CHASING] = 0, 0, 0, 0
                continue
            if (
                vs[VS_CLOSING] != 0
                and pend == 0
                and pos == 0
                and (vs[VS_PROT] == 0 or vs[VS_PROT] == 3)
            ):
                vs[VS_CLOSING] = 0
            if vs[VS_CLOSING] != 0 and pos == 0 and vs[VS_PROT] == 2:
                qmask, qhead, qtail, ok = _cmd_push(
                    ring, qmask, qhead, qtail, i + lat, K_UNPROTECT, vs[VS_GEN], -1
                )
                if not ok:
                    err, eb = ERR_RING_OVERFLOW, i
                vs[VS_PROT] = 3
            xside = pos if pos != 0 else pend
            exit_now = xg or (xside > 0 and xl) or (xside < 0 and xs)
            if pend != 0 and vs[VS_CLOSING] == 0 and not np.isnan(p_lim) and i == p_chase:
                vs[VS_CHASING] = 1
                qmask, qhead, qtail, ok = _cmd_push(
                    ring, qmask, qhead, qtail, i + lat, K_CANCEL, vs[VS_PD], -1
                )
                if not ok:
                    err, eb = ERR_RING_OVERFLOW, i
            if exit_now:
                vs[VS_CHASING] = 0
                if pend != 0:
                    if vs[VS_CLOSING] == 0 and not np.isnan(p_lim):
                        qmask, qhead, qtail, ok = _cmd_push(
                            ring, qmask, qhead, qtail, i + lat, K_CANCEL, vs[VS_PD], -1
                        )
                        if not ok:
                            err, eb = ERR_RING_OVERFLOW, i
                    vs[VS_CLOSING], vs[VS_REASON] = 1, REASON_SIGNAL
                if pos != 0:
                    if vs[VS_PROT] == 1 or vs[VS_PROT] == 2:
                        qmask, qhead, qtail, ok = _cmd_push(
                            ring, qmask, qhead, qtail, i + lat, K_UNPROTECT, vs[VS_GEN], -1
                        )
                        if not ok:
                            err, eb = ERR_RING_OVERFLOW, i
                        # SUBMITTED protection must not become matchable merely
                        # because its cancel was sent; acceptance still comes first.
                        if vs[VS_PROT] == 2:
                            vs[VS_PROT] = 3
                    if vs[VS_CLOSING] == 0:
                        qmask, qhead, qtail, ok = _cmd_push(
                            ring, qmask, qhead, qtail, i + lat, K_CLOSE, vs[VS_GEN], -1
                        )
                        if not ok:
                            err, eb = ERR_RING_OVERFLOW, i
                    vs[VS_CLOSING], vs[VS_REASON] = 1, REASON_SIGNAL
            if (
                pos == 0
                and pend == 0
                and not busy
                and vs[VS_CLOSING] == 0
                and s != 0
                and (reenter or not exited)
            ):
                lim = _lane_at(entry_limit, at, np.nan) if has_limit else np.nan
                cap = _lane_at(ioc_limits, at, np.nan) if has_ioc else np.nan
                pend, p_lim, p_chase = s, lim, i + chase
                vs[VS_PD], vs[VS_PIDENT] = i, _held_ident(core, i)
                if not np.isnan(cap):
                    due = i + lat + 1
                    if due >= n or (not cross_session and _any_in(new_day, i + 1, due + 1)):
                        if exited:
                            # The refused bar reports the state it started in: its exit row and
                            # count are put back.
                            _row_restore(
                                count_only, base + cnt - 1, prior_i, prior_f, out_entry_i,
                                out_exit_i, out_side, out_entry_px, out_exit_px, out_reason,
                                out_stop,
                            )  # fmt: skip
                            cnt -= 1
                            pos = swept
                        err, eb = ERR_NEXT_OPEN_SESSION_END, i
                        break
                    qmask, qhead, qtail, ok = _cmd_push(
                        ring, qmask, qhead, qtail, due, K_IOC, i, vs[VS_PIDENT]
                    )
                else:
                    qmask, qhead, qtail, ok = _cmd_push(
                        ring, qmask, qhead, qtail, i + lat,
                        K_MARKET if np.isnan(lim) else K_LIMIT, i, vs[VS_PIDENT],
                    )  # fmt: skip
                if not ok:
                    err, eb = ERR_RING_OVERFLOW, i
            if err != ERR_NONE:
                break
            while qmask != 0:
                ready, qmask, qhead = _ring_pop_cmd(ring, qmask, qhead, qtail, i, False)
                if ready < 0:
                    break
                kind = ring[1, ready]
                vd = ring[2, ready]
                if kind <= K_LIMIT:
                    if pend == 0 or vs[VS_PD] != vd:
                        continue
                    px = ci + pend * slip
                    if kind == K_MARKET or (pend > 0 and px <= p_lim) or (pend < 0 and px >= p_lim):
                        held = vs[VS_PIDENT]
                        st, tg = _entry_levels(
                            core,
                            pend,
                            _anchor(bracket_anchor, c[vd], px, px if kind == K_MARKET else p_lim),
                            px,
                            held,
                            entry_stop,
                            entry_target,
                            stop_pts,
                            target_pts,
                            has_es,
                            has_et,
                            has_sp,
                            has_tp,
                            tick,
                            tick_round,
                            combine,
                        )
                        if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                            _entry_quantity(core, vd, cnt)
                        pos, ei, ep, stop, target = pend, i, px, st, tg
                        sx, tx = _stop_sentinel(pos, st), _target_sentinel(pos, tg)
                        vs[VS_GEN], vs[VS_GSIDE] = vd, pos
                        pend = zero
                        vs[VS_RESTING], vs[VS_CHASING], vs[VS_REJECTED] = 0, 0, 0
                        vs[VS_PROT] = 0 if vs[VS_CLOSING] != 0 else 1
                        qmask, qhead, qtail, ok = _cmd_push(
                            ring, qmask, qhead, qtail, i + lat,
                            K_CLOSE if vs[VS_CLOSING] != 0 else K_PROTECT, vd, -1,
                        )  # fmt: skip
                        if not ok:
                            err, eb = ERR_RING_OVERFLOW, i
                    else:
                        vs[VS_RESTING] = 1
                elif kind == K_CANCEL:
                    if pend != 0 and vs[VS_PD] == vd:
                        vs[VS_RESTING] = 0
                        if vs[VS_CHASING] != 0:
                            # The chase: the limit is gone, the entry goes in at market.
                            vs[VS_CHASING] = 0
                            qmask, qhead, qtail, ok = _cmd_push(
                                ring, qmask, qhead, qtail, i + lat, K_MARKET, vd, -1
                            )
                            if not ok:
                                err, eb = ERR_RING_OVERFLOW, i
                        else:
                            pend = zero
                elif kind == K_PROTECT and vs[VS_GEN] == vd:
                    vs[VS_PROT] = 2
                    if pos != 0:
                        if _in_market(pos, stop, ci, True):
                            # The venue rejects a stop at or through the market (A10).
                            if not protect_flatten:
                                err, eb = ERR_PROTECT_REJECTED, i
                                break
                            vs[VS_REJECTED] = 1
                            if np.isnan(target):
                                vs[VS_PROT] = 0
                            else:
                                # The rejection callback clears the owned target even
                                # while its submission is in flight (V2 11e1a204e).
                                qmask, qhead, qtail, ok = _cmd_push(
                                    ring, qmask, qhead, qtail, i + lat, K_UNPROTECT, vd, -1
                                )
                                if not ok:
                                    err, eb = ERR_RING_OVERFLOW, i
                                vs[VS_PROT] = 3
                            if vs[VS_CLOSING] == 0:
                                vs[VS_REASON] = REASON_PROTECT_FAIL
                                qmask, qhead, qtail, ok = _cmd_push(
                                    ring, qmask, qhead, qtail, i + lat, K_CLOSE, vd, -1
                                )
                                if not ok:
                                    err, eb = ERR_RING_OVERFLOW, i
                            vs[VS_CLOSING] = 1
                        px = ci - pos * slip
                        if _in_market(pos, target, px, False):  # a marketable target fills (A11)
                            _venue_record(
                                count_only, base + cnt, ei, i, pos, ep, px, REASON_TARGET, stop,
                                out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                                out_reason, out_stop,
                            )  # fmt: skip
                            cnt += 1
                            pos = zero
                            qmask, qhead, qtail, ok = _cmd_push(
                                ring, qmask, qhead, qtail, i + lat, K_UNPROTECT, vd, -1
                            )
                            if not ok:
                                err, eb = ERR_RING_OVERFLOW, i
                            vs[VS_PROT] = 3
                elif kind == K_UNPROTECT and vs[VS_GEN] == vd:
                    vs[VS_PROT] = 0
                elif kind == K_CLOSE and pos != 0:
                    _venue_record(
                        count_only, base + cnt, ei, i, pos, ep, ci - pos * slip, vs[VS_REASON],
                        stop, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                        out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                    pos = zero
                if err != ERR_NONE:
                    break
            if err != ERR_NONE:
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
                    if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                        _entry_quantity(core, i, cnt)
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
            if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                _entry_quantity(core, i, cnt)
            pos, ei, ep, stop, target = s, i, ci, st, tg
            sx, tx = _stop_sentinel(s, st), _target_sentinel(s, tg)
    return cnt, err, eb, pos, ei, ep, stop


__all__ = [
    "COMPAT_SINGLE",
    "COMPAT_SLOTS",
    "COMPAT_VENUE",
    "C_IN_MARKET_MASK",
    "C_IN_MARKET_SHIFT",
    "C_NEW_DAY_MASK",
    "C_NEW_DAY_SHIFT",
    "C_ROUTE_MASK",
    "C_ROUTE_VENUE_BIT",
    "C_ROUTE_SHIFT",
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
    "IN_MARKET_ACCEPT",
    "IN_MARKET_REFUSE",
    "NEW_DAY_REFUSE",
    "NEW_DAY_REFUSE_RESTING",
    "ROUTE_SLOTS",
    "ROUTE_VENUE",
    "VS_CHASING",
    "VS_CLOSING",
    "VS_GEN",
    "VS_GSIDE",
    "VS_PD",
    "VS_PIDENT",
    "VS_PROT",
    "VS_REASON",
    "VS_REJECTED",
    "VS_RESTING",
    "VS_SIZE",
    "_book_loop",
    "_native_open_close",
    "_exit_flags",
    "_has_lane",
    "_held_ident",
    "_held_level",
    "_held_side",
    "_lane_at",
    "_put_trade",
    "_routes_here",
    "_entry_levels",
    "_rule_flags",
    "_scratch_at",
    "_stop_sentinel",
    "_sweep",
    "_target_sentinel",
    "_VENUE_IGNORES",
]
