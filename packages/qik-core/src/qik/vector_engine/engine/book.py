"""The unified matcher (single-matcher plan, stage 2): one flat, event-gated loop.

Single-route semantics only, and not routed to yet. ``book_column`` takes the numeric ABI of
``legacy.single._column`` and returns the same seven-field result. Latency and ONE_TICK without
next-open entries still go to the venue matcher, as in ``_column``; a column driven by a strategy
callback goes to the legacy matcher until the loop owns the callback (that fallback goes in D).

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
  ``(due, kind, decision bar)``, the next open settles the ready ones in FIFO order, and an
  overflow is a refusal, never a drop;
* rule bits are decoded once, before the loop, from one ``int64``. The route-semantics
  differences between the single, venue and slots matchers are named compat bits in that word
  (``C_*``), set by the route entry, never a second copy of the code.
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
from .fills import _no_reversal
from .legacy.single import _column
from .venue import _venue_driver

# Rule bits, decoded once from the settings record.
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
# Compat bits: what the three routes disagree on for identical rules (REPORT section 2). A bit
# is named here when the loop has a branch for it; the venue/slots arms land with their stages.
# C_REFUSE_IN_MARKET: a stop or target at or through the entry close refuses the entry (ERR 1/2),
# where venue and slots accept it and apply A10/A11 (ERR 4, or an immediate target trade).
# C_REFUSE_PENDING_NEW_DAY: a working limit entry across a New York date change refuses (ERR 5);
# V2 drops the chase and leaves a DAY limit resting, which is not modelled.
C_REFUSE_IN_MARKET = 1 << 16
C_REFUSE_PENDING_NEW_DAY = 1 << 17
COMPAT_SINGLE = C_REFUSE_IN_MARKET | C_REFUSE_PENDING_NEW_DAY

# Transition ring: next-open staging. A close enqueues, the next open settles in FIFO order.
K_OPEN_EXIT, K_OPEN_ENTRY = 1, 2
CAP = 8  # a power of two; single holds at most a staged exit and a staged entry
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


@nb.njit(cache=True, nogil=True)
def _rule_flags(rules, compat):
    """The rule bits and the route's compat bits as one word, read once per column."""
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
    entries, run on the venue matcher (R15), as in ``_column``."""
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
    """The loop alone, for a column that ``_routes_here`` accepts: no fallback, numeric ABI.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
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
    refuse_in_market = flags & C_REFUSE_IN_MARKET != 0
    refuse_new_day = flags & C_REFUSE_PENDING_NEW_DAY != 0
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

    pos, cnt, ei, ep = 0, 0, -1, 0.0
    stop, target = np.nan, np.nan  # the published levels
    sx, tx = -np.inf, np.inf  # sentinel copies the sweep gate reads
    # A working limit entry: its side (0 none), level and chase bar are hot; the protection
    # levels and distances it will trade with are cold and live in ``rec``.
    pend, p_lim, p_chase = 0, 0.0, -1
    rec = np.full(4, np.nan)  # stop, target, stop points, target points of the working entry
    # Transition ring: next-open staging. A close enqueues, the next open settles in FIFO order.
    ring = np.empty((3, CAP), np.int64)  # due bar, kind, decision bar
    qmask, qhead, qtail = 0, 0, 0
    err, eb = ERR_NONE, -1
    for i in range(n):
        s = int(side_sig[i])
        if pos == 0 and pend == 0 and qmask == 0 and s == 0:
            continue  # nothing rests, nothing is in flight, nothing to enter
        exited = False
        busy = False  # a working entry resolved on this bar: the sleeve could not act (A15)
        pin_side = 0  # side of a position a target filled this bar at a stale quote (A13)
        pin = 0.0
        oi = o[i]
        ci = c[i]
        xg = _lane_at(exit_sig, i, False) if has_exit else False
        xl = _lane_at(long_exits, i, False) if has_long else False
        xs = _lane_at(short_exits, i, False) if has_short else False
        cmask = _lane_at(force_close, i, 0) if has_close else 0
        oflat = _lane_at(force_open, i, False) if has_open else False
        if pend != 0:
            if day_cancel and (_lane_at(expiry_days, i, False) if has_expiry else new_day[i]):
                pend, p_chase = 0, -1
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
            pos, pend, qmask, qhead, qtail, exited = 0, 0, 0, 0, 0, True
        while qmask != 0:
            ready = -1
            for t in range(qhead, qtail):
                slot = t & (CAP - 1)
                if (qmask >> slot) & 1 and ring[0, slot] <= i:
                    ready = slot
                    break
            if ready < 0:
                break
            qmask &= ~(1 << ready)
            while qhead < qtail and not (qmask >> (qhead & (CAP - 1))) & 1:
                qhead += 1
            d = ring[2, ready]
            if ring[1, ready] == K_OPEN_EXIT:
                _put_trade(
                    count_only, base + cnt, ei, i, pos, ep, oi - pos * slip, REASON_SIGNAL, stop,
                    out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                    out_reason, out_stop,
                )  # fmt: skip
                cnt += 1
                pos = 0
                exited = True
                continue
            sd = int(side_sig[d])
            cap = _lane_at(ioc_limits, d, np.nan) if has_ioc else np.nan
            if not np.isnan(cap) and sd * (oi - cap) > 0:
                continue  # the IOC cap is through at the open: the entry is dropped
            fill = oi + sd * slip
            st, tg = levels(
                sd,
                _anchor(bracket_anchor, c[d], fill, fill),
                _lane_at(entry_stop, d, np.nan) if has_es else np.nan,
                _lane_at(entry_target, d, np.nan) if has_et else np.nan,
                _lane_at(stop_pts, d, np.nan) if has_sp else np.nan,
                _lane_at(target_pts, d, np.nan) if has_tp else np.nan,
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
        if pend != 0:
            # A resting limit fills at its own level once any path point reaches it (A5); its
            # protection meets this bar's close, not the rest of the bar (A15).
            filled = (pend > 0 and lo[i] <= p_lim) or (pend < 0 and h[i] >= p_lim)
            busy = filled or i == p_chase
            # A chase is a close command: an exit on this bar flattens instead of chasing.
            if filled or (i == p_chase and not (xg or (pend > 0 and xl) or (pend < 0 and xs))):
                side = pend
                pend = 0
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
                    pos = 0
                    exited = True
        # --- decisions at bar i's close (the strategy callback goes here) ---
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
                pos, pend = 0, 0
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
            if qtail - qhead >= CAP:
                err, eb = ERR_RING_OVERFLOW, i
                break
            slot = qtail & (CAP - 1)
            ring[0, slot], ring[1, slot], ring[2, slot] = i + 1, K_OPEN_EXIT, i
            qmask |= 1 << slot
            qtail += 1
        if exit_now and not exit_next_open and pos != 0:
            _put_trade(
                count_only, base + cnt, ei, i, pos, ep, ci - pos * slip, REASON_SIGNAL, stop,
                out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                out_reason, out_stop,
            )  # fmt: skip
            cnt += 1
            pos = 0
            exited = True
            pend = 0
        if next_open or not np.isnan(_lane_at(ioc_limits, i, np.nan) if has_ioc else np.nan):
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
                if qtail - qhead >= CAP:
                    err, eb = ERR_RING_OVERFLOW, i
                    break
                slot = qtail & (CAP - 1)
                ring[0, slot], ring[1, slot], ring[2, slot] = i + 1, K_OPEN_ENTRY, i
                qmask |= 1 << slot
                qtail += 1
            continue
        if exit_now and not exit_next_open:
            pend = 0  # an exit also cancels a working entry
        if pos == 0 and pend == 0 and not busy and s != 0 and (reenter or not exited):
            lim = _lane_at(entry_limit, i, np.nan) if has_limit else np.nan
            es = _lane_at(entry_stop, i, np.nan) if has_es else np.nan
            et = _lane_at(entry_target, i, np.nan) if has_et else np.nan
            sp = _lane_at(stop_pts, i, np.nan) if has_sp else np.nan
            tp = _lane_at(target_pts, i, np.nan) if has_tp else np.nan
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


@nb.njit(cache=True, nogil=True)
def book_column(
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
    """Simulate one column with single-route semantics; same ABI and result as ``_column``.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """
    # A column the callback drives goes to legacy until the loop owns the callback (stage D).
    if core is not None:
        return _column(
            o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, entry_limit, stop_pts,
            target_pts, new_day, rules, long_exits, short_exits, force_close, force_open,
            count_only, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
            out_reason, out_stop, base, core, ioc_limits, expiry_days,
        )  # fmt: skip
    # R15: latency, and ONE_TICK without next-open entries, route to the venue matcher.
    if not _routes_here(rules):
        venue = _venue_driver(core)
        return venue(
            o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, entry_limit, stop_pts,
            target_pts, new_day, rules, long_exits, short_exits, force_close, force_open,
            count_only, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
            out_reason, out_stop, base, core, ioc_limits, expiry_days,
        )  # fmt: skip
    return _book_loop(
        o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, entry_limit, stop_pts,
        target_pts, new_day, rules, long_exits, short_exits, force_close, force_open, count_only,
        out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop, base,
        core, ioc_limits, expiry_days,
    )  # fmt: skip


__all__: list[str] = []
