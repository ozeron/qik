"""Numba kernel: closed-form per-bar execution rules, one column (config) at a time.

The rules are written down in ``docs/vector-engine.md`` (A1-A20 and repro-only R1). ``reference.py`` is an
independent, slow implementation of the same spec that matches orders price point by price point;
``tests`` check the two agree exactly and that both reproduce Nautilus golden data.

All inputs are float64 / int8 / bool arrays with one row per column (config) and one column per bar.
"""

from __future__ import annotations

from collections import namedtuple

import numba as nb
import numpy as np
from numba.extending import overload

from .account import fill_cost, settle
from .account import quantity as _contracts_for
from .protection import (
    ERR_NONE,
    ERR_PROTECT_REJECTED,
    REASON_PROTECT_FAIL,
    REASON_TARGET,
    _anchor,
    _fill_then_protect,
    _in_market,
    _round_price,
    levels,
)

REASON_SIGNAL = 1  # market exit at the decision close (R1: the next open)
REASON_STOP = 2  # resting stop-market hit during a bar, filled at the stop level
REASON_STOP_GAP = 4  # resting stop already through at the bar's open, filled at the open

ERR_STOP_IN_MARKET = 1  # stop level not strictly beyond the entry close (Nautilus rejects it)
ERR_TARGET_IN_MARKET = 2  # target level not strictly beyond the entry close (would fill at once)
ERR_STALE_BOOK = 3  # new stop falls inside the stale quote a same-bar limit fill left (A13)
ERR_PENDING_NEW_DAY = 5  # a limit entry still working when the New York date changes (A15)
ERR_NEXT_OPEN_SESSION_END = 6  # R1 decision has no following bar on the same New York date
ERR_STOP_ENTRY_IN_MARKET = 7
ERR_MIT_IN_MARKET = 8
ERR_AMEND_REJECTED = 9
ERR_AMEND_NO_ORDER = 10
ERR_STOP_REPLACE_LATENCY = 11

_SLOT_SUBMIT = 1
_SLOT_PROTECT = 2
_SLOT_CANCEL = 3
_SLOT_CHASE = 4
_SLOT_REPLACE = 5
_SLOT_CLOSE = 6
_SLOT_PROTECT_FAIL = 7
_SLOT_AMEND_STOP = 8
_SLOT_AMEND_TARGET = 9


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
    if not count_only:
        out_entry_i[k], out_exit_i[k], out_side[k] = ei, i, side
        out_entry_px[k], out_exit_px[k] = ep, px
        out_reason[k], out_stop[k] = why, stop


@nb.njit(cache=True, nogil=True)
def _venue_column(
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
    chase,
    adaptive,
    reenter,
    protect_flatten,
    latency,
    slip,
    adverse_first,
    bracket_anchor,
    target_gap,
    tick_round,
    round_tick,
    long_exits,
    short_exits,
    force_close,
    force_open,
    cross_session,
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
    combine=0,
):
    """A16/A17 venue settlement, with a bounded-by-flight-time FIFO command list.

    Commands: 1 market, 2 limit, 3 protection, 4 cancel entry, 5 cancel protection, 6 close.
    A command's generation is its entry decision row; cancels cannot erase a later trade's legs.
    """
    commands = [(0, 0, 0)]
    commands.pop()
    state = VenueState(
        commands, 0, -1, -1, np.nan, np.nan, np.nan, -1,
        False, False, 0, False, REASON_SIGNAL, False, 0,
    )  # fmt: skip
    inputs = (
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
        chase,
        adaptive,
        reenter,
        protect_flatten,
        latency,
        slip,
        adverse_first,
        bracket_anchor,
        target_gap,
        tick_round,
        round_tick,
        long_exits,
        short_exits,
        force_close,
        force_open,
        cross_session,
        count_only,
        out_entry_i,
        out_exit_i,
        out_side,
        out_entry_px,
        out_exit_px,
        out_reason,
        out_stop,
        base,
        ioc_limits,
        combine,
    )
    for i in range(len(c)):
        state, err, eb = _venue_column_advance(state, i, inputs, core)
        if err:
            return state.cnt, err, eb, state.pos, state.ei, state.ep, state.stop
    return state.cnt, ERR_NONE, -1, state.pos, state.ei, state.ep, state.stop


VenueState = namedtuple(
    "VenueState",
    "commands pos ei gen ep stop target pending resting chasing protection closing close_reason rejected_stop cnt",
)


@nb.njit(cache=True, nogil=True, inline="always")
def _venue_column_advance(state, i, inputs, core=None):
    (
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
        chase,
        adaptive,
        reenter,
        protect_flatten,
        latency,
        slip,
        adverse_first,
        bracket_anchor,
        target_gap,
        tick_round,
        round_tick,
        long_exits,
        short_exits,
        force_close,
        force_open,
        cross_session,
        count_only,
        out_entry_i,
        out_exit_i,
        out_side,
        out_entry_px,
        out_exit_px,
        out_reason,
        out_stop,
        base,
        ioc_limits,
        combine,
    ) = inputs
    """Advance one bar; the batch and compiled-core drivers share this matcher."""
    (
        commands,
        pos,
        ei,
        gen,
        ep,
        stop,
        target,
        pending,
        resting,
        chasing,
        protection,
        closing,
        close_reason,
        rejected_stop,
        cnt,
    ) = state
    ci = c[i]
    long_exit_i = _optional_at(long_exits, i, False)
    short_exit_i = _optional_at(short_exits, i, False)
    close_mask = _optional_at(force_close, i, 0)
    open_flatten = _optional_at(force_open, i, False)
    exit_now = _exit_for(
        exit_sig[i],
        long_exit_i,
        short_exit_i,
        pos if pos else (side_sig[pending] if pending >= 0 else 0),
    )
    exited = False
    busy = pending >= 0
    if pending >= 0 and not np.isnan(entry_limit[pending]) and new_day[i]:
        return (
            VenueState(
                commands,
                pos,
                ei,
                gen,
                ep,
                stop,
                target,
                pending,
                resting,
                chasing,
                protection,
                closing,
                close_reason,
                rejected_stop,
                cnt,
            ),
            ERR_PENDING_NEW_DAY,
            i,
        )
    if open_flatten:
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                o[i] - pos * slip,
                REASON_SIGNAL,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
        commands.clear()
        pos, pending, protection, closing, resting, chasing = 0, -1, 0, False, False, False
    for q in range(len(commands) - 1, -1, -1):
        due, kind, d = commands[q]
        if kind == 7 and due <= i:
            commands.pop(q)
            if pending != d:
                continue
            s = side_sig[d]
            cap = _optional_at(ioc_limits, d, np.nan)
            if s * (o[i] - cap) > 0:
                pending, resting, chasing, closing = -1, False, False, False
                continue
            pos, ei, ep, gen = s, i, o[i], d
            stop, target = levels(
                s,
                _anchor(bracket_anchor, c[d], ep, ep),
                entry_stop[d],
                entry_target[d],
                stop_pts[d],
                target_pts[d],
                round_tick,
                tick_round,
            )
            pending, resting, chasing, rejected_stop = -1, False, False, False
            protection = 0 if closing else 1
            commands.append((i + latency, 6 if closing else 3, d))
    if closing and pending < 0 and pos == 0 and (protection == 0 or protection == 3):
        closing = False
    high_first = not adaptive or abs(o[i] - h[i]) < abs(o[i] - lo[i])
    # Locate the first matching point. A newly filled entry has no resting protection yet.
    if pending >= 0 and resting:
        d = pending
        s = side_sig[d]
        extreme = lo[i] if s > 0 else h[i]
        if (s > 0 and extreme + slip <= entry_limit[d]) or (
            s < 0 and extreme - slip >= entry_limit[d]
        ):
            px = entry_limit[d]
            if slip:
                for m in range(4):
                    pt = (
                        o[i]
                        if m == 0
                        else (c[i] if m == 3 else (h[i] if (m == 1) == high_first else lo[i]))
                    )
                    candidate = pt + s * slip
                    if (s > 0 and candidate <= entry_limit[d]) or (
                        s < 0 and candidate >= entry_limit[d]
                    ):
                        px = candidate
                        break
            pos, ei, ep, gen = s, i, px, d
            stop, target = levels(
                s,
                _anchor(bracket_anchor, c[d], px, entry_limit[d]),
                entry_stop[d],
                entry_target[d],
                stop_pts[d],
                target_pts[d],
                round_tick,
                tick_round,
                combine=combine,
                fill=px,
            )
            pending, resting, chasing = -1, False, False
            rejected_stop = False
            protection = 0 if closing else 1
            commands.append((i + latency, 6 if closing else 3, d))
    elif pos == 0 and protection == 2 and rejected_stop and not np.isnan(target):
        extreme = h[i] if side_sig[gen] > 0 else lo[i]
        if _in_market(side_sig[gen], target, extreme, False):
            protection = 0  # flat reduce-only target is cancelled before liquidity is checked
    elif pos and protection >= 2:
        adverse_hit = (
            adverse_first
            and not rejected_stop
            and not np.isnan(stop)
            and (lo[i] <= stop if pos > 0 else h[i] >= stop)
        )
        reason = 0
        px = 0.0
        for m in range(4):
            pt = (
                o[i]
                if m == 0
                else (c[i] if m == 3 else (h[i] if (m == 1) == high_first else lo[i]))
            )
            if not rejected_stop and _in_market(pos, stop, pt, True):
                reason = REASON_STOP_GAP if m == 0 else REASON_STOP
                px = pt - pos * slip if slip else (pt if m == 0 else stop)
                break
            candidate = pt - pos * slip
            if not adverse_hit and _in_market(pos, target, candidate, False):
                reason = REASON_TARGET
                px = candidate if slip else (pt if target_gap and m == 0 else target)
                break
        if reason:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                px,
                reason,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
            pos, exited = 0, True
            commands.append((i + latency, 5, gen))
            protection = 3
    if core is not None:
        _core_decision(
            core,
            i,
            o[i],
            h[i],
            lo[i],
            c[i],
            pos,
            ei,
            ep,
            (stop if protection >= 2 and not rejected_stop else np.nan),
            (target if protection >= 2 else np.nan),
            side_sig[pending] if pending >= 0 else 0,
            entry_limit[pending] if pending >= 0 else np.nan,
            stop,
            cnt,
            out_entry_i,
            out_exit_i,
            out_side,
            out_entry_px,
            out_exit_px,
            out_reason,
            out_stop,
        )
        long_exit_i, short_exit_i = core.long_exits[i], core.short_exits[i]
        close_mask, open_flatten = core.force_close[i], core.force_open[i]
    if close_mask & 8:
        if pos or pending >= 0:
            return (
                VenueState(
                    commands,
                    pos,
                    ei,
                    gen,
                    ep,
                    stop,
                    target,
                    pending,
                    resting,
                    chasing,
                    protection,
                    closing,
                    close_reason,
                    rejected_stop,
                    cnt,
                ),
                ERR_NEXT_OPEN_SESSION_END,
                i,
            )
        return (
            VenueState(
                commands,
                pos,
                ei,
                gen,
                ep,
                stop,
                target,
                pending,
                resting,
                chasing,
                protection,
                closing,
                close_reason,
                rejected_stop,
                cnt,
            ),
            ERR_NONE,
            -1,
        )
    if (
        _force_for(close_mask, pos if pos else (side_sig[pending] if pending >= 0 else 0))
        or open_flatten
    ):
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                ci - pos * slip,
                REASON_SIGNAL,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
        commands.clear()
        pos, pending, protection, closing, resting, chasing = 0, -1, 0, False, False, False
        return (
            VenueState(
                commands,
                pos,
                ei,
                gen,
                ep,
                stop,
                target,
                pending,
                resting,
                chasing,
                protection,
                closing,
                close_reason,
                rejected_stop,
                cnt,
            ),
            ERR_NONE,
            -1,
        )
    if closing and pending < 0 and pos == 0 and (protection == 0 or protection == 3):
        closing = False
    if closing and pos == 0 and protection == 2:
        commands.append((i + latency, 5, gen))
        protection = 3
    exit_now = _exit_for(
        exit_sig[i],
        long_exit_i,
        short_exit_i,
        pos if pos else (side_sig[pending] if pending >= 0 else 0),
    )
    if pending >= 0 and not closing and not np.isnan(entry_limit[pending]) and i == pending + chase:
        chasing = True
        commands.append((i + latency, 4, pending))
    if exit_now:
        chasing = False
        if pending >= 0:
            if not closing and not np.isnan(entry_limit[pending]):
                commands.append((i + latency, 4, pending))
            closing, close_reason = True, REASON_SIGNAL
        if pos:
            if protection == 2:
                commands.append((i + latency, 5, gen))
                protection = 3
            if not closing:
                commands.append((i + latency, 6, gen))
            closing, close_reason = True, REASON_SIGNAL
    if (
        pos == 0
        and pending < 0
        and not busy
        and not closing
        and side_sig[i] != 0
        and (reenter or not exited)
    ):
        pending = i
        if not np.isnan(_optional_at(ioc_limits, i, np.nan)):
            due = i + latency + 1
            if due >= len(c) or (not cross_session and np.any(new_day[i + 1 : due + 1])):
                return state, ERR_NEXT_OPEN_SESSION_END, i
            commands.append((due, 7, i))
        else:
            commands.append((i + latency, 1 if np.isnan(entry_limit[i]) else 2, i))
    while len(commands):
        ready = -1
        for q in range(len(commands)):
            if commands[q][0] <= i and commands[q][1] != 7:
                ready = q
                break
        if ready < 0:
            break
        _, kind, d = commands.pop(ready)
        if kind == 1 or kind == 2:
            if pending != d:
                continue
            s = side_sig[d]
            px = ci + s * slip
            if kind == 1 or (s > 0 and px <= entry_limit[d]) or (s < 0 and px >= entry_limit[d]):
                anchor = _anchor(bracket_anchor, c[d], px, px if kind == 1 else entry_limit[d])
                pos, ei, ep, gen = s, i, px, d
                stop, target = levels(
                    s,
                    anchor,
                    entry_stop[d],
                    entry_target[d],
                    stop_pts[d],
                    target_pts[d],
                    round_tick,
                    tick_round,
                    combine=combine,
                    fill=px,
                )
                pending, resting, chasing = -1, False, False
                rejected_stop = False
                protection = 0 if closing else 1
                commands.append((i + latency, 6 if closing else 3, d))
            else:
                resting = True
        elif kind == 4:
            if pending == d:
                pending, resting = -1, False
                if chasing:
                    chasing = False
                    pending = d
                    commands.append((i + latency, 1, d))
        elif kind == 3 and gen == d:
            protection = 2
            if pos:
                if _in_market(pos, stop, ci, True):
                    if not protect_flatten:
                        return (
                            VenueState(
                                commands,
                                pos,
                                ei,
                                gen,
                                ep,
                                stop,
                                target,
                                pending,
                                resting,
                                chasing,
                                protection,
                                closing,
                                close_reason,
                                rejected_stop,
                                cnt,
                            ),
                            ERR_PROTECT_REJECTED,
                            i,
                        )
                    rejected_stop = True
                    if np.isnan(target):
                        protection = 0
                    if not closing:
                        close_reason = REASON_PROTECT_FAIL
                        commands.append((i + latency, 6, d))
                    closing = True
                px = ci - pos * slip
                if _in_market(pos, target, px, False):
                    _venue_record(
                        count_only,
                        base + cnt,
                        ei,
                        i,
                        pos,
                        ep,
                        px,
                        REASON_TARGET,
                        stop,
                        out_entry_i,
                        out_exit_i,
                        out_side,
                        out_entry_px,
                        out_exit_px,
                        out_reason,
                        out_stop,
                    )
                    cnt += 1
                    pos, exited = 0, True
                    commands.append((i + latency, 5, d))
                    protection = 3
        elif kind == 5 and gen == d:
            protection = 0
        elif kind == 6 and pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                ci - pos * slip,
                close_reason,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
            pos, exited = 0, True
    return (
        VenueState(
            commands,
            pos,
            ei,
            gen,
            ep,
            stop,
            target,
            pending,
            resting,
            chasing,
            protection,
            closing,
            close_reason,
            rejected_stop,
            cnt,
        ),
        ERR_NONE,
        -1,
    )


@nb.njit(cache=True, nogil=True)
def _column(
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
    chase,
    adaptive,
    reenter,
    stale_drop,
    protect_flatten,
    next_open,
    exit_next_open,
    latency_bars,
    one_tick,
    tick,
    adverse_first,
    bracket_anchor,
    target_gap,
    tick_round,
    round_tick,
    long_exits,
    short_exits,
    force_close,
    force_open,
    cross_session,
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
    combine=0,
):
    """Simulate one column.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """

    if latency_bars or one_tick:
        venue_slip = tick if one_tick else 0.0
        venue = _venue_driver(core)
        return venue(
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
            chase,
            adaptive,
            reenter,
            protect_flatten,
            latency_bars,
            venue_slip,
            adverse_first,
            bracket_anchor,
            target_gap,
            tick_round,
            round_tick,
            long_exits,
            short_exits,
            force_close,
            force_open,
            cross_session,
            count_only,
            out_entry_i,
            out_exit_i,
            out_side,
            out_entry_px,
            out_exit_px,
            out_reason,
            out_stop,
            base,
            core,
            ioc_limits,
            combine=combine,
        )
    n = o.shape[0]
    # Flat position, no working limit or next-open market commands.
    state = ColumnState(
        0, -1, 0.0, np.nan, np.nan, 0, 0, 0.0, -1,
        np.nan, np.nan, np.nan, np.nan, -1, False,
    )  # fmt: skip
    inputs = (
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
        chase,
        adaptive,
        reenter,
        stale_drop,
        protect_flatten,
        next_open,
        exit_next_open,
        latency_bars,
        one_tick,
        tick,
        adverse_first,
        bracket_anchor,
        target_gap,
        tick_round,
        round_tick,
        long_exits,
        short_exits,
        force_close,
        force_open,
        cross_session,
        count_only,
        out_entry_i,
        out_exit_i,
        out_side,
        out_entry_px,
        out_exit_px,
        out_reason,
        out_stop,
        base,
        ioc_limits,
        combine,
    )
    for i in range(n):
        state, err, eb = _column_advance(state, i, inputs, core)
        if err:
            return state.cnt, err, eb, state.pos, state.entry_i, state.entry_px, state.stop
    return state.cnt, ERR_NONE, -1, state.pos, state.entry_i, state.entry_px, state.stop


ColumnState = namedtuple(
    "ColumnState",
    "pos entry_i entry_px stop target cnt pend p_lim p_chase p_st p_tg p_stp p_tgp market_entry market_exit",
)


@nb.njit(cache=True, nogil=True, inline="always")
def _column_advance(state, i, inputs, core=None):
    (
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
        chase,
        adaptive,
        reenter,
        stale_drop,
        protect_flatten,
        next_open,
        exit_next_open,
        latency_bars,
        one_tick,
        tick,
        adverse_first,
        bracket_anchor,
        target_gap,
        tick_round,
        round_tick,
        long_exits,
        short_exits,
        force_close,
        force_open,
        cross_session,
        count_only,
        out_entry_i,
        out_exit_i,
        out_side,
        out_entry_px,
        out_exit_px,
        out_reason,
        out_stop,
        base,
        ioc_limits,
        combine,
    ) = inputs
    """Advance one bar; the batch and compiled-core drivers share this matcher."""
    (
        pos,
        entry_i,
        entry_px,
        stop,
        target,
        cnt,
        pend,
        p_lim,
        p_chase,
        p_st,
        p_tg,
        p_stp,
        p_tgp,
        market_entry,
        market_exit,
    ) = state
    n = len(o)
    exited = False
    busy = False  # a working entry resolved on this bar: the sleeve could not act (A15)
    stale_dir = 0  # side of a position closed by a limit fill this bar whose quote is stale
    stale_px = 0.0
    ci = c[i]
    long_exit_i = _optional_at(long_exits, i, False)
    short_exit_i = _optional_at(short_exits, i, False)
    close_mask = _optional_at(force_close, i, 0)
    open_flatten = _optional_at(force_open, i, False)
    exit_now = _exit_for(exit_sig[i], long_exit_i, short_exit_i, pos if pos else pend)
    if pend != 0 and new_day[i]:
        # V2 drops the chase on a new day and leaves a DAY limit resting: not modelled
        return (
            ColumnState(
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                cnt,
                pend,
                p_lim,
                p_chase,
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                market_entry,
                market_exit,
            ),
            ERR_PENDING_NEW_DAY,
            i,
        )
    if open_flatten:
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                entry_i,
                i,
                pos,
                entry_px,
                o[i],
                REASON_SIGNAL,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
        pos, pend, market_entry, market_exit, exited = 0, 0, -1, False, True
    if market_exit:
        if not count_only:
            k = base + cnt
            out_entry_i[k] = entry_i
            out_exit_i[k] = i
            out_side[k] = pos
            out_entry_px[k] = entry_px
            out_exit_px[k] = o[i]
            out_reason[k] = REASON_SIGNAL
            out_stop[k] = stop
        cnt += 1
        pos = 0
        exited = True
        market_exit = False
    if market_entry >= 0:
        cap = _optional_at(ioc_limits, market_entry, np.nan)
        if not np.isnan(cap) and side_sig[market_entry] * (o[i] - cap) > 0:
            market_entry = -1
    if market_entry >= 0:
        d = market_entry
        s = side_sig[d]
        st, tg = levels(
            s,
            _anchor(bracket_anchor, c[d], o[i], o[i]),
            entry_stop[d],
            entry_target[d],
            stop_pts[d],
            target_pts[d],
            round_tick,
            tick_round,
            combine=combine,
            fill=o[i],
        )
        # Newly placed protection is validated at the actual entry open. A4's resting-stop
        # gap rule cannot be applied retroactively across this entry fill (R1).
        if _in_market(s, st, o[i], True):
            return (
                ColumnState(
                    pos,
                    entry_i,
                    entry_px,
                    stop,
                    target,
                    cnt,
                    pend,
                    p_lim,
                    p_chase,
                    p_st,
                    p_tg,
                    p_stp,
                    p_tgp,
                    market_entry,
                    market_exit,
                ),
                ERR_STOP_IN_MARKET,
                i,
            )
        if _in_market(s, tg, o[i], False):
            return (
                ColumnState(
                    pos,
                    entry_i,
                    entry_px,
                    stop,
                    target,
                    cnt,
                    pend,
                    p_lim,
                    p_chase,
                    p_st,
                    p_tg,
                    p_stp,
                    p_tgp,
                    market_entry,
                    market_exit,
                ),
                ERR_TARGET_IN_MARKET,
                i,
            )
        pos = s
        entry_i = i
        entry_px = o[i]
        stop = st
        target = tg
        market_entry = -1
    core_called = False
    if pend != 0:
        s = pend
        # A resting limit fills at its own level once any path point reaches it (A5); the
        # protection meets this bar's close, not the rest of the bar (A15).
        filled = (s > 0 and lo[i] <= p_lim) or (s < 0 and h[i] >= p_lim)
        busy = filled or i == p_chase  # the sleeve sees a working entry on this bar
        # A chase is a close command, so its cancellation decision must precede that fill.
        if core is not None and not filled and i == p_chase:
            _core_decision(
                core,
                i,
                o[i],
                h[i],
                lo[i],
                c[i],
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                pend,
                p_lim,
                stop,
                cnt,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            core_called = True
            long_exit_i, short_exit_i = core.long_exits[i], core.short_exits[i]
            close_mask, open_flatten = core.force_close[i], core.force_open[i]
            exit_now = _exit_for(exit_sig[i], long_exit_i, short_exit_i, pos if pos else pend)
        if filled or (i == p_chase and not exit_now):
            # chase: cancel, market at this close, levels from that fill (an exit on the
            # same bar flattens instead, and V2's flatten clears the chase)
            pend = 0
            px = p_lim if filled else ci
            st, tg = levels(
                s,
                _anchor(bracket_anchor, c[p_chase - chase], px, px),
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                round_tick,
                tick_round,
                combine=combine,
                fill=px,
            )
            if core is not None and filled:
                # The resting entry has printed, but its callback protection is a close
                # command. Observe the position before accepting/rejecting those legs.
                _core_decision(
                    core,
                    i,
                    o[i],
                    h[i],
                    lo[i],
                    ci,
                    s,
                    i,
                    px,
                    np.nan,
                    np.nan,
                    0,
                    np.nan,
                    st,
                    cnt,
                    out_entry_i,
                    out_exit_i,
                    out_side,
                    out_entry_px,
                    out_exit_px,
                    out_reason,
                    out_stop,
                )
                core_called = True
                long_exit_i, short_exit_i = core.long_exits[i], core.short_exits[i]
                close_mask, open_flatten = core.force_close[i], core.force_open[i]
            err, why = _fill_then_protect(
                s, i, px, st, tg, ci, protect_flatten, count_only, out_entry_i, out_exit_i,
                out_side, out_entry_px, out_exit_px, out_reason, out_stop, base + cnt,
            )  # fmt: skip
            if err:
                return (
                    ColumnState(
                        pos,
                        entry_i,
                        entry_px,
                        stop,
                        target,
                        cnt,
                        pend,
                        p_lim,
                        p_chase,
                        p_st,
                        p_tg,
                        p_stp,
                        p_tgp,
                        market_entry,
                        market_exit,
                    ),
                    err,
                    i,
                )
            # A target that fills at this close beats an exit signal on it: V2's flatten is a
            # reduce-only close_all_positions, which Nautilus then rejects (A15).
            if why:
                cnt += 1
                exited = True
            else:
                pos = s
                entry_i = i
                entry_px = px
                stop = st
                target = tg
    elif pos != 0:
        adverse_hit = (
            adverse_first and not np.isnan(stop) and (lo[i] <= stop if pos > 0 else h[i] >= stop)
        )
        reason = 0
        px = 0.0
        fill_point = 0  # path point (0=O, 1=first extreme, 2=second) that produced the fill
        high_first = True
        if adaptive:
            high_first = abs(o[i] - h[i]) < abs(o[i] - lo[i])
        # --- sweep of bar i: resting exits set up at or before bar i-1's close ---
        # Open point: a stop already through fills at the open; a limit already through
        # fills at its own price (Nautilus quirk A5).
        if pos > 0:
            if not np.isnan(stop) and o[i] <= stop:
                reason, px = REASON_STOP_GAP, o[i]
            elif not adverse_hit and not np.isnan(target) and o[i] >= target:
                reason, px = REASON_TARGET, (o[i] if target_gap else target)
        else:
            if not np.isnan(stop) and o[i] >= stop:
                reason, px = REASON_STOP_GAP, o[i]
            elif not adverse_hit and not np.isnan(target) and o[i] <= target:
                reason, px = REASON_TARGET, (o[i] if target_gap else target)
        if reason == 0:
            # Extremes: high then low, or the one nearer the open first when adaptive
            # (strictly nearer high -> high first; ties -> low first).
            for step in range(2):
                use_high = (step == 0) == high_first
                fill_point = step + 1
                if use_high:
                    if pos > 0:
                        if not adverse_hit and not np.isnan(target) and h[i] >= target:
                            reason, px = REASON_TARGET, target
                    else:
                        if not np.isnan(stop) and h[i] >= stop:
                            reason, px = REASON_STOP, stop
                else:
                    if pos > 0:
                        if not np.isnan(stop) and lo[i] <= stop:
                            reason, px = REASON_STOP, stop
                    else:
                        if not adverse_hit and not np.isnan(target) and lo[i] <= target:
                            reason, px = REASON_TARGET, target
                if reason != 0:
                    break
        if reason != 0:
            if not count_only:
                k = base + cnt
                out_entry_i[k] = entry_i
                out_exit_i[k] = i
                out_side[k] = pos
                out_entry_px[k] = entry_px
                out_exit_px[k] = px
                out_reason[k] = reason
                out_stop[k] = stop
            cnt += 1
            if reason == REASON_TARGET and fill_point > 0:  # an open-point (gap) fill pins nothing
                e1 = h[i] if high_first else lo[i]
                e2 = lo[i] if high_first else h[i]
                if _no_reversal(o[i], e1, e2, ci, fill_point, pos):
                    stale_dir = pos
                    stale_px = px
            pos = 0
            exited = True
    if core is not None and not core_called:
        _core_decision(
            core,
            i,
            o[i],
            h[i],
            lo[i],
            c[i],
            pos,
            entry_i,
            entry_px,
            stop,
            target,
            pend,
            p_lim if pend else np.nan,
            stop,
            cnt,
            out_entry_i,
            out_exit_i,
            out_side,
            out_entry_px,
            out_exit_px,
            out_reason,
            out_stop,
        )
        long_exit_i, short_exit_i = core.long_exits[i], core.short_exits[i]
        close_mask, open_flatten = core.force_close[i], core.force_open[i]
    if close_mask & 8:
        if pos or pend:
            return (
                ColumnState(
                    pos,
                    entry_i,
                    entry_px,
                    stop,
                    target,
                    cnt,
                    pend,
                    p_lim,
                    p_chase,
                    p_st,
                    p_tg,
                    p_stp,
                    p_tgp,
                    market_entry,
                    market_exit,
                ),
                ERR_NEXT_OPEN_SESSION_END,
                i,
            )
        return (
            ColumnState(
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                cnt,
                pend,
                p_lim,
                p_chase,
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                market_entry,
                market_exit,
            ),
            ERR_NONE,
            -1,
        )
    if _force_for(close_mask, pos if pos else pend):
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                entry_i,
                i,
                pos,
                entry_px,
                ci,
                REASON_SIGNAL,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
        pos, pend, market_entry, market_exit = 0, 0, -1, False
        return (
            ColumnState(
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                cnt,
                pend,
                p_lim,
                p_chase,
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                market_entry,
                market_exit,
            ),
            ERR_NONE,
            -1,
        )
    if open_flatten:
        return (
            ColumnState(
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                cnt,
                pend,
                p_lim,
                p_chase,
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                market_entry,
                market_exit,
            ),
            ERR_NONE,
            -1,
        )
    exit_now = _exit_for(exit_sig[i], long_exit_i, short_exit_i, pos if pos else pend)
    market_exit = exit_now and pos != 0 and exit_next_open
    if market_exit and (i + 1 == n or (new_day[i + 1] and not cross_session)):
        return (
            ColumnState(
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                cnt,
                pend,
                p_lim,
                p_chase,
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                market_entry,
                market_exit,
            ),
            ERR_NEXT_OPEN_SESSION_END,
            i,
        )
    if exit_now and not exit_next_open and pos != 0:
        _venue_record(
            count_only,
            base + cnt,
            entry_i,
            i,
            pos,
            entry_px,
            ci,
            REASON_SIGNAL,
            stop,
            out_entry_i,
            out_exit_i,
            out_side,
            out_entry_px,
            out_exit_px,
            out_reason,
            out_stop,
        )
        cnt += 1
        pos, exited, pend = 0, True, 0
    if next_open or not np.isnan(_optional_at(ioc_limits, i, np.nan)):
        # R1 decisions remain at the close; stage only actionable orders. At the following
        # open a market exit executes before a replacement entry and before the price sweep.
        enter = (
            (pos == 0 or (market_exit and reenter))
            and (next_open or (pend == 0 and not busy))
            and side_sig[i] != 0
            and (reenter or not exited)
        )
        if market_exit or enter:
            if i + 1 == n or (new_day[i + 1] and not cross_session):
                return (
                    ColumnState(
                        pos,
                        entry_i,
                        entry_px,
                        stop,
                        target,
                        cnt,
                        pend,
                        p_lim,
                        p_chase,
                        p_st,
                        p_tg,
                        p_stp,
                        p_tgp,
                        market_entry,
                        market_exit,
                    ),
                    ERR_NEXT_OPEN_SESSION_END,
                    i,
                )
            if enter:
                market_entry = i
        return (
            ColumnState(
                pos,
                entry_i,
                entry_px,
                stop,
                target,
                cnt,
                pend,
                p_lim,
                p_chase,
                p_st,
                p_tg,
                p_stp,
                p_tgp,
                market_entry,
                market_exit,
            ),
            ERR_NONE,
            -1,
        )
    # --- decisions at bar i's close (on_bar): market orders fill at c[i] ---
    if exit_now and not exit_next_open:
        if pos != 0:
            if not count_only:
                k = base + cnt
                out_entry_i[k] = entry_i
                out_exit_i[k] = i
                out_side[k] = pos
                out_entry_px[k] = entry_px
                out_exit_px[k] = ci
                out_reason[k] = REASON_SIGNAL
                out_stop[k] = stop
            cnt += 1
            pos = 0
            exited = True
        pend = 0  # an exit also cancels a working entry
    if (
        pos == 0
        and (next_open or (pend == 0 and not busy))
        and side_sig[i] != 0
        and (reenter or not exited)
    ):
        s = side_sig[i]
        lim = entry_limit[i]
        if not np.isnan(lim):
            # A same-bar target fill's stale quote does not touch a limit entry (A13: pinned by
            # ltm_bracket, 100+ such entries, all equal to Nautilus).
            if (s > 0 and lim >= ci) or (s < 0 and lim <= ci):
                # marketable: fills at the close (A11); V2 anchors point levels to the limit
                st, tg = levels(
                    s,
                    _anchor(bracket_anchor, ci, ci, lim),
                    entry_stop[i],
                    entry_target[i],
                    stop_pts[i],
                    target_pts[i],
                    round_tick,
                    tick_round,
                    combine=combine,
                    fill=ci,
                )
                err, why = _fill_then_protect(
                    s, i, ci, st, tg, ci, protect_flatten, count_only, out_entry_i,
                    out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop,
                    base + cnt,
                )  # fmt: skip
                if err:
                    return (
                        ColumnState(
                            pos,
                            entry_i,
                            entry_px,
                            stop,
                            target,
                            cnt,
                            pend,
                            p_lim,
                            p_chase,
                            p_st,
                            p_tg,
                            p_stp,
                            p_tgp,
                            market_entry,
                            market_exit,
                        ),
                        err,
                        i,
                    )
                if why:
                    cnt += 1
                    exited = True
                else:
                    pos = s
                    entry_i = i
                    entry_px = ci
                    stop = st
                    target = tg
            else:
                pend = s
                p_lim = lim
                p_chase = i + chase
                p_st = entry_stop[i]
                p_tg = entry_target[i]
                p_stp = stop_pts[i]
                p_tgp = target_pts[i]
            return (
                ColumnState(
                    pos,
                    entry_i,
                    entry_px,
                    stop,
                    target,
                    cnt,
                    pend,
                    p_lim,
                    p_chase,
                    p_st,
                    p_tg,
                    p_stp,
                    p_tgp,
                    market_entry,
                    market_exit,
                ),
                ERR_NONE,
                -1,
            )
        st, tg = levels(
            s,
            ci,
            entry_stop[i],
            entry_target[i],
            stop_pts[i],
            target_pts[i],
            round_tick,
            tick_round,
            combine=combine,
            fill=ci,
        )
        if _in_market(s, st, ci, True):
            return (
                ColumnState(
                    pos,
                    entry_i,
                    entry_px,
                    stop,
                    target,
                    cnt,
                    pend,
                    p_lim,
                    p_chase,
                    p_st,
                    p_tg,
                    p_stp,
                    p_tgp,
                    market_entry,
                    market_exit,
                ),
                ERR_STOP_IN_MARKET,
                i,
            )
        if _in_market(s, tg, ci, False):
            return (
                ColumnState(
                    pos,
                    entry_i,
                    entry_px,
                    stop,
                    target,
                    cnt,
                    pend,
                    p_lim,
                    p_chase,
                    p_st,
                    p_tg,
                    p_stp,
                    p_tgp,
                    market_entry,
                    market_exit,
                ),
                ERR_TARGET_IN_MARKET,
                i,
            )
        if (
            stale_dir == s
            and not np.isnan(st)
            and ((s > 0 and st >= stale_px) or (s < 0 and st <= stale_px))
        ):
            if not stale_drop:
                return (
                    ColumnState(
                        pos,
                        entry_i,
                        entry_px,
                        stop,
                        target,
                        cnt,
                        pend,
                        p_lim,
                        p_chase,
                        p_st,
                        p_tg,
                        p_stp,
                        p_tgp,
                        market_entry,
                        market_exit,
                    ),
                    ERR_STALE_BOOK,
                    i,
                )
            st = np.nan  # Nautilus rejects the stop and cancels the linked target (A13)
            tg = np.nan
        pos = s
        entry_i = i
        entry_px = ci
        stop = st
        target = tg
    return (
        ColumnState(
            pos,
            entry_i,
            entry_px,
            stop,
            target,
            cnt,
            pend,
            p_lim,
            p_chase,
            p_st,
            p_tg,
            p_stp,
            p_tgp,
            market_entry,
            market_exit,
        ),
        ERR_NONE,
        -1,
    )


@nb.njit(parallel=True, cache=True)
def run_columns(
    o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, entry_limit, stop_pts, target_pts,
    new_day, chase, adaptive, reenter, stale_drop, protect_flatten, next_open=False,
    latency_bars=0, one_tick=False, tick=0.0,
    entry_trigger: np.ndarray | None = None, stop_updates: np.ndarray | None = None,
    target_updates: np.ndarray | None = None, be_after: np.ndarray | None = None,
    be_offset: np.ndarray | None = None,
    mit=False, oto=False, amend_keep=False, ratchet=False, stop_skip=False,
    exit_next_open=None, adverse_first=False, bracket_anchor=0,
    target_gap=False, tick_round=0, exit_long=None, exit_short=None, force_close=None, force_open=None, cross_session=False, entry_limit_ioc=None, combine=0,
):  # fmt: skip
    """All columns in parallel. Signal arrays have shape ``(k, n)``; prices and ``new_day`` (the
    bar starts a New York date) shape ``(n,)``.

    Two passes over the same ``_column`` function (count, then fill), so the trade table is exact
    without over-allocating ``k * n`` rows.
    """

    # Keep absent masks as None so Numba removes their checks from each bar.
    exit_next_open = next_open if exit_next_open is None else exit_next_open
    if entry_trigger is not None:
        assert stop_updates is not None and target_updates is not None
        assert be_after is not None and be_offset is not None
    if tick_round:
        entry_limit = entry_limit.copy()
        for j in range(side_sig.shape[0]):
            for i in range(side_sig.shape[1]):
                entry_limit[j, i] = _round_price(
                    entry_limit[j, i], tick, tick_round, -side_sig[j, i]
                )
        if entry_trigger is not None:
            entry_trigger = entry_trigger.copy()
            for j in range(side_sig.shape[0]):
                for i in range(side_sig.shape[1]):
                    entry_trigger[j, i] = _round_price(
                        entry_trigger[j, i], tick, tick_round, side_sig[j, i]
                    )
    slot_slip = tick if one_tick else 0.0
    k = side_sig.shape[0]
    counts = np.zeros(k, np.int64)
    errs = np.zeros(k, np.int64)
    err_bar = np.full(k, -1, np.int64)
    final_side = np.zeros(k, np.int64)
    final_entry = np.full(k, -1, np.int64)
    final_px = np.full(k, np.nan)
    final_stop = np.full(k, np.nan)
    e0 = np.empty(0, np.int64)
    s0 = np.empty(0, np.int8)
    f0 = np.empty(0, np.float64)
    for j in nb.prange(k):  # ty: ignore[not-iterable]  (numba ships prange as a class)
        ioc_column = None if entry_limit_ioc is None else entry_limit_ioc[j]
        long_column = None if exit_long is None else exit_long[j]
        short_column = None if exit_short is None else exit_short[j]
        close_column = None if force_close is None else force_close[j]
        open_column = None if force_open is None else force_open[j]
        if (
            entry_trigger is not None
            and stop_updates is not None
            and target_updates is not None
            and be_after is not None
            and be_offset is not None
        ):
            cnt, err, eb, fs, fe, fpx, fst = _slots_column(
                o,
                h,
                lo,
                c,
                side_sig[j],
                exit_sig[j],
                entry_stop[j],
                entry_target[j],
                entry_limit[j],
                stop_pts[j],
                target_pts[j],
                new_day,
                chase,
                adaptive,
                reenter,
                stale_drop,
                protect_flatten,
                next_open,
                exit_next_open,
                latency_bars,
                slot_slip,
                entry_trigger[j],
                stop_updates[j],
                target_updates[j],
                be_after[j],
                be_offset[j],
                mit,
                oto,
                amend_keep,
                ratchet,
                stop_skip,
                adverse_first,
                bracket_anchor,
                target_gap,
                tick_round,
                tick,
                long_column,
                short_column,
                close_column,
                open_column,
                cross_session,
                True,
                e0,
                e0,
                s0,
                f0,
                f0,
                s0,
                f0,
                0,
                ioc_limits=ioc_column,
                combine=combine,
            )
        else:
            cnt, err, eb, fs, fe, fpx, fst = _column(
            o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
            entry_limit[j], stop_pts[j], target_pts[j], new_day, chase,
            adaptive, reenter, stale_drop, protect_flatten, next_open, exit_next_open, latency_bars, one_tick, tick, adverse_first, bracket_anchor, target_gap, tick_round, tick, long_column, short_column, close_column, open_column, cross_session, True,
            e0, e0, s0, f0, f0, s0, f0, 0, ioc_limits=ioc_column,
            combine=combine,
        )  # fmt: skip
        counts[j] = cnt
        errs[j] = err
        err_bar[j] = eb
        final_side[j] = fs
        final_entry[j] = fe
        final_px[j] = fpx
        final_stop[j] = fst
    offsets = np.zeros(k + 1, np.int64)
    for j in range(k):
        offsets[j + 1] = offsets[j] + counts[j]
    total = offsets[k]
    entry_i = np.empty(total, np.int64)
    exit_i = np.empty(total, np.int64)
    side = np.empty(total, np.int8)
    entry_px = np.empty(total, np.float64)
    exit_px = np.empty(total, np.float64)
    reason = np.empty(total, np.int8)
    stop = np.empty(total, np.float64)
    for j in nb.prange(k):  # ty: ignore[not-iterable]
        ioc_column = None if entry_limit_ioc is None else entry_limit_ioc[j]
        long_column = None if exit_long is None else exit_long[j]
        short_column = None if exit_short is None else exit_short[j]
        close_column = None if force_close is None else force_close[j]
        open_column = None if force_open is None else force_open[j]
        if errs[j] == 0:
            if (
                entry_trigger is not None
                and stop_updates is not None
                and target_updates is not None
                and be_after is not None
                and be_offset is not None
            ):
                _slots_column(
                    o,
                    h,
                    lo,
                    c,
                    side_sig[j],
                    exit_sig[j],
                    entry_stop[j],
                    entry_target[j],
                    entry_limit[j],
                    stop_pts[j],
                    target_pts[j],
                    new_day,
                    chase,
                    adaptive,
                    reenter,
                    stale_drop,
                    protect_flatten,
                    next_open,
                    exit_next_open,
                    latency_bars,
                    slot_slip,
                    entry_trigger[j],
                    stop_updates[j],
                    target_updates[j],
                    be_after[j],
                    be_offset[j],
                    mit,
                    oto,
                    amend_keep,
                    ratchet,
                    stop_skip,
                    adverse_first,
                    bracket_anchor,
                    target_gap,
                    tick_round,
                    tick,
                    long_column,
                    short_column,
                    close_column,
                    open_column,
                    cross_session,
                    False,
                    entry_i,
                    exit_i,
                    side,
                    entry_px,
                    exit_px,
                    reason,
                    stop,
                    offsets[j],
                    ioc_limits=ioc_column,
                    combine=combine,
                )
            else:
                _column(
                o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
                entry_limit[j], stop_pts[j], target_pts[j], new_day, chase,
                adaptive, reenter, stale_drop, protect_flatten, next_open, exit_next_open, latency_bars, one_tick, tick, adverse_first, bracket_anchor, target_gap, tick_round, tick, long_column, short_column, close_column, open_column, cross_session, False,
                entry_i, exit_i, side, entry_px,
                exit_px, reason,
                stop, offsets[j], ioc_limits=ioc_column,
                combine=combine,
            )  # fmt: skip
    return (
        counts, offsets, errs, err_bar, final_side, final_entry, final_px, final_stop,
        entry_i, exit_i, side, entry_px, exit_px, reason, stop,
    )  # fmt: skip


@nb.njit(cache=True, nogil=True)
def _slots_column(
    o,
    h,
    lo,
    c,
    sides,
    exits,
    stops,
    targets,
    limits,
    sp,
    tp,
    days,
    chase,
    adaptive,
    reenter,
    stale_drop,
    protect_flatten,
    next_open,
    exit_next_open,
    latency,
    slip,
    triggers,
    su,
    tu,
    ba,
    bo,
    mit,
    oto,
    amend_keep,
    ratchet,
    stop_skip,
    adverse_first,
    bracket_anchor,
    target_gap,
    tick_round,
    round_tick,
    long_exits,
    short_exits,
    force_close,
    force_open,
    cross_session,
    count_only,
    out_ei,
    out_xi,
    out_sd,
    out_ep,
    out_xp,
    out_why,
    out_st,
    base,
    core=None,
    ioc_limits=None,
    combine=0,
):
    """A18-A20: queued commands retain their entry generation and requested amendment price."""
    queue = [(0, 0, 0, 0.0)]
    queue.pop()
    state = SlotsState(
        queue, 0, -1, -1, -1, np.nan, np.nan, np.nan, np.nan,
        False, False, False, False, False, 0, ERR_NONE,
    )  # fmt: skip
    inputs = (
        o,
        h,
        lo,
        c,
        sides,
        exits,
        stops,
        targets,
        limits,
        sp,
        tp,
        days,
        chase,
        adaptive,
        reenter,
        stale_drop,
        protect_flatten,
        next_open,
        exit_next_open,
        latency,
        slip,
        triggers,
        su,
        tu,
        ba,
        bo,
        mit,
        oto,
        amend_keep,
        ratchet,
        stop_skip,
        adverse_first,
        bracket_anchor,
        target_gap,
        tick_round,
        round_tick,
        long_exits,
        short_exits,
        force_close,
        force_open,
        cross_session,
        count_only,
        out_ei,
        out_xi,
        out_sd,
        out_ep,
        out_xp,
        out_why,
        out_st,
        base,
        ioc_limits,
        combine,
    )
    for i in range(len(c)):
        state, err, eb = _slots_column_advance(state, i, inputs, core)
        if err:
            return state.cnt, err, eb, state.pos, state.ei, state.ep, state.risk_stop
    return state.cnt, ERR_NONE, -1, state.pos, state.ei, state.ep, state.risk_stop


SlotsState = namedtuple(
    "SlotsState",
    "queue pos ei gen pending ep stop target risk_stop live resting chasing closing be_done cnt error",
)


@nb.njit(cache=True, nogil=True, inline="always")
def _slots_column_advance(state, i, inputs, core=None):
    (
        o,
        h,
        lo,
        c,
        sides,
        exits,
        stops,
        targets,
        limits,
        sp,
        tp,
        days,
        chase,
        adaptive,
        reenter,
        stale_drop,
        protect_flatten,
        next_open,
        exit_next_open,
        latency,
        slip,
        triggers,
        su,
        tu,
        ba,
        bo,
        mit,
        oto,
        amend_keep,
        ratchet,
        stop_skip,
        adverse_first,
        bracket_anchor,
        target_gap,
        tick_round,
        round_tick,
        long_exits,
        short_exits,
        force_close,
        force_open,
        cross_session,
        count_only,
        out_ei,
        out_xi,
        out_sd,
        out_ep,
        out_xp,
        out_why,
        out_st,
        base,
        ioc_limits,
        combine,
    ) = inputs
    """Advance one bar; the batch and compiled-core drivers share this matcher."""
    (
        queue,
        pos,
        ei,
        gen,
        pending,
        ep,
        stop,
        target,
        risk_stop,
        live,
        resting,
        chasing,
        closing,
        be_done,
        cnt,
        error,
    ) = state
    ci = c[i]
    long_exit_i = _optional_at(long_exits, i, False)
    short_exit_i = _optional_at(short_exits, i, False)
    close_mask = _optional_at(force_close, i, 0)
    open_flatten = _optional_at(force_open, i, False)
    exit_now = _exit_for(
        exits[i],
        long_exit_i,
        short_exit_i,
        pos if pos else (sides[pending] if pending >= 0 else 0),
    )
    exited, busy = False, pending >= 0
    pinned, pin, prev = 0, np.nan, o[i]
    if (
        pending >= 0
        and days[i]
        and (not np.isnan(triggers[pending]) or not np.isnan(limits[pending]))
    ):
        return (
            SlotsState(
                queue,
                pos,
                ei,
                gen,
                pending,
                ep,
                stop,
                target,
                risk_stop,
                live,
                resting,
                chasing,
                closing,
                be_done,
                cnt,
                error,
            ),
            ERR_PENDING_NEW_DAY,
            i,
        )
    if open_flatten:
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                o[i] - pos * slip,
                REASON_SIGNAL,
                risk_stop,
                out_ei,
                out_xi,
                out_sd,
                out_ep,
                out_xp,
                out_why,
                out_st,
            )
            cnt += 1
        queue.clear()
        pos, pending, live, resting, chasing, closing = 0, -1, False, False, False, False
    for phase in range(2):
        if phase == 0 and not (next_open or exit_next_open or ioc_limits is not None):
            continue
        ci = o[i] if phase == 0 else c[i]
        if phase == 1:
            adverse_hit = (
                adverse_first
                and pos != 0
                and live
                and not np.isnan(stop)
                and (lo[i] <= stop if pos > 0 else h[i] >= stop)
            )
            high_first = not adaptive or abs(o[i] - h[i]) < abs(o[i] - lo[i])
            for m in range(4):
                pt = (
                    o[i]
                    if m == 0
                    else (ci if m == 3 else (h[i] if (m == 1) == high_first else lo[i]))
                )
                if (pinned > 0 and pt < prev) or (pinned < 0 and pt > prev):
                    pinned = 0
                prev = pt
                entered = False
                if pending >= 0 and resting:
                    d, s = pending, sides[pending]
                    stop_order = not np.isnan(triggers[d])
                    level = triggers[d] if stop_order else limits[d]
                    match = _in_market(-s, level, pt if stop_order else pt + s * slip, stop_order)
                    if match:
                        px = pt + s * slip if slip else (pt if stop_order and m == 0 else level)
                        anchor = _anchor(
                            bracket_anchor,
                            c[d],
                            px,
                            limits[d] if not np.isnan(limits[d]) else px,
                        )
                        stop, target = levels(
                            s,
                            anchor,
                            stops[d],
                            targets[d],
                            sp[d],
                            tp[d],
                            round_tick,
                            tick_round,
                            combine=combine,
                            fill=px,
                        )
                        pos, ei, ep, gen, risk_stop = s, i, px, d, stop
                        be_done = False
                        pending, resting, chasing = -1, False, False
                        live, entered = False, True
                        if oto:
                            if mit and _in_market(pos, target, pt, False):
                                error = ERR_MIT_IN_MARKET
                            elif _in_market(pos, stop, pt, True):
                                error = ERR_PROTECT_REJECTED
                            else:
                                live = True
                                if _in_market(pos, target, pt, False):
                                    _venue_record(
                                        count_only,
                                        base + cnt,
                                        ei,
                                        i,
                                        pos,
                                        ep,
                                        pt - pos * slip,
                                        REASON_TARGET,
                                        risk_stop,
                                        out_ei,
                                        out_xi,
                                        out_sd,
                                        out_ep,
                                        out_xp,
                                        out_why,
                                        out_st,
                                    )
                                    cnt += 1
                                    pos, exited = 0, True
                        else:
                            queue.append(
                                (
                                    i + latency + (exit_next_open if closing else 0),
                                    _SLOT_CLOSE if closing else _SLOT_PROTECT,
                                    d,
                                    0.0,
                                )
                            )
                if pos and live and not entered:
                    reason, px = 0, 0.0
                    if _in_market(pos, stop, pt, True):
                        reason = REASON_STOP_GAP if m == 0 else REASON_STOP
                        px = pt - pos * slip if slip else (pt if m == 0 else stop)
                    elif not adverse_hit and _in_market(
                        pos, target, pt if mit else pt - pos * slip, False
                    ):
                        reason = REASON_TARGET
                        px = (
                            pt - pos * slip
                            if slip
                            else (pt if (mit or target_gap) and m == 0 else target)
                        )
                        if not mit and not slip and (m == 1 or m == 2):
                            pinned, pin = pos, px
                    if reason:
                        _venue_record(
                            count_only,
                            base + cnt,
                            ei,
                            i,
                            pos,
                            ep,
                            px,
                            reason,
                            risk_stop,
                            out_ei,
                            out_xi,
                            out_sd,
                            out_ep,
                            out_xp,
                            out_why,
                            out_st,
                        )
                        cnt += 1
                        pos, exited, live = 0, True, False
            if error:
                return (
                    SlotsState(
                        queue,
                        pos,
                        ei,
                        gen,
                        pending,
                        ep,
                        stop,
                        target,
                        risk_stop,
                        live,
                        resting,
                        chasing,
                        closing,
                        be_done,
                        cnt,
                        error,
                    ),
                    error,
                    i,
                )
            if core is not None:
                _core_decision(
                    core,
                    i,
                    o[i],
                    h[i],
                    lo[i],
                    c[i],
                    pos,
                    ei,
                    ep,
                    (stop if live else np.nan),
                    (target if live else np.nan),
                    sides[pending] if pending >= 0 else 0,
                    (triggers[pending] if not np.isnan(triggers[pending]) else limits[pending])
                    if pending >= 0
                    else np.nan,
                    risk_stop,
                    cnt,
                    out_ei,
                    out_xi,
                    out_sd,
                    out_ep,
                    out_xp,
                    out_why,
                    out_st,
                )
                long_exit_i, short_exit_i = core.long_exits[i], core.short_exits[i]
                close_mask, open_flatten = core.force_close[i], core.force_open[i]
            if close_mask & 8:
                if pos or pending >= 0:
                    return (
                        SlotsState(
                            queue,
                            pos,
                            ei,
                            gen,
                            pending,
                            ep,
                            stop,
                            target,
                            risk_stop,
                            live,
                            resting,
                            chasing,
                            closing,
                            be_done,
                            cnt,
                            error,
                        ),
                        ERR_NEXT_OPEN_SESSION_END,
                        i,
                    )
                continue
            if (
                _force_for(close_mask, pos if pos else (sides[pending] if pending >= 0 else 0))
                or open_flatten
            ):
                if pos:
                    _venue_record(
                        count_only,
                        base + cnt,
                        ei,
                        i,
                        pos,
                        ep,
                        ci - pos * slip,
                        REASON_SIGNAL,
                        risk_stop,
                        out_ei,
                        out_xi,
                        out_sd,
                        out_ep,
                        out_xp,
                        out_why,
                        out_st,
                    )
                    cnt += 1
                queue.clear()
                pos, pending, live, resting, chasing, closing = (
                    0,
                    -1,
                    False,
                    False,
                    False,
                    False,
                )
                continue
            exit_now = _exit_for(
                exits[i],
                long_exit_i,
                short_exit_i,
                pos if pos else (sides[pending] if pending >= 0 else 0),
            )
            if pos == 0 and pending < 0:
                closing = False
            if (
                pending >= 0
                and not np.isnan(limits[pending])
                and not closing
                and i == pending + chase
            ):
                chasing = True
                queue.append(
                    (
                        i + latency + (exit_next_open if exit_now else next_open),
                        _SLOT_CANCEL,
                        pending,
                        0.0,
                    )
                )
            if exit_now:
                chasing = False
                if pending >= 0:
                    queue.append(
                        (
                            i + latency + (exit_next_open if exit_now else next_open),
                            _SLOT_CANCEL,
                            pending,
                            0.0,
                        )
                    )
                    closing = not oto
                if pos and not closing:
                    queue.append((i + latency + exit_next_open, _SLOT_CLOSE, gen, 0.0))
                    closing = True
            elif pos:
                update = su[i]
                if not be_done and not np.isnan(ba[gen]):
                    reached = h[i] >= ep + ba[gen] if pos > 0 else lo[i] <= ep - ba[gen]
                    if reached:
                        if not np.isnan(update):
                            error = ERR_AMEND_REJECTED
                        update = ep + pos * (0.0 if np.isnan(bo[gen]) else bo[gen])
                        be_done = True
                if not np.isnan(update):
                    update = _round_price(update, round_tick, tick_round, -pos)
                    bound = stop
                    for _, kind, d, value in queue:
                        if kind == _SLOT_AMEND_STOP and d == gen and pos * (value - bound) > 0:
                            bound = value
                    if np.isnan(stop):
                        error = ERR_AMEND_NO_ORDER
                    elif not ratchet or pos * (update - bound) > 0:
                        queue.append((i + latency + next_open, _SLOT_AMEND_STOP, gen, update))
                if not np.isnan(tu[i]):
                    if np.isnan(target):
                        error = ERR_AMEND_NO_ORDER
                    else:
                        queue.append(
                            (
                                i + latency + next_open,
                                _SLOT_AMEND_TARGET,
                                gen,
                                _round_price(tu[i], round_tick, tick_round, pos),
                            )
                        )
            if (
                pos == 0
                and pending >= 0
                and not np.isnan(triggers[pending])
                and sides[i]
                and not exit_now
            ):
                if latency:
                    return (
                        SlotsState(
                            queue,
                            pos,
                            ei,
                            gen,
                            pending,
                            ep,
                            stop,
                            target,
                            risk_stop,
                            live,
                            resting,
                            chasing,
                            closing,
                            be_done,
                            cnt,
                            error,
                        ),
                        ERR_STOP_REPLACE_LATENCY,
                        i,
                    )
                queue.append(
                    (
                        i + latency + (exit_next_open if exit_now else next_open),
                        _SLOT_CANCEL,
                        pending,
                        0.0,
                    )
                )
                queue.append((i + latency + next_open, _SLOT_REPLACE, i, 0.0))
            elif (
                pos == 0
                and pending < 0
                and not busy
                and not closing
                and sides[i]
                and (reenter or not exited)
            ):
                pending = i
                if not np.isnan(_optional_at(ioc_limits, i, np.nan)):
                    due = i + latency + 1
                    if due >= len(c) or (not cross_session and np.any(days[i + 1 : due + 1])):
                        error = ERR_NEXT_OPEN_SESSION_END
                queue.append(
                    (
                        i
                        + latency
                        + (1 if not np.isnan(_optional_at(ioc_limits, i, np.nan)) else next_open),
                        _SLOT_SUBMIT,
                        i,
                        0.0,
                    )
                )
            if next_open and exit_now and pos and reenter and sides[i] and pending < 0:
                pending = i
                queue.append((i + 1, _SLOT_SUBMIT, i, 0.0))
        if not error and not latency and (next_open or exit_next_open):
            for due, _, _, _ in queue:
                if due > i and (i + 1 == len(c) or (days[i + 1] and not cross_session)):
                    error = ERR_NEXT_OPEN_SESSION_END
        while len(queue) and error == 0:
            ready = -1
            for q in range(len(queue)):
                due, cmd_kind, cmd_d, _ = queue[q]
                ioc_command = cmd_kind == _SLOT_SUBMIT and not np.isnan(
                    _optional_at(ioc_limits, cmd_d, np.nan)
                )
                if due <= i and (
                    phase == 1
                    or next_open
                    or exit_next_open
                    or ioc_command
                    or (cmd_kind == _SLOT_PROTECT and gen == cmd_d and ei == i)
                ):
                    ready = q
                    break
            if ready < 0:
                break
            _, kind, d, value = queue.pop(ready)
            if kind == _SLOT_REPLACE:
                if pos or pending >= 0:
                    continue
                pending, kind = d, _SLOT_SUBMIT
            if kind == _SLOT_SUBMIT and pending == d:
                s = sides[d]
                cap = _optional_at(ioc_limits, d, np.nan)
                if not np.isnan(cap) and s * (ci - cap) > 0:
                    pending, resting, chasing, closing = -1, False, False, False
                    continue
                q = pin if pinned == -s else ci
                if not np.isnan(triggers[d]):
                    if not (phase == 0 and next_open) and _in_market(-s, triggers[d], q, True):
                        if stop_skip:
                            pending, resting = -1, False
                        else:
                            error = ERR_STOP_ENTRY_IN_MARKET
                    else:
                        resting = True
                    continue
                if not np.isnan(limits[d]) and s * (ci + s * slip - limits[d]) > 0:
                    resting = True
                    continue
                px = ci + s * slip
                anchor = _anchor(
                    bracket_anchor, c[d], px, limits[d] if not np.isnan(limits[d]) else px
                )
            elif kind == _SLOT_CHASE and pending == d:
                s = sides[d]
                px = ci + s * slip
                anchor = _anchor(bracket_anchor, c[d], px, px)
            else:
                s, px, anchor = 0, 0.0, 0.0
            if s:
                pos, ei, ep, gen = s, i, px, d
                stop, target = levels(
                    s,
                    anchor,
                    stops[d],
                    targets[d],
                    sp[d],
                    tp[d],
                    round_tick,
                    tick_round,
                    combine=combine,
                    fill=px,
                )
                risk_stop, be_done = stop, False
                pending, resting, chasing, live = -1, False, False, False
                if oto:
                    # Native child processing is synchronous within the entry fill.
                    q = pin if pinned == pos else ci
                    if mit and _in_market(pos, target, q, False):
                        error = ERR_MIT_IN_MARKET
                    elif _in_market(pos, stop, q, True):
                        if pinned == pos and not _in_market(pos, stop, ci, True):
                            if not stale_drop:
                                error = ERR_STALE_BOOK
                            else:
                                stop, target = np.nan, np.nan
                        else:
                            error = ERR_PROTECT_REJECTED
                    else:
                        live = True
                        if _in_market(pos, target, q, False):
                            _venue_record(
                                count_only,
                                base + cnt,
                                ei,
                                i,
                                pos,
                                ep,
                                q - pos * slip,
                                REASON_TARGET,
                                risk_stop,
                                out_ei,
                                out_xi,
                                out_sd,
                                out_ep,
                                out_xp,
                                out_why,
                                out_st,
                            )
                            cnt += 1
                            pos, exited, live = 0, True, False
                else:
                    queue.append((i + latency, _SLOT_CLOSE if closing else _SLOT_PROTECT, d, 0.0))
            elif kind == _SLOT_CANCEL and pending == d:
                pending, resting = -1, False
                if chasing:
                    pending = d
                    queue.append((i + latency, _SLOT_CHASE, d, 0.0))
            elif pos and gen == d:
                q = pin if pinned == pos else ci
                reason = 0
                if kind == _SLOT_PROTECT:
                    if mit and _in_market(pos, target, q, False):
                        error = ERR_MIT_IN_MARKET
                    elif _in_market(pos, stop, q, True):
                        if not protect_flatten:
                            error = ERR_PROTECT_REJECTED
                        else:
                            queue.append((i + latency, _SLOT_PROTECT_FAIL, gen, 0.0))
                    else:
                        live = True
                        if _in_market(pos, target, q, False):
                            reason = REASON_TARGET
                elif kind == _SLOT_CLOSE or kind == _SLOT_PROTECT_FAIL:
                    reason = REASON_SIGNAL if kind == _SLOT_CLOSE else REASON_PROTECT_FAIL
                elif kind == _SLOT_AMEND_STOP or kind == _SLOT_AMEND_TARGET:
                    is_stop = kind == _SLOT_AMEND_STOP
                    invalid = _in_market(pos, value, q, is_stop)
                    if invalid and (is_stop or mit):
                        if not amend_keep:
                            error = ERR_AMEND_REJECTED
                    elif is_stop:
                        stop = value
                    else:
                        target = value
                        if invalid:
                            reason = REASON_TARGET
                if reason:
                    _venue_record(
                        count_only,
                        base + cnt,
                        ei,
                        i,
                        pos,
                        ep,
                        q - pos * slip,
                        reason,
                        risk_stop,
                        out_ei,
                        out_xi,
                        out_sd,
                        out_ep,
                        out_xp,
                        out_why,
                        out_st,
                    )
                    cnt += 1
                    if next_open and pending >= 0 and pending != gen:
                        closing = False
                    pos, exited, live = 0, True, False
        if not error and not latency and (next_open or exit_next_open):
            for due, _, _, _ in queue:
                if due > i and (i + 1 == len(c) or (days[i + 1] and not cross_session)):
                    error = ERR_NEXT_OPEN_SESSION_END
    if error:
        return (
            SlotsState(
                queue,
                pos,
                ei,
                gen,
                pending,
                ep,
                stop,
                target,
                risk_stop,
                live,
                resting,
                chasing,
                closing,
                be_done,
                cnt,
                error,
            ),
            error,
            i,
        )
    return (
        SlotsState(
            queue,
            pos,
            ei,
            gen,
            pending,
            ep,
            stop,
            target,
            risk_stop,
            live,
            resting,
            chasing,
            closing,
            be_done,
            cnt,
            error,
        ),
        ERR_NONE,
        -1,
    )


# Stable numeric ABI shared with compiled strategy cores and live hosts.
V_SIDE, V_QTY, V_ENTRY_I, V_ENTRY_PX, V_STOP, V_TARGET, V_WORK_SIDE, V_WORK_LEVEL = range(8)
V_LAST_EXIT_I, V_LAST_EXIT_PX, V_LAST_EXIT_REASON, V_EQUITY = range(8, 12)

CoreContext = namedtuple(
    "CoreContext",
    "step state params feats volume position_view decision sides exits levels long_exits short_exits "
    "force_close force_open days ends economic account flatten_mode boundary tick rounding decisions exit_next_open",
)


@nb.njit(cache=False, inline="always")
def _core_decision(
    core,
    i,
    o,
    h,
    lo,
    c,
    pos,
    ei,
    ep,
    stop,
    target,
    work_side,
    work_level,
    risk_stop,
    cnt,
    out_ei,
    out_xi,
    out_sd,
    out_ep,
    out_xp,
    out_why,
    out_st,
):
    """Observe settled fills, call a core, and publish its close decision (no matching here)."""
    seen = int(core.account[0])
    pv, friction, bps, mode, amount, cap, lot, minimum = core.economic
    for k in range(seen, cnt):
        risk = abs(out_ep[k] - out_st[k]) * pv
        qty = _contracts_for(mode, amount, cap, core.account[1], risk, lot, minimum)
        if qty < 0:
            raise ValueError("compiled core: entry sizing refused")
        net = settle(
            out_sd[k] * (out_xp[k] - out_ep[k]),
            pv,
            friction,
            fill_cost(out_ep[k], out_xp[k], pv, bps),
        )
        core.account[1] += qty * net
        core.position_view[V_LAST_EXIT_I] = out_xi[k]
        core.position_view[V_LAST_EXIT_PX] = out_xp[k]
        core.position_view[V_LAST_EXIT_REASON] = out_why[k]
    core.account[0] = cnt
    view = core.position_view
    view[V_SIDE], view[V_QTY] = pos, 0
    view[V_ENTRY_I], view[V_ENTRY_PX] = (ei if pos else -1), (ep if pos else np.nan)
    view[V_STOP], view[V_TARGET] = (stop if pos else np.nan), (target if pos else np.nan)
    if pos:
        qty = _contracts_for(
            mode, amount, cap, core.account[1], abs(ep - risk_stop) * pv, lot, minimum
        )
        if qty < 0:
            raise ValueError("compiled core: open sizing refused")
        view[V_QTY] = qty
    view[V_WORK_SIDE], view[V_WORK_LEVEL] = work_side, work_level
    view[V_EQUITY] = core.account[1]
    d = core.decision
    d[:] = np.nan
    d[0], d[1], d[12], d[13], d[14] = 0, 0, 0, 0, 0
    core.step(core.state, core.params, i, core.feats, o, h, lo, c, core.volume[i], view, d)
    core.decisions[:, i] = d
    if d[0] != -1 and d[0] != 0 and d[0] != 1:
        raise ValueError("compiled core: side must be -1, 0, 1")
    for col in (1, 12, 13, 14):
        if d[col] != 0 and d[col] != 1:
            raise ValueError("compiled core: boolean decision must be 0 or 1")
    core.sides[i], core.exits[i] = int(d[0]), bool(d[1])
    for col in range(10):
        core.levels[col, i] = d[col + 2]
    core.levels[10, i] = d[15]
    core.levels[2, i] = _round_price(d[4], core.tick, core.rounding, -d[0])
    core.levels[5, i] = _round_price(d[7], core.tick, core.rounding, d[0])
    core.long_exits[i], core.short_exits[i] = bool(d[13]), bool(d[14])
    # Match simulate's boundary transformation on the current decision, without future signals.
    if core.boundary == 1 and core.ends[i] and core.exit_next_open:
        core.force_close[i] |= int(core.exits[i]) | (2 * int(d[13])) | (4 * int(d[14]))
        core.exits[i], core.long_exits[i], core.short_exits[i] = False, False, False
    if core.flatten_mode == 2:
        if core.ends[i]:
            core.force_close[i] |= 1
    elif d[12]:
        if core.flatten_mode == 1:
            core.force_close[i] |= 1
        elif i + 1 == len(core.sides) or (core.days[i + 1] and core.boundary != 2):
            core.force_close[i] |= 1 if core.boundary == 1 else 8
        else:
            core.force_open[i + 1] = True


@nb.njit(cache=True)
def _optional_at(a, i, default):
    if a is None:
        return default
    return a[i]


# Keep external callback/jitclass signatures out of numeric batch cache indices.
# Both dispatchers compile the same source; the matcher is never copied.
_compiled_venue_column = nb.njit(cache=False, nogil=True)(
    getattr(_venue_column, "py_func", _venue_column)
)
_compiled_column = nb.njit(cache=False, nogil=True)(getattr(_column, "py_func", _column))
_compiled_slots_column = nb.njit(cache=False, nogil=True)(
    getattr(_slots_column, "py_func", _slots_column)
)


def _venue_driver(core):
    return _venue_column if core is None else _compiled_venue_column


@overload(_venue_driver)
def _venue_driver_type(core):
    # Select a dispatcher at typing time: a first-class-function union loses
    # Numba's omitted-default argument specialization for the numeric driver.
    if isinstance(core, nb.types.NoneType):

        def numeric(core):
            return _venue_column

        return numeric

    def generic(core):
        return _compiled_venue_column

    return generic
