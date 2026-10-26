"""Single-order matcher (no slots) and its dispatcher."""

from __future__ import annotations

from collections import namedtuple

import numba as nb
import numpy as np

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
from .decision import _core_decision
from .fills import (
    _day_entry_expires,
    _exit_for,
    _force_for,
    _high_first,
    _no_reversal,
    _target_fill_price,
)
from .levels import _optional_at
from .venue import _venue_driver, _venue_record


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
    """Simulate one column.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """
    next_open = bool(rules[R_NEXT_OPEN])
    latency_bars = int(rules[R_LATENCY_BARS])
    one_tick = bool(rules[R_ONE_TICK])

    # R15: ONE_TICK with next-open entries stays on this scheduler, adding the tick per fill.
    if latency_bars or (one_tick and not next_open):
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
            core,
            ioc_limits,
            expiry_days,
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
        ioc_limits,
        expiry_days,
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
        ioc_limits,
        expiry_days,
    ) = inputs
    chase = int(rules[R_CHASE])
    adaptive = bool(rules[R_ADAPTIVE])
    reenter = bool(rules[R_REENTER])
    stale_drop = bool(rules[R_STALE_DROP])
    protect_flatten = bool(rules[R_PROTECT_FLATTEN])
    next_open = bool(rules[R_NEXT_OPEN])
    exit_next_open = bool(rules[R_EXIT_NEXT_OPEN])
    one_tick = bool(rules[R_ONE_TICK])
    tick = rules[R_TICK]
    adverse_first = bool(rules[R_ADVERSE_FIRST])
    bracket_anchor = int(rules[R_BRACKET_ANCHOR])
    target_gap = bool(rules[R_TARGET_GAP])
    tick_round = int(rules[R_TICK_ROUND])
    round_tick = rules[R_TICK]
    cross_session = bool(rules[R_CROSS_SESSION])
    combine = int(rules[R_COMBINE])
    day_cancel = bool(rules[R_DAY_CANCEL])
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
    slip = tick if one_tick else 0.0  # R15: one adverse tick on each market and stop fill
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
    if pend != 0 and _day_entry_expires(day_cancel, _optional_at(expiry_days, i, new_day[i])):
        pend, p_chase = 0, -1
    if pend != 0 and new_day[i] and not day_cancel:
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
        pos, pend, market_entry, market_exit, exited = 0, 0, -1, False, True
    if market_exit:
        if not count_only and base + cnt < len(out_entry_i):
            k = base + cnt
            out_entry_i[k] = entry_i
            out_exit_i[k] = i
            out_side[k] = pos
            out_entry_px[k] = entry_px
            out_exit_px[k] = o[i] - pos * slip
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
        fill = o[i] + s * slip
        st, tg = levels(
            s,
            _anchor(bracket_anchor, c[d], fill, fill),
            entry_stop[d],
            entry_target[d],
            stop_pts[d],
            target_pts[d],
            round_tick,
            tick_round,
            combine=combine,
            fill=fill,
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
        entry_px = fill
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
        high_first = _high_first(adaptive, o[i], h[i], lo[i])
        # --- sweep of bar i: resting exits set up at or before bar i-1's close ---
        # Open point: a stop already through fills at the open; a limit already through
        # fills at its own price (Nautilus quirk A5).
        if pos > 0:
            if not np.isnan(stop) and o[i] <= stop:
                reason, px = REASON_STOP_GAP, o[i] - slip
            elif not adverse_hit and not np.isnan(target) and o[i] >= target:
                reason = REASON_TARGET
                px = _target_fill_price(o[i], target, True, target_gap, pos, 0.0)
        else:
            if not np.isnan(stop) and o[i] >= stop:
                reason, px = REASON_STOP_GAP, o[i] + slip
            elif not adverse_hit and not np.isnan(target) and o[i] <= target:
                reason = REASON_TARGET
                px = _target_fill_price(o[i], target, True, target_gap, pos, 0.0)
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
                            reason, px = REASON_STOP, stop + slip
                else:
                    if pos > 0:
                        if not np.isnan(stop) and lo[i] <= stop:
                            reason, px = REASON_STOP, stop - slip
                    else:
                        if not adverse_hit and not np.isnan(target) and lo[i] <= target:
                            reason, px = REASON_TARGET, target
                if reason != 0:
                    break
        if reason != 0:
            if not count_only and base + cnt < len(out_entry_i):
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
            if not count_only and base + cnt < len(out_entry_i):
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


_compiled_column = nb.njit(cache=False, nogil=True)(getattr(_column, "py_func", _column))


__all__ = [
    "ColumnState",
    "_column",
    "_column_advance",
    "_compiled_column",
]
