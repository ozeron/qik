"""Slots matcher: trigger/stop/target updates, straddle legs and its dispatcher."""

from __future__ import annotations

from collections import namedtuple

import numba as nb
import numpy as np

from qik.vector_engine.protection import (
    ERR_NONE,
    ERR_PROTECT_REJECTED,
    REASON_PROTECT_FAIL,
    REASON_TARGET,
    _anchor,
    _in_market,
    _round_price,
    levels,
)

from .codes import (
    _SLOT_AMEND_STOP,
    _SLOT_AMEND_TARGET,
    _SLOT_ARM_STOP,
    _SLOT_ARM_TARGET,
    _SLOT_CANCEL,
    _SLOT_CHASE,
    _SLOT_CLOSE,
    _SLOT_PAIR_IOC,
    _SLOT_PROTECT,
    _SLOT_PROTECT_FAIL,
    _SLOT_REPLACE,
    _SLOT_SUBMIT,
    ERR_AMEND_NO_ORDER,
    ERR_AMEND_REJECTED,
    ERR_BUSY_ENTRY_DROPPED,
    ERR_MIT_IN_MARKET,
    ERR_NEXT_OPEN_SESSION_END,
    ERR_PENDING_NEW_DAY,
    ERR_REARM_BEFORE_CLOSE,
    ERR_STALE_BOOK,
    ERR_STOP_ENTRY_IN_MARKET,
    ERR_STOP_REPLACE_LATENCY,
    ERR_STRADDLE_SIDE_CANCEL,
    R_ADAPTIVE,
    R_ADVERSE_FIRST,
    R_AMEND_CLOSE,
    R_AMEND_KEEP,
    R_ARM_STOP,
    R_BRACKET_ANCHOR,
    R_CHASE,
    R_COMBINE,
    R_COMPLETED_CLOSE,
    R_CROSS_SESSION,
    R_DAY_CANCEL,
    R_EXIT_AND_REARM,
    R_EXIT_NEXT_OPEN,
    R_LATENCY_BARS,
    R_MIT,
    R_NEXT_BAR,
    R_NEXT_OPEN,
    R_ONE_TICK,
    R_OTO,
    R_PROTECT_FLATTEN,
    R_RATCHET,
    R_REENTER,
    R_STALE_DROP,
    R_STOP_SKIP,
    R_TARGET_GAP,
    R_TICK,
    R_TICK_ROUND,
    REASON_SIGNAL,
    REASON_STOP,
    REASON_STOP_GAP,
)
from .decision import _core_decision
from .fills import (
    _cancel_entry_commands,
    _day_entry_expires,
    _exit_for,
    _force_for,
    _high_first,
    _target_fill_price,
)
from .levels import _optional_at, _store_risk
from .pairs import (
    _has_pairs,
    _lifetime_crosses,
    _open_in_market,
    _pair_at,
    _pair_level,
    _pair_rests,
    _stop_pair,
)
from .venue import _venue_record


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
    rules,
    triggers,
    su,
    tu,
    ba,
    bo,
    long_exits,
    short_exits,
    force_close,
    force_open,
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
    expiry_days=None,
    straddle=None,
    from_open=None,
    stop_fracs=None,
    target_fracs=None,
    sizing_risks=None,
):
    """A18-A20: queued commands retain their entry generation and requested amendment price."""
    arm_stop = bool(rules[R_ARM_STOP])
    day_cancel = bool(rules[R_DAY_CANCEL])
    if straddle is not None and np.isfinite(straddle).any():
        if arm_stop:
            raise ValueError("straddle arm_stop=next_bar composition is unproved")
        if day_cancel:
            raise ValueError("straddle day_order_expiry=cancel composition is unproved")
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
        rules,
        triggers,
        su,
        tu,
        ba,
        bo,
        long_exits,
        short_exits,
        force_close,
        force_open,
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
        expiry_days,
        straddle,
        from_open,
        stop_fracs,
        target_fracs,
        sizing_risks,
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
        rules,
        triggers,
        su,
        tu,
        ba,
        bo,
        long_exits,
        short_exits,
        force_close,
        force_open,
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
        expiry_days,
        straddle,
        from_open,
        stop_fracs,
        target_fracs,
        sizing_risks,
    ) = inputs
    chase = int(rules[R_CHASE])
    adaptive = bool(rules[R_ADAPTIVE])
    reenter = bool(rules[R_REENTER])
    stale_drop = bool(rules[R_STALE_DROP])
    protect_flatten = bool(rules[R_PROTECT_FLATTEN])
    next_open = bool(rules[R_NEXT_OPEN])
    exit_next_open = bool(rules[R_EXIT_NEXT_OPEN])
    latency = int(rules[R_LATENCY_BARS])
    slip = rules[R_TICK] if rules[R_ONE_TICK] else 0.0
    mit = bool(rules[R_MIT])
    oto = bool(rules[R_OTO])
    amend_keep = bool(rules[R_AMEND_KEEP])
    ratchet = bool(rules[R_RATCHET])
    stop_skip = bool(rules[R_STOP_SKIP])
    adverse_first = bool(rules[R_ADVERSE_FIRST])
    bracket_anchor = int(rules[R_BRACKET_ANCHOR])
    target_gap = bool(rules[R_TARGET_GAP])
    tick_round = int(rules[R_TICK_ROUND])
    round_tick = rules[R_TICK]
    cross_session = bool(rules[R_CROSS_SESSION])
    combine = int(rules[R_COMBINE])
    amend_close = bool(rules[R_AMEND_CLOSE])
    completed_close = bool(rules[R_COMPLETED_CLOSE])
    arm_stop = bool(rules[R_ARM_STOP])
    day_cancel = bool(rules[R_DAY_CANCEL])
    next_bar = bool(rules[R_NEXT_BAR])
    exit_and_rearm = bool(rules[R_EXIT_AND_REARM])
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
        and _day_entry_expires(day_cancel, _optional_at(expiry_days, i, days[i]))
        and (not np.isnan(triggers[pending]) or not np.isnan(limits[pending]))
    ):
        _cancel_entry_commands(queue, pending)
        pending, resting, chasing, closing, busy = -1, False, False, False, False
    if (
        pending >= 0
        and days[i]
        and not day_cancel
        and not (next_bar and i == pending + 1)
        and (
            _pair_rests(straddle, pending)
            or not np.isnan(triggers[pending])
            or not np.isnan(limits[pending])
        )
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
        if phase == 0 and not (
            next_open
            or exit_next_open
            or (ioc_limits is not None and len(ioc_limits) > 0)
            or _has_pairs(straddle)
        ):
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
            high_first = _high_first(adaptive, o[i], h[i], lo[i])
            if (
                pending >= 0
                and resting
                and i == pending + 1
                and bool(_optional_at(from_open, pending, False))
                and _open_in_market(triggers, straddle, sides, pending, o[i])
            ):
                # The trigger is submitted at this open: at or through it is in the market.
                if stop_skip and not _pair_at(straddle, pending):
                    pending, resting = -1, False
                else:
                    return (
                        SlotsState(
                            queue, pos, ei, gen, pending, ep, stop, target, risk_stop, live,
                            resting, chasing, closing, be_done, cnt, error,
                        ),
                        ERR_STOP_ENTRY_IN_MARKET,
                        i,
                    )  # fmt: skip
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
                    paired = _pair_at(straddle, d)
                    stop_order = not np.isnan(triggers[d])
                    level = triggers[d] if stop_order else limits[d]
                    if straddle is not None and paired:
                        s = 0
                        for offset, direction in ((0, 1), (6, -1)):
                            trigger = straddle[offset, d]
                            if not np.isnan(trigger) and direction * (pt - trigger) >= 0:
                                s, stop_order, level = direction, True, trigger
                                break
                    match = s != 0 and _in_market(
                        -s, level, pt if stop_order else pt + s * slip, stop_order
                    )
                    if match:
                        if slip and next_open and stop_order:
                            # R15b: the A4 price (trigger on a touch, open on a gap) plus one tick.
                            px = _round_price(
                                (pt if m == 0 else level) + s * slip, round_tick, tick_round, s
                            )
                        else:
                            px = pt + s * slip if slip else (pt if stop_order and m == 0 else level)
                        anchor = _anchor(
                            bracket_anchor,
                            c[d],
                            px,
                            limits[d] if not np.isnan(limits[d]) else px,
                        )
                        leg_stop, leg_target, leg_sp, leg_tp = stops[d], targets[d], sp[d], tp[d]
                        if straddle is not None and paired:
                            offset = 0 if s > 0 else 6
                            leg_stop, leg_target = straddle[offset + 2, d], straddle[offset + 3, d]
                            leg_sp, leg_tp = straddle[offset + 4, d], straddle[offset + 5, d]
                        stop, target = levels(
                            s,
                            anchor,
                            leg_stop,
                            leg_target,
                            leg_sp,
                            leg_tp,
                            round_tick,
                            tick_round,
                            combine=combine,
                            fill=px,
                            stop_frac=_optional_at(stop_fracs, d, np.nan),
                            target_frac=_optional_at(target_fracs, d, np.nan),
                        )
                        if slip and next_open and stop_order:
                            # Quantity is submitted against the trigger geometry, before slippage.
                            submitted_stop, _ = levels(
                                s,
                                _anchor(bracket_anchor, c[d], level, level),
                                leg_stop,
                                leg_target,
                                leg_sp,
                                leg_tp,
                                round_tick,
                                tick_round,
                                combine=combine,
                                fill=level,
                            )
                            submitted_risk = abs(level - submitted_stop)
                            _store_risk(sizing_risks, i, submitted_risk)
                            if core is not None:
                                core.sizing_risks[i] = submitted_risk
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
                        if slip and next_open:
                            px = (pt if m == 0 else stop) - pos * slip
                        else:
                            px = pt - pos * slip if slip else (pt if m == 0 else stop)
                    elif not adverse_hit and _in_market(
                        pos, target, pt if mit else pt - pos * slip, False
                    ):
                        reason = REASON_TARGET
                        px = _target_fill_price(pt, target, m == 0, mit or target_gap, pos, slip)
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
            if (
                next_bar
                and pending >= 0
                and resting
                and i > pending
                and (not np.isnan(triggers[pending]) or _pair_rests(straddle, pending))
            ):
                # GTD next bar: an unfilled stop expires before this close's decision.
                for q in range(len(queue) - 1, -1, -1):
                    if queue[q][2] == pending:
                        queue.pop(q)
                pending, resting, busy = -1, False, False
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
                    _pair_level(straddle, pending, 0, i, latency),
                    _pair_level(straddle, pending, 6, i, latency),
                    o[i + 1] if i + 1 < len(o) else np.nan,
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
                if not (exit_and_rearm and reenter and (sides[i] or _pair_at(straddle, i))):
                    continue
                exited, busy = True, False
            exit_now = _exit_for(
                exits[i],
                long_exit_i,
                short_exit_i,
                pos if pos else (sides[pending] if pending >= 0 else 0),
            )
            if (
                pending >= 0
                and _pair_at(straddle, pending)
                and (long_exit_i or short_exit_i or close_mask & 6)
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
                    ERR_STRADDLE_SIDE_CANCEL,
                    i,
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
                if (
                    pending >= 0
                    and exit_and_rearm
                    and reenter
                    and (sides[i] or _pair_at(straddle, i))
                ):
                    # A23: this close cancels the working entry, then submits the new one.
                    for q in range(len(queue) - 1, -1, -1):
                        if queue[q][2] == pending:
                            queue.pop(q)
                    pending, resting, closing = -1, False, False
                elif pending >= 0:
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
                    has_stop = not np.isnan(stop)
                    for _, kind, d, value in queue:
                        if (kind == _SLOT_AMEND_STOP or kind == _SLOT_ARM_STOP) and d == gen:
                            has_stop = True
                            if np.isnan(bound) or pos * (value - bound) > 0:
                                bound = value
                    if not has_stop and arm_stop:
                        queue.append((i + latency, _SLOT_ARM_STOP, gen, update))
                    elif not has_stop:
                        error = ERR_AMEND_NO_ORDER
                    elif not ratchet or pos * (update - bound) > 0:
                        queue.append(
                            (
                                i + latency + (next_open and not amend_close),
                                _SLOT_AMEND_STOP,
                                gen,
                                update,
                            )
                        )
                if not np.isnan(tu[i]):
                    if np.isnan(target) and arm_stop and np.isnan(stop) and not np.isnan(update):
                        queue.append(
                            (
                                i + latency,
                                _SLOT_ARM_TARGET,
                                gen,
                                _round_price(tu[i], round_tick, tick_round, pos),
                            )
                        )
                    elif np.isnan(target):
                        error = ERR_AMEND_NO_ORDER
                    else:
                        queue.append(
                            (
                                i + latency + (next_open and not amend_close),
                                _SLOT_AMEND_TARGET,
                                gen,
                                _round_price(tu[i], round_tick, tick_round, pos),
                            )
                        )
            if (
                pos == 0
                and pending >= 0
                and (_pair_at(straddle, pending) or not np.isnan(triggers[pending]))
                and (sides[i] or _pair_at(straddle, i))
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
                (pos == 0 or (exit_and_rearm and exit_now and reenter))
                and pending < 0
                and (not busy or (exit_and_rearm and exit_now and reenter))
                and (not closing or (exit_and_rearm and exit_now and reenter))
                and (sides[i] or _pair_at(straddle, i))
                and (reenter or not exited)
            ):
                if (
                    exit_and_rearm
                    and exit_now
                    and pos
                    and exit_next_open
                    and not next_open
                    and not _pair_at(straddle, i)
                    and np.isnan(triggers[i])
                    and np.isnan(_optional_at(ioc_limits, i, np.nan))
                ):
                    error = ERR_REARM_BEFORE_CLOSE
                pending = i
                if _pair_at(straddle, i) or not np.isnan(_optional_at(ioc_limits, i, np.nan)):
                    due = i + latency + 1
                    if due >= len(c) or (
                        not cross_session
                        and not (next_bar and _stop_pair(straddle, i))
                        and np.any(days[i + 1 : due + 1])
                    ):
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
            elif (
                slip
                and next_open
                and pos == 0
                and pending < 0
                and busy
                and not closing
                and (sides[i] or _pair_at(straddle, i))
                and (reenter or not exited)
                and not error
            ):
                # R15b: refuse instead of letting the whole-bar busy guard drop this entry.
                error = ERR_BUSY_ENTRY_DROPPED
            if next_open and exit_now and pos and reenter and sides[i] and pending < 0:
                pending = i
                queue.append((i + 1, _SLOT_SUBMIT, i, 0.0))
        if not error and not latency and (next_open or exit_next_open):
            for due, kind, d, _ in queue:
                if due > i and (
                    i + 1 == len(c)
                    or (
                        days[i + 1]
                        and not cross_session
                        and not _lifetime_crosses(next_bar, kind, d, triggers, straddle)
                    )
                ):
                    error = ERR_NEXT_OPEN_SESSION_END
        while len(queue) and error == 0:
            ready = -1
            for q in range(len(queue)):
                due, cmd_kind, cmd_d, _ = queue[q]
                ioc_command = cmd_kind == _SLOT_PAIR_IOC or (
                    cmd_kind == _SLOT_SUBMIT
                    and not np.isnan(_optional_at(ioc_limits, cmd_d, np.nan))
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
                if straddle is not None and _pair_at(straddle, d):
                    due = d + latency + 1
                    if due >= len(c) or (
                        not cross_session
                        and not (next_bar and _stop_pair(straddle, d))
                        and np.any(days[d + 1 : due + 1])
                    ):
                        error = ERR_NEXT_OPEN_SESSION_END
                        continue
                    if (
                        not (phase == 0 and next_open)
                        and not bool(_optional_at(from_open, d, False))
                        and (
                            (
                                not np.isnan(straddle[0, d])
                                and (pin if pinned == -1 else ci) >= straddle[0, d]
                            )
                            or (
                                not np.isnan(straddle[6, d])
                                and (pin if pinned == 1 else ci) <= straddle[6, d]
                            )
                        )
                    ):
                        error = ERR_STOP_ENTRY_IN_MARKET
                    else:
                        resting = True
                        if not (np.isnan(straddle[1, d]) and np.isnan(straddle[7, d])):
                            queue.append((d + latency + 1, _SLOT_PAIR_IOC, d, 0.0))
                    continue
                s = sides[d]
                cap = _optional_at(ioc_limits, d, np.nan)
                if not np.isnan(cap) and s * (ci - cap) > 0:
                    pending, resting, chasing, closing = -1, False, False, False
                    continue
                q = pin if pinned == -s else ci
                if not np.isnan(triggers[d]):
                    if (
                        not (phase == 0 and next_open)
                        and not bool(_optional_at(from_open, d, False))
                        and _in_market(-s, triggers[d], q, True)
                    ):
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
            elif kind == _SLOT_PAIR_IOC:
                # Stable long-then-short priority at one opening price. A stop leg gapped
                # at this open wins in the path at m == 0; an eligible IOC fills here.
                s, px, anchor = 0, 0.0, 0.0
                if pending == d and resting and pos == 0 and phase == 0:
                    for offset, direction in ((0, 1), (6, -1)):
                        trigger, cap = straddle[offset, d], straddle[offset + 1, d]
                        if not np.isnan(trigger) and direction * (ci - trigger) >= 0:
                            break
                        if not np.isnan(cap) and direction * (ci - cap) <= 0:
                            s = direction
                            break
                    px = ci + s * slip
                    anchor = _anchor(bracket_anchor, c[d], px, px)
                    if s == 0 and np.isnan(straddle[0, d]) and np.isnan(straddle[6, d]):
                        pending, resting = -1, False
            elif kind == _SLOT_CHASE and pending == d:
                s = sides[d]
                px = ci + s * slip
                anchor = _anchor(bracket_anchor, c[d], px, px)
            else:
                s, px, anchor = 0, 0.0, 0.0
            if s:
                # A market entry at this open is settled before the completed-close
                # decision. Resting limit/stop callbacks keep their A15 busy contract.
                if completed_close and phase == 0 and np.isnan(limits[d]) and np.isnan(triggers[d]):
                    busy = False
                pos, ei, ep, gen = s, i, px, d
                leg_stop, leg_target, leg_sp, leg_tp = stops[d], targets[d], sp[d], tp[d]
                if _pair_at(straddle, d):
                    offset = 0 if s > 0 else 6
                    leg_stop, leg_target = straddle[offset + 2, d], straddle[offset + 3, d]
                    leg_sp, leg_tp = straddle[offset + 4, d], straddle[offset + 5, d]
                stop, target = levels(
                    s,
                    anchor,
                    leg_stop,
                    leg_target,
                    leg_sp,
                    leg_tp,
                    round_tick,
                    tick_round,
                    combine=combine,
                    fill=px,
                    stop_frac=_optional_at(stop_fracs, d, np.nan),
                    target_frac=_optional_at(target_fracs, d, np.nan),
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
                elif kind == _SLOT_ARM_STOP:
                    if _in_market(pos, value, q, True):
                        if not protect_flatten:
                            error = ERR_PROTECT_REJECTED
                        else:
                            queue.append((i + latency, _SLOT_PROTECT_FAIL, gen, 0.0))
                    else:
                        stop, live = value, True
                elif kind == _SLOT_ARM_TARGET and not np.isnan(stop):
                    if mit and _in_market(pos, value, q, False):
                        error = ERR_MIT_IN_MARKET
                    else:
                        target = value
                        if _in_market(pos, value, q, False):
                            reason = REASON_TARGET
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
                    if (next_open or exit_and_rearm) and pending >= 0 and pending != gen:
                        closing = False
                    pos, exited, live = 0, True, False
        if not error and not latency and (next_open or exit_next_open):
            for due, kind, d, _ in queue:
                if due > i and (
                    i + 1 == len(c)
                    or (
                        days[i + 1]
                        and not cross_session
                        and not _lifetime_crosses(next_bar, kind, d, triggers, straddle)
                    )
                ):
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


_compiled_slots_column = nb.njit(cache=False, nogil=True)(
    getattr(_slots_column, "py_func", _slots_column)
)


__all__ = [
    "SlotsState",
    "_compiled_slots_column",
    "_slots_column",
    "_slots_column_advance",
]
