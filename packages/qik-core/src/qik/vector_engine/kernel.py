"""Numba kernel: closed-form per-bar execution rules, one column (config) at a time.

The rules are written down in ``docs/vector-engine.md`` (A1-A20 and repro-only R1). ``reference.py`` is an
independent, slow implementation of the same spec that matches orders price point by price point;
``tests`` check the two agree exactly and that both reproduce Nautilus golden data.

All inputs are float64 / int8 / bool arrays with one row per column (config) and one column per bar.
"""

from __future__ import annotations

import numba as nb
import numpy as np

REASON_SIGNAL = 1  # market exit at the decision close (R1: the next open)
REASON_STOP = 2  # resting stop-market hit during a bar, filled at the stop level
REASON_TARGET = 3  # resting limit hit during a bar, filled at the limit level
REASON_STOP_GAP = 4  # resting stop already through at the bar's open, filled at the open
REASON_PROTECT_FAIL = 5  # limit-then-market entry's stop rejected; V2 flattens at that close

ERR_NONE = 0
ERR_STOP_IN_MARKET = 1  # stop level not strictly beyond the entry close (Nautilus rejects it)
ERR_TARGET_IN_MARKET = 2  # target level not strictly beyond the entry close (would fill at once)
ERR_STALE_BOOK = 3  # new stop falls inside the stale quote a same-bar limit fill left (A13)
ERR_PROTECT_REJECTED = 4  # limit-then-market entry's stop at or through its fill bar's close (A15)
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
def _level(side, anchor, level, pts, sign, tick=0.0, rounding=0):
    """A level as given, or ``pts`` points from ``anchor`` (``sign`` +1 target side, -1 stop side)."""

    value = level if np.isnan(pts) else anchor + sign * side * pts
    return _round_price(value, tick, rounding, sign * side)


@nb.njit(cache=True, nogil=True)
def _round_price(value, tick, rounding, direction):
    if rounding == 0 or np.isnan(value):
        return value
    steps = value / tick
    if rounding == 1:
        return np.floor(steps + 0.5) * tick
    return (np.ceil(steps - 1e-10) if direction > 0 else np.floor(steps + 1e-10)) * tick


@nb.njit(cache=True, nogil=True)
def _force_for(mask, side):
    return bool(mask & 1) or (side > 0 and bool(mask & 2)) or (side < 0 and bool(mask & 4))


@nb.njit(cache=True, nogil=True)
def _exit_for(global_exit, long_exit, short_exit, side):
    return global_exit or (side > 0 and long_exit) or (side < 0 and short_exit)


@nb.njit(cache=True, nogil=True)
def _anchor(mode, decision, fill, legacy):
    if mode == 1:
        return decision
    return fill if mode == 2 else legacy


@nb.njit(cache=True, nogil=True)
def _in_market(side, level, close, is_stop):
    """A stop at or through ``close``, or a limit target at or through it (A10 / A11)."""

    if np.isnan(level):
        return False
    if is_stop:
        return level >= close if side > 0 else level <= close
    return level <= close if side > 0 else level >= close


@nb.njit(cache=True, nogil=True)
def _fill_then_protect(
    s, i, fill_px, st, tg, ci, protect_flatten, count_only,
    out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop, k,
):  # fmt: skip
    """V2 pattern (A15): an entry filled at ``fill_px`` whose protection meets the close ``ci``.

    A stop at or through the close is rejected (refused, or V2's flatten at the close); a target at
    or through it fills at the close. Returns ``(err, reason)``: reason 0 = position opened, else
    the reason of the same-bar trade written at row ``k``.
    """

    reason = 0
    if _in_market(s, st, ci, True):
        if not protect_flatten:
            return ERR_PROTECT_REJECTED, 0
        reason = REASON_PROTECT_FAIL
    elif _in_market(s, tg, ci, False):
        reason = REASON_TARGET
    if reason != 0 and not count_only:
        out_entry_i[k] = i
        out_exit_i[k] = i
        out_side[k] = s
        out_entry_px[k] = fill_px
        out_exit_px[k] = ci
        out_reason[k] = reason
        out_stop[k] = st
    return ERR_NONE, reason


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
):
    """A16/A17 venue settlement, with a bounded-by-flight-time FIFO command list.

    Commands: 1 market, 2 limit, 3 protection, 4 cancel entry, 5 cancel protection, 6 close.
    A command's generation is its entry decision row; cancels cannot erase a later trade's legs.
    """
    commands = [(0, 0, 0)]
    commands.pop()
    pos, ei, gen = 0, -1, -1
    ep, stop, target = np.nan, np.nan, np.nan
    pending = -1
    resting = False
    chasing = False
    protection = 0
    closing = False
    close_reason = REASON_SIGNAL
    rejected_stop = False
    cnt = 0
    for i in range(len(c)):
        ci = c[i]
        exit_now = _exit_for(
            exit_sig[i],
            long_exits[i],
            short_exits[i],
            pos if pos else (side_sig[pending] if pending >= 0 else 0),
        )
        exited = False
        busy = pending >= 0
        if pending >= 0 and not np.isnan(entry_limit[pending]) and new_day[i]:
            return cnt, ERR_PENDING_NEW_DAY, i, pos, ei, ep, stop
        if force_open[i]:
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
                stop = _level(
                    s,
                    _anchor(bracket_anchor, c[d], px, entry_limit[d]),
                    entry_stop[d],
                    stop_pts[d],
                    -1.0,
                    round_tick,
                    tick_round,
                )
                target = _level(
                    s,
                    _anchor(bracket_anchor, c[d], px, entry_limit[d]),
                    entry_target[d],
                    target_pts[d],
                    1.0,
                    round_tick,
                    tick_round,
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
        if force_close[i] & 8:
            if pos or pending >= 0:
                return cnt, ERR_NEXT_OPEN_SESSION_END, i, pos, ei, ep, stop
            continue
        if (
            _force_for(force_close[i], pos if pos else (side_sig[pending] if pending >= 0 else 0))
            or force_open[i]
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
            continue
        if closing and pending < 0 and pos == 0 and (protection == 0 or protection == 3):
            closing = False
        if closing and pos == 0 and protection == 2:
            commands.append((i + latency, 5, gen))
            protection = 3
        exit_now = _exit_for(
            exit_sig[i],
            long_exits[i],
            short_exits[i],
            pos if pos else (side_sig[pending] if pending >= 0 else 0),
        )
        if (
            pending >= 0
            and not closing
            and not np.isnan(entry_limit[pending])
            and i == pending + chase
        ):
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
            commands.append((i + latency, 1 if np.isnan(entry_limit[i]) else 2, i))
        while len(commands) and commands[0][0] <= i:
            _, kind, d = commands.pop(0)
            if kind == 1 or kind == 2:
                if pending != d:
                    continue
                s = side_sig[d]
                px = ci + s * slip
                if (
                    kind == 1
                    or (s > 0 and px <= entry_limit[d])
                    or (s < 0 and px >= entry_limit[d])
                ):
                    anchor = _anchor(bracket_anchor, c[d], px, px if kind == 1 else entry_limit[d])
                    pos, ei, ep, gen = s, i, px, d
                    stop = _level(
                        s, anchor, entry_stop[d], stop_pts[d], -1.0, round_tick, tick_round
                    )
                    target = _level(
                        s, anchor, entry_target[d], target_pts[d], 1.0, round_tick, tick_round
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
                            return cnt, ERR_PROTECT_REJECTED, i, pos, ei, ep, stop
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
    return cnt, ERR_NONE, -1, pos, ei, ep, stop


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
):
    """Simulate one column.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """

    if latency_bars or one_tick:
        venue_slip = tick if one_tick else 0.0
        return _venue_column(
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
        )
    n = o.shape[0]
    pos = 0
    entry_i = -1
    entry_px = 0.0
    stop = np.nan
    target = np.nan
    cnt = 0
    # working limit-then-market entry (A15): side, limit, chase bar, and its level spec
    pend = 0
    p_lim = 0.0
    p_chase = -1
    p_st = np.nan
    p_tg = np.nan
    p_stp = np.nan
    p_tgp = np.nan
    market_entry = -1  # R1: decision bar whose market entry will fill at the next open
    market_exit = False
    for i in range(n):
        exited = False
        busy = False  # a working entry resolved on this bar: the sleeve could not act (A15)
        stale_dir = 0  # side of a position closed by a limit fill this bar whose quote is stale
        stale_px = 0.0
        ci = c[i]
        exit_now = _exit_for(exit_sig[i], long_exits[i], short_exits[i], pos if pos else pend)
        if pend != 0 and new_day[i]:
            # V2 drops the chase on a new day and leaves a DAY limit resting: not modelled
            return cnt, ERR_PENDING_NEW_DAY, i, pos, entry_i, entry_px, stop
        if force_open[i]:
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
            d = market_entry
            s = side_sig[d]
            st = _level(
                s,
                _anchor(bracket_anchor, c[d], o[i], o[i]),
                entry_stop[d],
                stop_pts[d],
                -1.0,
                round_tick,
                tick_round,
            )
            tg = _level(
                s,
                _anchor(bracket_anchor, c[d], o[i], o[i]),
                entry_target[d],
                target_pts[d],
                1.0,
                round_tick,
                tick_round,
            )
            # Newly placed protection is validated at the actual entry open. A4's resting-stop
            # gap rule cannot be applied retroactively across this entry fill (R1).
            if _in_market(s, st, o[i], True):
                return cnt, ERR_STOP_IN_MARKET, i, pos, entry_i, entry_px, stop
            if _in_market(s, tg, o[i], False):
                return cnt, ERR_TARGET_IN_MARKET, i, pos, entry_i, entry_px, stop
            pos = s
            entry_i = i
            entry_px = o[i]
            stop = st
            target = tg
            market_entry = -1
        if pend != 0:
            s = pend
            # A resting limit fills at its own level once any path point reaches it (A5); the
            # protection meets this bar's close, not the rest of the bar (A15).
            filled = (s > 0 and lo[i] <= p_lim) or (s < 0 and h[i] >= p_lim)
            busy = filled or i == p_chase  # the sleeve sees a working entry on this bar
            if filled or (i == p_chase and not exit_now):
                # chase: cancel, market at this close, levels from that fill (an exit on the
                # same bar flattens instead, and V2's flatten clears the chase)
                pend = 0
                px = p_lim if filled else ci
                st = _level(
                    s,
                    _anchor(bracket_anchor, c[p_chase - chase], px, px),
                    p_st,
                    p_stp,
                    -1.0,
                    round_tick,
                    tick_round,
                )
                tg = _level(
                    s,
                    _anchor(bracket_anchor, c[p_chase - chase], px, px),
                    p_tg,
                    p_tgp,
                    1.0,
                    round_tick,
                    tick_round,
                )
                err, why = _fill_then_protect(
                    s, i, px, st, tg, ci, protect_flatten, count_only, out_entry_i, out_exit_i,
                    out_side, out_entry_px, out_exit_px, out_reason, out_stop, base + cnt,
                )  # fmt: skip
                if err:
                    return cnt, err, i, pos, entry_i, entry_px, stop
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
                adverse_first
                and not np.isnan(stop)
                and (lo[i] <= stop if pos > 0 else h[i] >= stop)
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
                if (
                    reason == REASON_TARGET and fill_point > 0
                ):  # an open-point (gap) fill pins nothing
                    e1 = h[i] if high_first else lo[i]
                    e2 = lo[i] if high_first else h[i]
                    if _no_reversal(o[i], e1, e2, ci, fill_point, pos):
                        stale_dir = pos
                        stale_px = px
                pos = 0
                exited = True
        if force_close[i] & 8:
            if pos or pend:
                return cnt, ERR_NEXT_OPEN_SESSION_END, i, pos, entry_i, entry_px, stop
            continue
        if _force_for(force_close[i], pos if pos else pend):
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
            continue
        if force_open[i]:
            continue
        exit_now = _exit_for(exit_sig[i], long_exits[i], short_exits[i], pos if pos else pend)
        market_exit = exit_now and pos != 0 and exit_next_open
        if market_exit and (i + 1 == n or (new_day[i + 1] and not cross_session)):
            return cnt, ERR_NEXT_OPEN_SESSION_END, i, pos, entry_i, entry_px, stop
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
        if next_open:
            # R1 decisions remain at the close; stage only actionable orders. At the following
            # open a market exit executes before a replacement entry and before the price sweep.
            enter = (
                (pos == 0 or (market_exit and reenter))
                and side_sig[i] != 0
                and (reenter or not exited)
            )
            if market_exit or enter:
                if i + 1 == n or (new_day[i + 1] and not cross_session):
                    return cnt, ERR_NEXT_OPEN_SESSION_END, i, pos, entry_i, entry_px, stop
                if enter:
                    market_entry = i
            continue
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
        if pos == 0 and pend == 0 and not busy and side_sig[i] != 0 and (reenter or not exited):
            s = side_sig[i]
            lim = entry_limit[i]
            if not np.isnan(lim):
                # A same-bar target fill's stale quote does not touch a limit entry (A13: pinned by
                # ltm_bracket, 100+ such entries, all equal to Nautilus).
                if (s > 0 and lim >= ci) or (s < 0 and lim <= ci):
                    # marketable: fills at the close (A11); V2 anchors point levels to the limit
                    st = _level(
                        s,
                        _anchor(bracket_anchor, ci, ci, lim),
                        entry_stop[i],
                        stop_pts[i],
                        -1.0,
                        round_tick,
                        tick_round,
                    )
                    tg = _level(
                        s,
                        _anchor(bracket_anchor, ci, ci, lim),
                        entry_target[i],
                        target_pts[i],
                        1.0,
                        round_tick,
                        tick_round,
                    )
                    err, why = _fill_then_protect(
                        s, i, ci, st, tg, ci, protect_flatten, count_only, out_entry_i,
                        out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop,
                        base + cnt,
                    )  # fmt: skip
                    if err:
                        return cnt, err, i, pos, entry_i, entry_px, stop
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
                continue
            st = _level(s, ci, entry_stop[i], stop_pts[i], -1.0, round_tick, tick_round)
            tg = _level(s, ci, entry_target[i], target_pts[i], 1.0, round_tick, tick_round)
            if _in_market(s, st, ci, True):
                return cnt, ERR_STOP_IN_MARKET, i, pos, entry_i, entry_px, stop
            if _in_market(s, tg, ci, False):
                return cnt, ERR_TARGET_IN_MARKET, i, pos, entry_i, entry_px, stop
            if (
                stale_dir == s
                and not np.isnan(st)
                and ((s > 0 and st >= stale_px) or (s < 0 and st <= stale_px))
            ):
                if not stale_drop:
                    return cnt, ERR_STALE_BOOK, i, pos, entry_i, entry_px, stop
                st = np.nan  # Nautilus rejects the stop and cancels the linked target (A13)
                tg = np.nan
            pos = s
            entry_i = i
            entry_px = ci
            stop = st
            target = tg
    return cnt, ERR_NONE, -1, pos, entry_i, entry_px, stop


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
    target_gap=False, tick_round=0, exit_long=None, exit_short=None, force_close=None, force_open=None, cross_session=False,
):  # fmt: skip
    """All columns in parallel. Signal arrays have shape ``(k, n)``; prices and ``new_day`` (the
    bar starts a New York date) shape ``(n,)``.

    Two passes over the same ``_column`` function (count, then fill), so the trade table is exact
    without over-allocating ``k * n`` rows.
    """

    exit_next_open = next_open if exit_next_open is None else exit_next_open
    if entry_trigger is not None:
        assert stop_updates is not None and target_updates is not None
        assert be_after is not None and be_offset is not None
    if exit_long is None:
        exit_long = np.zeros_like(exit_sig)
    if exit_short is None:
        exit_short = np.zeros_like(exit_sig)
    if force_close is None:
        force_close = np.zeros(exit_sig.shape, np.int8)
    if force_open is None:
        force_open = np.zeros_like(exit_sig)
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
                exit_long[j],
                exit_short[j],
                force_close[j],
                force_open[j],
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
            )
        else:
            cnt, err, eb, fs, fe, fpx, fst = _column(
            o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
            entry_limit[j], stop_pts[j], target_pts[j], new_day, chase,
            adaptive, reenter, stale_drop, protect_flatten, next_open, exit_next_open, latency_bars, one_tick, tick, adverse_first, bracket_anchor, target_gap, tick_round, tick, exit_long[j], exit_short[j], force_close[j], force_open[j], cross_session, True,
            e0, e0, s0, f0, f0, s0, f0, 0,
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
                    exit_long[j],
                    exit_short[j],
                    force_close[j],
                    force_open[j],
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
                )
            else:
                _column(
                o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
                entry_limit[j], stop_pts[j], target_pts[j], new_day, chase,
                adaptive, reenter, stale_drop, protect_flatten, next_open, exit_next_open, latency_bars, one_tick, tick, adverse_first, bracket_anchor, target_gap, tick_round, tick, exit_long[j], exit_short[j], force_close[j], force_open[j], cross_session, False,
                entry_i, exit_i, side, entry_px,
                exit_px, reason,
                stop, offsets[j],
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
):
    """A18-A20: queued commands retain their entry generation and requested amendment price."""
    queue = [(0, 0, 0, 0.0)]
    queue.pop()
    pos, ei, gen, pending = 0, -1, -1, -1
    ep, stop, target, risk_stop = np.nan, np.nan, np.nan, np.nan
    live, resting, chasing, closing, be_done = False, False, False, False, False
    cnt, error = 0, ERR_NONE
    for i in range(len(c)):
        ci = c[i]
        exit_now = _exit_for(
            exits[i],
            long_exits[i],
            short_exits[i],
            pos if pos else (sides[pending] if pending >= 0 else 0),
        )
        exited, busy = False, pending >= 0
        pinned, pin, prev = 0, np.nan, o[i]
        if (
            pending >= 0
            and days[i]
            and (not np.isnan(triggers[pending]) or not np.isnan(limits[pending]))
        ):
            return cnt, ERR_PENDING_NEW_DAY, i, pos, ei, ep, risk_stop
        if force_open[i]:
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
            if phase == 0 and not (next_open or exit_next_open):
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
                        match = _in_market(
                            -s, level, pt if stop_order else pt + s * slip, stop_order
                        )
                        if match:
                            px = pt + s * slip if slip else (pt if stop_order and m == 0 else level)
                            anchor = _anchor(
                                bracket_anchor,
                                c[d],
                                px,
                                limits[d] if not np.isnan(limits[d]) else px,
                            )
                            stop = _level(s, anchor, stops[d], sp[d], -1.0, round_tick, tick_round)
                            target = _level(
                                s, anchor, targets[d], tp[d], 1.0, round_tick, tick_round
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
                    return cnt, error, i, pos, ei, ep, risk_stop
                if force_close[i] & 8:
                    if pos or pending >= 0:
                        return cnt, ERR_NEXT_OPEN_SESSION_END, i, pos, ei, ep, risk_stop
                    continue
                if (
                    _force_for(
                        force_close[i], pos if pos else (sides[pending] if pending >= 0 else 0)
                    )
                    or force_open[i]
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
                    long_exits[i],
                    short_exits[i],
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
                        return cnt, ERR_STOP_REPLACE_LATENCY, i, pos, ei, ep, risk_stop
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
                    queue.append((i + latency + next_open, _SLOT_SUBMIT, i, 0.0))
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
                    if queue[q][0] <= i:
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
                    stop = _level(s, anchor, stops[d], sp[d], -1.0, round_tick, tick_round)
                    target = _level(s, anchor, targets[d], tp[d], 1.0, round_tick, tick_round)
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
                        queue.append(
                            (i + latency, _SLOT_CLOSE if closing else _SLOT_PROTECT, d, 0.0)
                        )
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
            return cnt, error, i, pos, ei, ep, risk_stop
    return cnt, ERR_NONE, -1, pos, ei, ep, risk_stop
