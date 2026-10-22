"""Numba kernel: closed-form per-bar execution rules, one column (config) at a time.

The rules are written down in ``docs/vector-engine.md`` (A1-A15 and repro-only R1). ``reference.py`` is an
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
def _level(side, anchor, level, pts, sign):
    """A level as given, or ``pts`` points from ``anchor`` (``sign`` +1 target side, -1 stop side)."""

    if np.isnan(pts):
        return level
    return anchor + sign * side * pts


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
            st = _level(s, o[i], entry_stop[d], stop_pts[d], -1.0)
            tg = _level(s, o[i], entry_target[d], target_pts[d], 1.0)
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
        if pend != 0 and new_day[i]:
            # V2 drops the chase on a new day and leaves a DAY limit resting: not modelled
            return cnt, ERR_PENDING_NEW_DAY, i, pos, entry_i, entry_px, stop
        if pend != 0:
            s = pend
            # A resting limit fills at its own level once any path point reaches it (A5); the
            # protection meets this bar's close, not the rest of the bar (A15).
            filled = (s > 0 and lo[i] <= p_lim) or (s < 0 and h[i] >= p_lim)
            busy = filled or i == p_chase  # the sleeve sees a working entry on this bar
            if filled or (i == p_chase and not exit_sig[i]):
                # chase: cancel, market at this close, levels from that fill (an exit on the
                # same bar flattens instead, and V2's flatten clears the chase)
                pend = 0
                px = p_lim if filled else ci
                st = _level(s, px, p_st, p_stp, -1.0)
                tg = _level(s, px, p_tg, p_tgp, 1.0)
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
                elif not np.isnan(target) and o[i] >= target:
                    reason, px = REASON_TARGET, target
            else:
                if not np.isnan(stop) and o[i] >= stop:
                    reason, px = REASON_STOP_GAP, o[i]
                elif not np.isnan(target) and o[i] <= target:
                    reason, px = REASON_TARGET, target
            if reason == 0:
                # Extremes: high then low, or the one nearer the open first when adaptive
                # (strictly nearer high -> high first; ties -> low first).
                for step in range(2):
                    use_high = (step == 0) == high_first
                    fill_point = step + 1
                    if use_high:
                        if pos > 0:
                            if not np.isnan(target) and h[i] >= target:
                                reason, px = REASON_TARGET, target
                        else:
                            if not np.isnan(stop) and h[i] >= stop:
                                reason, px = REASON_STOP, stop
                    else:
                        if pos > 0:
                            if not np.isnan(stop) and lo[i] <= stop:
                                reason, px = REASON_STOP, stop
                        else:
                            if not np.isnan(target) and lo[i] <= target:
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
        if next_open:
            # R1 decisions remain at the close; stage only actionable orders. At the following
            # open a market exit executes before a replacement entry and before the price sweep.
            market_exit = exit_sig[i] and pos != 0
            enter = (
                (pos == 0 or (market_exit and reenter))
                and side_sig[i] != 0
                and (reenter or not exited)
            )
            if market_exit or enter:
                if i + 1 == n or new_day[i + 1]:
                    return cnt, ERR_NEXT_OPEN_SESSION_END, i, pos, entry_i, entry_px, stop
                if enter:
                    market_entry = i
            continue
        # --- decisions at bar i's close (on_bar): market orders fill at c[i] ---
        if exit_sig[i]:
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
                    st = _level(s, lim, entry_stop[i], stop_pts[i], -1.0)
                    tg = _level(s, lim, entry_target[i], target_pts[i], 1.0)
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
            st = _level(s, ci, entry_stop[i], stop_pts[i], -1.0)
            tg = _level(s, ci, entry_target[i], target_pts[i], 1.0)
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
):  # fmt: skip
    """All columns in parallel. Signal arrays have shape ``(k, n)``; prices and ``new_day`` (the
    bar starts a New York date) shape ``(n,)``.

    Two passes over the same ``_column`` function (count, then fill), so the trade table is exact
    without over-allocating ``k * n`` rows.
    """

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
        cnt, err, eb, fs, fe, fpx, fst = _column(
            o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
            entry_limit[j], stop_pts[j], target_pts[j], new_day, chase,
            adaptive, reenter, stale_drop, protect_flatten, next_open, True,
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
            _column(
                o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
                entry_limit[j], stop_pts[j], target_pts[j], new_day, chase,
                adaptive, reenter, stale_drop, protect_flatten, next_open, False,
                entry_i, exit_i, side, entry_px,
                exit_px, reason,
                stop, offsets[j],
            )  # fmt: skip
    return (
        counts, offsets, errs, err_bar, final_side, final_entry, final_px, final_stop,
        entry_i, exit_i, side, entry_px, exit_px, reason, stop,
    )  # fmt: skip
