"""Numba kernel: closed-form per-bar execution rules, one column (config) at a time.

The rules are written down in ``docs/vector-engine.md`` (assumptions A1-A12). ``reference.py`` is an
independent, slow implementation of the same spec that matches orders price point by price point;
``tests`` check the two agree exactly and that both reproduce Nautilus golden data.

All inputs are float64 / int8 / bool arrays with one row per column (config) and one column per bar.
"""

from __future__ import annotations

import numba as nb
import numpy as np

REASON_SIGNAL = 1  # market exit at the decision bar's close
REASON_STOP = 2  # resting stop-market hit during a bar, filled at the stop level
REASON_TARGET = 3  # resting limit hit during a bar, filled at the limit level
REASON_STOP_GAP = 4  # resting stop already through at the bar's open, filled at the open

ERR_NONE = 0
ERR_STOP_IN_MARKET = 1  # stop level not strictly beyond the entry close (Nautilus rejects it)
ERR_TARGET_IN_MARKET = 2  # target level not strictly beyond the entry close (would fill at once)
ERR_STALE_BOOK = 3  # new stop falls inside the stale quote a same-bar limit fill left (A13)


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
def _column(
    o,
    h,
    lo,
    c,
    side_sig,
    exit_sig,
    entry_stop,
    entry_target,
    adaptive,
    reenter,
    stale_drop,
    count_only,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    base,
):
    """Simulate one column. Returns ``(n_trades, err, err_bar, final_side, final_entry_i)``."""

    n = o.shape[0]
    pos = 0
    entry_i = -1
    entry_px = 0.0
    stop = np.nan
    target = np.nan
    cnt = 0
    for i in range(n):
        exited = False
        stale_dir = 0  # side of a position closed by a limit fill this bar whose quote is stale
        stale_px = 0.0
        if pos != 0:
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
                cnt += 1
                if (
                    reason == REASON_TARGET and fill_point > 0
                ):  # an open-point (gap) fill pins nothing
                    e1 = h[i] if high_first else lo[i]
                    e2 = lo[i] if high_first else h[i]
                    if _no_reversal(o[i], e1, e2, c[i], fill_point, pos):
                        stale_dir = pos
                        stale_px = px
                pos = 0
                exited = True
        # --- decisions at bar i's close (on_bar): market orders fill at c[i] ---
        if pos != 0 and exit_sig[i]:
            if not count_only:
                k = base + cnt
                out_entry_i[k] = entry_i
                out_exit_i[k] = i
                out_side[k] = pos
                out_entry_px[k] = entry_px
                out_exit_px[k] = c[i]
                out_reason[k] = REASON_SIGNAL
            cnt += 1
            pos = 0
            exited = True
        if pos == 0 and side_sig[i] != 0 and (reenter or not exited):
            s = side_sig[i]
            st = entry_stop[i]
            tg = entry_target[i]
            if not np.isnan(st):
                if (s > 0 and st >= c[i]) or (s < 0 and st <= c[i]):
                    return cnt, ERR_STOP_IN_MARKET, i, pos, entry_i
            if not np.isnan(tg):
                if (s > 0 and tg <= c[i]) or (s < 0 and tg >= c[i]):
                    return cnt, ERR_TARGET_IN_MARKET, i, pos, entry_i
            if (
                stale_dir == s
                and not np.isnan(st)
                and ((s > 0 and st >= stale_px) or (s < 0 and st <= stale_px))
            ):
                if not stale_drop:
                    return cnt, ERR_STALE_BOOK, i, pos, entry_i
                st = np.nan  # Nautilus rejects the stop and cancels the linked target (A13)
                tg = np.nan
            pos = s
            entry_i = i
            entry_px = c[i]
            stop = st
            target = tg
    return cnt, ERR_NONE, -1, pos, entry_i


@nb.njit(parallel=True, cache=True)
def run_columns(
    o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, adaptive, reenter, stale_drop
):
    """All columns in parallel. Signal arrays have shape ``(k, n)``; prices shape ``(n,)``.

    Two passes over the same ``_column`` function (count, then fill), so the trade table is exact
    without over-allocating ``k * n`` rows.
    """

    k = side_sig.shape[0]
    counts = np.zeros(k, np.int64)
    errs = np.zeros(k, np.int64)
    err_bar = np.full(k, -1, np.int64)
    final_side = np.zeros(k, np.int64)
    final_entry = np.full(k, -1, np.int64)
    e0 = np.empty(0, np.int64)
    s0 = np.empty(0, np.int8)
    f0 = np.empty(0, np.float64)
    for j in nb.prange(k):  # ty: ignore[not-iterable]  (numba ships prange as a class)
        cnt, err, eb, fs, fe = _column(
            o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
            adaptive, reenter, stale_drop, True, e0, e0, s0, f0, f0, s0, 0,
        )  # fmt: skip
        counts[j] = cnt
        errs[j] = err
        err_bar[j] = eb
        final_side[j] = fs
        final_entry[j] = fe
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
    for j in nb.prange(k):  # ty: ignore[not-iterable]
        if errs[j] == 0:
            _column(
                o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j],
                adaptive, reenter, stale_drop, False, entry_i, exit_i, side, entry_px, exit_px, reason,
                offsets[j],
            )  # fmt: skip
    return counts, offsets, errs, err_bar, final_side, final_entry, entry_i, exit_i, side, entry_px, exit_px, reason  # fmt: skip
