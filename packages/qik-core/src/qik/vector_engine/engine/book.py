"""The unified matcher (single-matcher plan, stage 2): one flat, event-gated loop.

Single-route semantics only, and not routed to yet. ``book_column`` takes the numeric ABI of
``legacy.single._column`` and returns the same seven-field result. A column it cannot match yet
(see ``_book_supported``) is handed to the legacy matcher, so the harness sees one byte-identical
candidate while the rules land; that fallback goes when the last of them does.

Shape (docs/reviews/single-matcher-design/REPORT.md section 3):

* hot state is scalars the loop carries in registers: ``pos``, ``cnt``, the order record
  ``(ei, ep, stop, target)`` and ``sx``/``tx``, sentinel copies of the live stop and target
  (+-inf when absent, so an absent order never touches). There is no state tuple, no phase
  function and no return per branch: a refusal sets ``err`` and breaks;
* a position bar costs one range test against the sentinels. Only a touching bar enters
  ``_sweep``, which takes scalars and returns scalars; a flat bar with no entry signal is two
  loads and two compares;
* rule bits are decoded once, before the loop, from one ``int64``. The route-semantics
  differences between the single, venue and slots matchers are named compat bits in that word
  (``C_*``), set by the route entry, never a second copy of the code.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba.extending import overload

from qik.vector_engine.protection import ERR_NONE, REASON_TARGET, _in_market, levels

from .codes import (
    ERR_STALE_BOOK,
    ERR_STOP_IN_MARKET,
    ERR_TARGET_IN_MARKET,
    R_ADAPTIVE,
    R_ADVERSE_FIRST,
    R_COMBINE,
    R_EXIT_NEXT_OPEN,
    R_LATENCY_BARS,
    R_NEXT_OPEN,
    R_ONE_TICK,
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

# Rule bits, decoded once from the settings record.
F_ADAPTIVE, F_ADVERSE_FIRST, F_REENTER, F_STALE_DROP, F_TARGET_GAP = 1, 2, 4, 8, 16
# Compat bits: what the three routes disagree on for identical rules (REPORT section 2). A bit
# is named here when the loop has a branch for it; the venue/slots arms land with their stages.
# C_REFUSE_IN_MARKET: a stop or target at or through the entry close refuses the entry (ERR 1/2),
# where venue and slots accept it and apply A10/A11 (ERR 4, or an immediate target trade).
C_REFUSE_IN_MARKET = 1 << 8
COMPAT_SINGLE = C_REFUSE_IN_MARKET


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
    if rules[R_ADAPTIVE] != 0.0:
        flags |= F_ADAPTIVE
    if rules[R_ADVERSE_FIRST] != 0.0:
        flags |= F_ADVERSE_FIRST
    if rules[R_REENTER] != 0.0:
        flags |= F_REENTER
    if rules[R_STALE_DROP] != 0.0:
        flags |= F_STALE_DROP
    if rules[R_TARGET_GAP] != 0.0:
        flags |= F_TARGET_GAP
    return flags


@nb.njit(cache=True, nogil=True)
def _has_level(a):
    """Any non-NaN value; an absent column has none."""
    if a is None or len(a) == 0:
        return False
    for k in range(len(a)):
        if not np.isnan(a[k]):
            return True
    return False


@nb.njit(cache=True, nogil=True)
def _has_flag(a):
    """Any set value; an absent column has none."""
    if a is None or len(a) == 0:
        return False
    for k in range(len(a)):
        if a[k]:
            return True
    return False


@nb.njit(cache=True, nogil=True)
def _book_supported(rules, entry_limit, ioc_limits, force_close, force_open):
    """Whether this column needs only rules the loop implements.

    Next-open fills, latency and ONE_TICK (the single-to-venue routing included), limit and IOC
    entries and the force masks are not in the loop yet. Dropped with the fallback.
    """
    if (
        rules[R_NEXT_OPEN] != 0.0
        or rules[R_EXIT_NEXT_OPEN] != 0.0
        or rules[R_LATENCY_BARS] != 0.0
        or rules[R_ONE_TICK] != 0.0
    ):
        return False
    return not (
        _has_level(entry_limit)
        or _has_level(ioc_limits)
        or _has_flag(force_close)
        or _has_flag(force_open)
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
    """The loop alone, for a column ``_book_supported`` accepts: no fallback, numeric ABI.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """
    n = o.shape[0]
    flags = _rule_flags(rules, COMPAT_SINGLE)
    adaptive = flags & F_ADAPTIVE != 0
    adverse_first = flags & F_ADVERSE_FIRST != 0
    reenter = flags & F_REENTER != 0
    stale_drop = flags & F_STALE_DROP != 0
    target_gap = flags & F_TARGET_GAP != 0
    refuse_in_market = flags & C_REFUSE_IN_MARKET != 0
    tick = rules[R_TICK]
    tick_round = int(rules[R_TICK_ROUND])
    combine = int(rules[R_COMBINE])
    # Absent columns (zero length) read as their default; decided once, not per bar.
    has_exit = _has_lane(exit_sig)
    has_long = _has_lane(long_exits)
    has_short = _has_lane(short_exits)
    has_es = _has_lane(entry_stop)
    has_et = _has_lane(entry_target)
    has_sp = _has_lane(stop_pts)
    has_tp = _has_lane(target_pts)

    pos, cnt, ei, ep = 0, 0, -1, 0.0
    stop, target = np.nan, np.nan  # the published levels
    sx, tx = -np.inf, np.inf  # sentinel copies the sweep reads
    err, eb = ERR_NONE, -1
    for i in range(n):
        exited = False
        pin_side = 0  # side of a position a target filled this bar at a stale quote (A13)
        pin = 0.0
        if pos != 0:
            oi = o[i]
            if pos > 0:
                touched = (oi <= sx) | (lo[i] <= sx) | (oi >= tx) | (h[i] >= tx)
            else:
                touched = (oi >= sx) | (h[i] >= sx) | (oi <= tx) | (lo[i] <= tx)
            if touched:
                reason, px, point, high_first = _sweep(
                    oi, h[i], lo[i], pos, stop, target, adaptive, adverse_first, target_gap
                )
                if reason != 0:
                    _put_trade(
                        count_only, base + cnt, ei, i, pos, ep, px, reason, stop,
                        out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                        out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                    if reason == REASON_TARGET and point > 0:  # a gap fill pins nothing
                        e1 = h[i] if high_first else lo[i]
                        e2 = lo[i] if high_first else h[i]
                        if _no_reversal(oi, e1, e2, c[i], point, pos):
                            pin_side = pos
                            pin = px
                    pos = 0
                    exited = True
        # --- decisions at bar i's close (the strategy callback goes here) ---
        if pos != 0:
            sig = _lane_at(exit_sig, i, False) if has_exit else False
            if pos > 0:
                sig = sig or (_lane_at(long_exits, i, False) if has_long else False)
            else:
                sig = sig or (_lane_at(short_exits, i, False) if has_short else False)
            if sig:
                _put_trade(
                    count_only, base + cnt, ei, i, pos, ep, c[i], REASON_SIGNAL, stop,
                    out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px,
                    out_reason, out_stop,
                )  # fmt: skip
                cnt += 1
                pos = 0
                exited = True
        s = int(side_sig[i])
        if pos == 0 and s != 0 and (reenter or not exited):
            ci = c[i]
            st, tg = levels(
                s,
                ci,
                _lane_at(entry_stop, i, np.nan) if has_es else np.nan,
                _lane_at(entry_target, i, np.nan) if has_et else np.nan,
                _lane_at(stop_pts, i, np.nan) if has_sp else np.nan,
                _lane_at(target_pts, i, np.nan) if has_tp else np.nan,
                tick,
                tick_round,
                combine=combine,
                fill=ci,
            )
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
    if core is not None or not _book_supported(
        rules, entry_limit, ioc_limits, force_close, force_open
    ):
        return _column(
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
    return _book_loop(
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


__all__: list[str] = []
