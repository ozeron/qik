"""The matcher entry points: one per loop signature, each a column on its loop's route.

``_column`` is the single-order ABI. Latency, and ONE_TICK without next-open entries, run the venue
arm of ``book._book_loop`` (R15); everything else is the single arm; the route word ``compat``
selects it. ``_slots_column`` is the slots ABI: every column that declares a trigger, a stop or
target update, breakeven, a straddle, an open-relative entry or fractional protection, OTO or MIT
protection, or exit-and-rearm runs ``book_slots._slots_loop`` under ``COMPAT_SLOTS``. Both take
the numeric ABI (numeric callers pass ``core=None``) or run a strategy callback through ``core``,
and both return the seven-field result.
"""

from __future__ import annotations

import numba as nb

from .book import COMPAT_SINGLE, COMPAT_SLOTS, COMPAT_VENUE, _book_loop, _routes_here
from .book_slots import _slots_loop


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
    """Simulate one column, on the route its rules select.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """
    # R15: latency, and ONE_TICK without next-open entries, run the venue arm of the loop.
    compat = COMPAT_SINGLE if _routes_here(rules) else COMPAT_VENUE
    return _book_loop(
        o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, entry_limit, stop_pts,
        target_pts, new_day, rules, long_exits, short_exits, force_close, force_open, count_only,
        out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px, out_reason, out_stop, base,
        core, ioc_limits, expiry_days, compat,
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
    """Simulate one column on the slots arm of the loop.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, risk_stop)``.
    """
    return _slots_loop(
        o, h, lo, c, sides, exits, stops, targets, limits, sp, tp, days, rules, triggers, su, tu,
        ba, bo, long_exits, short_exits, force_close, force_open, count_only, out_ei, out_xi,
        out_sd, out_ep, out_xp, out_why, out_st, base, core, ioc_limits, expiry_days, straddle,
        from_open, stop_fracs, target_fracs, sizing_risks, COMPAT_SLOTS,
    )  # fmt: skip


__all__ = ["_column", "_slots_column"]
