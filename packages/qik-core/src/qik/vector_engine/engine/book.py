"""The unified matcher (single-matcher plan, stage 2). Skeleton: delegates to the legacy matcher."""

from __future__ import annotations

import numba as nb

from .legacy.single import _column


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
    """Simulate one column with single-route semantics; same ABI and result as ``_column``."""
    return _column(
        o, h, lo, c, side_sig, exit_sig, entry_stop, entry_target, entry_limit, stop_pts,
        target_pts, new_day, rules, long_exits, short_exits, force_close, force_open,
        count_only, out_entry_i, out_exit_i, out_side, out_entry_px, out_exit_px, out_reason,
        out_stop, base, core, ioc_limits, expiry_days,
    )  # fmt: skip


__all__: list[str] = []
