"""The single-order matcher entry: route the column, then run the unified loop in ``book``.

Latency, and ONE_TICK without next-open entries, run on the venue matcher (R15); everything else
is single-route semantics in ``book._book_loop``, with or without a strategy callback. The previous
state-tuple matcher (``ColumnState``, ``_column_advance``) is ``legacy/single.py``: a differential
oracle for the tests, never imported here.
"""

from __future__ import annotations

import numba as nb

from .book import COMPAT_SINGLE, _book_loop, _routes_here
from .venue import _venue_driver


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
    """Simulate one column with single-route semantics.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px, final_stop)``.
    """
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
        core, ioc_limits, expiry_days, COMPAT_SINGLE,
    )  # fmt: skip


_compiled_column = nb.njit(cache=False, nogil=True)(getattr(_column, "py_func", _column))

__all__ = ["_column", "_compiled_column"]
