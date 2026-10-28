"""The slots matcher entry: the unified loop's slots arm in ``book_slots``.

Every column that declares a trigger, a stop or target update, breakeven, a straddle, an
open-relative entry or fractional protection, OTO or MIT protection, or exit-and-rearm runs
``book_slots._slots_loop`` under ``COMPAT_SLOTS`` (the compat word of ``book``). The previous
matcher (the state tuple and its per-bar advance) is ``legacy/slots.py``: the differential oracle
for the tests, never imported here.
"""

from __future__ import annotations

import numba as nb

from .book import COMPAT_SLOTS
from .book_slots import _slots_loop


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


_compiled_slots_column = nb.njit(cache=False, nogil=True)(
    getattr(_slots_column, "py_func", _slots_column)
)


__all__ = ["_compiled_slots_column", "_slots_column"]
