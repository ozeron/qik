"""Batch entry points: the parallel column runner and its Python-boundary ABI."""

from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any

import numba as nb
import numpy as np

from qik.vector_engine.protection import _round_price

from .codes import R_ARM_STOP, R_DAY_CANCEL, R_TICK, R_TICK_ROUND, RULES_WIDTH
from .entry import _column, _slots_column
from .levels import _optional_column, _straddle_column


@nb.njit(parallel=True, cache=True)
def _run_columns(
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
    entry_trigger: np.ndarray,
    stop_updates: np.ndarray,
    target_updates: np.ndarray,
    be_after: np.ndarray,
    be_offset: np.ndarray,
    exit_long,
    exit_short,
    force_close,
    force_open,
    entry_limit_ioc,
    expiry_days,
    straddle,
    entry_from_open,
    stop_fraction,
    target_fraction,
    sizing_risks,
    slots,
):
    """All columns in parallel. Signal arrays have shape ``(k, n)``; prices and ``new_day`` (the
    bar starts a New York date) shape ``(n,)``.

    Two passes over the same ``_column`` function (count, then fill), so the trade table is exact
    without over-allocating ``k * n`` rows.
    """
    tick = rules[R_TICK]
    tick_round = int(rules[R_TICK_ROUND])

    if tick_round:
        entry_limit = entry_limit.copy()
        for j in range(side_sig.shape[0]):
            for i in range(side_sig.shape[1]):
                entry_limit[j, i] = _round_price(
                    entry_limit[j, i], tick, tick_round, -side_sig[j, i]
                )
        if slots:
            entry_trigger = entry_trigger.copy()
            for j in range(side_sig.shape[0]):
                for i in range(side_sig.shape[1]):
                    entry_trigger[j, i] = _round_price(
                        entry_trigger[j, i], tick, tick_round, side_sig[j, i]
                    )
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
        pair_column = _straddle_column(straddle, j)
        risk_column = _optional_column(sizing_risks, j)
        rel_column = _optional_column(entry_from_open, j)
        sf_column = _optional_column(stop_fraction, j)
        tf_column = _optional_column(target_fraction, j)
        ioc_column = _optional_column(entry_limit_ioc, j)
        long_column = _optional_column(exit_long, j)
        short_column = _optional_column(exit_short, j)
        close_column = _optional_column(force_close, j)
        open_column = _optional_column(force_open, j)
        if slots:
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
                rules,
                entry_trigger[j],
                stop_updates[j],
                target_updates[j],
                be_after[j],
                be_offset[j],
                long_column,
                short_column,
                close_column,
                open_column,
                np.bool_(True),
                e0,
                e0,
                s0,
                f0,
                f0,
                s0,
                f0,
                np.int64(0),
                core=None,
                ioc_limits=ioc_column,
                expiry_days=expiry_days,
                straddle=pair_column,
                from_open=rel_column,
                stop_fracs=sf_column,
                target_fracs=tf_column,
                sizing_risks=risk_column,
            )
        else:
            cnt, err, eb, fs, fe, fpx, fst = _column(o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j], entry_limit[j], stop_pts[j], target_pts[j], new_day, rules, long_column, short_column, close_column, open_column, np.bool_(True), e0, e0, s0, f0, f0, s0, f0, np.int64(0), core=None, ioc_limits=ioc_column, expiry_days=expiry_days)  # fmt: skip
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
        pair_column = _straddle_column(straddle, j)
        risk_column = _optional_column(sizing_risks, j)
        rel_column = _optional_column(entry_from_open, j)
        sf_column = _optional_column(stop_fraction, j)
        tf_column = _optional_column(target_fraction, j)
        ioc_column = _optional_column(entry_limit_ioc, j)
        long_column = _optional_column(exit_long, j)
        short_column = _optional_column(exit_short, j)
        close_column = _optional_column(force_close, j)
        open_column = _optional_column(force_open, j)
        if errs[j] == 0:
            if slots:
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
                    rules,
                    entry_trigger[j],
                    stop_updates[j],
                    target_updates[j],
                    be_after[j],
                    be_offset[j],
                    long_column,
                    short_column,
                    close_column,
                    open_column,
                    np.bool_(False),
                    entry_i,
                    exit_i,
                    side,
                    entry_px,
                    exit_px,
                    reason,
                    stop,
                    offsets[j],
                    core=None,
                    ioc_limits=ioc_column,
                    expiry_days=expiry_days,
                    straddle=pair_column,
                    from_open=rel_column,
                    stop_fracs=sf_column,
                    target_fracs=tf_column,
                    sizing_risks=risk_column,
                )
            else:
                _column(o, h, lo, c, side_sig[j], exit_sig[j], entry_stop[j], entry_target[j], entry_limit[j], stop_pts[j], target_pts[j], new_day, rules, long_column, short_column, close_column, open_column, np.bool_(False), entry_i, exit_i, side, entry_px, exit_px, reason, stop, offsets[j], core=None, ioc_limits=ioc_column, expiry_days=expiry_days)  # fmt: skip
    return (
        counts, offsets, errs, err_bar, final_side, final_entry, final_px, final_stop,
        entry_i, exit_i, side, entry_px, exit_px, reason, stop,
    )  # fmt: skip


# Host boundary: defaults, missing buffers and flag literals do not identify a new
# matcher. Select the scheduler from declarations before replacing missing arrays.
_EMPTY_BOOL_COLUMNS = np.empty((1, 0), np.bool_)
_EMPTY_INT8_COLUMNS = np.empty((1, 0), np.int8)
_EMPTY_FLOAT_COLUMNS = np.empty((1, 0), np.float64)
_EMPTY_RISK_COLUMNS = np.empty((1, 0), np.float64)
_EMPTY_DAYS = np.empty(0, np.bool_)
_EMPTY_PAIRS = np.empty((1, 12, 0), np.float64)
for _empty_input in (
    _EMPTY_BOOL_COLUMNS,
    _EMPTY_INT8_COLUMNS,
    _EMPTY_FLOAT_COLUMNS,
    _EMPTY_DAYS,
    _EMPTY_PAIRS,
):
    _empty_input.flags.writeable = False


def _column_input(value: np.ndarray, dtype: Any) -> np.ndarray:
    """Use a read-only view to unify mutability without copying an aligned input."""
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


def _normalized_columns(args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    bound = _RUN_COLUMNS_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    rules = np.ascontiguousarray(values["rules"], dtype=np.float64)
    if rules.shape != (RULES_WIDTH,):
        raise ValueError(f"compiled Rules ABI requires exactly {RULES_WIDTH} values")
    values["rules"] = _column_input(rules, dtype=np.float64)
    slots = values["entry_trigger"] is not None
    if slots:
        assert values["stop_updates"] is not None and values["target_updates"] is not None
        assert values["be_after"] is not None and values["be_offset"] is not None
    pair = values["straddle"]
    if pair is not None and np.isfinite(pair).any():
        if values["rules"][R_ARM_STOP]:
            raise ValueError("straddle arm_stop=next_bar composition is unproved")
        if values["rules"][R_DAY_CANCEL]:
            raise ValueError("straddle day_order_expiry=cancel composition is unproved")
    if pair is not None and not slots:
        raise ValueError("straddle requires the slots scheduler")
    if not slots and any(
        values[name] is not None for name in ("entry_from_open", "stop_fraction", "target_fraction")
    ):
        raise ValueError("entry_from_open and fractional protection require the slots scheduler")
    values["slots"] = slots
    # Preserve integer and bool lanes: only prices/protection use float64.
    for name in ("o", "h", "lo", "c"):
        values[name] = _column_input(values[name], dtype=np.float64)
    values["new_day"] = _column_input(values["new_day"], dtype=np.bool_)
    values["side_sig"] = _column_input(values["side_sig"], dtype=np.int8)
    for name in (
        "exit_sig",
        "exit_long",
        "exit_short",
        "force_open",
        "entry_from_open",
    ):
        value = values[name]
        values[name] = (
            _EMPTY_BOOL_COLUMNS if value is None else _column_input(value, dtype=np.bool_)
        )
    value = values["force_close"]
    values["force_close"] = (
        _EMPTY_INT8_COLUMNS if value is None else _column_input(value, dtype=np.int8)
    )
    for name in (
        "entry_stop",
        "entry_target",
        "entry_limit",
        "stop_pts",
        "target_pts",
        "entry_trigger",
        "stop_updates",
        "target_updates",
        "be_after",
        "be_offset",
        "stop_fraction",
        "target_fraction",
        "entry_limit_ioc",
    ):
        value = values[name]
        values[name] = (
            _EMPTY_FLOAT_COLUMNS if value is None else _column_input(value, dtype=np.float64)
        )
    # This one buffer is output storage, so keep its writeable lane separate.
    value = values["sizing_risks"]
    values["sizing_risks"] = (
        _EMPTY_RISK_COLUMNS if value is None else np.ascontiguousarray(value, dtype=np.float64)
    )
    value = values["expiry_days"]
    values["expiry_days"] = _EMPTY_DAYS if value is None else _column_input(value, dtype=np.bool_)
    values["straddle"] = _EMPTY_PAIRS if pair is None else _column_input(pair, dtype=np.float64)
    return values


def run_columns(
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
    entry_trigger: np.ndarray | None = None,
    stop_updates: np.ndarray | None = None,
    target_updates: np.ndarray | None = None,
    be_after: np.ndarray | None = None,
    be_offset: np.ndarray | None = None,
    exit_long=None,
    exit_short=None,
    force_close=None,
    force_open=None,
    entry_limit_ioc=None,
    expiry_days=None,
    straddle=None,
    entry_from_open=None,
    stop_fraction=None,
    target_fraction=None,
    sizing_risks=None,
) -> tuple:
    """Normalize the numeric ABI before dispatch; declarations still select scheduling."""
    return _run_columns(**_normalized_columns((), locals()))


@wraps(run_columns)
def _python_run_columns(*args: Any, **kwargs: Any) -> tuple:
    runner = getattr(_run_columns, "py_func", _run_columns)
    return runner(**_normalized_columns(args, kwargs))


setattr(run_columns, "py_func", _python_run_columns)

_RUN_COLUMNS_SIGNATURE = signature(run_columns)


__all__ = [
    "_EMPTY_BOOL_COLUMNS",
    "_EMPTY_DAYS",
    "_EMPTY_FLOAT_COLUMNS",
    "_EMPTY_INT8_COLUMNS",
    "_EMPTY_PAIRS",
    "_EMPTY_RISK_COLUMNS",
    "_RUN_COLUMNS_SIGNATURE",
    "_column_input",
    "_empty_input",
    "_normalized_columns",
    "_python_run_columns",
    "_run_columns",
    "run_columns",
]
