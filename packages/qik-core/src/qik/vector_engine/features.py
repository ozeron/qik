"""Vectorised completed-bar features; calendars and indicators stay outside JIT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qik.indicators import session_vwap
from qik.indicators.session import Summation

from .schema import BoolArray, Features, FloatArray, IntArray

_ET = "America/New_York"


class SessionFeatures(Features):
    """One row per bar: NY calendar date/clock and scheduled RTH masks."""

    date: IntArray
    clock_min: IntArray
    active: BoolArray
    start: BoolArray
    end: BoolArray
    cutoff: BoolArray
    early_close: BoolArray


class RangeFeatures(Features):
    high: FloatArray
    low: FloatArray
    ready: BoolArray


def session_masks(
    timestamps: pd.DatetimeIndex,
    schedule: pd.DataFrame,
    *,
    cutoff_min: int,
    bar_minutes: int = 1,
) -> SessionFeatures:
    """Map exchange schedule to minute-open bars, without inferring missing bars.

    Supply calendar rows ``session_date, open_ts, close_ts, early_close`` (e.g.
    lake's NYSECalendar/CMECalendar in the caller). Bounds must be timezone-aware,
    same NY calendar date, nonoverlapping and ordered. Overnight schedules are
    refused. Dates absent from the schedule are inactive. Caller owns schedule
    horizon/completeness. Start/end mark exact scheduled first/last bar opens;
    cutoff marks active opens >= min(cutoff_min, scheduled last bar open).
    Missing terminal bars never move end/cutoff backwards. Early-close is a
    calendar attribute on all rows of that date, not inferred from tape length.

    Admitted: legacy load_data session clipping; calendar shortening
    is explicit caller input rather than their fixed-clock assumption.
    """
    if not 0 <= cutoff_min < 1440 or bar_minutes < 1:
        raise ValueError("cutoff must be 0..1439; bar_minutes must be positive")
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None or ts.hasnans or not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError("bars require ordered, unique, timezone-aware timestamps")
    if np.any(ts.as_unit("ns").asi8 % (60 * 10**9)):
        raise ValueError("bars must have minute-open timestamps")
    required = {"session_date", "open_ts", "close_ts", "early_close"}
    if not required.issubset(schedule.columns):
        raise ValueError("schedule requires session_date/open_ts/close_ts/early_close")
    dates = pd.DatetimeIndex(schedule.session_date)
    opens, closes = pd.DatetimeIndex(schedule.open_ts), pd.DatetimeIndex(schedule.close_ts)
    if (
        dates.tz is not None
        or dates.hasnans
        or not dates.equals(pd.DatetimeIndex(pd.Series(dates).dt.normalize()))
    ):
        raise ValueError("session_date must contain naive NY calendar dates")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("schedule dates must be unique and ordered")
    if opens.tz is None or closes.tz is None or opens.hasnans or closes.hasnans:
        raise ValueError("schedule bounds must be timezone-aware and nonmissing")
    if schedule.early_close.dtype != np.bool_:
        raise TypeError("early_close must be boolean")
    op, cl = opens.as_unit("ns").asi8, closes.as_unit("ns").asi8
    duration = bar_minutes * 60 * 10**9
    if np.any(cl - op < duration) or np.any(op[1:] < cl[:-1]):
        raise ValueError("schedule bounds must be ordered, nonoverlapping and fit a bar")
    for bound in (opens, closes):
        if not pd.DatetimeIndex(
            pd.Series(bound.tz_convert(_ET).tz_localize(None)).dt.normalize()
        ).equals(dates):
            raise ValueError("schedule bounds must share their NY session_date")
        if np.any(bound.as_unit("ns").asi8 % (60 * 10**9)):
            raise ValueError("schedule bounds must be minute aligned")
    local = ts.tz_convert(_ET)
    day = pd.DatetimeIndex(pd.Series(local.tz_localize(None)).dt.normalize())
    clock = np.asarray(pd.Series(local).dt.hour * 60 + pd.Series(local).dt.minute, dtype=np.int64)
    keys = pd.Index(dates)
    mapping = pd.Series(np.arange(len(dates)), index=keys).reindex(day)
    valid = mapping.notna().to_numpy()
    # Reindexing preserves every source row, including holidays and off-session bars.
    aligned = schedule.set_index(keys).reindex(day)
    start_ns = pd.DatetimeIndex(aligned.open_ts).as_unit("ns").asi8
    last_ns = np.where(valid, pd.DatetimeIndex(aligned.close_ts).as_unit("ns").asi8, 0) - duration
    now = ts.as_unit("ns").asi8
    active = valid & (now >= start_ns) & (now <= last_ns)
    # Compare wall-clock minutes without localizing an inactive DST Sunday's cutoff.
    return SessionFeatures(
        date=np.asarray(local.strftime("%Y%m%d"), dtype=np.int64),
        clock_min=clock,
        active=active,
        start=active & (now == start_ns),
        end=active & (now == last_ns),
        cutoff=active & ((clock >= cutoff_min) | (now >= last_ns)),
        early_close=aligned.early_close.eq(True).to_numpy(dtype=np.bool_),
    )


def session_range(
    high: FloatArray,
    low: FloatArray,
    session: IntArray,
    clock_min: IntArray,
    *,
    start_min: int,
    end_min: int,
    min_bars: int,
) -> RangeFeatures:
    """Freeze observed extrema in [start_min, end_min); reveal at opens >= end_min.

    Input contains only the caller's eligible session bars, ordered by clock
    within each unique session. Missing minutes are never synthesized; min_bars
    is an explicit admission choice. Empty/underfilled windows stay NaN/unready.
    Bounds can feed range_breakout directly; flat ranges remain caller policy.
    Admitted: legacy opening-range builders.
    """
    h, lo = np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64)
    key, clock = np.asarray(session), np.asarray(clock_min)
    if any(a.ndim != 1 or a.shape != h.shape for a in (h, lo, key, clock)):
        raise ValueError("range inputs must be equally sized 1-D arrays")
    if key.dtype != np.int64 or clock.dtype != np.int64:
        raise TypeError("session and clock_min must be int64")
    if not 0 <= start_min < end_min <= 1440 or min_bars < 1:
        raise ValueError("require 0 <= start_min < end_min <= 1440 and min_bars >= 1")
    if np.any((clock < 0) | (clock >= 1440)):
        raise ValueError("clock_min must be 0..1439")
    if np.any(key[1:] < key[:-1]) or np.any((key[1:] == key[:-1]) & (np.diff(clock) <= 0)):
        raise ValueError("sessions must be ordered with unique increasing clocks")
    if not np.isfinite(h).all() or not np.isfinite(lo).all() or np.any(h < lo):
        raise ValueError("range prices must be finite with high >= low")
    inside = (clock >= start_min) & (clock < end_min)
    count = pd.Series(inside.astype(np.int64)).groupby(key, sort=False).cumsum().to_numpy()
    ready = (clock >= end_min) & (count >= min_bars)
    hi = pd.Series(np.where(inside, h, -np.inf)).groupby(key, sort=False).cummax().to_numpy()
    lw = pd.Series(np.where(inside, lo, np.inf)).groupby(key, sort=False).cummin().to_numpy()
    return RangeFeatures(
        high=np.where(ready, hi, np.nan), low=np.where(ready, lw, np.nan), ready=ready
    )


def anchored_vwap(
    price: FloatArray,
    volume: FloatArray,
    reset: BoolArray,
    *,
    summation: Summation,
) -> FloatArray:
    """Completed-bar VWAP; reset BEFORE consuming the marked bar (first must reset).

    Caller supplies HLC3/close/OHLC4 price and explicit anchor/reset mask. Until
    cumulative volume is positive output is NaN; later zero-volume bars hold.
    Nonfinite inputs/negative volume are refused. Reuses qik.indicators sums:
    naive (sequential) or kahan (pandas compensated), never subtract global sums.
    Admitted: legacy session-VWAP features (naive and kahan).
    """
    px, vol, rst = (
        np.asarray(price, dtype=np.float64),
        np.asarray(volume, dtype=np.float64),
        np.asarray(reset),
    )
    if any(a.ndim != 1 or a.shape != px.shape for a in (px, vol, rst)):
        raise ValueError("VWAP inputs must be equally sized 1-D arrays")
    if rst.dtype != np.bool_ or (len(rst) and not rst[0]):
        raise ValueError("reset must be boolean and the first bar must reset")
    if not np.isfinite(px).all() or not np.isfinite(vol).all() or np.any(vol < 0):
        raise ValueError("VWAP requires finite price and nonnegative finite volume")
    p, v = pd.Series(px), pd.Series(vol)
    return session_vwap(
        p,
        p,
        p,
        v,
        pd.Series(np.cumsum(rst)),
        price="close",
        clip_negative_volume=False,
        summation=summation,
    ).to_numpy(dtype=np.float64)
