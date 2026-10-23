"""Causal higher-timeframe features from minute-open OHLCV frames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .schema import FloatArray

_ET = "America/New_York"
Clock = Sequence[int | str | datetime] | NDArray[np.int64] | pd.DatetimeIndex


def _timestamps(values: Clock) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.dtype == np.dtype("int64"):
        return raw
    dates = pd.DatetimeIndex(values)
    if dates.tz is None:
        raise ValueError("timestamps must be timezone-aware or int64 UTC nanoseconds")
    return dates.as_unit("ns").asi8


def htf_align(
    htf_values: FloatArray,
    htf_close_ts: Clock,
    base_ts: Clock,
) -> FloatArray:
    """Map last fully closed HTF value to each base close; prehistory is NaN.

    Both clocks are close timestamps, timezone-aware or int64 UTC ns. HTF closes
    must be strictly increasing; base closes must be nondecreasing. No fill across
    NaN feature values is performed.
    """
    values = np.asarray(htf_values)
    closes, base = _timestamps(htf_close_ts), _timestamps(base_ts)
    if values.dtype != np.float64 or values.ndim != 1 or closes.shape != values.shape:
        raise TypeError("HTF values must be 1-D float64 with one close per value")
    for clock, strict in ((closes, True), (base, False)):
        if clock.ndim != 1 or np.any(clock == np.iinfo(np.int64).min):
            raise ValueError("invalid timestamp clock")
        if np.any(np.diff(clock) <= 0 if strict else np.diff(clock) < 0):
            raise ValueError("timestamps must be ordered; HTF closes must be unique")
    indices = np.searchsorted(closes, base, side="right") - 1
    out = np.full(len(base), np.nan)
    valid = indices >= 0
    out[valid] = values[indices[valid]]
    return out


def htf(
    bars: pd.DataFrame,
    every: str,
    fn: Callable[[pd.DataFrame], FloatArray | pd.Series],
    *,
    session: str | Sequence[str | datetime] | None = None,
    label: Literal["close"] = "close",
) -> FloatArray:
    """Resample minute-open OHLCV, apply fn, align at base close without lookahead.

    Intraday sizes use lake's 09:30 New York grid, exclude premarket and drop
    incomplete buckets. Inputs must have unique, ordered minute-open timestamps.
    Daily bars aggregate observed source rows by CME trading date: 18:00 New
    York opens the following date's session, which closes at 17:00 New York.
    A session column name or date sequence can override labels (same 17:00 close
    convention). Missing minutes are never synthesized. The caller owns daily
    tape completeness/holiday QA. fn must itself be causal; use the prefix test.
    """
    if label != "close":
        raise ValueError("label must be 'close' for causal HTF features")
    required = {"datetime", "open", "high", "low", "close", "volume"}
    if missing := required.difference(bars.columns):
        raise ValueError(f"missing OHLCV columns: {sorted(missing)}")
    if every not in ("15min", "1h", "1D"):
        raise ValueError("every must be '15min', '1h' or '1D'")
    ts = pd.DatetimeIndex(bars.datetime)
    if ts.tz is None or ts.hasnans or not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError("bars require ordered, unique, timezone-aware timestamps")
    if np.any(ts.as_unit("ns").asi8 % (60 * 10**9)):
        raise ValueError("bars must have minute-open timestamps")
    if bars.empty:
        return np.empty(0, dtype=np.float64)
    local = ts.tz_convert(_ET)
    work = bars.reset_index(drop=True).copy()
    if every == "1D":
        if session is None:
            dates = pd.DatetimeIndex(
                [stamp.normalize() for stamp in local.tz_localize(None)]
            ) + pd.to_timedelta(
                np.array([stamp.hour >= 18 for stamp in local], dtype=int), unit="D"
            )
        else:
            dates = pd.DatetimeIndex(bars[session] if isinstance(session, str) else session)
            if len(dates) != len(bars) or dates.hasnans or dates.tz is not None:
                raise ValueError("session must supply one naive trading date per bar")
            dates = pd.DatetimeIndex([stamp.normalize() for stamp in dates])
        ends = (dates + pd.Timedelta(hours=17)).tz_localize(_ET)
        starts = ((dates - pd.Timedelta(days=1)) + pd.Timedelta(hours=18)).tz_localize(_ET)
        if np.any((ts < starts) | (ts >= ends)):
            raise ValueError("source bar lies outside its CME 18:00-17:00 trading session")
        work["_close"] = ends
    else:
        minutes = 15 if every == "15min" else 60
        clock = np.array([stamp.hour * 60 + stamp.minute for stamp in local])
        bucket = 570 + (clock - 570) // minutes * minutes
        starts = pd.DatetimeIndex(
            [stamp.normalize() for stamp in local.tz_localize(None)]
        ) + pd.to_timedelta(bucket, unit="min")
        work["_close"] = starts.tz_localize(_ET) + pd.Timedelta(minutes=minutes)
        work = work.loc[clock >= 570]
    frame = work.groupby("_close", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("datetime", "size"),
    )
    base_close = ts + pd.Timedelta(minutes=1)
    frame = frame.loc[frame.index <= base_close[-1]]
    if every != "1D":
        frame = frame.loc[frame["count"] == minutes]
    frame = frame.reset_index(names="datetime")
    # Datetime labels supplied to fn are actual close times, not bucket starts.
    values = np.asarray(fn(frame))
    return htf_align(values, pd.DatetimeIndex(frame.datetime), base_close)
