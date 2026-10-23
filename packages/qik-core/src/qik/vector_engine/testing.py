"""Deterministic test tapes and core assertions; no lake or execution-rule substitutes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import timedelta

import numpy as np
import pandas as pd

from .api import Bars
from .core import BarView, Core, Decision


def minute_bars(
    start: str | pd.Timestamp = "2023-01-10 09:30",
    periods: int = 30,
    *,
    price: float = 100.0,
    step: float = 0.0,
    open_offset: float = 0.0,
    wick: float = 0.0,
    volume: float | None = 100.0,
    tz: str | None = None,
    halt_hours: Sequence[int] = (),
    session_roll_hour: int = 0,
    gap_at: int | None = None,
    gap_points: float = 0.0,
    ohlc: Sequence[tuple[float, float, float, float]] | None = None,
) -> pd.DataFrame:
    """Build minute bars, with optional trend, gap, halt and explicit OHLC rows.

    ``periods`` counts clock minutes before removing ``halt_hours``. Prices advance
    only on retained bars. Session labels use the local wall date, rolling to the
    following date at ``session_roll_hour`` (0 means calendar dates). A timezone
    preserves actual elapsed minutes through DST; it never invents missing hours.
    Volume can be omitted explicitly for equivalence with legacy OHLC fixtures.
    """
    if periods < 1 or wick < 0 or not 0 <= session_roll_hour <= 23:
        raise ValueError("positive periods, nonnegative wick and roll hour 0..23 required")
    if any(not 0 <= hour <= 23 for hour in halt_hours):
        raise ValueError("halt hours must lie in 0..23")
    index = pd.date_range(start, periods=periods, freq="min", tz=tz)
    index = index[~pd.Series(index).dt.hour.isin(halt_hours).to_numpy()]
    if index.empty:
        raise ValueError("halt removed every bar")
    n = len(index)
    if gap_at is None and gap_points != 0:
        raise ValueError("gap_points requires gap_at")
    if gap_at is not None and not 1 <= gap_at < n:
        raise ValueError("gap_at must be inside the retained tape")
    if ohlc is not None:
        if len(ohlc) != n or gap_at is not None or step or open_offset or wick:
            raise ValueError("explicit OHLC requires one row per bar and no price transforms")
        prices = np.asarray(ohlc, dtype=float)
        if prices.shape != (n, 4):
            raise ValueError("OHLC rows must have four prices")
        opens, highs, lows, closes = prices.T
    else:
        closes = float(price) + float(step) * np.arange(n)
        if gap_at is not None:
            closes[gap_at:] += gap_points
        opens = closes + open_offset
        highs = np.maximum(opens, closes) + wick
        lows = np.minimum(opens, closes) - wick
    # Validate through the public engine input contract, without changing prices.
    Bars(
        opens,
        highs,
        lows,
        closes,
        index=index,
        volume=None if volume is None else np.full(n, volume),
    )
    sessions = [
        ts.date() + timedelta(days=int(bool(session_roll_hour) and ts.hour >= session_roll_hour))
        for ts in index
    ]
    frame = pd.DataFrame(
        {
            "datetime": index,
            "session_date": sessions,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )
    if volume is not None:
        frame["volume"] = float(volume)
    return frame


def bar_view(bars: Bars | pd.DataFrame, i: int = 0) -> BarView:
    """Return one completed bar, preserving the timestamp and missing-volume NaN."""
    if not 0 <= i < len(bars):
        raise IndexError(i)
    if isinstance(bars, pd.DataFrame):
        row = bars.iloc[i]
        timestamp = pd.Timestamp(row.datetime)
        if not isinstance(timestamp, pd.Timestamp):
            raise ValueError(f"bar {i}: timestamp must not be NaT")
        return BarView(
            i,
            timestamp,
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.get("volume", np.nan)),
        )
    timestamp = None if bars.index is None else bars.index[i]
    if timestamp is not None and not isinstance(timestamp, pd.Timestamp):
        raise ValueError(f"bar {i}: timestamp must not be NaT")
    return BarView(
        i,
        timestamp,
        bars.open[i],
        bars.high[i],
        bars.low[i],
        bars.close[i],
        np.nan if bars.volume is None else bars.volume[i],
    )


def decision_stream(core: Core, bars: Bars | pd.DataFrame) -> list[Decision]:
    """Advance the supplied core exactly once per bar; callers own fresh-core creation."""
    decisions = []
    for i in range(len(bars)):
        decision = core.step(bar_view(bars, i))
        if not isinstance(decision, Decision):
            raise TypeError(f"bar {i}: step must return a Decision")
        decisions.append(decision)
    return decisions


def assert_trades_equal(a: pd.DataFrame, b: pd.DataFrame, fields: Sequence[str]) -> None:
    """Exact ordered field comparison, ignoring storage dtype and frame index only.

    No sorting, rounding, timezone stripping or tolerance: normalize deliberately at
    the fixture boundary. NaNs compare equal; categorical storage is ignored.
    """
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("fields must be nonempty and unique")
    pd.testing.assert_frame_equal(
        a.loc[:, list(fields)].reset_index(drop=True),
        b.loc[:, list(fields)].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
        check_categorical=False,
    )
    for field in fields:
        if pd.api.types.is_datetime64_any_dtype(a[field]) or pd.api.types.is_datetime64_any_dtype(
            b[field]
        ):
            pd.testing.assert_series_equal(
                a[field].reset_index(drop=True),
                b[field].reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
            )


def assert_causal_feature(
    fn_builder: Callable[[pd.DataFrame], np.ndarray], bars: pd.DataFrame, n: int
) -> None:
    """Assert exact prefix stability (including NaN) for a feature builder."""
    if not 0 < n <= len(bars):
        raise ValueError("n must select a nonempty prefix")
    prefix = np.asarray(fn_builder(bars.iloc[:n].copy()))
    full = np.asarray(fn_builder(bars.copy()))
    if prefix.shape != (n,) or full.shape != (len(bars),):
        raise AssertionError("feature builder must return one value per base bar")
    np.testing.assert_array_equal(prefix, full[:n], err_msg="feature looks ahead")
