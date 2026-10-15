"""Session-anchored indicators: VWAP and opening range (batch).

Bars are labelled by their **open** time (repo-wide contract). ``clock_min`` is
minutes since midnight in the exchange's local clock (ET for CME equity index),
so 09:30 ET is 570. The session key is whatever the caller groups by
(``session_date`` = CME trade date in the lake; equal to the ET date on RTH bars).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

VwapPrice = Literal["typical", "close", "ohlc4"]
Summation = Literal["naive", "kahan"]
OrBy = Literal["clock", "bars"]

RTH_OPEN_MIN = 9 * 60 + 30


def vwap_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    price: VwapPrice = "typical",
    open_: pd.Series | None = None,
) -> pd.Series:
    """Per-bar price used by :func:`session_vwap`."""
    if price == "typical":
        return (high + low + close) / 3.0
    if price == "close":
        return close.astype(float)
    if price == "ohlc4":
        if open_ is None:
            raise ValueError("price='ohlc4' needs open_")
        return (open_ + high + low + close) / 4.0
    raise ValueError(f"unknown price: {price!r}")


def session_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    session: pd.Series,
    *,
    price: VwapPrice = "typical",
    open_: pd.Series | None = None,
    clip_negative_volume: bool = True,
    summation: Summation = "naive",
) -> pd.Series:
    """Cumulative ``sum(price*vol) / sum(vol)`` restarting at every ``session`` change.

    Rows must be time-ordered within each session. Until a session has positive
    cumulative volume the value is NaN; zero-volume bars hold the prior value.
    ``clip_negative_volume=True`` treats negative volume as 0 (streaming
    the common choice); ``False`` uses raw volume.
    ``summation="naive"`` is a plain running float sum (= ``Series.cumsum`` and the
    streaming ``SessionVWAP``, bit-for-bit); ``"kahan"`` is pandas'
    ``groupby().cumsum()`` (compensated). They differ by ~1e-11 on NQ prices.
    """
    vol = pd.to_numeric(volume, errors="coerce").astype(float)
    if clip_negative_volume:
        vol = vol.clip(lower=0.0)
    px = vwap_price(high, low, close, price=price, open_=open_)
    keys = pd.Series(np.asarray(session), index=vol.index)
    cum_pv = _session_cumsum(px * vol, keys, summation)
    cum_v = _session_cumsum(vol, keys, summation)
    return (cum_pv / cum_v.replace(0.0, np.nan)).rename("vwap")


def _session_cumsum(x: pd.Series, keys: pd.Series, summation: Summation) -> pd.Series:
    """Per-session running sum; NaN rows stay NaN and contribute 0 (pandas skipna)."""
    if summation == "kahan":
        return x.groupby(keys, sort=False).cumsum()
    if summation != "naive":
        raise ValueError(f"unknown summation: {summation!r}")
    vals = x.to_numpy(dtype=float)
    out = np.empty_like(vals)
    for idx in keys.groupby(keys, sort=False).indices.values():
        out[idx] = np.nancumsum(vals[idx])
    out[np.isnan(vals)] = np.nan
    return pd.Series(out, index=x.index)


def opening_range(
    bars: pd.DataFrame,
    length: int,
    *,
    by: OrBy = "clock",
    start_min: int = RTH_OPEN_MIN,
    session_col: str = "session_date",
    clock_col: str = "clock_min",
    order_col: str | None = "datetime",
    min_bars: int = 1,
) -> pd.DataFrame:
    """Opening range per session.

    ``by="clock"``: bars whose open minute is in ``[start_min, start_min + length)``
    (half-open: a 09:45 bar is NOT in a 15-minute OR). ``by="bars"``: the first
    ``length`` bars of each session in ``order_col`` order (a missing bar stretches
    the window). Sessions with fewer than ``min_bars`` bars in the
    window are dropped (repo copies use 1, 0.8·N, N/2, N — pass yours).

    Returns one row per session (index = session key): ``or_high, or_low, or_width,
    or_mid, or_width_bps, or_open, or_close, or_bars``.
    """
    length = int(length)
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    frame = bars.sort_values(order_col, kind="stable") if order_col else bars
    if by == "clock":
        clock = frame[clock_col]
        win = frame[(clock >= int(start_min)) & (clock < int(start_min) + length)]
    elif by == "bars":
        win = frame.groupby(session_col, sort=False).head(length)
    else:
        raise ValueError(f"unknown by: {by!r}")
    g = win.groupby(session_col, sort=True)
    out = pd.DataFrame(
        {
            "or_high": g["high"].max().astype(float),
            "or_low": g["low"].min().astype(float),
            "or_open": g["open"].first().astype(float) if "open" in win else np.nan,
            "or_close": g["close"].last().astype(float) if "close" in win else np.nan,
            "or_bars": g.size().astype(int),
        }
    )
    out = out[out["or_bars"] >= int(min_bars)]
    out["or_width"] = out["or_high"] - out["or_low"]
    out["or_mid"] = (out["or_high"] + out["or_low"]) / 2.0
    out["or_width_bps"] = (out["or_width"] / out["or_mid"].where(out["or_mid"] > 0)) * 10_000.0
    return out


__all__ = [
    "OrBy",
    "RTH_OPEN_MIN",
    "Summation",
    "VwapPrice",
    "opening_range",
    "session_vwap",
    "vwap_price",
]
