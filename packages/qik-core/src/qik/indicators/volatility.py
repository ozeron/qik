"""True range, ATR and Bollinger bands (batch)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from qik.indicators.moving_average import Seed, _check_length, ema, rma, sma

AtrSmoothing = Literal["wilder", "sma", "ema"]
FirstBar = Literal["hl", "nan"]
BandMid = Literal["sma", "ema"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="raise").astype(float)


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    prev_close: pd.Series | None = None,
    first: FirstBar = "hl",
) -> pd.Series:
    """``max(H-L, |H-Cprev|, |L-Cprev|)``.

    ``prev_close`` defaults to ``close.shift(1)``; pass your own to blank rolls
    (NaN → that bar is H-L) or to seed bar 0 with the open.
    ``first="hl"``: where the previous close is missing TR is ``H-L`` (skipna max,
    the repo-wide default). ``first="nan"``: TR is NaN there instead.
    """
    h, lo, c = _num(high), _num(low), _num(close)
    pc = c.shift(1) if prev_close is None else _num(prev_close)
    ranges = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1)
    tr = ranges.max(axis=1, skipna=True)
    if first == "nan":
        tr = tr.where(pc.notna())
    elif first != "hl":
        raise ValueError(f"unknown first: {first!r}")
    return tr.rename("true_range")


def smooth_range(
    tr: pd.Series,
    n: int,
    *,
    smoothing: AtrSmoothing = "wilder",
    seed: Seed = "sma",
    min_periods: int | None = None,
) -> pd.Series:
    """Average a (true) range series — the smoothing half of :func:`atr`."""
    n = _check_length(n)
    if smoothing == "wilder":
        return rma(tr, n, seed=seed, min_periods=min_periods)
    if smoothing == "sma":
        return sma(tr, n, min_periods=min_periods)
    if smoothing == "ema":
        return ema(tr, n, seed=seed, min_periods=min_periods)
    raise ValueError(f"unknown smoothing: {smoothing!r}")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int,
    *,
    smoothing: AtrSmoothing = "wilder",
    seed: Seed = "sma",
    min_periods: int | None = None,
    prev_close: pd.Series | None = None,
    first: FirstBar = "hl",
) -> pd.Series:
    """Average true range. Common variants, each selected explicitly:

    * ``smoothing="wilder", seed="sma"`` — Wilder / Pine ``ta.atr``.
    * ``smoothing="wilder", seed="first"`` — ``ewm(alpha=1/n, adjust=False)``.
    * ``smoothing="sma"`` — rolling mean of TR.
    * ``smoothing="ema", seed="sma"`` — ``alpha=2/(n+1)`` SMA-seeded.

    No shift: callers wanting a prior-bar value use ``.shift(1)`` themselves.
    """
    tr = true_range(high, low, close, prev_close=prev_close, first=first)
    return smooth_range(tr, n, smoothing=smoothing, seed=seed, min_periods=min_periods).rename(
        f"atr_{int(n)}"
    )


def bollinger(
    x: pd.Series,
    n: int,
    k: float = 2.0,
    *,
    ddof: int = 0,
    mid: BandMid = "sma",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Bollinger bands: ``mid ± k * rolling_std(n, ddof)``.

    Repo copies disagree on ``ddof`` (0 = population, Bollinger's original; 1 =
    pandas default) and on the mid (SMA, or an EMA seeded at bar 0) — pass both.
    Returns ``mid, upper, lower, width (= upper-lower), pct_b``.
    """
    n = _check_length(n)
    mp = n if min_periods is None else int(min_periods)
    if mid == "sma":
        centre = sma(x, n, min_periods=mp)
    elif mid == "ema":
        centre = ema(x, n, seed="first", min_periods=0)
    else:
        raise ValueError(f"unknown mid: {mid!r}")
    sd = x.rolling(n, min_periods=mp).std(ddof=int(ddof))
    upper = centre + float(k) * sd
    lower = centre - float(k) * sd
    width = upper - lower
    pct_b = (x - lower) / width.replace(0.0, np.nan)
    return pd.DataFrame(
        {"mid": centre, "upper": upper, "lower": lower, "width": width, "pct_b": pct_b}
    )


__all__ = ["AtrSmoothing", "BandMid", "FirstBar", "atr", "bollinger", "smooth_range", "true_range"]
