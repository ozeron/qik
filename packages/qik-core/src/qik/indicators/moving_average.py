"""Moving averages: SMA, EMA, Wilder RMA, WMA, HMA (batch, pandas in / pandas out).

Every warm-up / seeding choice found in the repo is an explicit parameter:

* ``seed="first"`` — recursion starts at the first finite sample (pandas
  ``ewm(adjust=False)``); ``min_periods`` only masks the warm-up.
* ``seed="sma"`` — Pine / TradingView / Wilder's book: the value at index
  ``n-1`` is the simple mean of the first ``n`` samples, then the recursion.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd

Seed = Literal["first", "sma"]


def _check_length(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError(f"length must be positive, got {n}")
    return n


def sma(x: pd.Series, n: int, *, min_periods: int | None = None) -> pd.Series:
    """Simple moving average; NaN until ``min_periods`` (default ``n``) samples."""
    n = _check_length(n)
    return x.rolling(window=n, min_periods=n if min_periods is None else int(min_periods)).mean()


def _sma_seeded(x: pd.Series, n: int, step: str, alpha: float) -> pd.Series:
    """SMA of the first ``n`` samples at index ``n-1``, then a recursion.

    ``step="wilder"``: ``(prev * (n - 1) + x) / n`` (Wilder's book form).
    ``step="alpha"``: ``alpha * x + (1 - alpha) * prev`` (EMA form).
    The two forms are algebraically equal for ``alpha = 1/n`` but round differently.
    """
    s = pd.to_numeric(x, errors="raise").astype(float)
    values = np.full(len(s), np.nan, dtype=float)
    if len(s) >= n:
        values[n - 1] = float(s.iloc[:n].mean())
        arr = s.to_numpy(dtype=float)
        prev = values[n - 1]
        if step == "wilder":
            for i in range(n, len(arr)):
                prev = (prev * (n - 1) + float(arr[i])) / n
                values[i] = prev
        else:
            for i in range(n, len(arr)):
                prev = alpha * float(arr[i]) + (1.0 - alpha) * prev
                values[i] = prev
    return pd.Series(values, index=s.index)


def ema(
    x: pd.Series,
    n: int,
    *,
    seed: Seed = "first",
    adjust: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """Exponential moving average, ``alpha = 2 / (n + 1)``.

    ``seed="first"`` → ``x.ewm(span=n, adjust=adjust, min_periods=...)``; repo copies
    disagree on ``min_periods`` (0 = value from bar 0, ``n`` = NaN warm-up) — pass it.
    ``seed="sma"`` → Pine ``ta.ema`` (``adjust`` must be False; NaN before ``n-1``).
    """
    n = _check_length(n)
    if seed == "first":
        mp = n if min_periods is None else int(min_periods)
        return x.ewm(span=n, adjust=adjust, min_periods=mp).mean()
    if seed == "sma":
        if adjust:
            raise ValueError("seed='sma' is a recursive EMA; adjust=True is undefined")
        return _sma_seeded(x, n, "alpha", 2.0 / (n + 1.0))
    raise ValueError(f"unknown seed: {seed!r}")


def rma(
    x: pd.Series,
    n: int,
    *,
    seed: Seed = "sma",
    min_periods: int | None = None,
) -> pd.Series:
    """Wilder's running moving average (``alpha = 1 / n``).

    ``seed="sma"`` → classic Wilder / Pine ``ta.rma``: mean of the first ``n`` at
    ``n-1``, then ``(prev*(n-1) + x)/n``. ``min_periods`` is ignored (always ``n``).
    ``seed="first"`` → ``x.ewm(alpha=1/n, adjust=False, min_periods=...)`` (default
    ``min_periods=n``): starts at the first finite sample, so the first ~3n values
    differ from ``seed="sma"``.
    """
    n = _check_length(n)
    if seed == "sma":
        return _sma_seeded(x, n, "wilder", 1.0 / n)
    if seed == "first":
        mp = n if min_periods is None else int(min_periods)
        return x.ewm(alpha=1.0 / n, min_periods=mp, adjust=False).mean()
    raise ValueError(f"unknown seed: {seed!r}")


def wma(x: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average (weights ``1..n``, newest heaviest)."""
    n = _check_length(n)
    weights = np.arange(1, n + 1, dtype=float)
    total = weights.sum()
    return x.rolling(n, min_periods=n).apply(lambda w: float(np.dot(w, weights) / total), raw=True)


def hma(x: pd.Series, n: int) -> pd.Series:
    """Hull MA: ``WMA(2*WMA(n//2) - WMA(n), int(sqrt(n)))`` (both lengths floored, ≥1)."""
    n = _check_length(n)
    half = max(1, n // 2)
    root = max(1, int(math.sqrt(n)))
    return wma(2.0 * wma(x, half) - wma(x, n), root)


__all__ = ["Seed", "ema", "hma", "rma", "sma", "wma"]
