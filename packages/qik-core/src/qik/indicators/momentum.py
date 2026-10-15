"""RSI family: Wilder / Cutler RSI, up-down streak, Connors RSI (batch)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from qik.indicators.moving_average import Seed, _check_length, rma

RsiSmoothing = Literal["wilder", "sma"]
ZeroLoss = Literal["guard", "nan"]
FirstDiff = Literal["skip", "zero"]


def rsi(
    close: pd.Series,
    n: int = 14,
    *,
    smoothing: RsiSmoothing = "wilder",
    seed: Seed = "first",
    zero_loss: ZeroLoss = "guard",
    first_diff: FirstDiff = "skip",
    min_periods: int | None = None,
) -> pd.Series:
    """Relative strength index on close-to-close changes.

    * ``smoothing="wilder"`` — RMA of gains/losses. ``seed="first"`` is
      ``ewm(alpha=1/n, adjust=False, min_periods=n)``;
      ``seed="sma"`` is Pine ``ta.rsi``.
    * ``smoothing="sma"`` — Cutler's RSI (rolling mean of gains/losses).
    * ``zero_loss="guard"``: avg loss 0 → 100 (gain > 0) or 50 (flat);
      ``"nan"``: leave NaN.
    * ``first_diff="zero"`` counts bar 0 as a 0/0 observation;
      ``"skip"`` (default) starts at the first real change.
    """
    n = _check_length(n)
    delta = close.diff()
    if first_diff == "zero":
        delta = delta.fillna(0.0)
    elif first_diff != "skip":
        raise ValueError(f"unknown first_diff: {first_diff!r}")
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    if smoothing == "wilder":
        avg_gain = rma(gain, n, seed=seed, min_periods=min_periods)
        avg_loss = rma(loss, n, seed=seed, min_periods=min_periods)
    elif smoothing == "sma":
        mp = n if min_periods is None else int(min_periods)
        avg_gain = gain.rolling(window=n, min_periods=mp).mean()
        avg_loss = loss.rolling(window=n, min_periods=mp).mean()
    else:
        raise ValueError(f"unknown smoothing: {smoothing!r}")
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    if zero_loss == "guard":
        out = out.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
        out = out.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    elif zero_loss != "nan":
        raise ValueError(f"unknown zero_loss: {zero_loss!r}")
    return out


def streak(close: pd.Series) -> pd.Series:
    """Signed run length of consecutive up (+) / down (−) closes; 0 on an unchanged close."""
    sign = np.sign(close.diff().fillna(0.0).to_numpy(dtype=float))
    out = np.zeros(len(sign), dtype=float)
    run = 0.0
    prev = 0.0
    for i, s in enumerate(sign):
        run = float(s) if s == 0 or s != prev else run + float(s)
        prev = float(s)
        out[i] = run
    return pd.Series(out, index=close.index)


def percent_rank(x: pd.Series, n: int) -> pd.Series:
    """Rank of the current value in the trailing ``n`` (incl. itself), ties averaged, ×100."""
    n = _check_length(n)
    return x.rolling(n, min_periods=n).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1] * 100.0, raw=False
    )


def connors_rsi(
    close: pd.Series,
    *,
    rsi_period: int = 3,
    streak_period: int = 2,
    pct_period: int = 100,
) -> pd.Series:
    """Connors RSI = mean(RSI(close), RSI(streak), percent_rank(1-bar ROC)).

    RSIs are Wilder ``seed="first"`` with the 100/50 zero-loss guard. The
    percent-rank includes today and averages ties — Connors' published version
    counts strictly-lower values only (open decision; no live user).
    """
    r = rsi(close, rsi_period)
    streak_rsi = rsi(streak(close), streak_period)
    pct = percent_rank(close.pct_change(1), pct_period)
    return (r + streak_rsi + pct) / 3.0


__all__ = ["FirstDiff", "RsiSmoothing", "ZeroLoss", "connors_rsi", "percent_rank", "rsi", "streak"]
