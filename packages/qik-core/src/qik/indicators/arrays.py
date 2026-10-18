"""Array indicators with explicit recursive arithmetic and missing-value semantics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from qik.indicators.moving_average import _check_length


def true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    session_start: np.ndarray | None = None,
    prev_close: np.ndarray | None = None,
    method: Literal["maximum", "envelope"] = "maximum",
) -> np.ndarray:
    prev = np.r_[np.nan, close[:-1]] if prev_close is None else prev_close
    if method == "envelope":
        out = np.fmax(high, prev) - np.fmin(low, prev)
    elif method == "maximum":
        out = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
        if len(out):
            out[0] = high[0] - low[0]
    else:
        raise ValueError(f"unknown true-range method: {method!r}")
    if session_start is not None:
        out = np.where(session_start, high - low, out)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    n = _check_length(n)
    tr = true_range(high, low, close, prev_close=np.roll(close, 1))
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1] = tr[:n].mean()
        for t in range(n, len(tr)):
            out[t] = (out[t - 1] * (n - 1) + tr[t]) / n
    return out


def ema(close: np.ndarray, n: int) -> np.ndarray:
    """NumPy SMA seed, then the increment form prev + alpha * (x - prev)."""
    n = _check_length(n)
    out = np.full(len(close), np.nan)
    if n <= len(close):
        out[n - 1] = close[:n].mean()
        for t in range(n, len(close)):
            out[t] = out[t - 1] + 2 / (n + 1) * (close[t] - out[t - 1])
    return out


def rsi(
    close: np.ndarray,
    n: int,
    *,
    seed: Literal["sma", "first"] = "sma",
    flat: float = 50.0,
    first_diff: Literal["skip", "zero"] = "skip",
    missing: Literal["propagate", "zero"] = "propagate",
) -> np.ndarray:
    n = _check_length(n)
    if seed not in ("sma", "first"):
        raise ValueError(f"unknown seed: {seed!r}")
    if first_diff not in ("skip", "zero"):
        raise ValueError(f"unknown first_diff: {first_diff!r}")
    if missing not in ("propagate", "zero"):
        raise ValueError(f"unknown missing policy: {missing!r}")
    if len(close) == 0:
        return np.full(0, np.nan)
    delta = (
        np.r_[np.nan, np.diff(close)] if first_diff == "skip" else np.diff(close, prepend=close[0])
    )
    gain, loss = np.maximum(delta, 0.0), np.maximum(-delta, 0.0)
    if missing == "zero":
        gain, loss = np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta, 0.0)
    if seed == "first":
        ag = pd.Series(gain).ewm(alpha=1.0 / n, adjust=False, min_periods=0).mean().to_numpy()
        al = pd.Series(loss).ewm(alpha=1.0 / n, adjust=False, min_periods=0).mean().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out = 100.0 - 100.0 / (1.0 + ag / al)
        out[al == 0] = 100.0
        out[(al == 0) & (ag == 0)] = flat
        out[:n] = np.nan
        return out
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    ag, al = float(gain[1 : n + 1].mean()), float(loss[1 : n + 1].mean())
    for t in range(n, len(close)):
        if t > n:
            up = max(delta[t], 0) if missing == "propagate" else gain[t]
            down = max(-delta[t], 0) if missing == "propagate" else loss[t]
            ag = (ag * (n - 1) + up) / n
            al = (al * (n - 1) + down) / n
        out[t] = flat if ag == al == 0 else 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def grouped_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    group: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Final HLC3 VWAP per integer group, using NumPy's sequential bincount sums."""
    tp = (high + low + close) / 3.0
    pv = np.bincount(group, weights=tp * volume, minlength=n_groups)
    vol = np.bincount(group, weights=volume, minlength=n_groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(vol > 0, pv / vol, np.nan)
