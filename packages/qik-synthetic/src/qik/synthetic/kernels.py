"""Vectorized numerical kernels for synthetic generation."""

from __future__ import annotations

from datetime import datetime
import numpy as np


def simulate_log_returns(
    n: int,
    *,
    seed: int | None,
    drift: float,
    volatility: float,
    noise_scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return drift + (volatility * noise_scale * rng.normal(size=n))


def simulate_returns_ar1(
    n: int,
    *,
    seed: int | None,
    phi: float,
    drift: float,
    volatility: float,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)

    if abs(phi) >= 1:
        raise ValueError("phi must satisfy |phi| < 1")

    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    out = np.empty(n, dtype=float)
    state = 0.0
    for idx, shock in enumerate(eps):
        state = (phi * state) + (volatility * shock)
        out[idx] = state
    if drift != 0.0:
        out = out + drift
    return out


def build_ohlcv(
    start_price: float,
    log_returns: np.ndarray,
    *,
    seed: int | None = None,
    wick_scale: float = 1.0,
    volume_loc: float = 12.0,
    volume_scale: float = 0.35,
) -> dict[str, np.ndarray]:
    close = np.cumsum(log_returns, dtype=np.float64)
    np.exp(close, out=close)
    close *= start_price
    if close.size == 0:
        empty = np.array([], dtype=float)
        return {"open": empty, "high": empty, "low": empty, "close": empty, "volume": empty}

    open_price = np.empty_like(close)
    open_price[0] = start_price
    open_price[1:] = close[:-1]

    rng = np.random.default_rng(seed)
    raw_width = rng.normal(loc=0.0, scale=0.35, size=close.size)
    np.abs(raw_width, out=raw_width)
    wick = close - open_price
    np.abs(wick, out=wick)
    np.maximum(wick, 1e-12, out=wick)
    wick *= 1.0 + raw_width
    wick *= max(float(wick_scale), 0.0)
    high = np.maximum(open_price, close)
    high += wick
    low = np.minimum(open_price, close)
    low -= wick
    np.maximum(low, 1e-12, out=low)
    volume = rng.normal(loc=float(volume_loc), scale=max(float(volume_scale), 0.0), size=close.size)
    np.exp(volume, out=volume)

    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def bars_to_rows(
    timestamps: tuple[datetime, ...],
    ohlcv: dict[str, np.ndarray],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "timestamp": ts,
            "open": float(ohlcv["open"][i]),
            "high": float(ohlcv["high"][i]),
            "low": float(ohlcv["low"][i]),
            "close": float(ohlcv["close"][i]),
            "volume": float(ohlcv["volume"][i]),
        }
        for i, ts in enumerate(timestamps)
    )
