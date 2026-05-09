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
    kernels = np.power(phi, np.arange(n, dtype=float))
    out = np.convolve(volatility * eps, kernels, mode="full")[:n]
    if drift != 0.0:
        out = out + drift
    return out


def build_ohlcv(
    start_price: float,
    log_returns: np.ndarray,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    close = start_price * np.exp(np.cumsum(log_returns))
    if close.size == 0:
        empty = np.array([], dtype=float)
        return {"open": empty, "high": empty, "low": empty, "close": empty, "volume": empty}

    open_price = np.empty_like(close)
    open_price[0] = start_price
    open_price[1:] = close[:-1]

    rng = np.random.default_rng(seed)
    raw_width = np.abs(rng.normal(loc=0.0, scale=0.35, size=close.size))
    half_body = np.clip(np.abs(close - open_price), 1e-12, np.inf)
    wick = half_body * (1.0 + raw_width)
    high = np.maximum(open_price, close) + wick
    low = np.maximum(1e-12, np.minimum(open_price, close) - wick)
    volume = np.exp(rng.normal(loc=12.0, scale=0.35, size=close.size))

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
