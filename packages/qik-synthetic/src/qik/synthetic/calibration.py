"""Calibration helpers for fitting synthetic parameters from OHLCV frames."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .errors import SyntheticValidationError


_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _finite_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise SyntheticValidationError(f"missing required column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if values.size == 0:
        raise SyntheticValidationError("calibration frame must contain at least one row")
    if not np.isfinite(values).all():
        raise SyntheticValidationError(f"column contains non-finite values: {column}")
    return values


def calibrate_ohlcv_params(frame: pd.DataFrame) -> dict[str, Any]:
    """Estimate model parameters from an OHLCV dataframe.

    The returned mapping is intended for `SyntheticRequest.model_params`.
    It calibrates the parameters already understood by built-in models and the
    OHLCV kernel: return drift/volatility, AR(1) phi, wick scale, and lognormal
    volume distribution.
    """

    if not isinstance(frame, pd.DataFrame):
        raise SyntheticValidationError("calibrate_ohlcv_params expects a pandas DataFrame")
    _reject_multi_identity(frame)
    order = None
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        if not timestamps.is_monotonic_increasing:
            order = np.argsort(timestamps.to_numpy())

    open_ = _finite_series(frame, "open")
    high = _finite_series(frame, "high")
    low = _finite_series(frame, "low")
    close = _finite_series(frame, "close")
    volume = _finite_series(frame, "volume")
    if order is not None:
        open_ = open_[order]
        high = high[order]
        low = low[order]
        close = close[order]
        volume = volume[order]

    if (open_ <= 0).any() or (high <= 0).any() or (low <= 0).any() or (close <= 0).any():
        raise SyntheticValidationError("OHLC prices must be positive")
    if (volume < 0).any():
        raise SyntheticValidationError("volume must be non-negative")
    if (high < np.maximum(open_, close)).any() or (low > np.minimum(open_, close)).any():
        raise SyntheticValidationError("OHLC rows violate high/low bounds")

    log_returns = np.diff(np.log(close))
    if log_returns.size == 0:
        drift = 0.0
        volatility = 0.0
        phi = 0.0
    else:
        drift = float(np.mean(log_returns))
        if log_returns.size > 4:
            q05, q25, q75, q95 = np.quantile(log_returns, [0.05, 0.25, 0.75, 0.95])
            iqr_vol = (q75 - q25) / 1.349
            q90_vol = (q95 - q05) / 3.289707253902945
            mad_vol = np.median(np.abs(log_returns - np.median(log_returns))) / 0.67448975
            volatility = float(max(iqr_vol, q90_vol, mad_vol, 0.0))
        else:
            volatility = float(np.std(log_returns, ddof=1 if log_returns.size > 1 else 0))
        if log_returns.size > 2:
            prev = log_returns[:-1]
            curr = log_returns[1:]
            prev_std = float(np.std(prev))
            curr_std = float(np.std(curr))
            phi = float(np.corrcoef(prev, curr)[0, 1]) if prev_std > 0 and curr_std > 0 else 0.0
            phi = float(np.clip(phi, -0.99, 0.99))
        else:
            phi = 0.0
    ar1_volatility = float(volatility * np.sqrt(max(1.0 - (phi * phi), 0.0)))

    rel_range = (high - low) / close
    rel_body = np.abs(close - open_) / close
    median_range = float(np.median(rel_range))
    median_body = float(np.median(rel_body))
    if median_body <= 0:
        wick_scale = 1.0
    else:
        # build_ohlcv uses wick ~= body/2 * (1 + abs(N(0, 0.35))) * wick_scale
        # on both sides, so this estimates the scale needed to reach the median
        # observed relative range above the candle body.
        wick_scale = max((median_range - median_body) / (2.0 * median_body * 1.28), 0.0)

    positive_volume = volume[volume > 0]
    if positive_volume.size == 0:
        raise SyntheticValidationError("calibration requires at least one positive volume")
    else:
        log_volume = np.log(positive_volume)
        volume_loc = float(np.median(log_volume))
        volume_scale = float(np.std(log_volume, ddof=1 if log_volume.size > 1 else 0))

    return {
        "drift": drift,
        "volatility": max(volatility, 0.0),
        "phi": phi,
        "ar1_volatility": max(ar1_volatility, 0.0),
        "wick_scale": wick_scale,
        "volume_loc": volume_loc,
        "volume_scale": max(volume_scale, 0.0),
    }


def calibrate_many_ohlcv_params(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("symbol", "venue", "dataset", "interval"),
) -> dict[tuple[object, ...], dict[str, Any]]:
    """Estimate calibration params for each identity group in an OHLCV frame."""

    if not isinstance(frame, pd.DataFrame):
        raise SyntheticValidationError("calibrate_many_ohlcv_params expects a pandas DataFrame")
    available = [column for column in group_columns if column in frame.columns]
    if not available:
        return {tuple(): calibrate_ohlcv_params(frame)}
    out: dict[tuple[object, ...], dict[str, Any]] = {}
    for key, group in frame.groupby(available, sort=True, observed=True):
        normalized_key = key if isinstance(key, tuple) else (key,)
        out[normalized_key] = calibrate_ohlcv_params(group)
    return out


__all__ = ["calibrate_ohlcv_params", "calibrate_many_ohlcv_params"]


def _reject_multi_identity(frame: pd.DataFrame) -> None:
    identity_columns = [column for column in ("symbol", "venue", "dataset", "interval") if column in frame.columns]
    if not identity_columns:
        return
    if frame[identity_columns].drop_duplicates().shape[0] > 1:
        raise SyntheticValidationError("calibrate_ohlcv_params expects a single identity; group multi-symbol frames first")
