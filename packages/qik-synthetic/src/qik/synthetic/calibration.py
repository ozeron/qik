"""Calibration helpers for fitting synthetic parameters from OHLCV frames."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import fingerprint_candle_table
from .errors import SyntheticValidationError


@dataclass(frozen=True)
class OhlcvCalibrationProfile:
    """Structured OHLCV calibration profile for realism models."""

    identity: dict[str, str | None]
    row_count: int
    timestamp_start: object | None
    timestamp_end: object | None
    interval: str | None
    return_stats: dict[str, float]
    dependence_stats: dict[str, float]
    candle_stats: dict[str, float]
    volume_stats: dict[str, float]
    feature_frame_fingerprint: str
    model_params: dict[str, Any]
    features: dict[str, np.ndarray]


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
    """Estimate scalar model parameters from an OHLCV dataframe."""

    return dict(fit_ohlcv_profile(frame).model_params)


def calibrate_many_ohlcv_params(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("symbol", "venue", "dataset", "interval"),
) -> dict[tuple[object, ...], dict[str, Any]]:
    """Estimate scalar calibration params for each identity group."""

    return {key: dict(profile.model_params) for key, profile in fit_many_ohlcv_profiles(frame, group_columns=group_columns).items()}


def fit_ohlcv_profile(frame: pd.DataFrame, *, model: str = "generic") -> OhlcvCalibrationProfile:
    """Fit a structured calibration profile from one OHLCV identity."""

    if not isinstance(frame, pd.DataFrame):
        raise SyntheticValidationError("fit_ohlcv_profile expects a pandas DataFrame")
    _reject_multi_identity(frame)
    prepared = _prepare_frame(frame)
    open_ = _finite_series(prepared, "open")
    high = _finite_series(prepared, "high")
    low = _finite_series(prepared, "low")
    close = _finite_series(prepared, "close")
    volume = _finite_series(prepared, "volume")
    _validate_ohlcv_arrays(open_, high, low, close, volume)
    features = extract_ohlcv_features(prepared)
    log_returns = features["close_to_close_return"]
    drift, volatility, robust_volatility, phi, ar1_volatility = _return_params(log_returns)
    positive_volume = volume[volume > 0]
    if positive_volume.size == 0:
        raise SyntheticValidationError("calibration requires at least one positive volume")
    log_volume = np.log(positive_volume)
    volume_loc = float(np.median(log_volume))
    volume_scale = float(np.std(log_volume, ddof=1 if log_volume.size > 1 else 0))
    rel_range = (high - low) / close
    rel_body = np.abs(close - open_) / close
    median_body = float(np.median(rel_body))
    median_range = float(np.median(rel_range))
    wick_scale = 1.0 if median_body <= 0 else max((median_range - median_body) / (2.0 * median_body * 1.28), 0.0)
    return_stats = {
        "drift": drift,
        "volatility_raw": volatility,
        "volatility_robust": robust_volatility,
        "phi": phi,
        "return_q01": _quantile(log_returns, 0.01),
        "return_q05": _quantile(log_returns, 0.05),
        "return_q50": _quantile(log_returns, 0.50),
        "return_q95": _quantile(log_returns, 0.95),
        "return_q99": _quantile(log_returns, 0.99),
    }
    dependence_stats = {
        "return_acf_lag_1": _acf(log_returns, 1),
        "abs_return_acf_lag_1": _acf(np.abs(log_returns), 1),
        "abs_return_acf_lag_24": _acf(np.abs(log_returns), 24),
    }
    candle_stats = {
        "range_median": median_range,
        "range_q95": float(np.quantile(rel_range, 0.95)),
        "upper_wick_median": float(np.median(features["upper_wick_rel"])),
        "lower_wick_median": float(np.median(features["lower_wick_rel"])),
    }
    volume_stats = {
        "volume_loc": volume_loc,
        "volume_scale": max(volume_scale, 0.0),
        "volume_median": float(np.median(volume)),
        "log_volume_q95": float(np.quantile(np.log(np.maximum(volume, 1e-12)), 0.95)),
        "hourly_volume_profile_distance": 0.0,
    }
    model_params = {
        "drift": drift,
        "volatility": max(robust_volatility, 0.0),
        "volatility_raw": max(volatility, 0.0),
        "phi": phi,
        "ar1_volatility": max(ar1_volatility, 0.0),
        "wick_scale": wick_scale,
        "volume_loc": volume_loc,
        "volume_scale": max(volume_scale, 0.0),
        "ohlcv_features": features,
        "bootstrap_mean_block_size": 48,
    }
    identity = _identity(prepared)
    return OhlcvCalibrationProfile(
        identity=identity,
        row_count=int(len(prepared)),
        timestamp_start=prepared["timestamp"].iloc[0] if "timestamp" in prepared.columns else None,
        timestamp_end=prepared["timestamp"].iloc[-1] if "timestamp" in prepared.columns else None,
        interval=identity.get("interval"),
        return_stats=return_stats,
        dependence_stats=dependence_stats,
        candle_stats=candle_stats,
        volume_stats=volume_stats,
        feature_frame_fingerprint=_feature_frame_fingerprint(prepared),
        model_params=model_params,
        features=features,
    )


def fit_many_ohlcv_profiles(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("symbol", "venue", "dataset", "interval"),
) -> dict[tuple[object, ...], OhlcvCalibrationProfile]:
    """Fit calibration profiles for each identity group."""

    if not isinstance(frame, pd.DataFrame):
        raise SyntheticValidationError("fit_many_ohlcv_profiles expects a pandas DataFrame")
    available = [column for column in group_columns if column in frame.columns]
    if not available:
        return {tuple(): fit_ohlcv_profile(frame)}
    out: dict[tuple[object, ...], OhlcvCalibrationProfile] = {}
    for key, group in frame.groupby(available, sort=True, observed=True, dropna=False):
        out[_normalize_group_key(key)] = fit_ohlcv_profile(group)
    return out


def extract_ohlcv_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract model-ready OHLCV feature arrays."""

    prepared = _prepare_frame(frame)
    open_ = _finite_series(prepared, "open")
    high = _finite_series(prepared, "high")
    low = _finite_series(prepared, "low")
    close = _finite_series(prepared, "close")
    volume = _finite_series(prepared, "volume")
    _validate_ohlcv_arrays(open_, high, low, close, volume)
    prev_close = np.empty_like(close)
    prev_close[0] = open_[0]
    prev_close[1:] = close[:-1]
    timestamp = pd.to_datetime(prepared["timestamp"], utc=True) if "timestamp" in prepared.columns else None
    return {
        "close_to_close_return": np.diff(np.log(close), prepend=np.log(open_[0])),
        "open_gap_return": np.log(open_ / np.maximum(prev_close, 1e-12)),
        "intrabar_return": np.log(close / open_),
        "upper_wick_rel": np.maximum(high / np.maximum(open_, close) - 1.0, 0.0),
        "lower_wick_rel": np.maximum(np.minimum(open_, close) / np.maximum(low, 1e-12) - 1.0, 0.0),
        "rel_range": (high - low) / close,
        "log_volume": np.log(np.maximum(volume, 1e-12)),
        "volume": volume.astype(float),
        "hour_of_day": timestamp.dt.hour.to_numpy(dtype=float) if timestamp is not None else np.zeros(len(close), dtype=float),
        "day_of_week": timestamp.dt.dayofweek.to_numpy(dtype=float) if timestamp is not None else np.zeros(len(close), dtype=float),
    }


def _feature_frame_fingerprint(frame: pd.DataFrame) -> str:
    if "timestamp" in frame.columns:
        return fingerprint_candle_table(frame)
    columns = ("open", "high", "low", "close", "volume")
    values = frame.loc[:, columns].to_numpy(dtype=float)
    return blake2b(values.tobytes(), digest_size=16).hexdigest()


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
        if not prepared["timestamp"].is_monotonic_increasing:
            prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    return prepared


def _validate_ohlcv_arrays(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> None:
    if (open_ <= 0).any() or (high <= 0).any() or (low <= 0).any() or (close <= 0).any():
        raise SyntheticValidationError("OHLC prices must be positive")
    if (volume < 0).any():
        raise SyntheticValidationError("volume must be non-negative")
    if (high < np.maximum(open_, close)).any() or (low > np.minimum(open_, close)).any():
        raise SyntheticValidationError("OHLC rows violate high/low bounds")


def _return_params(log_returns: np.ndarray) -> tuple[float, float, float, float, float]:
    if log_returns.size <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    returns = log_returns[1:] if log_returns.size > 1 else log_returns
    drift = float(np.mean(returns))
    volatility = float(np.std(returns, ddof=1 if returns.size > 1 else 0))
    if returns.size > 4:
        q05, q25, q75, q95 = np.quantile(returns, [0.05, 0.25, 0.75, 0.95])
        iqr_vol = (q75 - q25) / 1.349
        q90_vol = (q95 - q05) / 3.289707253902945
        mad_vol = np.median(np.abs(returns - np.median(returns))) / 0.67448975
        robust_volatility = float(max(iqr_vol, q90_vol, mad_vol, 0.0))
    else:
        robust_volatility = volatility
    if returns.size > 2:
        prev = returns[:-1]
        curr = returns[1:]
        phi = float(np.corrcoef(prev, curr)[0, 1]) if np.std(prev) > 0 and np.std(curr) > 0 else 0.0
        phi = float(np.clip(phi, -0.99, 0.99))
    else:
        phi = 0.0
    ar1_volatility = float(robust_volatility * np.sqrt(max(1.0 - (phi * phi), 0.0)))
    return drift, volatility, robust_volatility, phi, ar1_volatility


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _acf(values: np.ndarray, lag: int) -> float:
    if values.size <= lag or lag <= 0:
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _normalize_group_key(key: object) -> tuple[object, ...]:
    raw = key if isinstance(key, tuple) else (key,)
    return tuple(None if pd.isna(value) else value for value in raw)


def _identity(frame: pd.DataFrame) -> dict[str, str | None]:
    return {
        column: (str(frame[column].iloc[0]) if column in frame.columns and pd.notna(frame[column].iloc[0]) else None)
        for column in ("symbol", "venue", "dataset", "interval")
    }


def _reject_multi_identity(frame: pd.DataFrame) -> None:
    identity_columns = [column for column in ("symbol", "venue", "dataset", "interval") if column in frame.columns]
    if not identity_columns:
        return
    if frame[identity_columns].drop_duplicates().shape[0] > 1:
        raise SyntheticValidationError("calibrate_ohlcv_params expects a single identity; group multi-symbol frames first")


__all__ = [
    "OhlcvCalibrationProfile",
    "calibrate_ohlcv_params",
    "calibrate_many_ohlcv_params",
    "extract_ohlcv_features",
    "fit_ohlcv_profile",
    "fit_many_ohlcv_profiles",
]
