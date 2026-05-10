"""Calibration helpers for fitting synthetic parameters from OHLCV frames."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import fingerprint_candle_table
from .errors import SyntheticValidationError
from .regimes import RegimeLabelConfig, RegimeState, fit_regime_state


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
    regime_state: RegimeState | None = None


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


def fit_ohlcv_profile(
    frame: pd.DataFrame,
    *,
    model: str = "generic",
    regime_config: RegimeLabelConfig | None = None,
) -> OhlcvCalibrationProfile:
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
    calendar_features_present = bool(features.get("calendar_features_present", np.asarray([False], dtype=bool))[0])
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
        "log_volume_hour_iqr": float(np.quantile(features["log_volume_hour_effect"], 0.75) - np.quantile(features["log_volume_hour_effect"], 0.25)),
        "volume_level_last_30d": float(np.median(volume[-720:])) if volume.size else 0.0,
        "volume_level_last_90d": float(np.median(volume[-2160:])) if volume.size else 0.0,
        "hourly_volume_profile_distance": 0.0,
    }
    recent_volume = volume[-min(volume.size, 2160) :]
    volume_loc_recent = float(np.median(np.log(np.maximum(recent_volume, 1e-12)))) if recent_volume.size else volume_loc
    beta_abs, beta_range = _volume_coupling_params(features)
    model_params = {
        "realism_profile_version": "btc_1h_v1",
        "drift": drift,
        "volatility": max(robust_volatility, 0.0),
        "volatility_raw": max(volatility, 0.0),
        "phi": phi,
        "ar1_volatility": max(ar1_volatility, 0.0),
        "wick_scale": wick_scale,
        "volume_loc": volume_loc,
        "volume_loc_recent": volume_loc_recent,
        "volume_scale": max(volume_scale, 0.0),
        "empirical_sv_volume_beta_abs_return": beta_abs,
        "empirical_sv_volume_beta_range": beta_range,
        "ohlcv_features": features,
        "bootstrap_mean_block_size": 48,
        "time_align_bootstrap": calendar_features_present,
        "time_alignment_available": calendar_features_present,
        "student_t_df": _student_t_df_estimate(log_returns),
        "empirical_sv_return_clip": float(np.quantile(np.abs(log_returns[np.isfinite(log_returns)]), 0.995)) if np.isfinite(log_returns).any() else 0.35,
    }
    identity = _identity(prepared)
    timestamps_present = "timestamp" in prepared.columns
    regime_state = (
        fit_regime_state(
            features,
            interval=identity.get("interval") or None,
            timestamps_present=timestamps_present,
            config=regime_config,
        )
        if regime_config is not None
        else None
    )
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
        regime_state=regime_state,
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
    close_to_close = np.diff(np.log(close), prepend=np.log(open_[0]))
    rel_range = (high - low) / close
    body_rel = np.abs(close - open_) / close
    upper_wick_rel = np.maximum(high / np.maximum(open_, close) - 1.0, 0.0)
    lower_wick_rel = np.maximum(np.minimum(open_, close) / np.maximum(low, 1e-12) - 1.0, 0.0)
    rolling_24h_vol = _rolling_std(close_to_close, 24)
    rolling_168h_vol = _rolling_std(close_to_close, 168)
    abs_returns = np.abs(close_to_close)
    regime_basis = np.maximum(rolling_24h_vol, abs_returns)
    volatility_regime = _volatility_regimes(regime_basis)
    log_volume = np.log(np.maximum(volume, 1e-12))
    hours = timestamp.dt.hour.to_numpy(dtype=float) if timestamp is not None else None
    days = timestamp.dt.dayofweek.to_numpy(dtype=float) if timestamp is not None else None
    hour_effect = _hour_effect(log_volume, hours)
    log_volume_baseline = float(np.median(log_volume)) if log_volume.size else 0.0
    if hours is None:
        log_volume_hour_residual = log_volume - log_volume_baseline
    else:
        log_volume_hour_residual = log_volume - log_volume_baseline - np.take(hour_effect, hours.astype(int))
    volume_level_24h = _rolling_median(volume, 24)
    volume_level_168h = _rolling_median(volume, 168)
    features = {
        "close_to_close_return": close_to_close,
        "open_gap_return": np.log(open_ / np.maximum(prev_close, 1e-12)),
        "intrabar_return": np.log(close / open_),
        "upper_wick_rel": upper_wick_rel,
        "lower_wick_rel": lower_wick_rel,
        "rel_range": rel_range,
        "body_rel": body_rel,
        "body_to_range": body_rel / np.maximum(rel_range, 1e-12),
        "upper_wick_share": upper_wick_rel / np.maximum(rel_range, 1e-12),
        "lower_wick_share": lower_wick_rel / np.maximum(rel_range, 1e-12),
        "rolling_24h_vol": rolling_24h_vol,
        "rolling_168h_vol": rolling_168h_vol,
        "volatility_regime": volatility_regime.astype(float),
        "shock_flag": (volatility_regime >= 3).astype(float),
        "log_volume": log_volume,
        "log_volume_hour_effect": hour_effect.astype(float),
        "log_volume_hour_residual": log_volume_hour_residual,
        "volume_level_24h": volume_level_24h,
        "volume_level_168h": volume_level_168h,
        "volume": volume.astype(float),
        "calendar_features_present": np.asarray([timestamp is not None], dtype=bool),
    }
    if hours is not None and days is not None:
        features["hour_of_day"] = hours
        features["day_of_week"] = days
    return features


def _feature_frame_fingerprint(frame: pd.DataFrame) -> str:
    if "timestamp" in frame.columns:
        return fingerprint_candle_table(frame)
    columns = ("open", "high", "low", "close", "volume")
    values = frame.loc[:, columns].to_numpy(dtype=float)
    return blake2b(values.tobytes(), digest_size=16).hexdigest()


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "timestamp" in prepared.columns:
        try:
            prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], utc=True)
        except (TypeError, ValueError) as exc:
            raise SyntheticValidationError("timestamp column contains invalid values") from exc
        if prepared["timestamp"].isna().any():
            raise SyntheticValidationError("timestamp column contains missing values")
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


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)
    series = pd.Series(values)
    out = series.rolling(window=window, min_periods=2).std().bfill().fillna(0.0).to_numpy(dtype=float)
    return out


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)
    return pd.Series(values).rolling(window=window, min_periods=1).median().to_numpy(dtype=float)


def _volatility_regimes(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=int)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.size, dtype=int)
    q30, q75, q95 = np.quantile(finite, [0.30, 0.75, 0.95])
    return np.digitize(values, [q30, q75, q95], right=True).astype(int)


def _hour_effect(log_volume: np.ndarray, hours: np.ndarray | None) -> np.ndarray:
    baseline = float(np.median(log_volume)) if log_volume.size else 0.0
    out = np.zeros(24, dtype=float)
    if hours is None:
        return out
    hour_int = hours.astype(int)
    for hour in range(24):
        mask = hour_int == hour
        out[hour] = float(np.median(log_volume[mask]) - baseline) if mask.any() else 0.0
    return out


def _student_t_df_estimate(log_returns: np.ndarray) -> float:
    returns = log_returns[np.isfinite(log_returns)]
    if returns.size < 32:
        return 5.0
    centered = returns - np.median(returns)
    robust = np.median(np.abs(centered)) / 0.67448975
    if robust <= 0:
        return 8.0
    tail_ratio = float(np.quantile(np.abs(centered), 0.95) / robust)
    if tail_ratio > 3.5:
        return 3.5
    if tail_ratio > 2.8:
        return 5.0
    return 8.0


def _volume_coupling_params(features: dict[str, np.ndarray]) -> tuple[float, float]:
    log_residual = np.asarray(features.get("log_volume_hour_residual", np.zeros(0)), dtype=float)
    returns = np.abs(np.asarray(features.get("close_to_close_return", np.zeros(log_residual.size)), dtype=float))
    ranges = np.asarray(features.get("rel_range", np.zeros(log_residual.size)), dtype=float)
    n = min(log_residual.size, returns.size, ranges.size)
    if n < 8:
        return 0.08, 0.04
    y = log_residual[:n] - float(np.median(log_residual[:n]))
    x_abs = _standardize(returns[:n])
    x_range = _standardize(ranges[:n])
    design = np.column_stack([x_abs, x_range])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.08, 0.04
    return float(np.clip(beta[0], -0.5, 0.5)), float(np.clip(beta[1], -0.5, 0.5))


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.std(values))
    if scale <= 0:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.median(values))) / scale


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
    "RegimeLabelConfig",
    "RegimeState",
    "calibrate_ohlcv_params",
    "calibrate_many_ohlcv_params",
    "extract_ohlcv_features",
    "fit_ohlcv_profile",
    "fit_many_ohlcv_profiles",
]
