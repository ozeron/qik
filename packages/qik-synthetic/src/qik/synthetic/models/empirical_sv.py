"""Hybrid empirical stochastic-volatility OHLCV model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..calibration import OhlcvCalibrationProfile, fit_ohlcv_profile
from ..errors import SyntheticValidationError
from ..kernels import bootstrap_feature_indices, build_ohlcv_from_features
from ..types import SyntheticContext


@dataclass
class EmpiricalSVModel:
    """Empirical SV/regime bootstrap with coupled candle and volume features."""

    model_name: str = "empirical_sv"
    required_trading_hours: tuple[str, ...] | None = None
    _features: dict[str, np.ndarray] = field(default_factory=dict)
    _params: dict[str, Any] = field(default_factory=dict)
    _mean_block_size: float = 48.0

    def fit(
        self,
        data: Sequence[Mapping[str, Any]] | Any | None = None,
        *,
        context: SyntheticContext | None = None,
    ):
        if data is None:
            return self
        if isinstance(data, OhlcvCalibrationProfile):
            self._features = {key: np.asarray(value) for key, value in data.features.items()}
            self._params = dict(data.model_params)
            self._mean_block_size = float(data.model_params.get("bootstrap_mean_block_size", 48.0))
            return self
        if hasattr(data, "columns"):
            profile = fit_ohlcv_profile(data)
            self._features = {key: np.asarray(value) for key, value in profile.features.items()}
            self._params = dict(profile.model_params)
            self._mean_block_size = float(profile.model_params.get("bootstrap_mean_block_size", 48.0))
            return self
        rows = list(data)
        if rows:
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise SyntheticValidationError("pandas is required to fit empirical_sv model from rows") from exc
            return self.fit(pd.DataFrame.from_records(rows), context=context)
        return self

    def simulate_returns(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        params = self._resolve_params(params)
        features = self._resolve_features(params)
        return _simulate_empirical_returns(features, n, seed=seed, params=params)

    def simulate_batch(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        if n <= 0:
            empty = np.array([], dtype=float)
            return {"open": empty, "high": empty, "low": empty, "close": empty, "volume": empty}
        params = self._resolve_params(params)
        if "start_price" not in params:
            raise SyntheticValidationError("simulate_batch requires start_price")
        features = self._resolve_features(params)
        indices = _time_aligned_indices(features, n, params=params, mean_block_size=self._mean_block_size, seed=seed)
        returns = _simulate_empirical_returns(features, n, seed=seed, params=params, indices=indices)
        volume = _conditional_volume(features, returns, indices, params=params)
        return build_ohlcv_from_features(
            float(params["start_price"]),
            open_gap_returns=np.zeros(n, dtype=float),
            intrabar_returns=returns,
            upper_wick_rel=np.asarray(features["upper_wick_rel"], dtype=float)[indices],
            lower_wick_rel=np.asarray(features["lower_wick_rel"], dtype=float)[indices],
            volume=volume,
        )

    def _resolve_features(self, params: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        params = params or {}
        raw_features = params.get("ohlcv_features") or self._features
        if not raw_features:
            raise SyntheticValidationError("empirical_sv requires fitted features or model_params['ohlcv_features']")
        return {key: np.asarray(value) for key, value in raw_features.items()}

    def _resolve_params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(self._params)
        if params:
            merged.update(params)
        return merged


def _time_aligned_indices(
    features: dict[str, np.ndarray],
    n: int,
    *,
    params: Mapping[str, Any],
    mean_block_size: float,
    seed: int | None,
) -> np.ndarray:
    return bootstrap_feature_indices(
        features,
        n,
        params=dict(params),
        mean_block_size=mean_block_size,
        seed=seed,
    )


def _simulate_empirical_returns(
    features: dict[str, np.ndarray],
    n: int,
    *,
    seed: int | None,
    params: Mapping[str, Any],
    indices: np.ndarray | None = None,
) -> np.ndarray:
    source_returns = np.asarray(features["close_to_close_return"], dtype=float)
    if source_returns.size <= 0:
        raise SyntheticValidationError("empirical_sv requires return features")
    rng = np.random.default_rng(seed)
    if indices is None:
        indices = rng.integers(0, source_returns.size, size=n, dtype=int)
    drift = float(params.get("drift", np.median(source_returns)))
    phi1 = float(params.get("empirical_sv_phi1", _acf(source_returns, 1)))
    phi6 = float(params.get("empirical_sv_phi6", 0.25 * _acf(source_returns, 6)))
    phi24 = float(params.get("empirical_sv_phi24", 0.10 * _acf(source_returns, 24)))
    phi1 = float(np.clip(phi1, -0.35, 0.35))
    phi6 = float(np.clip(phi6, -0.15, 0.15))
    phi24 = float(np.clip(phi24, -0.08, 0.08))
    centered = source_returns - drift
    rolling_vol = np.asarray(features.get("rolling_24h_vol", np.full(source_returns.size, np.std(centered))), dtype=float)
    scale_floor = max(float(np.median(rolling_vol[rolling_vol > 0])) if (rolling_vol > 0).any() else float(np.std(centered)), 1e-12)
    standardized = centered / np.maximum(rolling_vol, scale_floor)
    shocks = standardized[indices]
    regimes = np.asarray(features.get("volatility_regime", np.ones(source_returns.size)), dtype=int)
    regime_scale = np.ones(4, dtype=float)
    for regime in range(4):
        mask = regimes == regime
        if mask.any():
            regime_scale[regime] = max(float(np.median(np.abs(centered[mask]))) / 0.67448975, scale_floor)
    state_scale = regime_scale[np.clip(regimes[indices], 0, 3)]
    out = np.empty(n, dtype=float)
    for idx in range(n):
        lag1 = out[idx - 1] - drift if idx >= 1 else 0.0
        lag6 = out[idx - 6] - drift if idx >= 6 else 0.0
        lag24 = out[idx - 24] - drift if idx >= 24 else 0.0
        out[idx] = drift + (phi1 * lag1) + (phi6 * lag6) + (phi24 * lag24) + (state_scale[idx] * shocks[idx])
    max_abs = float(params.get("empirical_sv_return_clip", params.get("empirical_sv_max_abs_return", 0.35)))
    return np.clip(out, -max_abs, max_abs)


def _conditional_volume(
    features: dict[str, np.ndarray],
    returns: np.ndarray,
    indices: np.ndarray,
    *,
    params: Mapping[str, Any],
) -> np.ndarray:
    log_volume = _required_feature(features, "log_volume")
    base_loc = float(params.get("volume_loc_recent", params.get("volume_loc", np.median(log_volume))))
    hour_effect = np.asarray(features.get("log_volume_hour_effect", np.zeros(24)), dtype=float)
    source_hours = np.asarray(features.get("hour_of_day", np.zeros(log_volume.size)), dtype=int)
    sampled_hours = np.clip(source_hours[indices], 0, 23)
    residual_source = np.asarray(features.get("log_volume_hour_residual", log_volume - np.median(log_volume)), dtype=float)
    residual = residual_source[indices].copy()
    abs_returns = np.abs(returns)
    std_abs = (abs_returns - np.median(abs_returns)) / max(float(np.std(abs_returns)), 1e-12)
    source_range = _required_feature(features, "rel_range")
    rel_range = source_range[indices]
    std_range = (rel_range - np.median(rel_range)) / max(float(np.std(rel_range)), 1e-12)
    rho = float(params.get("empirical_sv_volume_residual_rho", 0.35))
    out_resid = np.empty_like(residual)
    prev = 0.0
    for idx, value in enumerate(residual):
        prev = (rho * prev) + ((1.0 - rho) * value)
        out_resid[idx] = prev
    beta_abs = float(params.get("empirical_sv_volume_beta_abs_return", 0.08))
    beta_range = float(params.get("empirical_sv_volume_beta_range", 0.04))
    log_out = base_loc + hour_effect[sampled_hours] + out_resid + beta_abs * std_abs + beta_range * std_range
    return np.exp(np.clip(log_out, -32.0, 32.0))


def _acf(values: np.ndarray, lag: int) -> float:
    if values.size <= lag or lag <= 0:
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    if float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _required_feature(features: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in features:
        raise SyntheticValidationError(f"empirical_sv requires fitted ohlcv_features['{name}']")
    return np.asarray(features[name], dtype=float)
