"""Empirical stationary bootstrap OHLCV feature model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..calibration import OhlcvCalibrationProfile, extract_ohlcv_features, fit_ohlcv_profile
from ..errors import SyntheticValidationError
from ..kernels import build_ohlcv_from_features, stationary_bootstrap_indices
from ..types import SyntheticContext


@dataclass
class StationaryBootstrapModel:
    model_name: str = "stationary_bootstrap"
    required_trading_hours: tuple[str, ...] | None = None
    _features: dict[str, np.ndarray] = field(default_factory=dict)
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
            self._mean_block_size = float(data.model_params.get("bootstrap_mean_block_size", 48.0))
            return self
        if hasattr(data, "columns"):
            profile = fit_ohlcv_profile(data)
            self._features = {key: np.asarray(value) for key, value in profile.features.items()}
            self._mean_block_size = float(profile.model_params.get("bootstrap_mean_block_size", 48.0))
            return self
        rows = list(data)
        if rows:
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise SyntheticValidationError("pandas is required to fit bootstrap model from rows") from exc
            profile = fit_ohlcv_profile(pd.DataFrame.from_records(rows))
            self._features = {key: np.asarray(value) for key, value in profile.features.items()}
            self._mean_block_size = float(profile.model_params.get("bootstrap_mean_block_size", 48.0))
        return self

    def simulate_returns(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        features = self._resolve_features(params)
        indices = stationary_bootstrap_indices(
            len(features["intrabar_return"]),
            n,
            mean_block_size=float((params or {}).get("bootstrap_mean_block_size", self._mean_block_size)),
            seed=seed,
        )
        return np.asarray(features["open_gap_return"])[indices] + np.asarray(features["intrabar_return"])[indices]

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
        params = params or {}
        if "start_price" not in params:
            raise SyntheticValidationError("simulate_batch requires start_price")
        features = self._resolve_features(params)
        source_n = len(features["intrabar_return"])
        if source_n <= 0:
            raise SyntheticValidationError("stationary_bootstrap requires fitted OHLCV features")
        indices = stationary_bootstrap_indices(
            source_n,
            n,
            mean_block_size=float(params.get("bootstrap_mean_block_size", self._mean_block_size)),
            seed=seed,
        )
        return build_ohlcv_from_features(
            float(params["start_price"]),
            open_gap_returns=np.asarray(features["open_gap_return"])[indices],
            intrabar_returns=np.asarray(features["intrabar_return"])[indices],
            upper_wick_rel=np.asarray(features["upper_wick_rel"])[indices],
            lower_wick_rel=np.asarray(features["lower_wick_rel"])[indices],
            volume=np.asarray(features["volume"])[indices],
        )

    def _resolve_features(self, params: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        params = params or {}
        raw_features = params.get("ohlcv_features") or self._features
        if not raw_features:
            raise SyntheticValidationError("stationary_bootstrap requires fitted features or model_params['ohlcv_features']")
        return {key: np.asarray(value) for key, value in raw_features.items()}
