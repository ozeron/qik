"""Student-t random walk baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..calibration import OhlcvCalibrationProfile, fit_ohlcv_profile
from ..errors import SyntheticValidationError
from ..kernels import bootstrap_feature_indices, build_ohlcv, build_ohlcv_from_features
from ..types import SyntheticContext


@dataclass
class StudentTRandomWalkModel:
    model_name: str = "student_t_rw"
    required_trading_hours: tuple[str, ...] | None = None
    _features: dict[str, np.ndarray] = field(default_factory=dict)
    _params: dict[str, object] = field(default_factory=dict)

    def fit(self, data: Sequence[Mapping[str, object]] | None = None, *, context: SyntheticContext | None = None):
        if isinstance(data, OhlcvCalibrationProfile):
            self._features = {key: np.asarray(value) for key, value in data.features.items()}
            self._params = dict(data.model_params)
        elif hasattr(data, "columns"):
            profile = fit_ohlcv_profile(data)
            self._features = {key: np.asarray(value) for key, value in profile.features.items()}
            self._params = dict(profile.model_params)
        elif data is not None:
            rows = list(data)
            if rows:
                try:
                    import pandas as pd
                except ModuleNotFoundError as exc:  # pragma: no cover
                    raise SyntheticValidationError("pandas is required to fit student_t_rw model from rows") from exc
                profile = fit_ohlcv_profile(pd.DataFrame.from_records(rows))
                self._features = {key: np.asarray(value) for key, value in profile.features.items()}
                self._params = dict(profile.model_params)
        return self

    def simulate_returns(self, ctx: SyntheticContext, n: int, *, seed: int | None = None, params: Mapping[str, object] | None = None) -> np.ndarray:
        if n <= 0:
            raise SyntheticValidationError("n must be positive")
        params = self._resolve_params(params)
        drift = float(params.get("drift", 0.0))
        volatility = float(params.get("volatility", 0.01))
        df = max(float(params.get("student_t_df", 5.0)), 2.1)
        rng = np.random.default_rng(seed)
        noise = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))
        return drift + volatility * noise

    def simulate_batch(self, ctx: SyntheticContext, n: int, *, seed: int | None = None, params: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
        params = self._resolve_params(params)
        if "start_price" not in params:
            raise SyntheticValidationError("simulate_batch requires start_price")
        returns = self.simulate_returns(ctx, n, seed=seed, params=params)
        features = params.get("ohlcv_features") or self._features
        if features and params.get("use_feature_candles", True):
            indices = bootstrap_feature_indices(features, n, params=params, seed=seed)
            return build_ohlcv_from_features(
                float(params["start_price"]),
                open_gap_returns=np.zeros(n, dtype=float),
                intrabar_returns=returns,
                upper_wick_rel=np.asarray(features["upper_wick_rel"])[indices],
                lower_wick_rel=np.asarray(features["lower_wick_rel"])[indices],
                volume=np.asarray(features["volume"])[indices],
            )
        return build_ohlcv(
            float(params["start_price"]),
            returns,
            seed=seed,
            wick_scale=float(params.get("wick_scale", 1.0)),
            volume_loc=float(params.get("volume_loc", 12.0)),
            volume_scale=float(params.get("volume_scale", 0.35)),
        )

    def _resolve_params(self, params: Mapping[str, object] | None) -> dict[str, object]:
        merged = dict(self._params)
        if params:
            merged.update(params)
        return merged
