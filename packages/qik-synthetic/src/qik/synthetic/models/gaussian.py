"""Gaussian random-walk style models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import SyntheticValidationError
from ..kernels import bootstrap_feature_indices, build_ohlcv, build_ohlcv_from_features, simulate_log_returns
from ..types import SyntheticContext


@dataclass
class GaussianRandomWalkModel:
    model_name: str = "gaussian_rw"
    required_trading_hours: tuple[str, ...] | None = None

    def fit(
        self,
        data: Sequence[Mapping[str, object]] | None = None,
        *,
        context: SyntheticContext | None = None,
    ):
        return self

    def simulate_returns(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, object] | None = None,
    ) -> np.ndarray:
        if n <= 0:
            raise SyntheticValidationError("n must be positive")
        params = params or {}
        drift = float(params.get("drift", 0.0))
        volatility = float(params.get("volatility", 0.01))
        return simulate_log_returns(n, seed=seed, drift=drift, volatility=volatility)

    def simulate_batch(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, object] | None = None,
    ) -> dict[str, np.ndarray]:
        params = params or {}
        if "start_price" not in params:
            raise SyntheticValidationError("simulate_batch requires start_price")
        start_price = float(params["start_price"])
        returns = self.simulate_returns(ctx, n, seed=seed, params=params)
        features = params.get("ohlcv_features")
        if features is not None and params.get("use_feature_candles", True):
            indices = bootstrap_feature_indices(features, n, params=params, seed=seed)
            return build_ohlcv_from_features(
                start_price,
                open_gap_returns=np.zeros(n, dtype=float),
                intrabar_returns=returns,
                upper_wick_rel=np.asarray(features["upper_wick_rel"])[indices],
                lower_wick_rel=np.asarray(features["lower_wick_rel"])[indices],
                volume=np.asarray(features["volume"])[indices],
            )
        return build_ohlcv(
            start_price,
            returns,
            seed=seed,
            wick_scale=float(params.get("wick_scale", 1.0)),
            volume_loc=float(params.get("volume_loc", 12.0)),
            volume_scale=float(params.get("volume_scale", 0.35)),
        )
