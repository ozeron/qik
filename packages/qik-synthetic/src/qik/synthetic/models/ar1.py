"""AR(1) return model baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import SyntheticValidationError
from ..kernels import build_ohlcv, simulate_returns_ar1
from ..types import SyntheticContext


@dataclass
class AR1Model:
    model_name: str = "ar1"
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
        volatility = float(params.get("ar1_volatility", params.get("volatility", 0.01)))
        phi = float(params.get("phi", 0.2))
        return simulate_returns_ar1(n, seed=seed, phi=phi, drift=drift, volatility=volatility)

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
        return build_ohlcv(
            start_price,
            returns,
            seed=seed,
            wick_scale=float(params.get("wick_scale", 1.0)),
            volume_loc=float(params.get("volume_loc", 12.0)),
            volume_scale=float(params.get("volume_scale", 0.35)),
        )
