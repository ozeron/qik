"""Student-t random walk baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import SyntheticValidationError
from ..kernels import build_ohlcv
from ..types import SyntheticContext


@dataclass
class StudentTRandomWalkModel:
    model_name: str = "student_t_rw"
    required_trading_hours: tuple[str, ...] | None = None

    def fit(self, data: Sequence[Mapping[str, object]] | None = None, *, context: SyntheticContext | None = None):
        return self

    def simulate_returns(self, ctx: SyntheticContext, n: int, *, seed: int | None = None, params: Mapping[str, object] | None = None) -> np.ndarray:
        if n <= 0:
            raise SyntheticValidationError("n must be positive")
        params = params or {}
        drift = float(params.get("drift", 0.0))
        volatility = float(params.get("volatility", 0.01))
        df = max(float(params.get("student_t_df", 5.0)), 2.1)
        rng = np.random.default_rng(seed)
        noise = rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))
        return drift + volatility * noise

    def simulate_batch(self, ctx: SyntheticContext, n: int, *, seed: int | None = None, params: Mapping[str, object] | None = None) -> dict[str, np.ndarray]:
        params = params or {}
        if "start_price" not in params:
            raise SyntheticValidationError("simulate_batch requires start_price")
        returns = self.simulate_returns(ctx, n, seed=seed, params=params)
        return build_ohlcv(
            float(params["start_price"]),
            returns,
            seed=seed,
            wick_scale=float(params.get("wick_scale", 1.0)),
            volume_loc=float(params.get("volume_loc", 12.0)),
            volume_scale=float(params.get("volume_scale", 0.35)),
        )
