"""Optional GARCH-family model path.

The module keeps imports of optional statistical dependencies lazy to keep the
core package install lightweight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import SyntheticDependencyError, SyntheticValidationError
from ..kernels import build_ohlcv
from ..types import SyntheticContext


try:
    from arch import arch_model  # type: ignore[unused-ignore]
except Exception:  # pragma: no cover - optional dependency guard
    arch_model = None  # type: ignore[assignment]


def _require_arch() -> None:
    if arch_model is None:
        raise SyntheticDependencyError(
            "Optional dependency 'arch' is required for garch model. "
            "Install with `pip install qik-synthetic[models]`."
        )


@dataclass
class GARCHModel:
    model_name: str = "garch"
    required_trading_hours: tuple[str, ...] | None = None
    _volatility: float = 0.01

    def fit(
        self,
        data: Sequence[Mapping[str, object]] | None = None,
        *,
        context: SyntheticContext | None = None,
    ):
        _require_arch()
        if data:
            closes = np.asarray([float(item.get("close")) for item in data if "close" in item], dtype=float)
            if closes.size > 1:
                rets = np.diff(np.log(np.maximum(1e-12, closes)))
                if rets.size > 1:
                    self._volatility = float(np.std(rets, ddof=1) + 1e-12)
                else:
                    self._volatility = float(np.abs(rets[0]) + 1e-12)
            elif closes.size == 1:
                self._volatility = 0.01
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
        _require_arch()
        params = params or {}
        drift = float(params.get("drift", 0.0))
        volatility = float(params.get("volatility", self._volatility))
        rng = np.random.default_rng(seed)
        return drift + volatility * rng.normal(size=n)

    def simulate_batch(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, object] | None = None,
    ) -> dict[str, np.ndarray]:
        if n <= 0:
            return {
                "open": np.array([], dtype=float),
                "high": np.array([], dtype=float),
                "low": np.array([], dtype=float),
                "close": np.array([], dtype=float),
                "volume": np.array([], dtype=float),
            }

        _require_arch()
        params = params or {}
        start_price = params.get("start_price")
        if start_price is None:
            raise SyntheticValidationError("simulate_batch requires start_price")
        returns = self.simulate_returns(ctx, n, seed=seed, params=params)
        return build_ohlcv(
            float(start_price),
            returns,
            seed=seed,
            wick_scale=float(params.get("wick_scale", 1.0)),
            volume_loc=float(params.get("volume_loc", 12.0)),
            volume_scale=float(params.get("volume_scale", 0.35)),
        )
