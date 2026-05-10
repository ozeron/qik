"""Optional GARCH-family model path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..calibration import OhlcvCalibrationProfile
from ..errors import SyntheticDependencyError, SyntheticValidationError
from ..kernels import bootstrap_feature_indices, build_ohlcv, build_ohlcv_from_features
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
    max_abs_log_return: float = 0.25
    _volatility: float = 0.01
    _drift: float = 0.0
    _result: Any | None = None
    _dropped_return_count: int = 0
    _scale: float = 100.0

    def fit(
        self,
        data: Sequence[Mapping[str, object]] | Any | None = None,
        *,
        context: SyntheticContext | None = None,
    ):
        _require_arch()
        raw_returns = _log_returns_from_data(data)
        returns = _clean_fit_returns(raw_returns, max_abs_log_return=self.max_abs_log_return)
        self._dropped_return_count = int(raw_returns.size - returns.size)
        if returns.size == 0:
            return self
        self._drift = float(np.mean(returns))
        self._volatility = float(np.std(returns, ddof=1 if returns.size > 1 else 0) + 1e-12)
        try:
            model = arch_model(
                returns * self._scale,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="t",
                rescale=False,
            )
            self._result = model.fit(disp="off")
            _validate_garch_stationarity(self._result)
        except Exception as exc:
            self._result = None
            if isinstance(exc, SyntheticValidationError):
                raise
            raise SyntheticValidationError("GARCH fit failed") from exc
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
        if self._result is None:
            raise SyntheticValidationError("GARCH model must be fit before simulation")
        random_state = np.random.get_state()
        try:
            if seed is not None:
                np.random.seed(int(seed) % (2**32 - 1))
            sim_model = arch_model(
                None,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="t",
                rescale=False,
            )
            sim = sim_model.simulate(self._result.params, nobs=n, burn=100)
            values = np.asarray(sim["data"], dtype=float) / self._scale
            return values[:n]
        except Exception as exc:
            raise SyntheticValidationError("GARCH simulation failed") from exc
        finally:
            np.random.set_state(random_state)

    def simulate_batch(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, object] | None = None,
    ) -> dict[str, np.ndarray]:
        if n <= 0:
            empty = np.array([], dtype=float)
            return {"open": empty, "high": empty, "low": empty, "close": empty, "volume": empty}
        _require_arch()
        params = params or {}
        start_price = params.get("start_price")
        if start_price is None:
            raise SyntheticValidationError("simulate_batch requires start_price")
        returns = self.simulate_returns(ctx, n, seed=seed, params=params)
        features = params.get("ohlcv_features")
        if features is not None and params.get("use_feature_candles", True):
            indices = bootstrap_feature_indices(features, n, params=params, seed=seed)
            return build_ohlcv_from_features(
                float(start_price),
                open_gap_returns=np.zeros(n, dtype=float),
                intrabar_returns=returns,
                upper_wick_rel=np.asarray(features["upper_wick_rel"])[indices],
                lower_wick_rel=np.asarray(features["lower_wick_rel"])[indices],
                volume=np.asarray(features["volume"])[indices],
            )
        return build_ohlcv(
            float(start_price),
            returns,
            seed=seed,
            wick_scale=float(params.get("wick_scale", 1.0)),
            volume_loc=float(params.get("volume_loc", 12.0)),
            volume_scale=float(params.get("volume_scale", 0.35)),
        )


def _log_returns_from_data(data: object) -> np.ndarray:
    if data is None:
        return np.zeros(0, dtype=float)
    if isinstance(data, OhlcvCalibrationProfile):
        returns = data.features.get("close_to_close_return")
        if returns is None:
            return np.zeros(0, dtype=float)
        return _finite_returns(np.asarray(returns, dtype=float))
    if hasattr(data, "columns") and "close" in data.columns:
        closes = np.asarray(data["close"], dtype=float)
        if closes.size <= 1:
            return np.zeros(0, dtype=float)
        return _finite_returns(np.diff(np.log(np.maximum(1e-12, closes))))
    rows = list(data) if data is not None else []
    closes = np.asarray([float(item.get("close")) for item in rows if "close" in item], dtype=float)
    if closes.size <= 1:
        return np.zeros(0, dtype=float)
    return _finite_returns(np.diff(np.log(np.maximum(1e-12, closes))))


def _finite_returns(returns: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    return returns


def _clean_fit_returns(returns: np.ndarray, *, max_abs_log_return: float) -> np.ndarray:
    returns = _finite_returns(returns)
    if max_abs_log_return <= 0:
        return returns
    return returns[np.abs(returns) <= float(max_abs_log_return)]


def _validate_garch_stationarity(result: Any) -> None:
    persistence = _garch_persistence(getattr(result, "params", None))
    if persistence is not None and persistence >= 1.0:
        raise SyntheticValidationError(f"GARCH fit is non-stationary: alpha + beta = {persistence:.6f}")


def _garch_persistence(params: object) -> float | None:
    if params is None:
        return None
    alpha = _param_value(params, "alpha[1]")
    beta = _param_value(params, "beta[1]")
    if alpha is None or beta is None:
        return None
    return float(alpha) + float(beta)


def _param_value(params: object, name: str) -> float | None:
    if hasattr(params, "get"):
        value = params.get(name)
        if value is not None:
            return float(value)
    if hasattr(params, "index"):
        try:
            return float(params[name])
        except Exception:
            return None
    return None
