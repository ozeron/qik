"""Regime-conditioned bootstrap OHLCV model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..calibration import OhlcvCalibrationProfile, fit_ohlcv_profile
from ..errors import SyntheticValidationError
from ..kernels import build_ohlcv_from_features
from ..regimes import (
    RegimeGenerationDiagnostics,
    RegimeLabelConfig,
    RegimeState,
    fit_regime_state,
    merge_generation_diagnostics,
    sample_regime_indices,
)
from ..types import SyntheticContext


@dataclass
class RegimeConditionedBootstrapModel:
    """Bootstrap fitted OHLCV feature rows with explicit regime diagnostics."""

    model_name: str = "regime_conditioned_bootstrap"
    required_trading_hours: tuple[str, ...] | None = None
    _features: dict[str, np.ndarray] = field(default_factory=dict)
    _params: dict[str, Any] = field(default_factory=dict)
    _regime_state: RegimeState | None = None
    generation_diagnostics: RegimeGenerationDiagnostics = field(default_factory=RegimeGenerationDiagnostics)

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
            self._regime_state = data.regime_state or fit_regime_state(
                self._features,
                interval=data.interval or (context.interval if context is not None else "1h"),
                timestamps_present=data.timestamp_start is not None,
                config=RegimeLabelConfig(calendar_conditioning=data.timestamp_start is not None),
            )
            return self
        if hasattr(data, "columns"):
            profile = fit_ohlcv_profile(data, regime_config=RegimeLabelConfig(calendar_conditioning="timestamp" in data.columns))
            return self.fit(profile, context=context)
        rows = list(data)
        if rows:
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise SyntheticValidationError("pandas is required to fit regime_conditioned_bootstrap from rows") from exc
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
        features = self._resolve_features(params)
        state = self._resolve_regime_state()
        indices, relaxations = sample_regime_indices(
            state,
            n,
            seed=_sampling_seed(params or {}, seed),
            target_timestamps=tuple((params or {}).get("_timestamps", ())) or None,
            sparse_policy=(params or {}).get("regime_sparse_policy"),
            offset=int((params or {}).get("_block_start", 0)),
        )
        self.generation_diagnostics = merge_generation_diagnostics(
            self.generation_diagnostics,
            state=state,
            indices=indices,
            block_start=int((params or {}).get("_block_start", 0)),
            relaxation_counts=relaxations,
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
        merged = self._resolve_params(params)
        if "start_price" not in merged:
            raise SyntheticValidationError("simulate_batch requires start_price")
        features = self._resolve_features(merged)
        state = self._resolve_regime_state()
        indices, relaxations = sample_regime_indices(
            state,
            n,
            seed=_sampling_seed(merged, seed),
            target_timestamps=tuple(merged.get("_timestamps", ())) or None,
            sparse_policy=merged.get("regime_sparse_policy"),
            offset=int(merged.get("_block_start", 0)),
        )
        block_start = int(merged.get("_block_start", 0))
        self.generation_diagnostics = merge_generation_diagnostics(
            self.generation_diagnostics,
            state=state,
            indices=indices,
            block_start=block_start,
            relaxation_counts=relaxations,
        )
        return build_ohlcv_from_features(
            float(merged["start_price"]),
            open_gap_returns=np.asarray(features["open_gap_return"])[indices],
            intrabar_returns=np.asarray(features["intrabar_return"])[indices],
            upper_wick_rel=np.asarray(features["upper_wick_rel"])[indices],
            lower_wick_rel=np.asarray(features["lower_wick_rel"])[indices],
            volume=np.asarray(features["volume"])[indices],
        )

    def _resolve_params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(self._params)
        if params:
            merged.update(params)
        return merged

    def _resolve_features(self, params: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
        params = params or {}
        raw_features = params.get("ohlcv_features") or self._features
        if not raw_features:
            raise SyntheticValidationError("regime_conditioned_bootstrap requires fitted OHLCV features")
        return {key: np.asarray(value) for key, value in raw_features.items()}

    def _resolve_regime_state(self) -> RegimeState:
        if self._regime_state is None:
            raise SyntheticValidationError("regime_conditioned_bootstrap requires fitted RegimeState")
        return self._regime_state


def _sampling_seed(params: Mapping[str, Any], seed: int | None) -> int | None:
    if "_run_seed" in params:
        return int(params["_run_seed"])
    if seed is not None:
        return int(seed)
    request_seed = params.get("_request_seed")
    return int(request_seed) if request_seed is not None else None
