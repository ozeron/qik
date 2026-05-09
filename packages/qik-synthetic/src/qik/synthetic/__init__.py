"""QIK Synthetic package."""

from __future__ import annotations

from .backtesting import (
    to_backtesting_frame,
    to_backtesting_payload,
    to_backtesting_payload_from_frame,
    validate_backtesting_payload,
)
from .calibration import calibrate_many_ohlcv_params, calibrate_ohlcv_params
from .context import DefaultMarketContextProvider
from .diagnostics import OhlcvComparisonReport, compare_ohlcv, compare_ohlcv_by_identity, fingerprint_candle_table
from .engine import SyntheticEngine
from .errors import (
    SyntheticConversionError,
    SyntheticContextError,
    SyntheticDependencyError,
    SyntheticError,
    SyntheticValidationError,
)
from .horizon import bars_from_span, generate_timestamps, parse_interval
from .protocols import BacktestingAdapter, MarketContextProvider, PathModel
from .types import (
    DriftConfig,
    MarketSeries,
    NoiseGenerator,
    SyntheticBatch,
    SyntheticChunk,
    SyntheticContext,
    SyntheticMetadata,
    SyntheticRequest,
    build_market_path,
    gaussian_noise,
    market_series_to_batch,
    as_request,
)
from .registry import list_models, resolve_model

__all__ = [
    "SyntheticEngine",
    "DefaultMarketContextProvider",
    "bars_from_span",
    "generate_timestamps",
    "parse_interval",
    "PathModel",
    "MarketContextProvider",
    "BacktestingAdapter",
    "to_backtesting_frame",
    "to_backtesting_payload",
    "to_backtesting_payload_from_frame",
    "validate_backtesting_payload",
    "calibrate_ohlcv_params",
    "calibrate_many_ohlcv_params",
    "fingerprint_candle_table",
    "compare_ohlcv",
    "compare_ohlcv_by_identity",
    "OhlcvComparisonReport",
    "resolve_model",
    "list_models",
    "SyntheticContext",
    "SyntheticRequest",
    "SyntheticBatch",
    "SyntheticChunk",
    "SyntheticMetadata",
    "SyntheticError",
    "SyntheticValidationError",
    "SyntheticContextError",
    "SyntheticConversionError",
    "SyntheticDependencyError",
    "DriftConfig",
    "MarketSeries",
    "NoiseGenerator",
    "build_market_path",
    "gaussian_noise",
    "market_series_to_batch",
    "as_request",
]
