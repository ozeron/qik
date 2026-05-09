"""Core contracts and compatibility shims for synthetic generation.

This module intentionally keeps the existing public helpers used by older callers
while exposing the new protocol-oriented contracts used by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone, timedelta
from hashlib import blake2b
from typing import Any, Mapping, Protocol

from .errors import SyntheticValidationError
from .horizon import generate_timestamps


class NoiseGenerator(Protocol):
    """Noise function signature compatible with legacy callers and adapters."""

    def __call__(self, n: int, *, seed: int | None = None) -> tuple[float, ...]:
        ...


@dataclass(frozen=True)
class DriftConfig:
    """Configuration for a simple drift-and-volatility walk."""

    symbol: str
    start_price: float
    steps: int
    drift: float = 0.0
    volatility: float = 0.01
    seed: int | None = None


@dataclass(frozen=True)
class MarketSeries:
    """Generated synthetic series output."""

    symbol: str
    values: tuple[float, ...]
    seed: int | None = None


@dataclass(frozen=True)
class SyntheticContext:
    """Context used for generation and downstream backtesting."""

    symbol: str | None
    venue: str | None
    dataset: str | None
    interval: str
    asset_class: str
    trading_hours: str = "24x7"
    timezone: str = "UTC"


@dataclass(frozen=True)
class SyntheticRequest:
    """Input contract for a generation request."""

    start_ts: datetime
    interval: str
    start_price: float
    model: str = "gaussian_rw"
    seed: int | None = None
    symbol: str | None = None
    venue: str | None = None
    dataset: str | None = None
    n_bars: int | None = None
    duration: timedelta | None = None
    end_ts: datetime | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    def with_context(self, context: SyntheticContext) -> "SyntheticRequest":
        if self.symbol is None:
            return replace(self, symbol=context.symbol)
        return self


@dataclass(frozen=True)
class SyntheticMetadata:
    """Trace metadata for generated outputs."""

    request_id: str
    run_id: str
    seed: int | None
    model_name: str
    context_id: str


@dataclass(frozen=True)
class SyntheticChunk:
    """One streamed chunk from an engine generation call."""

    bars: tuple[dict[str, Any], ...]
    chunk_index: int
    run_id: str


@dataclass(frozen=True)
class SyntheticBatch:
    """Materialized synthetic output bundle."""

    bars: tuple[dict[str, Any], ...]
    context: SyntheticContext
    request: SyntheticRequest
    metadata: SyntheticMetadata

    @property
    def n_rows(self) -> int:
        return len(self.bars)

    @property
    def columns(self) -> tuple[str, ...]:
        if not self.bars:
            return tuple()
        return tuple(self.bars[0].keys())


def _default_noise(n: int, seed: int | None = None) -> tuple[float, ...]:
    """Compatibility noise helper kept for legacy callers."""

    import numpy as np

    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(-1.0, 1.0, size=n).tolist())


# Compatibility function kept for current callers and docs.
def gaussian_noise(n: int, seed: int | None = None) -> tuple[float, ...]:
    return _default_noise(n, seed)


def _stable_id(*parts: object) -> str:
    payload = repr(tuple(parts)).encode("utf-8")
    return blake2b(payload, digest_size=8).hexdigest()


def build_market_path(config: DriftConfig, *, noise_fn: NoiseGenerator | None = None) -> MarketSeries:
    """Build a synthetic price path using a multiplicative random walk."""

    if config.steps <= 0:
        raise ValueError("steps must be greater than 0")
    if config.volatility < 0:
        raise ValueError("volatility must be non-negative")

    if config.start_price <= 0:
        raise ValueError("start_price must be positive")

    resolver = noise_fn or gaussian_noise
    try:
        noise = resolver(config.steps, seed=config.seed)
    except TypeError:
        noise = resolver(config.steps)
    if len(noise) != config.steps:
        raise ValueError("noise_fn must return exactly config.steps values")

    values: list[float] = [config.start_price]
    for step in noise:
        next_price = values[-1] * (1.0 + config.drift + (config.volatility * step))
        if next_price <= 0:
            raise ValueError("generated non-positive price")
        values.append(next_price)

    return MarketSeries(
        symbol=config.symbol,
        values=tuple(values),
        seed=config.seed,
    )


def market_series_to_batch(series: MarketSeries, request: SyntheticRequest | None = None) -> SyntheticBatch:
    """Adapt legacy `MarketSeries` output into the new batch contract."""

    if request is None:
        request = SyntheticRequest(
            start_ts=datetime(1970, 1, 1, tzinfo=timezone.utc),
            interval="1h",
            start_price=series.values[0] if series.values else 1.0,
            model="gaussian_rw",
            seed=series.seed,
        )

    if request.start_price <= 0:
        raise SyntheticValidationError("start_price must be positive")

    n_bars = max(len(series.values) - 1, 0)
    timestamps = generate_timestamps(
        request.start_ts,
        request.interval,
        n_bars,
        trading_hours=request.model_params.get("trading_hours", "24x7"),
    )

    close_values = series.values[1:] if len(series.values) > 1 else tuple()
    bars: list[dict[str, Any]] = []
    prev = request.start_price
    for idx, close in enumerate(close_values):
        bars.append(
            {
                "timestamp": timestamps[idx],
                "open": float(prev),
                "high": float(max(prev, close) * 1.0001),
                "low": float(max(1e-12, min(prev, close) * 0.9999)),
                "close": float(close),
                "volume": 1.0,
            }
        )
        prev = close

    context = SyntheticContext(
        symbol=request.symbol,
        venue=request.venue,
        dataset=request.dataset,
        interval=request.interval,
        asset_class=request.model_params.get("asset_class", "crypto"),
        trading_hours=request.model_params.get("trading_hours", "24x7"),
        timezone=request.model_params.get("timezone", "UTC"),
    )
    req = replace(
        request,
        n_bars=n_bars,
        symbol=request.symbol,
        venue=request.venue,
        dataset=request.dataset,
    )

    request_id = _stable_id("legacy", series.symbol, len(series.values), request.seed)
    metadata = SyntheticMetadata(
        request_id=request_id,
        run_id=request_id,
        seed=series.seed,
        model_name="legacy-build_market_path",
        context_id=context.symbol or "legacy",
    )
    return SyntheticBatch(
        bars=tuple(bars),
        context=context,
        request=req,
        metadata=metadata,
    )


def as_request(
    *,
    start_ts: datetime,
    interval: str,
    start_price: float,
    model: str = "gaussian_rw",
    seed: int | None = None,
    symbol: str | None = None,
    venue: str | None = None,
    dataset: str | None = None,
    n_bars: int | None = None,
    duration: timedelta | None = None,
    end_ts: datetime | None = None,
    model_params: Mapping[str, Any] | None = None,
) -> SyntheticRequest:
    """Small helper for creating validated requests from plain parameters."""

    return SyntheticRequest(
        start_ts=start_ts,
        interval=interval,
        start_price=start_price,
        model=model,
        seed=seed,
        symbol=symbol,
        venue=venue,
        dataset=dataset,
        n_bars=n_bars,
        duration=duration,
        end_ts=end_ts,
        model_params=dict(model_params or {}),
    )
