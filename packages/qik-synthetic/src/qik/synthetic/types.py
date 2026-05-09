"""Core types and deterministic path builders for synthetic series."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Protocol, Sequence, Tuple


class NoiseGenerator(Protocol):
    """Protocol for deterministic noise sources used in synthetic path generation."""

    def __call__(self, n: int) -> Sequence[float]:
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
    values: Tuple[float, ...]
    seed: int | None = None


def gaussian_noise(n: int, seed: int | None = None) -> Tuple[float, ...]:
    """Generate deterministic pseudo-normal-like noise values in [-1, 1]."""
    rng = Random(seed)
    return tuple(rng.uniform(-1.0, 1.0) for _ in range(n))


def build_market_path(config: DriftConfig, *, noise_fn: NoiseGenerator = gaussian_noise) -> MarketSeries:
    """Build a synthetic price path using a multiplicative random walk."""
    if config.steps <= 0:
        raise ValueError("steps must be greater than 0")
    if config.volatility < 0:
        raise ValueError("volatility must be non-negative")

    noise = noise_fn(config.steps)
    if len(noise) != config.steps:
        raise ValueError("noise_fn must return exactly config.steps values")

    values: list[float] = [config.start_price]
    for step in noise:
        next_price = values[-1] * (1.0 + config.drift + (config.volatility * step))
        values.append(next_price)

    return MarketSeries(
        symbol=config.symbol,
        values=tuple(values),
        seed=config.seed,
    )

