"""Command-line entrypoint for quick synthetic path generation."""

from __future__ import annotations

from .types import DriftConfig, MarketSeries, build_market_path, gaussian_noise


def build_demo_series() -> MarketSeries:
    config = DriftConfig(
        symbol="BTCUSDT",
        start_price=100.0,
        steps=5,
        drift=0.0007,
        volatility=0.02,
        seed=42,
    )
    return build_market_path(config, noise_fn=lambda n: gaussian_noise(n, seed=config.seed))


def main() -> None:
    series = build_demo_series()
    print(f"{series.symbol}: {series.values}")
