"""Public contracts for pluggable models and adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .types import SyntheticContext


class PathModel(Protocol):
    """Simulation model interface."""

    model_name: str
    required_trading_hours: tuple[str, ...] | None

    def fit(
        self,
        data: Sequence[Mapping[str, Any]] | None = None,
        *,
        context: SyntheticContext | None = None,
    ) -> "PathModel":
        ...

    def simulate_returns(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        ...

    def simulate_batch(
        self,
        ctx: SyntheticContext,
        n: int,
        *,
        seed: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Mapping[str, np.ndarray]:
        ...


class MarketContextProvider(Protocol):
    """Protocol for identity/context resolution."""

    def resolve(
        self,
        *,
        symbol: str,
        venue: str | None = None,
        dataset: str | None = None,
        interval: str | None = None,
    ) -> SyntheticContext:
        ...


class BacktestingAdapter(Protocol):
    """Optional output adapter protocol for integrations."""

    def to_backtesting_frame(self, batch):
        ...

    def to_backtesting_payload(
        self,
        batch,
        *,
        identity_override: Mapping[str, Any] | None = None,
        run_context: Mapping[str, Any] | None = None,
    ):
        ...
