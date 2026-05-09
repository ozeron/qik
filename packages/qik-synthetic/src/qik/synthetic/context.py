"""Market context resolution for generation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .types import SyntheticContext


_ASSET_DEFAULTS: dict[str, dict[str, str]] = {
    "BTC": {
        "symbol": "BTCUSDT",
        "venue": "binance",
        "dataset": "trade_bars",
        "asset_class": "crypto",
        "trading_hours": "24x7",
    },
    "SPY": {
        "symbol": "SPY",
        "venue": "alpaca",
        "dataset": "stock_bars",
        "asset_class": "equity",
        "trading_hours": "equity_sessions",
    },
}

_SYMBOL_ALIASES = {
    "BTCUSDT": "BTC",
    "XBT": "BTC",
    "BTCUSDC": "BTC",
    "BITCOIN": "BTC",
    "SPY": "SPY",
}


@dataclass(frozen=True)
class DefaultMarketContextProvider:
    """Resolve symbol-aware context defaults used by generators."""

    defaults: dict[str, dict[str, str]] | None = None

    def resolve(
        self,
        *,
        symbol: str,
        venue: str | None = None,
        dataset: str | None = None,
        interval: str | None = None,
    ) -> SyntheticContext:
        if not symbol:
            raise ValueError("symbol must be set or resolvable")

        key = self._normalize_symbol(symbol)
        base = dict((self.defaults or _ASSET_DEFAULTS).get(key, {}))

        resolved_symbol = base.get("symbol", symbol)
        resolved_venue = venue or base.get("venue")
        resolved_dataset = dataset or base.get("dataset")

        return SyntheticContext(
            symbol=resolved_symbol,
            venue=resolved_venue,
            dataset=resolved_dataset,
            interval=interval or base.get("interval", "1h"),
            asset_class=base.get("asset_class", "generic"),
            trading_hours=base.get("trading_hours", "24x7"),
            timezone=base.get("timezone", "UTC"),
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        candidate = symbol.strip().upper()
        return _SYMBOL_ALIASES.get(candidate, candidate)

    def with_overrides(self, **values: Mapping[str, str] | str) -> "DefaultMarketContextProvider":
        merged: dict[str, dict[str, str]] = {key: dict(value) for key, value in _ASSET_DEFAULTS.items()}
        if self.defaults:
            for key, value in self.defaults.items():
                merged[key] = {**merged.get(key, {}), **value}

        for key, value in values.items():
            if isinstance(value, Mapping):
                merged[str(key)] = {**merged.get(str(key), {}), **dict(value)}
                continue

            merged.setdefault(str(key), {})["symbol"] = str(value)

        return DefaultMarketContextProvider(defaults=merged)
