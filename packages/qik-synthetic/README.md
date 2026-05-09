QIK Synthetic
=============

This package is part of **QIK (Quant Infrastructure Kit)** and provides
model-oriented synthetic OHLCV generation in the `qik.*` namespace.

Install/use with:

- Package import: `from qik.synthetic import SyntheticEngine, as_request`
- CLI: `qik-synthetic`

Quick start
-----------

```python
from datetime import datetime, timezone
from qik.synthetic import as_request, DefaultMarketContextProvider, SyntheticEngine

request = as_request(
    start_ts=datetime(2020, 1, 1, tzinfo=timezone.utc),
    interval="1h",
    start_price=50000.0,
    model="gaussian_rw",
    n_bars=24,
    seed=123,
    symbol="BTC",
    model_params={"drift": 0.0001, "volatility": 0.01},
)

engine = SyntheticEngine(model="gaussian_rw", context_provider=DefaultMarketContextProvider())
batch = engine.generate(request)
```

The BTC default provider resolves to:
- `symbol="BTCUSDT"`, `venue="binance"`, `dataset="trade_bars"`, `trading_hours="24x7"`

SPY defaults resolve to:
- `symbol="SPY"`, `venue="alpaca"`, `dataset="stock_bars"`, `trading_hours="equity_sessions"`

Streaming and horizons
----------------------

```python
from datetime import datetime, timedelta, timezone
from qik.synthetic import as_request, DefaultMarketContextProvider, SyntheticEngine

request = as_request(
    start_ts=datetime(2020, 1, 1, tzinfo=timezone.utc),
    interval="1m",
    start_price=70000.0,
    symbol="BTCUSDT",
    duration=timedelta(days=10),
    model="ar1",
    model_params={"phi": 0.12, "drift": 0.0, "volatility": 0.014},
    seed=7,
)

engine = SyntheticEngine(model="ar1")
for chunk in engine.iter_generate(request, chunk_size=100_000):
    # consume chunk.bars without materializing the full series
    pass
```

The same helper accepts `end_ts` too, and enforces exactly one time-horizon source:
`n_bars` OR `duration` OR `end_ts`.

Class-based workflow
--------------------

```python
from datetime import datetime, timezone
from qik.synthetic import SyntheticEngine, as_request, DefaultMarketContextProvider

engine = SyntheticEngine(model="garch", context_provider=DefaultMarketContextProvider())
request = as_request(
    start_ts=datetime(2021, 1, 1, tzinfo=timezone.utc),
    interval="15m",
    start_price=50000.0,
    symbol="BTCUSDT",
    n_bars=200,
    seed=13,
)
batch = engine.generate(request)
```

`garch` is optional and only resolves when optional dependencies are installed.
Use `qik-synthetic[models]` for optional math stacks.

Backtesting handoff
-------------------

Identity fields are enforced when converting.

```python
from qik.synthetic import to_backtesting_payload

payload = to_backtesting_payload(batch)
# payload contains: candle_table, metadata (request/run identity), required columns.
```

Compatibility notes
------------------

Legacy helpers are retained for older callers:
- `build_market_path`
- `DriftConfig`
- `MarketSeries`
- `gaussian_noise`

These produce deterministic, old-style outputs and can be wrapped with
`market_series_to_batch` for the new contract.
