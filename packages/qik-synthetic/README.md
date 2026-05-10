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

Model recommendations
---------------------

Use synthetic BTC candles as robustness tests, not as extra truth. Fit realism
models on a training window, select on validation, and keep a sealed real test
window for reporting.

Current BTCUSDT 1h recommendation:

| Use case | Model | Recommendation | Why |
| --- | --- | --- | --- |
| Default calibrated realism | `recommended_realism` / `realistic_btc_1h` / `student_t_rw` | Use first | Best validation-selected score in the current BTCUSDT 1h benchmark; strongest robust volatility and candle-range behavior among the calibrated defaults. |
| Dependent-return baseline | `stationary_bootstrap` | Compare against it | Preserves local historical dependence better than independent random walks, but can replay unstable raw volatility and should not be the only robustness test. |
| Experimental regime robustness | `regime_conditioned_bootstrap` | Keep, but do not make default yet | The wide sweep found `calendar_conditioning=false` with `2d` blocks as the best candidate, including strong report-only test behavior, but validation still lags `student_t_rw` and trend/drawdown labels are not yet influencing paths enough. |
| Optional econometric baseline | `garch` | Keep as a comparison | Useful when `qik-synthetic[models]` is installed, but it was not the closest BTCUSDT 1h model on validation. |
| Smoke tests and adapters | `gaussian_rw`, `ar1` | Use for speed/contract checks | Fast and deterministic, but they miss important tail, dependence, and volume behavior. |
| Early coupled SV research | `empirical_sv` | Keep experimental | Improves some coupling metrics, but current return distribution quality is not good enough for default use. |

Do not drop `regime_conditioned_bootstrap` yet. It is not the recommended
default, but it is useful as a robustness generator and diagnostics surface. The
next research target is to coarsen calendar/regime buckets and sample explicit
segments so trend, drawdown, and volume states affect path continuity more
directly.

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
