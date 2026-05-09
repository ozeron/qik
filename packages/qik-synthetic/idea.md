# qik-synthetic idea

This package is an additive, protocol-first synthetic data engine in the QIK
monorepo.

## Design choices

- Orchestrator layer (`SyntheticEngine`) is class-based and owns request/context
  lifecycles, metadata, seed derivation, and stream/materialize mode.
- Compute layer (`kernels.py`) is function-based and vectorized for
  deterministic and fast paths.
- Context is provider-driven so BTC/SPY defaults stay in `context.py` instead of
  model internals.
- Backtesting conversion is explicit and validated at adapter boundaries.

## Package API contract

1. Create request with `as_request`.
2. Run through `SyntheticEngine` (`generate` or `iter_generate`).
3. Convert with `to_backtesting_payload` only when identity is required.
4. Keep fixtures and payload templates outside package source under
   `tests/fixtures`.

## Compatibility and external integrations

- Existing helper shims remain (`build_market_path`, `DriftConfig`,
  `MarketSeries`, `NoiseGenerator`, `gaussian_noise`).
- Context contracts are intentionally field-compatible with external typing contracts
  (`symbol`, `venue`, `dataset`, `interval`) without importing that package.
- Identity is optional for core generation and validated only when constructing
  backtesting payloads.

## Multi-asset defaults

`DefaultMarketContextProvider` resolves:

- BTC aliases -> `BTCUSDT` / `binance` / `trade_bars` / `crypto` / `24x7`
- SPY alias -> `SPY` / `alpaca` / `stock_bars` / `equity` / `equity_sessions`

## Performance notes

- `iter_generate` should be used for long horizons (`chunk_size` tuned by caller).
- Chunks are seeded from `(request + engine seed + chunk index)` to remain
  reproducible and resumable.
- `generate_many` batches request loops and reuses resolved contexts where
  possible.

## Optional models

- Core package does not hard-require statsmodels/arch.
- `garch` is registered and lazily validated via `qik-synthetic[models]`.

## Progress

Implemented:

- Protocol-first generation API with `SyntheticEngine`, request normalization,
  deterministic metadata, materialized batches, and chunked iteration.
- BTC/SPY context defaults with explicit identity fields for backtesting handoff.
- Backtesting payload conversion and validation for candle-table consumers.
- Diagnostics for comparing OHLCV tables, including finite-value validation,
  candle-shape drift, volume drift, and price-level drift checks.
- Package tests for generation, model registry behavior, payload conversion, and
  diagnostics.

Still open:

- Decide whether diagnostics should stay pandas-only or expose a smaller typed
  protocol for dataframe-like inputs.
- Add a documented real-data calibration/evaluation workflow once the cache
  provenance is confirmed.
- Keep optional heavy model dependencies behind extras unless they become part
  of the default API.

## Local BTC 1m cache

Found cache root:

- `/Users/ozeron/.cache/candle_research/ohlcv_catalog/ohlcv`

Available BTC paths include:

- `BTCUSD.BINANCE/1m`
- `BTCUSDT.BINANCE/1m`
- `BTCUSDT.BINANCE/4h`
- `BTCUSDT.BINANCE/1h`
- `BTCUSDT.BINANCE/1d`

Current BTCUSDT 1m sample:

- File count: `165` parquet files.
- Full local-cache load: `0.854322s`.
- Rows after timestamp de-duplication: `420`.
- Sparse cache range: `2025-09-01 00:00:00+00:00` to
  `2026-01-01 00:59:00+00:00`.
- The sparse range spans `175740` minutes, but only has `420` unique rows, so
  the local cache is not a full one-year 1m dataset.
- Sample file:
  `/Users/ozeron/.cache/candle_research/ohlcv_catalog/ohlcv/BTCUSDT.BINANCE/1m/20260101T000000Z_20260101T010000Z_f6c9593364dce2e45fc0_1.parquet`
- Shape: `60 x 9`.
- Columns: `timestamp`, `symbol`, `venue`, `interval`, `open`, `high`, `low`,
  `close`, `volume`.
- Sample range: `2026-01-01 00:00:00+00:00` to
  `2026-01-01 00:59:00+00:00`.
- First row values are `open=100.0`, `close=100.5`, `volume=1.0`.

This is useful for adapter/diagnostic smoke checks because the schema matches the
backtesting candle contract. The sampled prices look fixture-like rather than
real BTC market levels, so it should not be used as a realism benchmark until the
cache provenance is verified.

## Cache-backed verification

Using the cached BTCUSDT/BINANCE 1m schema:

- Generated a synthetic payload with the same row count as the de-duplicated
  local cache: `420` rows.
- Validated the generated backtesting payload successfully.
- Compared cached real data to itself: diagnostics passed with `0` warnings.
- Compared cached real data to generated synthetic data: diagnostics produced
  `12` metrics and expected drift warnings.

One-year synthetic generation from the cached BTC identity/start price:

- Bars: `525600`.
- Shape: `525600 x 10`.
- Range: `2025-09-01 00:00:00+00:00` to
  `2026-08-31 23:59:00+00:00`.
- Generation + backtesting payload validation runtime: `4.62s` wall clock under
  `/usr/bin/time -l`.
- Maximum resident set size: `629932032` bytes, about `601 MiB`.

The package works for a one-year 1m synthetic generation, but the current local
BTC cache is too sparse to prove one-year real-data calibration.

Preferred local real-data verification target:

- Dataset: `BTCUSDT.BINANCE/1h`.
- Cache path:
  `/Users/ozeron/.cache/candle_research/ohlcv_catalog/ohlcv/BTCUSDT.BINANCE/1h`.
- Files: `322`.
- Rows after timestamp de-duplication: `27022`.
- Range: `2023-04-01 02:00:00+00:00` to
  `2026-04-30 23:00:00+00:00`.
- Gap check: all `27021` timestamp gaps are exactly `1h`.
- Full-cache load runtime: `1.365147s`.
- Synthetic generation + backtesting payload conversion for matching row count:
  `0.177436s`, about `152291` rows/second.
- Generated payload shape: `27022 x 10`.
- Real-vs-real diagnostics: passed with `0` warnings.
- Real-vs-synthetic diagnostics: produced `12` metrics and expected drift
  warnings.

This is the current best cache-backed verification path because it covers about
three years of contiguous BTCUSDT hourly candles.

Model comparison on the same `BTCUSDT.BINANCE/1h` cache:

- `gaussian_rw`: generated `27022` rows in `0.242868s`; diagnostics produced
  `7` warnings.
- `ar1`: generated `27022` rows in `0.556018s`; diagnostics produced `7`
  warnings.
- `garch`: generated `27022` rows in `0.268076s`; diagnostics produced `7`
  warnings.

Current interpretation: these model choices all work operationally, but none are
calibrated to the BTC cache by default. `ar1` improves median price-level drift
versus the default Gaussian run in this sample, but volatility, candle-range, and
volume warnings remain large across all three. The next realism step is
calibration, not switching default model names.

Calibration progress:

- Added `calibrate_ohlcv_params(frame)` for pandas OHLCV frames.
- The calibration estimates robust return volatility, drift, AR(1) phi, candle
  wick scale, and lognormal volume parameters for `SyntheticRequest.model_params`.
- On the preferred `BTCUSDT.BINANCE/1h` cache with `gaussian_rw`, default
  generation produced `7` diagnostics warnings.
- Using calibrated params reduced warnings to `5`.
- Improvements: volume median drift dropped from `209.818209` to `0.000412`,
  candle range median drift dropped from `3.883173` to `0.498297`, and central
  return quantile differences dropped materially.
- Remaining warnings are mostly return standard deviation and price-level drift.
  The cache has fat-tailed/outlier returns, so robust calibration intentionally
  avoids matching raw standard deviation exactly because doing so over-widens
  normal simulated tails.

Open calibration work:

- Add a price-anchoring or bridge mode when synthetic output must preserve real
  start/end price levels.
- Add a heavy-tail or bootstrapped return model if matching both robust quantiles
  and raw volatility is required.
