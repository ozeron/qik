## Why

`qik-synthetic` can now generate OHLCV payloads quickly, but the generated paths from `gaussian_rw`, `ar1`, and the current `garch` model remain statistically unlike the real BTCUSDT/BINANCE 1h cache they are intended to model. Diagnostics currently report large drift warnings without clearly separating model-realism failures from path-anchoring expectations, making it hard to know whether the synthetic data is useful for research or backtesting validation.

## What Changes

- Introduce a realism evaluation workflow for BTCUSDT 1h data using chronological train/validation/test splits and multi-seed model comparisons.
- Refactor OHLCV diagnostics into interpretable categories: data quality, return distribution, temporal dependence, candle shape, volume, and price path.
- Add diagnostic price modes so unanchored stochastic generation is not failed by raw price-level drift unless anchored comparison is explicitly requested.
- Add feature-based OHLCV calibration/profile support that captures returns, wick/range behavior, volume behavior, hourly effects, and dependence statistics.
- Add realistic model baselines, starting with stationary/block bootstrap over calibrated OHLCV feature vectors and then parametric fat-tail/volatility models.
- Correct the existing `garch` model so it either performs real GARCH-family simulation through the optional `arch` dependency or is not presented as a GARCH realism model.
- Add benchmark/evaluation scripts and tests that compare model realism by category without making cache-dependent metrics mandatory in normal CI.

## Capabilities

### New Capabilities
- `synthetic-realism-evaluation`: Real-data calibration, model comparison, and diagnostic evaluation workflow for OHLCV synthetic realism.
- `synthetic-realism-models`: Feature-driven candle construction and realistic return/volume model families for OHLCV generation.

### Modified Capabilities

## Impact

- Affected package: `packages/qik-synthetic`.
- Affected APIs: diagnostics, calibration, model registry, model implementations, and optional frame-generation workflows.
- New optional behavior: cache-backed realism benchmark scripts requiring local OHLCV parquet data via environment variable.
- Optional dependency behavior: `garch` may rely more substantially on `qik-synthetic[models]` / `arch` when selected.
- Non-goals: no breaking change to existing `SyntheticEngine.generate`, `iter_generate`, `to_backtesting_payload`, or existing baseline model names unless explicitly documented with compatibility shims.
