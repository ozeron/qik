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
