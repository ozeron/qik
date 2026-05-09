## Why

`qik-synthetic` is currently a toy path generator and does not map cleanly to existing research and backtesting workflows. This change turns it into a model-first synthetic engine that supports long-run BTC and SPY experiments, while keeping the package lightweight.

The goal is to:
- generate fast multi-year candles with deterministic seeds,
- support multiple model families behind a common protocol,
- keep generation core decoupled from market metadata,
- enforce strict validation only when converting to backtesting-ready payloads.

## What Changes

- Keep public generation API class-first where callers pass model families and run requests, while keeping hot path execution function-first and vectorized:
  - `SyntheticEngine` for generation lifecycle.
  - `PathModel` Protocol for swappable families (Gaussian, AR, bootstrap, future GARCH variants).
- Add a concrete request/response contract:
  - `SyntheticRequest`, `SyntheticContext`, `SyntheticBatch`.
- Add `iter_generate` for streaming output (chunked) and `generate` for materialized output.
- Add `generate_many` for grouped batch execution and reduced overhead on multi-N jobs.
- Support time-span oriented generation: `start_ts` + `interval` + either `n_bars` or horizon inputs.
- Add adapter functions for backtesting handoff (`to_backtesting_payload`, `to_backtesting_frame`) with required metadata checks.
- Preserve compatibility with existing symbols (`build_market_path`, `DriftConfig`, `NoiseGenerator`, `MarketSeries`, `gaussian_noise`) as compatibility shims.
- Keep model families optional via extras so core remains small.
- Add a provider-based market context abstraction for BTC/SPY defaults without hard-coding profile behavior in generators.

### Performance decision

- Classes are used as orchestration boundaries (`SyntheticEngine`, model objects, context providers).  
- The inner loops of each model run through function modules that return NumPy arrays and assemble OHLCV in bulk.  
- No per-bar Python object construction in default generation path.

### Breaking Changes

- No mandatory breaking changes in phase 1. Existing toy API remains exported for compatibility.
- New API is additive and namespaced under `qik.synthetic` for new workflow adoption.

## Capabilities

### New Capabilities

- `qik-synthetic` gains a composable model architecture with class-based generators and stream-safe execution.
- Contract-first bridge helpers produce backtesting-compatible payloads with explicit identity and provenance checks.
- Multi-asset support becomes context-provider driven (`BTC`/`SPY` via defaults, custom providers possible).

### Modified Capabilities

- Baseline generators become deterministic, validation-aware, and convertible to research-ready artifacts with less glue code.

## Impact

- Affected code: `packages/qik-synthetic/src/qik/synthetic/*`, package exports, and OpenSpec artifacts.
- Data systems: direct and cleaner ingestion into `~/code/backtesting` pipelines that expect canonical OHLCV contracts.
- External typing: protocol-shape alignment with `~/data` contracts without hard runtime dependency.
