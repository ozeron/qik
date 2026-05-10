## 1. Core contracts and compatibility baseline

- [x] 1.1 Add `packages/qik-synthetic/src/qik/synthetic/types.py` dataclasses:
  - `SyntheticContext`
  - `SyntheticRequest`
  - `SyntheticBatch`
  - `SyntheticChunk`
  - `SyntheticMetadata`
- [x] 1.2 Add `packages/qik-synthetic/src/qik/synthetic/errors.py`:
  - `SyntheticError`
  - `SyntheticValidationError`
  - `SyntheticContextError`
  - `SyntheticConversionError`
- [x] 1.3 Add `packages/qik-synthetic/src/qik/synthetic/protocols.py`:
  - `PathModel` Protocol with `fit`, `simulate_returns`, `simulate_batch`
  - `MarketContextProvider` Protocol
  - `BacktestingAdapter` Protocol (optional adapters)
- [x] 1.4 Keep legacy API compatibility in `packages/qik-synthetic/src/qik/synthetic/types.py`:
  - continue export of `DriftConfig`, `MarketSeries`, `NoiseGenerator`, `build_market_path`, `gaussian_noise`
  - add deprecation guidance where needed.

## 2. Engine + generator workflow

- [x] 2.1 Add `packages/qik-synthetic/src/qik/synthetic/engine.py`:
  - `SyntheticEngine` with `fit`, `generate`, `iter_generate`, `generate_many`
  - deterministic seed derivation from request seed + model + run parameters
  - grouped request optimizations (same interval/model/context) for timestamp reuse
- [x] 2.2 Add `packages/qik-synthetic/src/qik/synthetic/horizon.py`:
  - `bars_from_span(start_ts, interval, *, end_ts=None, duration=None, n_bars=None, trading_hours="24x7")`
  - one-of validation: exactly one of (`n_bars`, `duration`, `end_ts`) unless request already has explicit `n_bars`
- [x] 2.3 Add model registry in `packages/qik-synthetic/src/qik/synthetic/registry.py`:
  - built-ins: `"gaussian_rw"` and `"ar1"` baseline families.
- [x] 2.4 Add `packages/qik-synthetic/src/qik/synthetic/kernels.py` with vectorized compute helpers (noise, returns path, OHLCV reconstruction, volume).
- [x] 2.5 Add built-in model implementations under `packages/qik-synthetic/src/qik/synthetic/models/` using the kernel layer and no frame-by-frame loops.
- [x] 2.6 Add internal benchmarks/growth checks in tasks (or comments) for `iter_generate` memory behavior on >1e7 rows and chunk boundary reproducibility.

## 3. Context and multi-asset compatibility

- [x] 3.1 Add `packages/qik-synthetic/src/qik/synthetic/context.py`:
  - `DefaultMarketContextProvider` resolving BTC + SPY defaults and custom provider hooks.
- [x] 3.2 Validate context/request mismatch and raise `SyntheticContextError` on calendar/model assumption conflicts.
- [x] 3.3 Ensure context protocol fields are compatible with `market_data_lake.contracts` naming, without importing that dependency.

## 4. Backtesting bridge

- [x] 4.1 Add `packages/qik-synthetic/src/qik/synthetic/backtesting.py`:
  - `to_backtesting_frame(batch)`
  - `to_backtesting_payload(batch, *, identity_override=None, run_context=None)`
  - `validate_backtesting_payload(payload)`
- [x] 4.2 Enforce required identity fields (`symbol`, `venue`, `interval`, `dataset`) before conversion.
- [x] 4.3 Add run metadata contract:
  - `run_id`, `seed`, `model_name`, `context_id`, request fingerprint.

## 5. Optional statsmodels/garch path

- [x] 5.1 Add `packages/qik-synthetic/pyproject.toml` extras:
  - `[project.optional-dependencies] models = ["statsmodels", "arch"]`
- [x] 5.2 Add `packages/qik-synthetic/src/qik/synthetic/models/garch.py` behind conditional imports.
- [x] 5.3 Register optional model names in registry with clear runtime error when optional dependencies missing.

## 6. Packaging, docs, and external fixtures

- [x] 6.1 Update exports in `packages/qik-synthetic/src/qik/synthetic/__init__.py` for new primary API.
- [x] 6.2 Add `packages/qik-synthetic/README.md` usage snippets:
  - BTC/SPY generation using `SyntheticEngine` + `iter_generate`
- [x] 6.3 Add `packages/qik-synthetic/idea.md` implementation notes:
  - compatibility-first architecture, no internal hard-coded fixture payloads
- [x] 6.4 Add/adjust tests to load fixtures from repo-level `tests/fixtures/` paths (outside package source).
