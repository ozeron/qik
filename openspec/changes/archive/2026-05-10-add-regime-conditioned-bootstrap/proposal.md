## Why

Current synthetic BTC generation has fitted OHLCV profiles, feature bootstraps,
fat-tail baselines, GARCH, and cache-backed benchmarking, but it does not yet
model a durable sequence of market regimes. That limits usefulness for strategy
robustness because block sampling can preserve local candle structure while still
mixing trend, volatility, drawdown, volume, and calendar states too freely.

This change adds a regime-conditioned generation strategy and a two-round
research/benchmark workflow: first define and validate interval-aware regime and
block classification, then fit the model and compare it against the existing
synthetic strategies.

## What Changes

- Add an interval-aware regime labeling layer for OHLCV profiles covering:
  volatility buckets, trend buckets, drawdown/recovery state, volume buckets,
  calendar buckets, and source block metadata.
- Add a new fitted `regime_conditioned_bootstrap` model that simulates regime
  paths and samples dependent OHLCV feature blocks from matching fitted regimes.
- Require explicit validation errors for sparse or impossible regime buckets;
  lower-fidelity relaxation must be opt-in and named.
- Extend the realism benchmark to compare the new model against existing
  `gaussian_rw`, `ar1`, `garch`, `stationary_bootstrap`, `student_t_rw`, and
  `empirical_sv` models.
- Add research tasks for block length, segment length, interval scaling, and
  regime classification before judging model performance.
- Preserve the fit-once/generate-many contract: fitted profiles/models must carry
  regime state needed for repeated generation.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `synthetic-realism-models`: Add interval-aware regime classification and a
  fitted regime-conditioned bootstrap generator.
- `synthetic-realism-evaluation`: Add benchmark comparison and reporting for
  regime-conditioned bootstrap, including regime/block diagnostics.

## Impact

- Affected package code:
  - `packages/qik-synthetic/src/qik/synthetic/calibration.py`
  - `packages/qik-synthetic/src/qik/synthetic/kernels.py`
  - `packages/qik-synthetic/src/qik/synthetic/models/`
  - `packages/qik-synthetic/src/qik/synthetic/registry.py`
  - `packages/qik-synthetic/src/qik/synthetic/diagnostics.py`
- Affected benchmark code:
  - `packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py`
- Affected tests:
  - `tests/test_qik_synthetic.py`
- No new mandatory heavy dependencies.
- No strategy optimization logic is added to `qik.synthetic`; downstream
  backtesting remains outside the package.
