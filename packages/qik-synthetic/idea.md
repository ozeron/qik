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
   `packages/qik-synthetic/tests/fixtures`.

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

Current quality benchmark on the same `BTCUSDT.BINANCE/1h` cache:

- Benchmark script:
  `packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py`.
- Data: `27022` contiguous hourly BTCUSDT rows from
  `2023-04-01 02:00:00+00:00` to `2026-04-30 23:00:00+00:00`.
- Split: train `18915` rows, validation `4053` rows, test `4054` rows.
- Each model is fit/calibrated on train only and evaluated over `10` seeds on
  validation and test.
- Real-vs-real validation baseline: `0` warnings.
- Real-vs-real test baseline: `7` warnings across `return_distribution`,
  `temporal_dependence`, `volume`, and `price_path`.

Quality result, measured as median held-out test metric deltas versus the real
test window. Lower is better. `rel_diff` values are relative differences, so
`0.184` means about an `18.4%` difference from the real metric. `abs_diff`
values are absolute differences in already-normalized statistics such as
autocorrelation or total path return.

| Metric | Meaning | `stationary_bootstrap` | `ar1` | `gaussian_rw` | `garch` | `student_t_rw` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Return scale | robust return volatility `rel_diff` | `0.184` | `0.341` | `0.335` | `0.648` | `0.117` |
| Tail/raw volatility | raw return standard deviation `rel_diff` | `0.281` | `0.968` | `0.968` | `0.950` | `0.968` |
| Return IQR | interquartile return range `rel_diff` | `0.090` | `0.065` | `0.070` | `0.370` | `0.108` |
| Lag-1 return ACF | return autocorrelation `abs_diff` | `0.060` | `0.047` | `0.490` | `0.497` | `0.501` |
| Lag-1 squared-return ACF | volatility clustering `abs_diff` | `0.185` | `0.300` | `0.501` | `0.496` | `0.493` |
| 24h rolling vol | median rolling volatility `rel_diff` | `0.087` | `0.025` | `0.027` | `0.405` | `0.062` |
| True range | median candle true range `rel_diff` | `0.163` | `0.066` | `0.081` | `0.213` | `0.147` |
| Range/volume corr | range to log-volume correlation `abs_diff` | `0.190` | `0.026` | `0.028` | `0.161` | `0.502` |
| Abs-return/volume corr | absolute-return to log-volume correlation `abs_diff` | `0.033` | `0.029` | `0.026` | `0.031` | `0.033` |
| Volume median | median volume `rel_diff` | `0.445` | `0.554` | `0.510` | `0.541` | `0.484` |
| Volume seasonality | hour-of-day volume profile distance | `1.652` | `1.603` | `1.702` | `1.654` | `1.652` |
| Endpoint return | total path return `abs_diff` | `0.856` | `0.638` | `0.522` | `1.424` | `0.508` |
| Normalized upper price quantile | normalized close q95 `rel_diff` | `0.902` | `0.464` | `0.430` | `2.521` | `0.400` |

Current interpretation: speed is no longer the main blocker. No method is close
to the original dataset across all important dimensions. `stationary_bootstrap`
is the best quality baseline overall because it is much closer on raw volatility,
volatility clustering, and robust return behavior than the parametric random
walks. `ar1` and `gaussian_rw` are closer on some candle/path metrics, but they
miss tail volatility and dependence badly. `student_t_rw` has the closest robust
return scale and normalized price quantile, but it misses dependence and volume
coupling. `garch` now runs, but it is not the closest model on this BTCUSDT 1h
benchmark.

The largest shared gap is volume realism: median volume is still about
`44%`-`55%` away across models and the hourly volume profile distance is around
`1.6` for every method. Price-path similarity is also weak for unanchored
generators. Because the real-vs-real test baseline already fails, the benchmark
is also detecting regime shift between train and test, not only synthetic-model
error. Future model decisions should therefore optimize these held-out metric
deltas and real-vs-real calibrated thresholds, not raw generation speed.

## Realism iteration: timestamp-aware feature candles and objective scoring

Implemented next-step realism changes:

- `SyntheticEngine` now generates target timestamps before model simulation and
  passes them privately through `model_params["_timestamps"]`. This preserves
  the public request API while allowing models to condition on hour/day.
- Feature calibration now includes rolling volatility, volatility regimes,
  shock flags, log-volume hour effects/residuals, recent volume levels, and
  candle body/wick/range shares.
- Calibrated feature-candle models now use time-aligned bootstrap sampling when
  target timestamps are available.
- Added `empirical_sv` as a first coupled regime/SV path with empirical shocks,
  AR return lags, regime scales, conditional candle geometry, and conditional
  volume.
- Added `realistic_btc_1h` / `recommended_realism` aliases for the current
  validation-selected fit-once model path. Generation remains optimizer-free:
  fit the profile once, then generate many deterministic samples from the fitted
  distributions.
- The BTCUSDT benchmark now emits per-seed JSON and a validation-only weighted
  objective score. Warning count is still reported but is no longer the model
  selection target.

Verification command:

```bash
uv run --with pytest pytest packages/qik-synthetic/tests/test_qik_synthetic.py -q
```

Result:

- `52 passed, 1 skipped`.

Cache benchmark command:

```bash
QIK_SYNTHETIC_OHLCV_CACHE_PATH=/Users/ozeron/.cache/candle_research/ohlcv_catalog/ohlcv \
  uv run --with pandas --with pyarrow --with arch \
  python packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py \
  --json-out /tmp/qik_synthetic_realism_benchmark.json
```

Validation weighted score, lower is better:

| Model | Score | Warnings median | Return std | Robust vol | IQR | Sq-ret ACF1 | 24h vol | True range | Volume median | Volume ACF1 | Volume ACF24 | Hourly volume | Abs-ret/vol corr | Range/vol corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `student_t_rw` | `37.006` | `3.0` | `0.218` | `0.253` | `0.158` | `0.122` | `0.268` | `0.276` | `0.751` | `0.122` | `0.145` | `0.571` | `0.563` | `0.294` |
| `gaussian_rw` | `44.278` | `3.5` | `0.204` | `0.732` | `0.323` | `0.122` | `0.401` | `0.473` | `0.705` | `0.103` | `0.074` | `0.498` | `0.566` | `0.279` |
| `garch` | `47.587` | `5.5` | `0.812` | `0.029` | `0.188` | `0.153` | `0.119` | `0.217` | `0.752` | `0.106` | `0.109` | `0.511` | `0.572` | `0.452` |
| `ar1` | `48.211` | `4.5` | `0.199` | `0.716` | `0.311` | `0.044` | `0.396` | `0.481` | `0.734` | `0.109` | `0.062` | `0.557` | `0.576` | `0.263` |
| `stationary_bootstrap` | `63.217` | `6.0` | `25.201` | `0.081` | `0.339` | `0.261` | `0.427` | `0.198` | `0.726` | `0.117` | `0.091` | `0.578` | `0.613` | `0.043` |
| `empirical_sv` | `66.533` | `8.5` | `1.119` | `0.237` | `1.116` | `0.080` | `1.341` | `0.246` | `0.467` | `0.130` | `0.090` | `0.967` | `0.079` | `0.027` |

Selected validation config:

- Model: `student_t_rw`.
- Selection score: `37.00575468265533`.
- Selection window: validation only.
- Test was report-only, not used for selection.

Test weighted score, report-only:

| Model | Score | Warnings median | Return std | Robust vol | IQR | Sq-ret ACF1 | 24h vol | True range | Volume median | Volume ACF1 | Volume ACF24 | Hourly volume | Abs-ret/vol corr | Range/vol corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ar1` | `30.124` | `6.0` | `0.970` | `0.341` | `0.068` | `0.353` | `0.024` | `0.078` | `0.544` | `0.098` | `0.030` | `0.649` | `0.029` | `0.313` |
| `stationary_bootstrap` | `33.026` | `4.0` | `0.525` | `0.163` | `0.045` | `0.103` | `0.046` | `0.131` | `0.526` | `0.099` | `0.059` | `0.606` | `0.014` | `0.099` |
| `student_t_rw` | `36.239` | `7.5` | `0.970` | `0.024` | `0.168` | `0.507` | `0.123` | `0.054` | `0.498` | `0.078` | `0.069` | `0.610` | `0.027` | `0.360` |
| `garch` | `37.320` | `6.0` | `0.959` | `0.215` | `0.177` | `0.306` | `0.241` | `0.126` | `0.540` | `0.091` | `0.073` | `0.612` | `0.032` | `0.509` |
| `gaussian_rw` | `37.734` | `7.0` | `0.970` | `0.335` | `0.056` | `0.494` | `0.025` | `0.078` | `0.535` | `0.097` | `0.052` | `0.609` | `0.030` | `0.329` |
| `empirical_sv` | `53.670` | `6.5` | `0.947` | `0.040` | `0.601` | `0.306` | `0.619` | `0.042` | `0.296` | `0.109` | `0.064` | `0.816` | `0.516` | `0.087` |

Main improvement versus the earlier held-out test table:

- Hourly volume profile distance dropped from roughly `1.60`-`1.70` across all
  models to `0.60`-`0.65` for timestamp-aware feature-candle models, meeting the
  v1 `<= 0.75` target.
- `student_t_rw` test robust volatility improved from `0.117` to `0.030`.
- `student_t_rw` test true-range median improved from `0.147` to `0.053`.
- `garch` test score improved materially after fitted profile defaults and
  return clipping: `52.116` to `37.320`, but validation still does not select
  it.
- `stationary_bootstrap` test hourly volume improved from `1.652` to `0.606`,
  but validation still rejects it because return standard deviation is unstable.

Remaining gaps:

- Selected `student_t_rw` still misses raw return standard deviation on test
  (`0.970`) even though robust volatility is good.
- Volume median remains too high on the selected validation/test config
  (`0.751` validation, `0.498` test) versus the v1 `<= 0.25` target.
- Range/log-volume correlation is still too weak for selected `student_t_rw`
  (`0.294` validation, `0.360` test) versus the v1 `<= 0.08` target.
- `empirical_sv` improves volume seasonality and correlation coupling on
  validation, but its return distribution is too wide and is not selected.
- Endpoint and normalized price-path metrics should remain guardrails for
  unanchored generation; do not optimize them directly until adding a separate
  anchored bridge mode.

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

## Performance task plan

Implemented performance tasks:

- Add columnar generation APIs:
  - `SyntheticEngine.generate_frame`
  - `SyntheticEngine.iter_generate_frames`
  - `SyntheticEngine.generate_backtesting_payload`
- Keep `SyntheticEngine.generate` and `iter_generate` compatible while sharing
  the same internal block generation path.
- Replace AR(1) convolution with an equivalent linear recurrence.
- Make `to_backtesting_frame` use `DataFrame.from_records` and avoid rebuilding
  rows before pandas conversion.
- Add `compact_identity` support for categorical `symbol`, `venue`, `dataset`,
  and `interval` columns on generated frames/payloads.
- Replace JSON fingerprinting with canonical byte hashing.
- Combine diagnostics/calibration quantile calls where possible.
- Reject multi-identity diagnostics/calibration inputs unless callers group
  first.
- Add grouped multi-identity helpers:
  - `compare_ohlcv_by_identity`
  - `calibrate_many_ohlcv_params`
- Avoid full-frame timestamp sorting in calibration; sort extracted arrays by
  timestamp order instead.
- Cache parsed intervals with `lru_cache`.
- Add local benchmark script:
  `packages/qik-synthetic/benchmarks/bench_synthetic.py`.

Measured impact:

- `100000` rows:
  - `generate_frame`: `0.135569s`.
  - legacy `generate`: `0.159400s`.
  - direct frame payload: `0.120913s`.
  - legacy generate + payload: `0.273482s`.
  - direct payload speedup versus legacy total: `2.26x`.
- One-year `1m` / `525600` rows:
  - `generate_frame`: `0.673187s` to `0.776726s`.
  - direct frame payload: `0.576792s` to `0.683421s`.
  - compact frame memory: `27331554` bytes, about `26 MiB`.
  - `/usr/bin/time -l` direct payload wall time including Python startup/imports:
    `1.44s`.
  - `/usr/bin/time -l` maximum resident set size: `171458560` bytes, about
    `164 MiB`.
  - Previous one-year legacy payload measurement was `4.62s` wall clock and
    `629932032` bytes max RSS, about `601 MiB`.
- One-year `1m` diagnostics/support timings:
  - Fast fingerprint: `1.785277s`.
  - Self comparison diagnostics: `0.616781s`.
  - Calibration: `0.167782s`.
  - AR(1) frame generation after recurrence rewrite: `0.742274s`.

Remaining realism/modeling tasks:

- Add price-bridge and heavy-tail/bootstrapped return models for realism rather
  than speed.
- Decide whether `compact_identity=True` should become the default only for the
  new frame APIs after downstream adapter compatibility is confirmed.

## Regime-conditioned bootstrap direction

`regime_conditioned_bootstrap` is the next robustness model. It is intentionally
separate from the existing `empirical_sv` aliases so existing callers keep their
behavior.

The model contract is:

- Fit once from a single-identity OHLCV profile.
- Build explicit `RegimeState` from interval-aware labels:
  volatility, trend, drawdown/recovery, volume, and calendar buckets.
- Generate many paths by sampling fitted feature rows through strict or named
  relaxed regime/calendar policies.
- Expose generation diagnostics for sampled regimes, source indices, internal
  block starts, relaxation counts, and repeated-block concentration.

Strict behavior remains the default. If a target calendar/regime bucket has no
source support, generation raises unless the caller explicitly sets a named
policy such as `regime_sparse_policy="relax_calendar"`.

Benchmarking now treats `regime_conditioned_bootstrap` as a profile-fitted model
and reports regime diagnostics alongside the existing OHLCV metric summaries.
Cache-backed interval runs are still opt-in; unavailable intervals should be
reported through explicit skip/fail behavior rather than hidden fallback.

Bounded BTCUSDT 1h benchmark, `2` seeds, compact output:

- Cache: `27022` rows, train `18915`, validation `4053`, test `4054`.
- Validation winner remained `student_t_rw` with score `33.841`.
- `regime_conditioned_bootstrap` validation score was `115.907` with median
  warning count `7.0`; it should not become the recommended realism default.
- Report-only test score for `regime_conditioned_bootstrap` was `45.833`, behind
  `stationary_bootstrap`, `ar1`, `student_t_rw`, `gaussian_rw`, and `garch`.
- Diagnostics showed low repeated-block concentration and no relaxation events,
  so the current weakness is model quality/regime granularity rather than obvious
  block replay.

Current recommendation: keep the implementation as an experimental robustness
generator and diagnostics baseline, but keep `student_t_rw` /
`realistic_btc_1h` as the selected validation default until regime buckets are
coarsened and block/segment candidates are compared.

Block-length candidate check on the same BTCUSDT 1h cache:

| Block length | Source blocks | Sparse buckets | Repeated concentration | Runtime |
| --- | ---: | ---: | ---: | ---: |
| `1d` | `789` | `4228` | `0.00099` | `0.44s` |
| `2d` | `395` | `4228` | `0.00099` | `0.45s` |
| `7d` | `113` | `4228` | `0.00099` | `0.40s` |

This confirms the first implementation is too row/calendar driven: block length
changes source block metadata but not sampled-row concentration. The next
research iteration should coarsen regime buckets and make segment/block state
actually influence sampled path continuity before promoting this model.

Wider sweep on the same BTCUSDT 1h cache, `48` configs and `5` seeds, compared
calendar conditioning on/off, trend windows `1d`/`3d`/`7d`/`14d`, drawdown
windows `7d`/`30d`, and block lengths `1d`/`2d`/`7d`:

| Dimension | Best/Signal | Validation score | Runtime |
| --- | --- | ---: | ---: |
| Calendar conditioning | `false` | mean `59.074` | mean `0.326s` |
| Calendar conditioning | `true` | mean `67.580` | mean `0.893s` |
| Block length | `2d` | mean `60.854` | mean `0.606s` |
| Block length | `1d` | mean `62.330` | mean `0.610s` |
| Block length | `7d` | mean `66.796` | mean `0.612s` |

Best validation configs were all calendar-free `2d` block runs with score
`54.128`, median warning count `7.0`, and runtimes around `0.30s`-`0.35s`.
Report-only test checks for the top five validation configs all scored `27.802`
with median warning count `5.0`, which is a stronger test outcome than the
earlier strict calendar/default run. Trend and drawdown windows did not move the
score in this sweep, which means the current sampler is mostly responding to
block length and the calendar bucket explosion rather than to trend/drawdown
state. For the next iteration, use `calendar_conditioning=false` and `2d` blocks
as the benchmark candidate, then spend research effort on coarser calendar
buckets or explicit segment-continuity sampling instead of adding more raw
trend/drawdown window combinations.

## Synthetic realism architecture

The next realism layer separates speed from market similarity. Fast frame
creation remains useful for adapter and benchmark workloads, while realism is
measured through explicit calibration profiles, feature-based candle generation,
and categorized diagnostics.

Diagnostic modes:

- `price_mode="ignore"`: compare distribution, candle, and volume behavior
  without path-level price checks.
- `price_mode="normalized"`: compare normalized price path behavior and return
  features; this is the default for unanchored stochastic simulations.
- `price_mode="anchored"`: include raw price-level checks when a synthetic path
  is intentionally conditioned on historical price levels or endpoints.

Realism model taxonomy:

- Baselines: `gaussian_rw`, `ar1`.
- Fat-tail baseline: `student_t_rw`.
- Feature bootstrap baseline: `stationary_bootstrap` / `empirical_bootstrap`.
- Optional parametric volatility: `garch` through `qik-synthetic[models]`.

Calibration/profile workflow:

1. Fit `OhlcvCalibrationProfile` from a single-identity OHLCV frame.
2. Use profile features for feature-driven candles or stationary bootstrap.
3. Use scalar `calibrate_ohlcv_params` as a compatibility projection for older
   baseline models.
4. Evaluate with many seeds over chronological train/validation/test splits.

Cache-backed realism benchmark:

- Script: `packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py`.
- Env var: `QIK_SYNTHETIC_OHLCV_CACHE_PATH`.
- Normal package tests skip cache-backed checks when the env var is unset.

Bootstrap overfitting policy:

- Block-overlap and memorization signals are warning-only initially.
- Chronological train/validation/test splits are required for benchmark output.
- Hard failure thresholds should only be added after benchmark output is stable.
