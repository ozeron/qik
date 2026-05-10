## 1. Research Round One: Regime and Block Design

- [x] 1.1 Inspect current OHLCV feature extraction, bootstrap sampling, interval parsing, and benchmark entry points.
- [x] 1.2 Define a compact `RegimeLabelConfig` covering volatility, trend, drawdown/recovery, volume, and calendar dimensions.
- [x] 1.3 Define a typed `RegimeState` contract for aligned labels, transition or duration summaries, bucket support, source block metadata, and selected config.
- [x] 1.4 Define interval-scaling rules that convert duration-based lookbacks into validated bar counts for 1m, 1h, 4h, and 1d-style candles using shared interval parsing.
- [x] 1.5 Define strict and named relaxed sparse-bucket policies, including telemetry fields for every relaxation.
- [x] 1.6 Add a research helper or benchmark mode that reports bucket support, occupancy, transition or duration summaries, and sparse-bucket risks for candidate configs.
- [x] 1.7 Add explicit candidate, seed, and runtime controls for research sweeps so large interval/block grids are bounded.
- [x] 1.8 Compare candidate block and segment lengths on available cache intervals and record dependence, overlap, repeated-block concentration, runtime, and memory-sensitive summary results.
- [x] 1.9 Select a first implementation config from validation evidence and document the selected defaults in package notes or benchmark output.

## 2. Regime Classification Implementation

- [x] 2.1 Add shared regime-label extraction helpers for OHLCV feature arrays.
- [x] 2.2 Replace 1h-biased fixed rolling windows in regime-related feature paths with interval-aware duration-to-bars helpers.
- [x] 2.3 Add trend, drawdown, recovery, volume-state, and calendar-state labels in addition to existing volatility and shock features.
- [x] 2.4 Extend fitted OHLCV profiles or model-local fitted state with explicit `RegimeState` rather than storing regime artifacts only in generic `features` or `model_params`.
- [x] 2.5 Add support validation helpers for regime/calendar buckets with explicit errors for unsupported states.
- [x] 2.6 Treat missing source timestamps as an error for calendar-conditioned classification unless calendar conditioning is explicitly disabled.
- [x] 2.7 Split strict and relaxed time/regime bucket sampling helpers so day/hour to hour-only fallback cannot occur without a named policy.
- [x] 2.8 Add tests for interval-scaled windows, aligned regime arrays, sparse-bucket validation, missing timestamp calendar errors, and strict-vs-relaxed bucket behavior.

## 3. Regime-Conditioned Bootstrap Model

- [x] 3.1 Implement shared helpers for regime path simulation or bootstrapping and source block indexing.
- [x] 3.2 Implement `RegimeConditionedBootstrapModel` as a distinct model rather than reusing `EmpiricalSVModel`.
- [x] 3.3 Preserve fit-once/generate-many state, including `RegimeState`, source features, source block index, and selected sparse-bucket policy.
- [x] 3.4 Make generation stable across `SyntheticEngine` internal block boundaries by preserving request-level regime path state or passing deterministic block offsets.
- [x] 3.5 Register `regime_conditioned_bootstrap` in the registry and `list_models()` without remapping legacy `regime_bootstrap` or `seasonal_bootstrap` aliases.
- [x] 3.6 Define the diagnostics channel for sampled regime sequence, source block ranges, relaxation counters, overlap metrics, and repeated-block concentration.
- [x] 3.7 Return or expose sampled regime and block metadata through the chosen diagnostics channel for benchmark output.
- [x] 3.8 Add deterministic generation tests for repeated seed output, valid OHLCV bounds, row count, no hidden global fallback, and multi-internal-block regime continuity.
- [x] 3.9 Add tests for explicit named sparse-bucket relaxation policies and relaxation reporting.

## 4. Research Round Two: Benchmark and Compare

- [x] 4.1 Replace the hard-coded BTCUSDT 1h-only benchmark loader with an interval-aware loader that can evaluate configured intervals and report unavailable or non-contiguous intervals.
- [x] 4.2 Add explicit skip/fail mode for unavailable intervals and clear dependency validation for parquet backends before benchmark work starts.
- [x] 4.3 Extend benchmark defaults and profile-fit model routing to include `regime_conditioned_bootstrap`.
- [x] 4.4 Add benchmark reporting for regime occupancy, transition or duration, sparse-bucket, relaxation, source-overlap, repeated-block concentration, and memorization diagnostics.
- [x] 4.5 Compare the new model against `gaussian_rw`, `ar1`, `garch`, `stationary_bootstrap`, `student_t_rw`, and `empirical_sv` over bounded multi-seed runs.
- [x] 4.6 Preserve train-only fitting, validation-driven selection, and report-only test evaluation.
- [x] 4.7 Add interval coverage reporting for evaluated, skipped, unavailable, or non-contiguous cache intervals.
- [x] 4.8 Keep per-candidate benchmark summaries compact or streamed enough to avoid storing unnecessary full per-seed artifacts during large sweeps.

## 5. Verification and Documentation

- [x] 5.1 Update package docs or `idea.md` with the regime-conditioned model scope, fit/generate contract, and benchmark interpretation.
- [x] 5.2 Add fast deterministic unit tests for regime classification, model generation, registry behavior, benchmark helper routing, and diagnostics metadata.
- [x] 5.3 Keep cache-backed or long-horizon integration checks opt-in through explicit cache configuration or slow-test selection.
- [x] 5.4 Run focused unit tests for regime classification, model generation, registry behavior, and benchmark helpers.
- [x] 5.5 Run the bounded cache-backed research and benchmark commands when local OHLCV cache data is configured.
- [x] 5.6 Summarize whether `regime_conditioned_bootstrap` is better, worse, or complementary to existing models based on validation and report-only test metrics.
