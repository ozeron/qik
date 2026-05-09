## 1. Realism Evaluation Harness

- [x] 1.1 Add `packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py` with cache loading from `QIK_SYNTHETIC_OHLCV_CACHE_PATH`.
- [x] 1.2 Implement BTCUSDT/BINANCE 1h continuity validation and chronological train/validation/test splitting.
- [x] 1.3 Implement multi-seed model evaluation with metric median, lower quantile, upper quantile, pass rate, warning categories, runtime, and row count.
- [x] 1.4 Ensure fitting and calibration use train only, model/threshold selection uses validation output, and final held-out metrics are reported separately for test.
- [x] 1.5 Add real-vs-real baseline comparisons for validation/test interpretation.
- [x] 1.6 Pre-normalize/slice reference windows once and reuse prepared arrays or frames across seed sweeps to avoid repeated sorting/copying.
- [x] 1.7 Add a skipped integration smoke test guarded by `QIK_SYNTHETIC_OHLCV_CACHE_PATH` availability so normal package tests pass without local cache data.

## 2. Diagnostic Refactor

- [x] 2.1 Refactor `diagnostics.py` metrics into data quality, return distribution, temporal dependence, candle shape, volume, and price path categories.
- [x] 2.2 Add `price_mode` support with `ignore`, `normalized`, and `anchored` modes.
- [x] 2.3 Add temporal dependence metrics including return ACF, absolute-return ACF, squared-return ACF, and rolling volatility summaries.
- [x] 2.4 Add candle geometry metrics for true range, upper wick, lower wick, body-to-range ratio, and gap behavior.
- [x] 2.5 Add volume realism metrics for log-volume distribution, volume ACF, zero-volume rate, range-volume correlation, abs-return-volume correlation, and hourly profile distance.
- [x] 2.6 Add optional data-calibrated threshold generation from real-vs-real block resampling.
- [x] 2.7 Preserve existing flat diagnostic fields (`passed`, `metrics`, `warnings`, `thresholds`, `row_counts`) while adding category-level results.
- [x] 2.8 Update diagnostics tests for categorized reports, flat compatibility, normalized price mode, anchored price mode, and new quality checks.

## 3. Calibration Profiles and Feature Extraction

- [x] 3.1 Add an `OhlcvCalibrationProfile` data contract with identity, row count, time range, stats categories, model params, and source fingerprint.
- [x] 3.2 Add feature extraction for close-to-close return, open-gap return, intrabar return, upper wick relative size, lower wick relative size, relative range, log volume, hour-of-day, and day-of-week.
- [x] 3.3 Add `fit_ohlcv_profile(frame, model=...)` while preserving `calibrate_ohlcv_params(frame)` compatibility.
- [x] 3.4 Add grouped profile fitting for multi-identity frames.
- [x] 3.5 Store raw volatility, robust volatility, return quantiles, wick/range quantiles, volume profile, correlations, and dependence stats in profiles.
- [x] 3.6 Keep `calibrate_ohlcv_params(frame)` as a scalar compatibility projection from the richer profile path where feasible.
- [x] 3.7 Add tests proving profile fitting is deterministic, does not mutate input frames, rejects ambiguous single-profile multi-identity input, supports grouped fitting, and preserves scalar calibration compatibility.

## 4. Feature-Driven Candle Construction

- [x] 4.1 Add `build_ohlcv_from_features` to construct OHLCV arrays from open-gap returns, intrabar returns, upper/lower wick features, and volume.
- [x] 4.2 Keep the existing `build_ohlcv` path for compatibility with baseline models.
- [x] 4.3 Add model hooks or params so calibrated Gaussian/AR(1)-style models can use feature-driven candle decoration instead of body-derived ranges.
- [x] 4.4 Add tests for high/low bounds, deterministic seeded output, and improved range/volume behavior versus the legacy candle renderer.

## 5. Stationary Bootstrap Model

- [x] 5.1 Add stationary/block bootstrap index generation with deterministic seeding and configurable mean block size.
- [x] 5.2 Add `models/bootstrap.py` with a fitted stationary bootstrap OHLCV feature model.
- [x] 5.3 Register `stationary_bootstrap` and any intended aliases in the model registry.
- [x] 5.4 Add engine/model fit support for pandas frames or feature profiles without breaking existing `fit(history)` callers.
- [x] 5.5 Add a compatibility-safe model-fit adapter path that accepts `pd.DataFrame` and `OhlcvCalibrationProfile` for models that support them while preserving generic history fitting.
- [x] 5.6 Add warning-only bootstrap memorization diagnostics such as block-overlap or duplicate-window metrics.
- [x] 5.7 Add tests for requested row count, structural OHLCV validity, deterministic seeding, feature relationship preservation, and warning-only memorization reporting.

## 6. Parametric Realism Models

- [x] 6.1 Correct `models/garch.py` to fit and simulate real GARCH-family conditional volatility when optional dependencies are installed.
- [x] 6.2 Preserve the optional dependency error path for GARCH when `arch` is unavailable.
- [x] 6.3 Add at least one lower-dependency fat-tail or volatility-clustering baseline such as `student_t_rw`, `ewma_t`, or empirical-residual bootstrap.
- [x] 6.4 Register new model names and add model-list coverage.
- [x] 6.5 Add tests for seeded determinism, volatility clustering behavior, optional dependency handling, and diagnostic separation of return-tail behavior from candle/volume behavior.

## 7. Evaluation and Documentation

- [x] 7.1 Update `packages/qik-synthetic/idea.md` with the realism architecture, benchmark workflow, model taxonomy, and interpretation guidance.
- [x] 7.2 Document when to use unanchored, normalized, anchored, and bridge/endpoint-conditioned comparisons.
- [x] 7.3 Document overfitting risks for bootstrap models and how chronological splits and memorization diagnostics mitigate them.
- [x] 7.4 Run `uv run --with pytest pytest tests/test_qik_synthetic.py -q`.
- [x] 7.5 Run `openspec validate improve-synthetic-realism --strict`.
- [x] 7.6 Run `uv build packages/qik-synthetic`.
