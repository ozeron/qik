## Context

`qik-synthetic` currently has fast generation and backtesting payload paths, but its realism story is incomplete. Baseline models (`gaussian_rw`, `ar1`, and current `garch`) can generate structurally valid OHLCV tables, yet BTCUSDT 1h comparisons still show large differences in volatility, tails, candle geometry, volume behavior, and path-level drift. Current diagnostics are useful for smoke checks, but they mix raw price-level checks with stochastic realism checks, which makes unanchored simulations look wrong even when the intended comparison should be distributional.

The package already has calibration, diagnostics, frame generation, and local BTC cache notes. The next architecture step is to make realism measurable and model-driven: evaluate models against held-out real data, add diagnostics that identify failure categories, and introduce models capable of generating the features those diagnostics measure.

## Goals / Non-Goals

**Goals:**

- Provide a repeatable local realism evaluation workflow for BTCUSDT/BINANCE 1h data using chronological train/validation/test splits.
- Make diagnostics explain why a synthetic series is unrealistic by separating data quality, return distribution, temporal dependence, candle shape, volume behavior, and price-path checks.
- Support unanchored, normalized, and anchored price comparison modes.
- Add feature-based OHLCV profiles that capture returns, intrabar behavior, wick/range geometry, volume effects, hourly seasonality, and dependence statistics.
- Add realistic model baselines, starting with stationary/block bootstrap over OHLCV feature vectors.
- Correct `garch` so it is either a real GARCH-family simulation path or clearly not used as a GARCH realism model.
- Preserve existing generation and payload APIs for compatibility.

**Non-Goals:**

- No neural generator or GAN/diffusion model in this change.
- No mandatory dependency on local BTC cache data in normal tests or builds.
- No promise that one model will pass every diagnostic on every seed.
- No default endpoint/path anchoring unless explicitly requested by model parameters or diagnostic mode.
- No breaking change to `SyntheticEngine.generate`, `iter_generate`, `generate_frame`, `to_backtesting_payload`, or existing baseline model names.

## Decisions

### Use chronological train/validation/test evaluation

The realism workflow will split real OHLCV by time, fit/calibrate on train, select/tune on validation, and reserve test for final comparison. This avoids evaluating a model on the same window used for calibration. Random splits were rejected because they leak market regimes across time.

### Report ensemble diagnostics across many seeds

The evaluation workflow will generate many synthetic paths per model and report metric medians, quantiles, warning categories, and pass rates. Single-seed comparisons remain useful for smoke tests, but they are not sufficient evidence of realism.

### Refactor diagnostics into categories

Diagnostics will keep row-count and OHLC validity checks, but realism metrics will be categorized. This makes results actionable: a model can fail tail behavior, volatility clustering, candle shape, or volume coupling independently. A flat warning list was rejected because it does not identify which modeling component is deficient.

### Add price comparison modes

Diagnostics will support `price_mode="ignore"`, `price_mode="normalized"`, and `price_mode="anchored"`. `normalized` will be the realism default because unanchored stochastic simulations should not fail solely because their raw price path wanders away from one historical path. `anchored` remains available for payload or bridge-mode tests that intentionally condition on real price levels.

### Introduce feature-based candle modeling

The existing candle renderer ties range to body size and cannot naturally create small-body/large-range candles. New realism models will use feature vectors: close-to-close return, open gap return, intrabar return, upper wick relative size, lower wick relative size, range, volume, hour-of-day, and day-of-week. Feature-based construction preserves candle geometry better than a scalar wick multiplier.

### Implement stationary/block bootstrap first

A stationary/block bootstrap model over joint OHLCV feature vectors is the first realism model because it can preserve fat tails, volatility clusters, volume/range coupling, wick behavior, and hourly patterns without adding heavy dependencies. Real GARCH and Student-t/EWMA models are valuable follow-ups, but bootstrap provides the strongest immediate baseline per engineering effort.

### Keep compatibility APIs and add richer profile APIs

`calibrate_ohlcv_params(frame)` will remain a scalar compatibility helper. Richer calibration will be exposed through profile APIs such as `fit_ohlcv_profile`, with model-specific conversion into params. This avoids forcing advanced models through a single flat dict.

### Keep cache-backed realism benchmarks opt-in

The BTCUSDT cache path is local and environment-specific, so realism benchmarks will be scripts or integration tests guarded by environment variables. Unit tests will use small deterministic fixtures and assert behavioral direction rather than brittle real-cache thresholds.

The cache environment variable is standardized as `QIK_SYNTHETIC_OHLCV_CACHE_PATH`. Normal package tests must pass without this variable; cache-backed tests and benchmarks must skip cleanly unless the variable is present and points to usable data.

### Keep bootstrap memorization diagnostics warning-only initially

Stationary/block bootstrap can look realistic by replaying training behavior. The first implementation will report block-overlap and memorization metrics as warnings/diagnostic output only, not as normal-CI failures or hard benchmark gates. This keeps early research iteration unblocked while still making leakage visible. Hard failure thresholds can be added later after the benchmark output is stable.

### Preserve compatibility while adding categorized diagnostics

Categorized diagnostics will be added without removing the current flat report fields. Existing callers that read `metrics`, `warnings`, `thresholds`, `row_counts`, and `passed` should continue to work while new callers can inspect category-level results.

### Add a model-fit adapter layer

Realism models need to fit from pandas frames and richer OHLCV profiles, while existing model protocols accept generic history rows. The implementation will add a compatibility-safe adapter path that accepts `pd.DataFrame` and `OhlcvCalibrationProfile` where supported, while preserving `fit(history)` behavior for current callers.

## Risks / Trade-offs

- Bootstrap memorization risk -> Use chronological holdouts, block-overlap/memorization diagnostics, and report validation/test behavior separately. Memorization signals are warning-only in the initial implementation.
- More diagnostics can create noisy results -> Group metrics into categories and report aggregate category status plus raw metrics.
- Price modes may hide real path failures -> Make `anchored` explicit for path-conditioned workflows and keep raw endpoint/path metrics available.
- Real GARCH can be optimizer-sensitive -> Keep it optional behind `qik-synthetic[models]`, add deterministic seeded tests, and compare against simpler baselines.
- Data-calibrated thresholds can become too permissive -> Report raw metrics, real-vs-real baselines, and fixed smoke thresholds separately.
- Rich profiles add API surface -> Preserve existing scalar calibration API and introduce profile APIs additively.
