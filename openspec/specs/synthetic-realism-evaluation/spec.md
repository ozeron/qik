# synthetic-realism-evaluation Specification

## Purpose
TBD - created by archiving change improve-synthetic-realism. Update Purpose after archive.
## Requirements
### Requirement: Cache-backed realism evaluation workflow
The system SHALL provide an opt-in evaluation workflow that loads real OHLCV cache data, splits it chronologically into train, validation, and test windows, and compares synthetic model outputs against held-out real data.

#### Scenario: Evaluate BTCUSDT 1h cache when configured
- **WHEN** a user runs the realism evaluation with `QIK_SYNTHETIC_OHLCV_CACHE_PATH` pointing to OHLCV cache data containing contiguous BTCUSDT/BINANCE 1h parquet data
- **THEN** the system SHALL load the data, verify timestamp continuity for the selected interval, create chronological splits, and report model metrics for validation and test windows

#### Scenario: Skip cache-backed evaluation when cache is unavailable
- **WHEN** `QIK_SYNTHETIC_OHLCV_CACHE_PATH` is not set or the expected data is unavailable
- **THEN** normal unit tests SHALL skip cache-backed realism evaluation rather than failing package builds

#### Scenario: Preserve validation and test separation
- **WHEN** cache-backed realism evaluation is run
- **THEN** model fitting and scalar/profile calibration SHALL use only the train split, model or threshold selection SHALL use validation output, and final held-out metrics SHALL be reported separately for the test split

### Requirement: Ensemble model comparison
The system SHALL evaluate stochastic synthetic models over multiple seeds and summarize metric distributions instead of relying on a single generated path.

#### Scenario: Compare model across seeds
- **WHEN** a realism benchmark evaluates a model for a validation or test horizon
- **THEN** the system SHALL generate multiple seeded paths and report metric median, lower quantile, upper quantile, warning categories, pass rate, runtime, and row count

#### Scenario: Avoid repeated reference normalization
- **WHEN** a realism benchmark evaluates many seeds against the same reference window
- **THEN** the system SHALL prepare the reference window once and reuse prepared arrays or frames rather than repeatedly sorting and copying the same real data for each seed

### Requirement: Categorized OHLCV diagnostics
The system SHALL categorize OHLCV diagnostics into data quality, return distribution, temporal dependence, candle shape, volume behavior, and price path categories.

#### Scenario: Report diagnostic categories
- **WHEN** synthetic OHLCV is compared with reference OHLCV
- **THEN** the report SHALL identify which categories passed or failed and expose the underlying metric values for each category

#### Scenario: Preserve flat diagnostic compatibility
- **WHEN** categorized diagnostics are returned
- **THEN** existing flat report fields for overall pass status, metrics, warnings, thresholds, and row counts SHALL remain available for compatibility

### Requirement: Price comparison modes
The system SHALL support explicit price comparison modes for diagnostics: ignore, normalized, and anchored.

#### Scenario: Unanchored stochastic comparison
- **WHEN** diagnostics run with normalized price mode
- **THEN** the system SHALL compare normalized price-path behavior and SHALL NOT fail solely because raw close levels differ from the historical path

#### Scenario: Anchored path comparison
- **WHEN** diagnostics run with anchored price mode
- **THEN** the system SHALL include raw price-level and endpoint-path checks capable of detecting close-level drift

### Requirement: Data-calibrated diagnostic thresholds
The system SHALL support optional threshold calibration from real-vs-real resampling so realism warnings can be interpreted relative to historical sampling variation.

#### Scenario: Calibrate thresholds from reference data
- **WHEN** a user calibrates diagnostic thresholds from a reference OHLCV frame and target horizon
- **THEN** the system SHALL return metric thresholds derived from real-vs-real comparisons while preserving hard data-quality validation rules

### Requirement: Regime bootstrap research workflow
The system SHALL provide a repeatable research workflow for selecting
regime-classification, block-length, segment-length, and interval-scaling choices
before treating `regime_conditioned_bootstrap` as a candidate realism model.

#### Scenario: Evaluate candidate regime definitions
- **WHEN** the research workflow runs against configured OHLCV cache data
- **THEN** it SHALL report bucket support, regime occupancy, transition or duration summaries, and sparse-bucket warnings for each candidate configuration

#### Scenario: Evaluate block and segment sizing
- **WHEN** the research workflow compares candidate block and segment sizes
- **THEN** it SHALL report dependence preservation, source overlap, repeated-block concentration, and runtime so the selected configuration is evidence-based

#### Scenario: Preserve chronological separation
- **WHEN** regime research uses train, validation, and test windows
- **THEN** fitting and candidate selection SHALL use train and validation only, with test reserved for final report-only evaluation

#### Scenario: Bound candidate sweep cost
- **WHEN** the research workflow evaluates candidate regime or block configurations
- **THEN** it SHALL support explicit candidate limits, seed limits, and compact summary output to avoid unbounded runtime or memory use

### Requirement: Regime-conditioned benchmark comparison
The system SHALL include `regime_conditioned_bootstrap` in the cache-backed
realism benchmark and compare it against the existing model families over
multiple seeds.

#### Scenario: Compare against existing models
- **WHEN** cache-backed realism evaluation runs with the full model set
- **THEN** the benchmark SHALL include `regime_conditioned_bootstrap` alongside `gaussian_rw`, `ar1`, `garch`, `stationary_bootstrap`, `student_t_rw`, and `empirical_sv` when optional dependencies are available

#### Scenario: Report regime diagnostics with OHLCV metrics
- **WHEN** the benchmark evaluates `regime_conditioned_bootstrap`
- **THEN** the output SHALL include existing OHLCV metric summaries plus regime occupancy, transition or duration, sparse-bucket, relaxation, and memorization diagnostics

#### Scenario: Keep test report-only
- **WHEN** the benchmark selects or ranks model configurations
- **THEN** validation results SHALL drive selection and test results SHALL remain report-only

#### Scenario: Fit regime model from profile
- **WHEN** the benchmark evaluates `regime_conditioned_bootstrap`
- **THEN** it SHALL fit the model from the train-only profile before generation, matching the fit-once behavior used by other profile-aware models

### Requirement: Interval coverage reporting
The system SHALL make interval applicability visible when benchmarking or
researching regime-conditioned generation.

#### Scenario: Run on available intervals
- **WHEN** cache data is available for multiple intervals
- **THEN** the research or benchmark workflow SHALL report which intervals were evaluated and the interval-specific configuration used

#### Scenario: Skip unavailable interval explicitly
- **WHEN** a requested interval is unavailable or non-contiguous
- **THEN** the workflow SHALL skip or fail according to the caller's explicit mode and SHALL report the exact unavailable interval reason

#### Scenario: Fail clearly for missing parquet backend
- **WHEN** the workflow needs to read parquet cache data and the required parquet backend is unavailable
- **THEN** it SHALL fail before model evaluation with an actionable dependency error

### Requirement: Verification cost control
The system SHALL keep expensive cache-backed and long-horizon regime benchmark
checks opt-in while preserving fast deterministic unit coverage by default.

#### Scenario: Run default tests quickly
- **WHEN** the normal package test suite runs without cache configuration
- **THEN** it SHALL verify regime classification and generation behavior using small deterministic fixtures without requiring full cache-backed benchmarks

#### Scenario: Run long benchmarks explicitly
- **WHEN** cache-backed or long-horizon benchmark checks are requested
- **THEN** the workflow SHALL require explicit cache configuration or slow-test selection and SHALL report the evaluated intervals and seed counts

