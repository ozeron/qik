## ADDED Requirements

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
