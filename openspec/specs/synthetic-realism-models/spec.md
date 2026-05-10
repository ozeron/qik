# synthetic-realism-models Specification

## Purpose
TBD - created by archiving change improve-synthetic-realism. Update Purpose after archive.
## Requirements
### Requirement: Feature-based OHLCV calibration profile
The system SHALL provide a calibration profile API that captures return statistics, temporal dependence, candle geometry, volume behavior, hourly effects, identity metadata, and source-data fingerprints for an OHLCV frame.

#### Scenario: Fit profile from single-identity frame
- **WHEN** a user fits an OHLCV profile from a valid single-identity candle frame
- **THEN** the system SHALL return a profile containing feature statistics and model-ready parameters without mutating the input frame

#### Scenario: Reject ambiguous multi-identity profile input
- **WHEN** a user fits a single OHLCV profile from a frame containing multiple identities
- **THEN** the system SHALL raise a validation error or require grouped profile fitting

#### Scenario: Preserve scalar calibration compatibility
- **WHEN** a user calls the existing scalar calibration API
- **THEN** the system SHALL continue returning model parameter mappings compatible with existing baseline models, either directly or as a projection from the richer profile API

### Requirement: Profile-aware model fitting
The system SHALL provide a compatibility-safe fitting path for realism models that can consume pandas OHLCV frames or `OhlcvCalibrationProfile` objects without breaking existing generic history fitting.

#### Scenario: Fit model from frame or profile
- **WHEN** a realism model supports frame/profile fitting and receives a pandas OHLCV frame or calibration profile
- **THEN** the model SHALL fit using the richer data source while preserving existing behavior for generic row-history callers

### Requirement: Feature-driven candle construction
The system SHALL provide a candle construction path that builds OHLCV arrays from calibrated feature vectors rather than deriving candle range only from open-close body size.

#### Scenario: Build candles from features
- **WHEN** a model supplies open-gap returns, intrabar returns, upper wick sizes, lower wick sizes, and volume features
- **THEN** the system SHALL construct OHLCV arrays that satisfy high/low bounds and preserve the supplied candle geometry

### Requirement: Stationary bootstrap realism model
The system SHALL provide a stationary or block bootstrap model that fits joint OHLCV feature rows from historical data and generates synthetic candles by sampling dependent feature blocks.

#### Scenario: Generate from fitted bootstrap model
- **WHEN** a stationary bootstrap model is fitted on training OHLCV data and asked to generate a target horizon
- **THEN** the system SHALL generate structurally valid OHLCV with sampled feature dependence, deterministic seeded output, and the requested row count

#### Scenario: Preserve joint feature relationships
- **WHEN** the bootstrap model generates synthetic data
- **THEN** the sampled features SHALL preserve observed relationships among absolute returns, ranges, wicks, volume, and time-of-day effects better than independently sampled scalar defaults

#### Scenario: Report bootstrap memorization signals
- **WHEN** the bootstrap model is evaluated against held-out data
- **THEN** the system SHALL report block-overlap or memorization diagnostics as warnings/metrics and SHALL NOT fail normal package tests solely because those warning-only diagnostics are present

### Requirement: Real GARCH-family model behavior
The system SHALL ensure any model exposed as `garch` performs a real GARCH-family fit/simulation when optional model dependencies are installed, or otherwise clearly fails with an optional-dependency error.

#### Scenario: Simulate fitted GARCH model
- **WHEN** optional model dependencies are installed and the GARCH model is fitted on historical close returns
- **THEN** the model SHALL simulate seeded returns using fitted conditional volatility dynamics rather than only a scalar iid Gaussian volatility

#### Scenario: Missing optional GARCH dependency
- **WHEN** a user selects the GARCH model without the required optional dependency
- **THEN** the system SHALL raise the existing optional-dependency error path without importing heavy dependencies during baseline model usage

### Requirement: Fat-tail and volatility-clustering baseline models
The system SHALL provide at least one lower-dependency realism baseline beyond Gaussian and AR(1), such as Student-t random walk, EWMA-t, or empirical-residual bootstrap.

#### Scenario: Generate fat-tail baseline
- **WHEN** a fat-tail baseline is calibrated and generated for a validation horizon
- **THEN** the system SHALL produce deterministic seeded output and expose diagnostics showing return-tail behavior separately from candle and volume behavior

### Requirement: Interval-aware regime classification
The system SHALL classify fitted single-identity OHLCV history into interval-aware
regime labels covering volatility, trend, drawdown or recovery state, volume
state, and calendar state when timestamps are available.

#### Scenario: Fit regime labels from valid OHLCV history
- **WHEN** a user fits a realism profile or regime bootstrap model from valid single-identity OHLCV history
- **THEN** the fitted state SHALL include regime label arrays aligned one-to-one with the fitted feature rows

#### Scenario: Scale classification windows by candle interval
- **WHEN** regime labels are fitted for different candle intervals
- **THEN** the system SHALL derive lookback windows from explicit time durations or validated bar counts rather than assuming BTC 1h-only constants

#### Scenario: Reject missing timestamp calendar classification
- **WHEN** calendar-conditioned classification is requested for OHLCV history without timestamps
- **THEN** the system SHALL raise a validation error unless calendar conditioning is explicitly disabled

### Requirement: Explicit regime state contract
The system SHALL expose fitted regime artifacts through an explicit typed state
contract rather than only through generic feature or parameter dictionaries.

#### Scenario: Fit profile with regime state
- **WHEN** a profile or regime bootstrap model is fitted with regime classification enabled
- **THEN** the fitted state SHALL include typed regime labels, bucket support, transition or duration summaries, and source block metadata

#### Scenario: Reject malformed regime state
- **WHEN** a regime-conditioned model receives missing, unaligned, or malformed regime state
- **THEN** the system SHALL raise a validation error before generation starts

### Requirement: Regime-conditioned bootstrap model
The system SHALL provide a fitted `regime_conditioned_bootstrap` model that
generates OHLCV paths by simulating target regime sequences and sampling
dependent source feature blocks from matching fitted regimes.

#### Scenario: Generate deterministic regime-conditioned OHLCV
- **WHEN** a fitted `regime_conditioned_bootstrap` model receives a valid generation request with a seed and horizon
- **THEN** the system SHALL generate deterministic structurally valid OHLCV with the requested row count and matching repeated runs for the same fitted state and seed

#### Scenario: Preserve fit-once generate-many contract
- **WHEN** a user fits the model once and generates multiple paths
- **THEN** the fitted model SHALL carry the regime labels, transition or duration state, block index, and source features needed for generation without requiring callers to resend duplicated model parameters

#### Scenario: Report sampled regime and block metadata
- **WHEN** the model generates a path
- **THEN** the generated batch or benchmark output SHALL expose enough metadata to inspect sampled regimes, sampled source block ranges, relaxation counts, and repeated-block concentration

#### Scenario: Preserve regime path across engine chunks
- **WHEN** generation spans multiple internal engine blocks
- **THEN** the model SHALL preserve deterministic regime path continuity across block boundaries rather than restarting transitions for each block

#### Scenario: Register as first-class model
- **WHEN** a user lists or selects built-in models
- **THEN** `regime_conditioned_bootstrap` SHALL be available as a first-class model name without silently remapping existing legacy aliases

### Requirement: Explicit sparse-regime handling
The system SHALL fail fast for missing or under-supported regime buckets unless
the caller selects a named lower-fidelity relaxation policy.

#### Scenario: Reject sparse default bucket
- **WHEN** generation requests a regime or calendar bucket whose fitted support is below the configured minimum
- **THEN** the system SHALL raise a validation error describing the missing bucket and required relaxation policy

#### Scenario: Use named relaxation policy
- **WHEN** the caller sets an explicit sparse-bucket relaxation policy
- **THEN** the system SHALL apply only that named policy and SHALL report each relaxation in generation or benchmark diagnostics

#### Scenario: Disable hidden calendar fallback
- **WHEN** strict calendar-conditioned sampling is enabled
- **THEN** the system SHALL NOT automatically weaken a day-and-hour bucket to an hour-only bucket unless a named relaxation policy authorizes it

### Requirement: Shared regime block sampling helpers
The system SHALL keep regime path simulation, bucket support validation, and
block-index sampling behind shared helpers rather than duplicating fallback trees
inside individual models.

#### Scenario: Validate bucket support through shared helper
- **WHEN** a regime-conditioned model prepares source blocks
- **THEN** it SHALL validate bucket support through a shared helper that raises explicit validation errors for unsupported states

#### Scenario: Avoid hidden global fallback
- **WHEN** a matching regime bucket is unavailable
- **THEN** the system SHALL NOT silently sample from the global feature history

### Requirement: Regime diagnostics channel
The system SHALL provide a testable diagnostics channel for regime-conditioned
generation metadata, including sampled regimes, source block ranges, relaxation
counts, overlap metrics, and repeated-block concentration.

#### Scenario: Collect regime diagnostics
- **WHEN** `regime_conditioned_bootstrap` generates data for a benchmark run
- **THEN** the benchmark SHALL be able to collect regime diagnostics without parsing OHLCV prices heuristically

#### Scenario: Preserve generic model compatibility
- **WHEN** non-regime models generate OHLCV through the same engine
- **THEN** they SHALL continue to work without providing regime diagnostics

