## ADDED Requirements

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
