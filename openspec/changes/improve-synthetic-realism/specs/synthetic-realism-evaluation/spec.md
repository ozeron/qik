## ADDED Requirements

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
