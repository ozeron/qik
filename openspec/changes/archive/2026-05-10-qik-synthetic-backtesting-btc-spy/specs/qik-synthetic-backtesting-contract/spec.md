## ADDED Requirements

### Requirement: Request contract includes time-aware horizon
The system SHALL expose a request contract that can describe a synthetic generation run by either explicit `n_bars` or time span (`end_ts`/`duration`) plus `start_ts` and `interval`.

#### Scenario: resolve_horizon helper
- **WHEN** a request has `start_ts` + `interval` + `duration`
- **THEN** the engine SHALL resolve `n_bars` using the chosen calendar policy.
- **WHEN** both `duration`/`end_ts` and `n_bars` are provided
- **THEN** the system SHALL reject the request as invalid.

### Requirement: Deterministic backtesting-compatible output
The system SHALL generate synthetic OHLCV using a canonical schema:
`timestamp`, `open`, `high`, `low`, `close`, `volume`.

#### Scenario: stable column order and dtypes
- **WHEN** `SyntheticEngine.generate` is called successfully
- **THEN** output SHALL be deterministic for fixed request and seed.
- **AND** every generated row SHALL satisfy:
  - `timestamp` monotonic non-decreasing and UTC-normalized,
  - `open`, `high`, `low`, `close` > 0,
  - `low <= min(open, close)` and `high >= max(open, close)`,
  - `volume >= 0`.

### Requirement: Streamed generation with chunk-level determinism
The system SHALL provide chunked generation and keep per-request reproducibility.

#### Scenario: chunk iteration
- **WHEN** `iter_generate(request, chunk_size=N)` is called with a positive `N`
- **THEN** it SHALL yield iterator chunks whose concatenation is equivalent to `generate(request)`.
- **AND** re-running with same request and seed SHALL return identical chunks byte-for-byte.
- **AND** chunks SHALL be produced from contiguous array blocks without per-bar Python iteration in core model implementations.

### Requirement: Class orchestration vs function compute boundary
The system SHALL keep class-based orchestration (`SyntheticEngine`, model objects) separate from vectorized compute kernels.

#### Scenario: internal implementation shape
- **WHEN** a built-in model is executed
- **THEN** its hot path SHALL call pure functions that work on preallocated arrays and return NumPy arrays, then transform them into batches once per chunk.
- **AND** no Python loop over bars may be required for the default fast path.

### Requirement: Explicit backtesting handoff boundary
The system SHALL expose an adapter that converts a synthetic batch into backtesting payload only after metadata identity checks.

#### Scenario: identity validation on handoff
- **WHEN** converting to payload
- **THEN** required backtesting metadata SHALL be present (`symbol`, `venue`, `interval`, `dataset`), either from request or from resolved context.
- **WHEN** metadata is missing
- **THEN** a `SyntheticConversionError` SHALL be raised and no payload SHALL be returned.

### Requirement: Backtesting payload schema
The backtesting payload SHALL include:
- canonical OHLCV table/dataframe,
- `request_id`, `run_id`, `seed`, `model_name`, `profile_id/context_id`,
- a strict `required_columns` declaration.

#### Scenario: payload reproducibility
- **WHEN** the same request + seed + model family are executed twice
- **THEN** payload metadata (`run_id`, row count, first/last timestamp, first close) SHALL be identical.

### Requirement: Compatibility utility for existing callers
The system SHALL keep old functions available and allow old call patterns to still produce a valid synthetic batch that can be adapted for conversion.

#### Scenario: legacy compatibility path
- **WHEN** legacy `build_market_path` is used
- **THEN** it SHALL still return the existing `MarketSeries` shape and support wrapping into `SyntheticBatch` through a compatibility helper.
