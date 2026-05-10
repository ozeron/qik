## ADDED Requirements

### Requirement: Provider-driven multi-asset context
The system SHALL support market context resolution through a pluggable provider instead of hard-coded profile branching inside generation models.

#### Scenario: resolve BTC defaults
- **WHEN** a BTC request is resolved through default provider
- **THEN** context SHALL match:
  - `symbol="BTCUSDT"`, `venue="binance"`, `dataset="trade_bars"`, `asset_class="crypto"`, `trading_hours="24x7"`.

#### Scenario: resolve SPY defaults
- **WHEN** a SPY request is resolved through default provider
- **THEN** context SHALL match:
  - `symbol="SPY"`, `venue="alpaca"`, `dataset="stock_bars"`, `asset_class="equity"`, `trading_hours="equity_sessions"`.

### Requirement: Context provider as lightweight boundary
The system SHALL not require a class for context resolution; any callable matching the protocol contract may be used.

#### Scenario: fast context resolution
- **WHEN** `DefaultMarketContextProvider` is used repeatedly in `generate_many`
- **THEN** context resolution may be cached or memoized, and generation remains model-centric.

### Requirement: Context mismatch validation
The system SHALL fail fast when model generation assumptions conflict with market context profile.

#### Scenario: equity profile with 24x7 expectation
- **WHEN** a model requested with equity trading assumptions is run with `trading_hours="24x7"` or vice versa
- **THEN** a `SyntheticContextError` SHALL explain the mismatch.

### Requirement: Optional compatibility with external typing contracts
The system SHALL support local Protocols that match key `market_data_lake` style fields without importing that package.

#### Scenario: protocol-compatible context
- **WHEN** a consumer builds a context provider from external typing-compatible objects
- **THEN** the engine SHALL consume it via attribute names (`symbol`, `venue`, `dataset`, `interval`) and identifier fields.

### Requirement: External payloads and fixtures policy
The system SHALL keep large synthetic fixture payloads out of package source paths.

#### Scenario: fixture location
- **WHEN** tests run or examples need payload files
- **THEN** those files SHALL be loaded from repo-level test fixtures or external package fixtures, not from `packages/qik-synthetic/*` data directories.

### Requirement: model-family extensibility
The system SHALL allow adding new generator families without changing engine core interfaces.

#### Scenario: add a new family
- **WHEN** a new `PathModel` implementation (e.g. GARCH) is registered through model registry
- **THEN** existing callers SHALL invoke it through the same `SyntheticEngine` API with only configuration change.
