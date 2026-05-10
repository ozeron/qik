## Context

`qik-synthetic` should be a small, composable package that cleanly serves two needs:
- generation quality (multiple model families),
- research interoperability (OHLCV backtesting payloads).

To keep this clean, generation is model-centric and identity-agnostic; backtesting identity is attached by an adapter step.

## Goals / Non-Goals

**Goals**
- Provide class-oriented generation with minimal coupling.
- Enable fast, deterministic generation for short and very long horizons.
- Make conversion to backtesting data explicit and validated.
- Keep test fixtures and integration payload data outside package source.
- Keep compatibility path for current callers.

**Non-Goals**
- Full exchange/venue data ingestion.
- Strategy optimization logic.
- Mandatory dependency on `~/data` package or any heavy stats packages in the core package.

## API and architecture

Use an additive, protocol-driven design with dataclasses and functions only where needed.

### Performance strategy

- Model execution must be split into two layers:
  - orchestration layer (light classes)
  - compute layer (pure, vectorized functions and minimal state objects)
- Every built-in model implementation MUST avoid Python loops over bars in the default path.
- The compute layer SHOULD expose:
  - vectorized returns simulation (arrays),
  - vectorized price reconstruction (`cumprod`/`cumlog` style),
  - vectorized OHLCV conversion from close path and range multipliers.
- `iter_generate` must stream chunks as array blocks and only materialize Python objects at chunk boundaries.

### New modules (planned)

- `packages/qik-synthetic/src/qik/synthetic/protocols.py`
  - `class PathModel(Protocol)`
    - `fit(data: Sequence[Mapping[str, Any]], *, context: SyntheticContext | None = None) -> PathModel`
    - `simulate_returns(ctx: SyntheticContext, n: int, *, seed: int | None = None, params: Mapping[str, Any] | None = None) -> np.ndarray`
    - `simulate_batch(ctx: SyntheticContext, n: int, *, seed: int | None = None, params: Mapping[str, Any] | None = None) -> np.ndarray`
  - `class MarketContextProvider(Protocol)`
    - `resolve(symbol: str, *, venue: str | None = None, dataset: str | None = None, interval: str | None = None) -> SyntheticContext`
  - `class BacktestingAdapter(Protocol)` for optional integration adapters.
- `packages/qik-synthetic/src/qik/synthetic/kernels.py`
  - Pure NumPy helpers for:
    - noise generation with deterministic seeding,
    - close/path reconstruction,
    - open/high/low synthesis from close and volatility envelopes,
    - volume sampling.
- `packages/qik-synthetic/src/qik/synthetic/types.py`
  - `@dataclass(frozen=True) SyntheticContext`
    - `symbol, venue, dataset, interval, asset_class, trading_hours, timezone`
  - `@dataclass(frozen=True) SyntheticRequest`
    - `start_price: float`, `interval: str`, `seed: int | None`, `model: str`, `symbol: str | None`, `venue: str | None`, `dataset: str | None`
    - time inputs: `start_ts: datetime`, `n_bars: int | None`, `duration: timedelta | None`, `end_ts: datetime | None`
    - optional model params dict
  - `@dataclass(frozen=True) SyntheticChunk`
    - `bars: tuple[dict, ...]`, `chunk_index`, `run_id`
  - `@dataclass(frozen=True) SyntheticBatch`
    - `bars: tuple[dict, ...]`, `context: SyntheticContext`, `request: SyntheticRequest`, `metadata: SyntheticMetadata`
- `packages/qik-synthetic/src/qik/synthetic/errors.py`
  - `SyntheticError`, `SyntheticValidationError`, `SyntheticContextError`, `SyntheticConversionError`
- `packages/qik-synthetic/src/qik/synthetic/horizon.py`
  - `bars_from_span(start_ts, interval, *, end_ts=None, duration=None, n_bars=None, trading_hours="24x7") -> int`
- `packages/qik-synthetic/src/qik/synthetic/engine.py`
  - `@dataclass class SyntheticEngine`
    - `model: PathModel`
    - `context_provider: MarketContextProvider | None`
    - `seed: int | None`
    - `def fit(self, history: Sequence[Mapping[str, Any]]) -> "SyntheticEngine"`
    - `def generate(self, request: SyntheticRequest) -> SyntheticBatch`
    - `def iter_generate(self, request: SyntheticRequest, *, chunk_size: int = 200_000) -> Iterator[SyntheticChunk]`
    - `def generate_many(self, requests: Sequence[SyntheticRequest], *, strategy: str = "materialize") -> Sequence[SyntheticBatch]`
- `packages/qik-synthetic/src/qik/synthetic/backtesting.py`
  - `to_backtesting_frame(batch: SyntheticBatch) -> pd.DataFrame`
  - `to_backtesting_payload(batch: SyntheticBatch, *, run_context: Mapping[str, Any] | None = None) -> Mapping[str, Any]`
  - `validate_backtesting_payload(payload: Mapping[str, Any]) -> None`
- `packages/qik-synthetic/src/qik/synthetic/registry.py`
  - model registry for named families, e.g. `"gaussian_rw"`, `"ar1"`, `"bootstrap"`
- `packages/qik-synthetic/src/qik/synthetic/models/` for built-ins and optional heavy models.
  - built-in: gaussian random walk and optional AR1 family.
  - optional gated models under `qik-synthetic[models]`: GARCH variants.

### Profile/context strategy

- No hard-coded profile list in core model logic.
- Context defaults are provider-driven.
- `DefaultMarketContextProvider` supports:
  - BTC -> `symbol="BTCUSDT"`, `venue="binance"`, `dataset="trade_bars"`/`trading_hours="24x7"`
  - SPY -> `symbol="SPY"`, `venue="alpaca"`, `dataset="stock_bars"`/`trading_hours="market_sessions"`
- Custom providers can be supplied by callers for in-house metadata or venue overrides.

### Identity policy

- Core generation can operate with `symbol`/`venue`/`dataset` absent.
- Backtesting helpers require resolved identity; they fail fast if any of required identity fields are missing.
- This keeps the core focused and still guarantees backtesting safety at handoff.

### Streaming performance contract

- `iter_generate` produces chunks of OHLCV rows using vectorized numpy/state math.
- The engine must avoid per-row Python loops in default path.
- Chunks are deterministically reproducible using request seed plus chunk index.
- `generate_many` groups requests by `(model family, interval, context profile)` when possible to reduce repeated setup and timestamp calculations.
- Deterministic chunking MUST use independent child seeds derived from `(run_seed, chunk_index)` so resume/replay remains stable.

### Optional heavy dependencies

- Core package dependencies remain minimal.
- `project.optional-dependencies.models` includes optional installs for `statsmodels` and `arch` to enable GARCH/statistical families.
- Modules under `models/` import heavy deps lazily inside call sites.

## Risks / Trade-offs

- [Risk] Divergence between custom adapters and package defaults.
  - [Mitigation] define a strict `SyntheticContext` contract and validate it at adapter boundaries.
- [Risk] Very large chunk output may exhaust memory if always materialized.
  - [Mitigation] prefer `iter_generate` for long horizons and expose chunk tuning.
- [Risk] Optional model deps not available in target env.
  - [Mitigation] clean failure message + lazy import boundaries + no runtime import in core.

## Migration and compatibility

- Existing functions remain available and should delegate into the new contracts.
- New API is additive; old `build_market_path` path remains supported as a simple baseline.
- Add short deprecation notes where old patterns become less explicit for context handling.

## Non-functional requirements

- Determinism across runs with same `SyntheticRequest` + model family.
- No framework-specific dataset shapes inside core models.
- Fast-path generation for >1e7 rows should use chunk mode by default in scripts.
- `generate_many` should preserve vectorized execution by batching requests before fallback to per-request loops.
