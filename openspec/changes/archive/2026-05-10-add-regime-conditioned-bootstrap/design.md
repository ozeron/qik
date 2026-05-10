## Context

`qik.synthetic` already has a fit-once/generate-many realism pipeline:

- `fit_ohlcv_profile` extracts OHLCV feature arrays and scalar model params.
- `stationary_bootstrap` samples dependent feature blocks.
- `empirical_sv`, `student_t_rw`, and `garch` provide alternative baselines.
- `evaluate_btc_1h_realism.py` compares models on chronological train,
  validation, and test windows over multiple seeds.

The gap is not generic bootstrapping. The gap is stateful market regime
composition. A BTC strategy can depend on whether a sampled block belongs to a
high-volatility trend, a low-volatility chop, a drawdown, a recovery, a high
volume liquidation period, or a calendar-specific liquidity regime. Current
feature bootstraps preserve local rows but do not explicitly simulate those
regime paths.

The design must preserve repo constraints:

- no silent degradation when regime buckets are missing or malformed,
- fitted profiles/models must carry generation state,
- shared helpers own sampling/timestamp/candle behavior,
- downstream strategy optimization stays outside `qik.synthetic`.

## Goals / Non-Goals

**Goals:**

- Add an interval-aware regime classification layer for OHLCV history.
- Add a fitted `regime_conditioned_bootstrap` model that simulates regime paths
  and samples matching dependent OHLCV blocks.
- Benchmark the new model against the existing realism model set.
- Report regime/block diagnostics so we can judge whether the approach actually
  improves synthetic BTC path realism.
- Keep block-size and segment-size choices explicit, fitted, and reproducible.

**Non-Goals:**

- Do not add strategy optimization or trading rules to `qik.synthetic`.
- Do not claim generated paths are new truth or unlimited real BTC data.
- Do not add mandatory heavy dependencies.
- Do not silently fall back from sparse regime buckets to global sampling.
- Do not replace the existing bootstrap, Student-t, GARCH, or empirical SV
  models.

## Decisions

### Decision: Regime labels are fitted features, not caller-provided ad hoc params

The profile fitting path will derive regime labels from OHLCV features and store
them in the fitted profile/model state. Generation reuses those fitted labels
without requiring callers to pass duplicated `model_params`.

Alternative considered: pass regime config only through request params. That
keeps code smaller initially, but it breaks the fit-once/generate-many contract
and makes repeated generation easy to misconfigure.

### Decision: Regime state is explicit typed profile state

Regime labels, transition or duration summaries, bucket support, and source
block metadata will live in an explicit typed state object, for example
`RegimeState`, attached to the fitted profile or model. Regime data must not be
hidden only inside generic `features` or `model_params`.

Alternative considered: embed regime artifacts in `features`/`model_params`.
That is lower effort, but it weakens validation and makes malformed or missing
state harder to detect before generation.

### Decision: Start with explicit rule-based regimes before statistical state models

The first research round will define deterministic labels from interval-scaled
features:

- volatility bucket from rolling realized volatility and absolute returns,
- trend bucket from interval-scaled lookback return or moving-average slope,
- drawdown bucket from rolling peak-to-close drawdown and recovery state,
- volume bucket from log-volume residuals and rolling volume level,
- calendar bucket from hour/day where timestamps exist.

Alternative considered: start with HMM or clustering. That may become useful
later, but it makes the first implementation harder to inspect and harder to
debug when benchmark behavior is poor.

### Decision: Regime path simulation is separate from block-row sampling

Generation will have two stages:

1. simulate or bootstrap a target regime sequence,
2. sample source blocks whose fitted regime labels match the target sequence.

This keeps the model inspectable. We can diagnose whether poor output came from
bad regime transitions, sparse buckets, bad block lengths, or candle sampling.

Alternative considered: sample only blocks and infer regimes afterward. That is
close to current stationary bootstrap behavior and does not let us control or
measure path-level regime composition.

### Decision: Generation must be stable across engine block boundaries

`SyntheticEngine` may call `simulate_batch` in internal chunks for large
horizons. The regime-conditioned model must preserve request-level regime
continuity across those internal calls. Implementation may do this by
precomputing deterministic path state for the request, passing block offsets and
run identifiers into model params, or adding a narrow model-state hook, but it
must not restart regime transitions at each internal block.

Alternative considered: accept independent per-block regime paths. That would be
simpler, but it invalidates regime duration and transition diagnostics on long
horizons.

### Decision: Sparse buckets raise unless relaxation is explicit

Default sampling must fail when a requested regime/calendar bucket has too few
source rows or blocks. Relaxed behavior must be opt-in with a named policy such
as `relax_calendar`, `relax_volume`, or `nearest_regime`, and diagnostics must
report every relaxation.

Alternative considered: automatically widen buckets. That would make benchmark
runs easier but violates the repo rule against hidden degradation.

Existing time-aligned bootstrap helpers currently have permissive behavior, such
as trying a weaker calendar bucket when a stricter bucket is empty. The new
regime path must split strict and relaxed behavior into shared helpers so
fallbacks only happen through a named policy and always produce telemetry.

### Decision: Block and segment sizing are interval-aware research outputs

The first round will compare candidate scales, not hard-code one BTC 1h value:

- block length candidates in bars and time duration,
- regime segment length candidates,
- overlap and memorization rates,
- sensitivity across at least one smaller and one larger candle interval when
  cache data is available.

The model can default to fitted empirical durations, with explicit params to
override block and segment sizing for experiments.

### Decision: Model registration preserves legacy aliases

`regime_conditioned_bootstrap` will be added as a first-class model name and
included in `list_models()` and benchmark defaults. Existing `regime_bootstrap`
and `seasonal_bootstrap` aliases will remain mapped to their current behavior
unless a separate migration decision is made.

Alternative considered: remap old aliases to the new model. That is semantically
cleaner, but it silently changes behavior for existing callers.

### Decision: Regime/block diagnostics use an explicit diagnostics channel

The implementation must define where sampled regime sequences, source block
ranges, relaxation counts, overlap metrics, and repeated-block concentration are
returned. A dedicated benchmark diagnostics payload is acceptable for the first
implementation if changing `SyntheticBatch` would create broad API churn. The
contract must be testable either way.

Alternative considered: require every generated frame or batch to carry regime
metadata. That is richer, but it may over-couple generic generation outputs to a
single model family.

### Decision: Benchmark selection remains validation-first

The benchmark will fit only on train, compare models on validation, and report
test separately. The new model must be compared with the existing model set using
the same multi-seed summary style.

Alternative considered: optimize directly on test or only run visual inspection.
That would overstate improvements and make regressions hard to detect.

### Decision: Research sweeps are bounded and opt-in for expensive coverage

Candidate regime/block/interval sweeps must have explicit runtime controls:
model/candidate limits, seed limits, optional wall-clock caps, and compact
summary output. Fast deterministic unit tests remain the default verification
path; cache-backed and long-horizon research runs stay opt-in.

Alternative considered: run exhaustive multi-interval, multi-seed grids by
default. That improves confidence but risks slow, memory-heavy, or un-runnable
local checks.

### Decision: Benchmark data loading is interval-aware and dependency-checked

The benchmark workflow will move beyond a hard-coded BTCUSDT 1h loader. It must
load configured intervals, report unavailable or non-contiguous intervals, and
support explicit skip/fail behavior. Parquet backend availability must be checked
before benchmark work begins so missing optional dependencies fail clearly.

## Risks / Trade-offs

- [Risk] Regime labels become too granular and produce sparse buckets.
  - Mitigation: include bucket support diagnostics and require explicit
    relaxation policies before generation can widen buckets.
- [Risk] Rule-based regimes are too simple to capture BTC dynamics.
  - Mitigation: treat them as the first inspectable baseline and keep HMM or
    clustering as a future extension after benchmark evidence.
- [Risk] Regime-conditioned bootstrap overfits by replaying recognizable source
  blocks.
  - Mitigation: report source overlap, repeated-block concentration, and
    memorization warnings in the benchmark.
- [Risk] Interval scaling works for 1h but not 1m/4h/1d.
  - Mitigation: define windows in duration-first terms, convert to bars through
    shared interval helpers, and benchmark at multiple intervals when data is
    available.
- [Risk] New metadata increases profile size.
  - Mitigation: store compact arrays and summaries, avoid Python object rows in
    fitted model state.
- [Risk] Engine chunking hides model-state resets in long paths.
  - Mitigation: add tests that force multiple internal generation blocks and
    verify regime continuity metadata across the boundary.
- [Risk] Research sweeps become too slow to run during normal development.
  - Mitigation: keep small deterministic tests in the default suite and make
    cache-backed sweeps explicitly configured with bounded candidates/seeds.
