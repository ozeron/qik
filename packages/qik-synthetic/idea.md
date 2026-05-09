Alex, build **Synthetic Data Generator** as a small, model-agnostic candle simulation library where the simulator is treated as a research instrument, not as evidence of profitability. The first serious MVP should be: **validated OHLCV input → log-return transforms → simple linear return model / bootstrap baseline → empirical OHLCV reconstruction → real-vs-synthetic diagnostics → strategy robustness runner**. Then add **GARCH(1,1) with Student-t innovations** as the second major model, because BTC’s volatility clustering and fat tails will make simple linear models look fake very quickly. 🚀

## 1. Restate the goal clearly

The package should take historical OHLCV candles, fit one or more explicit statistical or simulation models to the candle-generating process, and produce many synthetic OHLCV paths that preserve selected statistical features of the original data closely enough to support strategy robustness research, parameter sensitivity analysis, stress testing, and anti-overfitting workflows.

The important distinction is this:

| Activity                              | What it does                                                                                                 | What it does **not** prove                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Fitting a price/candle model          | Estimates a model of returns, volatility, ranges, volume, regimes, or candle shapes from historical candles. | It does not discover a trading edge by itself.                                      |
| Generating synthetic paths            | Samples alternative market histories from the fitted model.                                                  | It does not create “more truth”; it creates more scenarios under model assumptions. |
| Testing strategies on synthetic paths | Evaluates how strategy behavior changes across many plausible paths.                                         | It does not replace real out-of-sample testing.                                     |
| Optimizing without overfitting        | Selects parameters that are stable across many paths, nearby parameter values, and stress cases.             | It does not guarantee the future realized path will reward those parameters.        |

The package should therefore be designed around this principle: **fit the data-generating process first; use synthetic paths to punish fragile strategies second; validate once on untouched real out-of-sample data last.**

## 2. Extract the key principles from Dr. Chan’s transcript

### Principles directly supported by the transcript

Chan’s core critique is that traditional backtesting often optimizes strategy parameters against one historical path, where the number of actual trading decisions is much smaller than the number of available price observations. Because the optimizer has few strategy signals to work with, it can easily cherry-pick parameters that happened to fit historical coincidences.

He argues that fitting a **time-series model to prices** is less overfit-prone than fitting a **trading strategy directly to trades**, because price data usually gives many more observations than trade signals. That does not make the fitted price model “true,” but it shifts the optimization problem to a statistically richer target.

He distinguishes simple analytical models from simulation models. Analytical models can be elegant, but they usually require simple price processes, simple trading rules, and simple linear objectives. Simulation-based models can handle nonlinear strategies, transaction costs, drawdown constraints, Sharpe-like objectives, volatility clustering, jumps, tails, autocorrelation, and other practical features because they are not limited by closed-form tractability.

He explicitly recommends fitting a discrete time-series model, generating many synthetic paths, evaluating a strategy across those paths, and optimizing the strategy objective across the path ensemble. In his example, he starts with AR(1), then notes that adding GARCH changed the optimal strategy parameter, implying that volatility modeling mattered.

He also stresses a subtle but crucial point: even if a parameter maximizes average Sharpe across simulated paths, the future is one realized path. That realized path can still make a supposedly “suboptimal” parameter look better. The method improves robustness; it does not eliminate randomness.

Finally, Chan says the method becomes more statistically reliable when applied across many instruments, because one realized path can be unlucky, but a diversified ensemble gives the law of averages more room to work.

### Reasonable implementation interpretations

For this package, the implementation interpretation should be: **do not optimize strategy parameters against raw BTC history first**. Fit a return/volatility/candle model on a training window, generate many alternative BTC-like histories, evaluate parameter stability across those histories, then lock parameters before touching the real out-of-sample window.

A second interpretation: the simulator should not be one model. Chan’s argument is not “AR(1) is enough.” It is “fit a discrete model to the data-generating process, and make the model as realistic as needed.” So your package should expose a common interface behind which you can plug in bootstrap models, linear models, GARCH, jump models, regime models, and later neural generators if they earn their place.

A third interpretation: model selection should be driven primarily by **statistical diagnostics of the generated data**, not by whether the synthetic data makes your strategy look profitable. Choosing the simulator because it flatters the strategy is just overfitting with extra steps.

For BTC specifically, Chan’s mean-reversion example should not be copied blindly. BTC is often non-stationary in price level, volatile, jumpy, regime-dependent, and historically upward-trending. A BTC simulator should usually model **log returns, volatility, jumps, ranges, volume, and regimes**, not raw log price as a stationary AR(1) process.

### Open questions or assumptions

The transcript does not specify how to generate OHLCV candles from simulated close prices. That is a package design problem you need to solve explicitly.

The transcript does not discuss BTC. Applying the methodology to BTC is a reasonable extension, but not directly supported by the source material.

The transcript does not define a model-selection protocol beyond “use established statistical procedures.” You will need concrete diagnostics and acceptance criteria.

The transcript does not solve the model-risk problem. If the fitted simulator misses the market behavior your strategy depends on, synthetic-path optimization can produce false confidence.

The transcript does not say that synthetic data makes a strategy profitable. In fact, the transcript’s example shows the opposite lesson: the optimized parameter can underperform on one out-of-sample realization.

## 3. Define the minimal viable package

Use a real package structure from day one. Not a notebook blob. The notebooks should call package functions, not contain the core implementation.

A practical initial layout:

```text
synthetic_data_generator/
  pyproject.toml
  README.md

  src/
    synthetic_data_generator/
      __init__.py

      schema.py
      config.py
      exceptions.py

      data/
        __init__.py
        validation.py
        calendar.py
        loaders.py
        splitting.py

      transforms/
        __init__.py
        returns.py
        candle_shapes.py
        volume.py
        scaling.py

      models/
        __init__.py
        base.py
        iid_bootstrap.py
        block_bootstrap.py
        linear_returns.py
        gbm.py
        garch.py
        jumps.py
        regimes.py

      simulation/
        __init__.py
        engine.py
        timestamps.py
        ohlcv.py
        random.py

      diagnostics/
        __init__.py
        distribution.py
        autocorrelation.py
        volatility.py
        drawdowns.py
        candle_stats.py
        volume.py
        strategy_probes.py
        report.py

      strategy/
        __init__.py
        protocols.py
        evaluation.py
        metrics.py
        optimization.py

      io/
        __init__.py
        serialization.py

      visualization/
        __init__.py
        plots.py

  notebooks/
    01_data_loading_and_cleaning.py
    02_baseline_bootstrap.py
    03_fit_simple_return_models.py
    04_garch_or_volatility_model.py
    05_ohlcv_reconstruction.py
    06_synthetic_vs_real_diagnostics.py
    07_strategy_robustness_test.py
    08_walk_forward_synthetic_optimization.py

  tests/
    test_schema.py
    test_transforms.py
    test_iid_bootstrap.py
    test_block_bootstrap.py
    test_linear_returns.py
    test_ohlcv_reconstruction.py
    test_diagnostics.py
    test_serialization.py
    test_strategy_evaluation.py
```

Responsibilities:

| Module                        | Responsibility                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `schema.py`                   | Defines canonical candle schema, validation settings, metadata types, model configs.                   |
| `data/validation.py`          | Checks OHLCV invariants, monotonic timestamps, missing bars, duplicates, symbol/timeframe consistency. |
| `data/calendar.py`            | Generates timestamps for crypto 24/7 data first; later handles equities/futures calendars.             |
| `data/splitting.py`           | Train/test splits, rolling windows, walk-forward windows, no-leakage helpers.                          |
| `transforms/returns.py`       | Log returns, price reconstruction, demeaning, scaling, winsorization flags.                            |
| `transforms/candle_shapes.py` | Encodes candle shapes: gaps, body, upper/lower wick, range, normalized range.                          |
| `transforms/volume.py`        | Log-volume transforms, volume seasonality, volume-return conditioning.                                 |
| `models/base.py`              | Common model interface.                                                                                |
| `models/iid_bootstrap.py`     | Naive but necessary lower-bound baseline.                                                              |
| `models/block_bootstrap.py`   | Time-series bootstrap preserving local dependence.                                                     |
| `models/linear_returns.py`    | Simple drift / AR / ARMA-style return model.                                                           |
| `models/garch.py`             | Volatility clustering model.                                                                           |
| `simulation/ohlcv.py`         | Converts simulated returns into valid OHLCV candles.                                                   |
| `diagnostics/*`               | Real-vs-synthetic statistical comparisons.                                                             |
| `strategy/*`                  | Strategy protocol, evaluation over many paths, metrics aggregation.                                    |
| `io/serialization.py`         | Save/load fitted model artifacts with versioning.                                                      |
| `visualization/plots.py`      | Lightweight plotting wrappers; notebooks decide presentation.                                          |

Keep the first version pandas-native. You can accept Polars later, but `statsmodels` and `arch` workflows are naturally pandas/numpy-friendly, so pandas is the path of least friction for the MVP. `statsmodels` supports AR/ARMA/ARIMA-style modeling, while `arch` supports common GARCH-style volatility models and time-series bootstraps. ([Statsmodels][1])

## 4. Define the core abstraction

The core abstraction should separate **unfitted model config**, **fitted model**, and **simulation request**.

A compact sketch:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class CandleSchema:
    timestamp_column: str = 'timestamp'
    open_column: str = 'open'
    high_column: str = 'high'
    low_column: str = 'low'
    close_column: str = 'close'
    volume_column: str = 'volume'
    symbol_column: str | None = 'symbol'
    timeframe_column: str | None = 'timeframe'


@dataclass(frozen=True)
class SimulationRequest:
    n_bars: int
    n_paths: int = 1
    start_price: float | None = None
    start_timestamp: pd.Timestamp | None = None
    timeframe: str | None = None
    symbol: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DiagnosticReport:
    summary: pd.DataFrame
    warnings: list[str]
    metrics: dict[str, float]


class SyntheticCandleModel(Protocol):
    model_name: str

    def fit(self, candles: pd.DataFrame, schema: CandleSchema = CandleSchema()) -> FittedSyntheticCandleModel:
        ...


class FittedSyntheticCandleModel(Protocol):
    model_name: str
    fitted_at: pd.Timestamp

    def simulate(self, request: SimulationRequest) -> pd.DataFrame:
        ...

    def diagnostics(self, real_candles: pd.DataFrame, synthetic_candles: pd.DataFrame) -> DiagnosticReport:
        ...

    def save(self, path: Path) -> None:
        ...

    @classmethod
    def load(cls, path: Path) -> FittedSyntheticCandleModel:
        ...
```

Expected input: a pandas `DataFrame` with one row per candle. MVP schema:

| Column      |                       Type | Rule                                 |
| ----------- | -------------------------: | ------------------------------------ |
| `timestamp` | datetime64, preferably UTC | monotonic within symbol/timeframe    |
| `open`      |                      float | positive                             |
| `high`      |                      float | positive, `high >= max(open, close)` |
| `low`       |                      float | positive, `low <= min(open, close)`  |
| `close`     |                      float | positive                             |
| `volume`    |                      float | non-negative                         |
| `symbol`    |           string, optional | required for multi-asset data        |
| `timeframe` |           string, optional | useful metadata: `15m`, `1h`, `1d`   |

Expected output: a long-form pandas `DataFrame` with columns:

```text
path_id, timestamp, open, high, low, close, volume, symbol, timeframe
```

Avoid returning a giant 3D numpy array as the public interface. Arrays are fine internally, but downstream strategy evaluation, plotting, diagnostics, and serialization will be easier with long-form frames.

What models should fit:

| Component             | MVP choice                                                                  | Why                                              |
| --------------------- | --------------------------------------------------------------------------- | ------------------------------------------------ |
| Price level           | Usually **do not** fit raw price level directly                             | BTC price level is non-stationary.               |
| Close-to-close return | Fit `log(close_t / close_{t-1})`                                            | Stable core transform for most models.           |
| Volatility            | Fit rolling vol or GARCH later                                              | BTC needs volatility dynamics.                   |
| Open                  | For crypto intraday, default `open_t ≈ close_{t-1}` plus optional gap       | Crypto is 24/7, but exchange gaps/outages exist. |
| High/low range        | Fit empirical candle-shape distribution                                     | Close-only paths are not enough.                 |
| Volume                | Fit `log1p(volume)` distribution, preferably conditional on volatility/time | Volume is non-Gaussian and regime-dependent.     |
| Intrabar shape        | Bootstrap from empirical shapes conditional on return/volatility            | Fast, realistic, hard to beat in MVP.            |

Useful transforms:

```text
return_t = log(close_t / close_{t-1})
gap_t = log(open_t / close_{t-1})
body_t = log(close_t / open_t)

upper_wick_t = log(high_t / max(open_t, close_t))
lower_wick_t = log(min(open_t, close_t) / low_t)

range_t = log(high_t / low_t)
log_volume_t = log1p(volume_t)
```

For crypto 15m/1h candles, `gap_t` will often be near zero but should still be measured instead of assumed away.

## 5. Recommend model families to support

Do not try to support everything. Support a staged set where every model teaches you something.

| Stage | Model                                   | Captures                                                           | Fails                                                | BTC suitability                                                     | Fit parameters                                            | Candle generation                                            |
| ----- | --------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| 0     | IID return bootstrap                    | Empirical return distribution, some tails                          | Destroys autocorrelation and volatility clustering   | Useful as a dumb baseline only                                      | Empirical distribution of returns and candle shapes       | Sample historical return rows or return + shape rows         |
| 1     | GBM / random walk with drift            | Drift and unconditional variance                                   | No fat tails, no vol clustering, no regimes          | Useful null model, not realistic BTC                                | Mean and std of log returns                               | Simulate returns, reconstruct candles empirically            |
| 2     | Linear return model: AR(1), AR(p), ARMA | Linear autocorrelation, short-term momentum/reversal               | Poor tails, poor volatility dynamics                 | Good first “Chan-style” fitted model, but use returns not raw price | Intercept, AR/MA coefficients, innovation distribution    | Simulate returns recursively                                 |
| 3     | Block / stationary bootstrap            | Local dependency, empirical candle shapes, fat tails inside blocks | Cannot invent unseen regimes; block length sensitive | Very strong baseline for BTC 15m/1h                                 | Block length, sampling method                             | Resample blocks of transformed candles                       |
| 4     | GARCH / EGARCH / GJR-GARCH              | Volatility clustering, conditional heteroskedasticity              | Weak on regime shifts and jumps unless extended      | Highly relevant for BTC                                             | Mean model, `omega`, `alpha`, `beta`, distribution params | Simulate conditional volatility and returns                  |
| 5     | Heavy-tail / jump extension             | Extreme moves and crash/rally discontinuities                      | Jump timing may be arbitrary                         | Relevant for BTC                                                    | Student-t df, mixture weights, jump intensity/size        | Inject jump innovations into return model                    |
| 6     | Regime-aware mixture                    | Bull/bear/high-vol/low-vol dynamics                                | Easy to overfit regimes; label instability           | Very relevant, but not MVP                                          | Regime classifier/HMM params, regime-specific models      | Simulate regime path, then regime-conditional returns/shapes |

The **first model** I’d implement is not pure GBM. It is a small `LinearReturnModel`:

```text
r_t = μ + φ_1 r_{t-1} + ε_t
```

with two innovation options:

```text
ε_t ~ Normal(0, σ)
ε_t sampled from fitted residuals
```

That gives you a clean linear model first, while the empirical-residual option avoids pretending Gaussian shocks are realistic.

The **second model** should be `GarchReturnModel`:

```text
r_t = μ + ε_t
ε_t = σ_t z_t
σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²
z_t ~ Student-t or Normal
```

The `arch` package’s documentation presents this basic GARCH structure and supports common univariate volatility modeling workflows, so it is justified once your linear baseline is working. ([Arch Documentation][2])

### Model-by-model notes

**IID return bootstrap baseline**

This is your “how bad is naive?” baseline. Sample historical log returns with replacement. For OHLCV, either sample entire transformed candle-shape rows with the return or sample return and shape conditionally. It captures the empirical marginal return distribution, including some real fat tails, but destroys temporal structure. For BTC, this will usually produce paths with too little volatility clustering.

Diagnostics: return distribution, tail quantiles, ACF of returns, ACF of absolute returns, rolling volatility distribution.

**GBM / random-walk-style baseline**

Fit mean and volatility of log returns. Optionally fit Student-t innovations instead of Gaussian. This gives you a null model: “what if BTC is basically drift plus noise?” It is useful because any fancy model should beat it diagnostically. But GBM is too smooth in its assumptions: no volatility clustering, weak tails unless extended, no volume dynamics, no regime shifts.

Diagnostics: compare real vs synthetic rolling volatility, drawdown distribution, and extreme-return frequency. Expect failure.

**AR / ARMA-style return model**

Fit returns, not prices, unless you have strong evidence that the price/log-price level is stationary. For BTC, raw log-price AR(1) will often just encode non-stationary trend and produce fake mean reversion or fake drift. Return AR models are cleaner:

```text
r_t = μ + φ_1 r_{t-1} + ... + φ_p r_{t-p} + ε_t
```

This captures linear momentum/reversal effects. It fails on conditional volatility unless paired with residual resampling or GARCH. `statsmodels` supports ARIMA-type models, including AR, MA, ARMA, ARIMA, and seasonal variants. ([Statsmodels][1])

Diagnostics: residual ACF, Ljung-Box-style checks, residual distribution, squared residual ACF.

**Block bootstrap / stationary bootstrap**

This may become your most useful nonparametric baseline. Resample contiguous blocks of transformed candles so you preserve local dependence, volatility clusters, and realistic candle shapes. Stationary bootstrap uses random block lengths; moving/circular block bootstraps use fixed-length blocks. The `arch` package includes stationary, circular, and moving block bootstraps, and `tsbootstrap` is another open-source package focused on time-series bootstrapping. ([bashtage.github.io][3])

Failure mode: it recombines history rather than generating genuinely new dynamics. It can accidentally preserve too much of the original path. It also cannot simulate a crash regime that was absent from the training window.

Diagnostics: block-length sensitivity, repeated subsequence detection, rolling-vol distribution, return ACF, drawdown distribution.

**GARCH or similar volatility clustering model**

GARCH is the first serious BTC upgrade after linear returns. Use constant mean or AR mean plus GARCH(1,1), preferably with Student-t innovations. BTC returns often exhibit volatility clustering, so a model with time-varying conditional variance is much more plausible than a homoskedastic AR model.

Failure mode: GARCH can match volatility clustering while still underestimating regime breaks, jump clustering, leverage/asymmetry, liquidity shocks, and long-memory volatility.

Diagnostics: standardized residual distribution, residual ACF, squared standardized residual ACF, volatility forecast calibration, tail exceedance frequency.

**Jump or heavy-tail extension**

Start with Student-t innovations before implementing explicit jumps. Student-t gets you heavier tails with very little machinery. Later, add a mixture innovation:

```text
ε_t =
  normal_or_t_shock_t + jump_t

jump_t = Bernoulli(λ) * jump_size_t
```

This helps generate sudden discontinuities. BTC needs this, but jump models are easy to overfit because rare events are rare by definition.

Diagnostics: exceedance counts at 3σ/5σ, left/right tail asymmetry, max single-bar move distribution, drawdown onset speed.

**Regime-aware or mixture model**

Only add this after you have diagnostics. Start simple: classify each historical bar/window by rolling volatility and rolling trend, then fit separate return/shape distributions per regime. Later use Markov-switching/HMM-like models. `statsmodels` includes nonlinear time-series tools such as Markov switching dynamic regression/autoregression, but I would not start there. ([Statsmodels][4])

Failure mode: regime labels can become a new overfitting surface. If your regime model is selected because it improves your strategy, you are leaking strategy preference into the simulator.

## 6. Handle OHLCV generation carefully

Close-only simulation is not enough. Many strategies depend on highs/lows, stops, intrabar range, volatility estimates, and volume filters. A synthetic candle package must produce valid and statistically plausible OHLCV, not just plausible closes.

The best MVP approach is:

1. Simulate close-to-close log returns.
2. Reconstruct close prices.
3. Generate open from previous close plus sampled gap.
4. Generate high/low using empirical wick/range shapes conditional on return magnitude or volatility regime.
5. Generate volume from empirical or conditional volume model.
6. Enforce candle invariants.

A robust reconstruction formula:

```text
previous_close = C_{t-1}
simulated_return = r_t

open_t = previous_close * exp(gap_t)
close_t = previous_close * exp(simulated_return)

high_t = max(open_t, close_t) * exp(upper_wick_t)
low_t = min(open_t, close_t) * exp(-lower_wick_t)
```

Where `gap_t`, `upper_wick_t`, and `lower_wick_t` can be sampled from empirical candle-shape rows. For crypto intraday bars, `gap_t` should usually be tiny, but keeping it in the model makes the architecture work later for stocks/futures.

Better than unconditional shape sampling: condition candle shapes on:

```text
return_sign: up / down / flat
abs_return_quantile: small / medium / large / extreme
volatility_regime: low / medium / high
hour_of_week: optional for BTC
```

This matters because large return bars usually have different wick/range/volume profiles than quiet bars.

For volume, do not start with a parametric fantasy model. Use:

```text
log_volume_t = log1p(volume_t)
```

Then sample log-volume conditionally on volatility regime, absolute return bucket, and optionally hour-of-week. Later you can model volume with AR/lognormal/mixture models.

Hard invariants every synthetic candle must satisfy:

```text
high >= max(open, close)
low <= min(open, close)
open > 0
high > 0
low > 0
close > 0
volume >= 0
timestamp is monotonic within each path/symbol/timeframe
no duplicate timestamps within each path/symbol/timeframe
no NaN or infinite values
```

Soft invariants to monitor:

```text
high / low range should not be absurd
volume should not collapse to constant values
open gaps should match asset/timeframe behavior
synthetic candles should not have too many doji candles
synthetic paths should not have identical repeated chunks unless using bootstrap knowingly
```

## 7. Design diagnostics

Your diagnostics should answer one question: **what properties of the real data does this simulator preserve, and what does it destroy?**

Treat diagnostics as part of the product, not as notebook garnish.

### Real-vs-synthetic statistical diagnostics

| Area                       | Diagnostic                                                     | Why it matters                                                    |
| -------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| Returns                    | Histogram, ECDF, QQ plot, mean/std/skew/kurtosis               | Checks marginal distribution.                                     |
| Tails                      | 1%, 5%, 95%, 99% quantiles; exceedance counts; max bar move    | BTC tails matter.                                                 |
| Autocorrelation            | ACF/PACF of returns                                            | Detects linear dependence.                                        |
| Volatility clustering      | ACF of absolute and squared returns                            | Detects conditional heteroskedasticity.                           |
| Rolling volatility         | Distribution of rolling vol over multiple windows              | Checks regime-like vol behavior.                                  |
| Drawdowns                  | Max drawdown distribution, drawdown duration, recovery time    | Strategy risk depends on path shape.                              |
| Trend persistence          | Run lengths, directional streaks, rolling drift distribution   | Important for trend strategies.                                   |
| OHLC ranges                | `log(high / low)`, body/range, wick ratios                     | Stops and volatility filters depend on this.                      |
| Volume                     | `log1p(volume)` distribution, volume ACF, volume seasonality   | Volume filters can be fragile.                                    |
| Return-volume relationship | Corr/conditional distribution of volume vs `abs(return)`       | Large moves often come with volume changes.                       |
| Regimes                    | Rolling trend vs rolling vol scatter, regime transition counts | Checks whether simulated market “weather” resembles real weather. |
| Cross-path diversity       | Distribution across paths, not just average path stats         | Avoids generating clones.                                         |

Financial return diagnostics should explicitly check fat tails and volatility clustering; these are standard stylized facts in asset returns, and they are exactly the kind of behavior simple Gaussian random walks miss. ([Wharton Statistics][5])

### BTC-specific stylized diagnostics

For BTC 15m/1h candles, include:

```text
hour-of-day and day-of-week volume profile
hour-of-day and day-of-week volatility profile
weekend vs weekday volatility/volume
extreme bar frequency
large move clustering
rolling 7d / 30d / 90d volatility distribution
large drawdown frequency
post-jump continuation/reversal behavior
```

High-frequency Bitcoin studies have reported regime differences and abrupt volatility changes, so rolling and regime-level diagnostics are more informative than one global summary statistic. ([arXiv][6])

### Strategy-facing diagnostics

These are not proof of edge. They are behavioral probes.

Run simple strategies on real and synthetic data:

| Probe strategy                                              | Purpose                                                    |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| Simple trend-following: moving-average or Donchian breakout | Does synthetic data preserve trendability or destroy it?   |
| Simple mean-reversion: z-score reversal after large move    | Does synthetic data create fake reversion?                 |
| Volatility breakout                                         | Are high/low ranges realistic enough for breakout testing? |
| Random-entry with same holding period                       | Detects whether synthetic paths are mechanically too easy. |
| Buy-and-hold / long-only BTC                                | Checks drift realism and bull-market bias.                 |

Questions to ask:

```text
Do synthetic paths make every trend strategy profitable?
Do synthetic paths destroy all momentum?
Are paths too smooth?
Are paths too noisy?
Do drawdowns look too shallow?
Are generated highs/lows making stops unrealistically easy or hard?
Does the simulator preserve or erase the exact feature my strategy relies on?
```

If the strategy works only on one simulator family, be suspicious. If it works across multiple imperfect simulators and real out-of-sample data, that is more interesting.

## 8. Explain how this should be used for strategy optimization

The workflow should be strict:

```text
1. Split real data into train / validation / untouched out-of-sample.
2. Fit simulator only on the training window.
3. Generate many synthetic paths from the fitted simulator.
4. Evaluate candidate strategy parameters across synthetic paths.
5. Select parameters using robust aggregate criteria, not best single-path performance.
6. Optionally check on a real validation window once.
7. Lock the full workflow.
8. Test once on untouched real out-of-sample data.
9. Repeat using walk-forward windows.
```

A good robust objective is not:

```text
maximize average Sharpe
```

A better objective is something like:

```text
maximize median return
subject to:
  10th percentile return > threshold
  max drawdown p90 < threshold
  ruin probability < threshold
  turnover < threshold
  performance stable near selected parameters
```

Collect these metrics per parameter set:

| Metric                                   | Why                                                       |
| ---------------------------------------- | --------------------------------------------------------- |
| Median PnL / CAGR-like return            | Central tendency across paths.                            |
| 5th/10th percentile PnL                  | Lower-tail robustness.                                    |
| Worst decile mean PnL / CVaR-like metric | “How bad are bad synthetic worlds?”                       |
| Max drawdown distribution                | Path-dependent risk.                                      |
| Drawdown duration                        | Capital lockup and psychological risk.                    |
| Sharpe / Sortino-like metrics            | Useful, but unstable for non-normal returns.              |
| Ruin probability                         | Fraction of paths breaching loss threshold.               |
| Turnover                                 | Cost sensitivity.                                         |
| Parameter stability                      | Nearby parameters should behave similarly.                |
| Rank stability                           | Parameter choice should not depend strongly on path seed. |
| Constraint pass rate                     | Fraction of paths meeting predefined risk constraints.    |

Do not let synthetic data become a second training set you peek at forever. The simulator is a robustness filter. The real untouched out-of-sample window remains the test that matters.

Walk-forward version:

```text
for each window:
  fit simulator on train_window
  generate synthetic paths
  select robust parameters
  trade/test on next real window
  record real result
aggregate walk-forward real results
```

This is where Chan’s point about refitting matters. A fixed BTC simulator trained on ancient history can be worse than no simulator.

## 9. BTC-specific warnings

BTC is exactly the kind of asset where this approach is useful and dangerous.

Useful because BTC gives many candles, trades continuously, has rich volatility structure, and offers enough data for model fitting.

Dangerous because BTC is non-stationary, historically upward-trending, regime-dependent, liquidity-dependent, and prone to jumps. A simulator fit to “all BTC history” may simply learn “number goes up with violent interruptions,” then generate fake bull markets that make long-biased strategies look robust.

Crypto markets are often described as 24/7/365, unlike traditional session-based markets; that means your timestamp generator should support continuous UTC candles and should not inject equity-style overnight gaps by default. But exchange-specific outages, maintenance windows, bad candles, and venue fragmentation still matter. ([Kraken][7])

BTC-specific issues:

| Issue                        | Simulator implication                                                      |
| ---------------------------- | -------------------------------------------------------------------------- |
| Long-term upward drift       | Do not blindly fit one global drift and call it realistic.                 |
| Non-stationarity             | Prefer rolling-window refits and regime models.                            |
| Regime shifts                | Fit bull/bear/high-vol scenarios separately or conditionally.              |
| Volatility clustering        | GARCH/block bootstrap should beat IID/GBM diagnostics.                     |
| Jumps                        | Student-t or jump extensions are needed after MVP.                         |
| 24/7 trading                 | Use continuous UTC timestamp generation.                                   |
| Weekend effects              | Diagnose weekend vs weekday vol/volume instead of assuming equality.       |
| Exchange microstructure      | Venue-specific OHLCV can contain bad wicks, thin books, outages.           |
| Stablecoin/liquidity effects | BTC/USD, BTC/USDT, BTC/USDC venues may behave differently.                 |
| Derivatives/funding          | Spot candles may miss futures/perp dynamics relevant to strategy behavior. |

Stablecoins are structurally important in crypto trading and can interact with crypto volatility/liquidity; if your BTC data is BTC/USDT from one exchange, the simulator should not be assumed equivalent to BTC/USD spot from another venue. ([European Central Bank][8])

Practical recommendations for BTC 1h or 15m:

```text
Use UTC timestamps.
Validate missing bars aggressively.
Remove or flag impossible/bad-wick candles.
Fit on log returns, not raw prices.
Keep open-gap modeling but expect tiny gaps for clean crypto data.
Use rolling windows: e.g. 6, 12, 24 months.
Compare all-history fit vs rolling fit.
Fit shape/volume conditionally on volatility and abs-return buckets.
Run simulations with zero-drift, positive-drift, and negative-drift scenarios.
Always include a bear/high-vol stress scenario.
Do not let the simulator generate only long-term bull drift.
```

For 15m BTC, microstructure noise matters more; for 1h BTC, you lose some microstructure detail but get cleaner volatility/trend behavior. I would start with **1h BTC** for package development and move to **15m** once OHLCV reconstruction and diagnostics are stable.

## 10. Marimo notebook plan

Use Marimo notebooks as reproducible research frontends over your package. Marimo notebooks are stored as pure Python, are reactive, and can be run as scripts/apps, which fits this workflow well. ([Marimo][9])

| Notebook                                    | Purpose                                                  | Inputs                                | Outputs                    | Plots/tables                                                    | Success criteria                                               |
| ------------------------------------------- | -------------------------------------------------------- | ------------------------------------- | -------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------- |
| `01_data_loading_and_cleaning.py`           | Load BTC OHLCV, validate schema, detect gaps/outliers.   | Raw CSV/Parquet.                      | Clean candles.             | Missing-bar table, OHLC invariant failures, price/volume chart. | Clean frame passes validation; bad rows explicitly handled.    |
| `02_transform_exploration.py`               | Explore returns, ranges, wick shapes, volume transforms. | Clean candles.                        | Transform summary.         | Return hist/QQ, range hist, wick ratios, log-volume.            | Transform functions are stable and invertible where needed.    |
| `03_baseline_bootstrap.py`                  | Implement IID and simple block bootstrap baselines.      | Clean candles/transforms.             | Synthetic baseline paths.  | Real vs synthetic return and range plots.                       | Baselines generate valid candles with deterministic seeds.     |
| `04_fit_simple_return_models.py`            | Fit drift/AR(1)/AR(p) return model.                      | Training candles.                     | Fitted linear model.       | Residual diagnostics, ACF/PACF, simulated closes.               | Linear model simulates paths and exposes clear failure modes.  |
| `05_garch_or_volatility_model.py`           | Fit GARCH after linear model works.                      | Training returns.                     | Fitted volatility model.   | Conditional vol, standardized residuals, squared-return ACF.    | GARCH improves volatility-clustering diagnostics.              |
| `06_ohlcv_reconstruction.py`                | Convert simulated returns into OHLCV.                    | Simulated returns + shape sampler.    | Synthetic OHLCV paths.     | Range/body/wick/volume comparisons.                             | All invariants pass; range/volume distributions are plausible. |
| `07_synthetic_vs_real_diagnostics.py`       | Full diagnostic report.                                  | Real candles + synthetic paths.       | Diagnostic tables/report.  | ECDF, ACF, rolling vol, drawdowns, volume.                      | Report makes model failures obvious.                           |
| `08_strategy_robustness_test.py`            | Evaluate simple trend and mean-reversion probes.         | Synthetic paths + strategy functions. | Metrics by parameter/path. | Heatmaps, percentile bands, stability plots.                    | Fragile params are visibly penalized.                          |
| `09_walk_forward_synthetic_optimization.py` | Rolling fit/simulate/optimize/test pipeline.             | Full real dataset.                    | Walk-forward results.      | Window-by-window selected params, OOS equity curve.             | No leakage; one locked evaluation per OOS window.              |

Keep notebooks thin. Any function you use twice belongs in the package.

## 11. Implementation steps

### Phase 1: minimal baseline

| Item          | Details                                                                                                                                        |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Objective     | Load real candles, validate them, and generate basic synthetic candles.                                                                        |
| Tasks         | Implement schema validation, log-return transform, simple IID return bootstrap, empirical candle-shape bootstrap, timestamp generation.        |
| Artifacts     | `schema.py`, `transforms/returns.py`, `transforms/candle_shapes.py`, `models/iid_bootstrap.py`, `simulation/ohlcv.py`, notebook `01` and `02`. |
| Tests         | Schema validation, OHLC invariants, deterministic seed, simulation length.                                                                     |
| Done criteria | Given clean BTC 1h candles, generate 100 synthetic paths of equal length with no NaNs and valid OHLCV.                                         |

### Phase 2: model interface

| Item          | Details                                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Objective     | Make models plug-compatible.                                                                                                  |
| Tasks         | Define `SyntheticCandleModel`, `FittedSyntheticCandleModel`, `SimulationRequest`, `DiagnosticReport`, serialization contract. |
| Artifacts     | `models/base.py`, `config.py`, `io/serialization.py`.                                                                         |
| Tests         | Mock model implements interface; save/load roundtrip; seed reproducibility.                                                   |
| Done criteria | IID bootstrap and any future model use the same `.fit().simulate()` flow.                                                     |

### Phase 3: first fitted model

| Item          | Details                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| Objective     | Implement the first simple linear fitted model.                                                                  |
| Tasks         | Fit AR(0)/AR(1)/AR(p) on log returns; support Gaussian and empirical residual innovations; simulate recursively. |
| Artifacts     | `models/linear_returns.py`, notebook `04_fit_simple_return_models.py`.                                           |
| Tests         | Known synthetic AR(1) process recovers approximate coefficient; deterministic simulation; length correctness.    |
| Done criteria | Linear model produces synthetic closes and passes basic diagnostics, even if it fails volatility clustering.     |

### Phase 4: OHLCV reconstruction

| Item          | Details                                                                                                                      |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Objective     | Make synthetic paths useful for candle-based strategies.                                                                     |
| Tasks         | Build shape sampler; condition shapes on abs-return/vol buckets; reconstruct open/high/low/close/volume; enforce invariants. |
| Artifacts     | `simulation/ohlcv.py`, `transforms/volume.py`, notebook `06_ohlcv_reconstruction.py`.                                        |
| Tests         | OHLC invariants, no negative/zero prices, volume non-negative, timestamp monotonic.                                          |
| Done criteria | Synthetic candles look plausible enough to compare ranges, wicks, and volume against real data.                              |

### Phase 5: diagnostics

| Item          | Details                                                                                             |
| ------------- | --------------------------------------------------------------------------------------------------- |
| Objective     | Build model evaluation before adding more models.                                                   |
| Tasks         | Implement distribution, tail, ACF, rolling-vol, drawdown, range, volume, return-volume diagnostics. |
| Artifacts     | `diagnostics/*`, notebook `07_synthetic_vs_real_diagnostics.py`.                                    |
| Tests         | Diagnostics run on small frames; no crashes on edge cases; stable output schema.                    |
| Done criteria | You can produce a report showing exactly where IID and linear models fail.                          |

### Phase 6: strategy testing integration

| Item          | Details                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| Objective     | Evaluate strategy robustness across synthetic paths.                                                                 |
| Tasks         | Define strategy protocol; implement metrics; run parameter grid across paths; aggregate robust metrics.              |
| Artifacts     | `strategy/protocols.py`, `strategy/evaluation.py`, `strategy/metrics.py`, notebook `08_strategy_robustness_test.py`. |
| Tests         | Reproducible strategy results, path grouping correctness, cost/slippage hooks.                                       |
| Done criteria | One simple trend strategy can be evaluated across 100+ paths with median/p10/drawdown metrics.                       |

### Phase 7: rolling-window / walk-forward workflow

| Item          | Details                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| Objective     | Prevent one-time simulator fitting from becoming hidden leakage.                                                 |
| Tasks         | Implement rolling train/test windows; fit simulator per window; optimize on synthetic; test on next real window. |
| Artifacts     | `data/splitting.py`, `strategy/optimization.py`, notebook `09_walk_forward_synthetic_optimization.py`.           |
| Tests         | No overlap leaks; selected params use only past data; timestamps align.                                          |
| Done criteria | Full walk-forward result table with selected params and real OOS performance per window.                         |

### Phase 8: more realistic models

| Item          | Details                                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Objective     | Add models only after diagnostics prove the baseline failures.                                                                |
| Tasks         | Implement GARCH(1,1) Student-t; stationary/block bootstrap; heavy-tail/jump extension; simple regime mixture.                 |
| Artifacts     | `models/garch.py`, `models/block_bootstrap.py`, `models/jumps.py`, `models/regimes.py`.                                       |
| Tests         | Model-specific parameter sanity; simulation correctness; diagnostics improve where expected.                                  |
| Done criteria | GARCH improves volatility-clustering diagnostics vs linear model; block bootstrap preserves local dependency better than IID. |

## 12. Python package recommendations

Use fewer libraries than you think at first.

| Package       | Use                                                            | When not needed                                                 |
| ------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| `numpy`       | Core numerical simulation, RNG, arrays.                        | Always useful.                                                  |
| `pandas`      | MVP DataFrame format, timestamps, rolling windows.             | Use Polars later for large-scale processing.                    |
| `polars`      | Large OHLCV datasets, lazy pipelines, fast transformations.    | Not necessary for MVP if it complicates statsmodel integration. |
| `scipy`       | Distribution fitting, statistical tests, optimization helpers. | Avoid until you need specific distributions/tests.              |
| `statsmodels` | AR/ARMA/ARIMA-style models, ACF/PACF, time-series tools.       | Do not use as a black box without diagnostics.                  |
| `arch`        | GARCH models and time-series bootstraps.                       | Not needed for Phase 1 linear MVP.                              |
| `pandera`     | DataFrame schema validation.                                   | Could be replaced by custom validation initially.               |
| `pydantic`    | Config/model metadata validation.                              | Not ideal for validating whole large DataFrames.                |
| `matplotlib`  | Static diagnostic plots.                                       | Enough for MVP.                                                 |
| `plotly`      | Interactive diagnostics in Marimo.                             | Nice, not required.                                             |
| `marimo`      | Research notebooks and interactive reports.                    | Keep core code outside notebooks.                               |
| `pytest`      | Unit/research validation tests.                                | Required.                                                       |

`pandera` is useful if you want explicit DataFrame schemas; it supports pandas schemas and also has Polars support. `pydantic` is better for configs and typed metadata than for validating millions of candle rows. ([pandera.readthedocs.io][10])

### Open-source packages worth knowing

I found relevant open-source pieces, but I would not outsource your full package to any one of them.

| Package/project             | What it is                                                                                               | My take                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `arch`                      | Univariate volatility models plus time-series bootstraps.                                                | Strong dependency candidate for GARCH and bootstrap methods.                  |
| `statsmodels`               | Time-series modeling toolkit including AR/ARMA/ARIMA-style models.                                       | Strong candidate for linear models/diagnostics.                               |
| `tsbootstrap`               | Time-series bootstrap library.                                                                           | Worth inspecting, but `arch` may already cover enough.                        |
| `PriceGenerator`            | Generates OHLCV candlesticks with controllable statistical parameters.                                   | Useful reference, but not enough for fitted, diagnostic-first BTC simulation. |
| `market-data-source`        | Rust/Python synthetic OHLC/tick generator with random walk drift and seed support.                       | Interesting engineering reference, but not your fitted-model workflow.        |
| `fsynth`                    | Financial synthetic generator using Heston-style stochastic volatility and Merton jump diffusion claims. | Interesting later reference; verify maturity before relying on it.            |
| `SDV`                       | General synthetic data library for tabular/sequential/multi-table data.                                  | More privacy/tabular oriented than quant candle simulation.                   |
| `YData-Synthetic` / TimeGAN | Time-series GAN tooling.                                                                                 | Later experiment, not MVP. Harder to diagnose and easier to fool yourself.    |
| `TSGM`                      | Framework for time-series generative modeling using TensorFlow.                                          | Later research path, not initial package core.                                |

These projects confirm that pieces exist, but your specific goal—**fitted OHLCV candle simulation for strategy robustness with quant diagnostics**—still justifies a custom package. ([GitHub][11])

## 13. Testing strategy

Split tests into **software correctness tests** and **research validation tests**.

### Unit tests

| Test                     | Example                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| Candle schema validation | Reject `high < close`, negative price, duplicate timestamp.        |
| Deterministic simulation | Same seed + same fitted model = same synthetic output.             |
| OHLC invariants          | Every generated candle satisfies high/low constraints.             |
| No NaNs/infs             | All generated numeric columns finite.                              |
| Simulation length        | `n_paths * n_bars` rows exactly.                                   |
| Timestamp generation     | Correct 15m/1h increments, monotonic per path.                     |
| Transform roundtrip      | Close prices → returns → reconstructed closes approximately equal. |
| Serialization            | Save/load fitted model produces same simulation for same seed.     |
| Strategy reproducibility | Same paths + same strategy config = same metrics.                  |
| Multi-symbol guardrails  | Either reject unsupported multi-symbol input or group correctly.   |

### Research validation tests

These are not strict unit tests with exact answers. Use tolerances and seeded synthetic fixtures.

| Test                                                    | Purpose                                                   |
| ------------------------------------------------------- | --------------------------------------------------------- |
| Fit known AR(1) process                                 | Recover approximate AR coefficient.                       |
| Fit known GARCH process                                 | Recover volatility clustering behavior, not exact params. |
| Bootstrap preserves marginal distribution               | Synthetic quantiles roughly match real quantiles.         |
| Block bootstrap preserves short-lag ACF better than IID | Proves block bootstrap is doing something useful.         |
| GARCH improves squared-return ACF vs linear Gaussian    | Validates intended model advantage.                       |
| Shape reconstruction preserves OHLC range distribution  | Prevents close-only realism.                              |
| Volume sampler preserves rough volume quantiles         | Avoids flat/absurd volume.                                |
| Diagnostics do not crash on small windows               | Research notebooks stay usable.                           |
| Walk-forward no-leakage test                            | Train window end < test window start for every split.     |

Use `pytest.mark.parametrize` heavily for schema/invariant cases. Pytest has built-in parametrization support, which is useful for testing many invalid candle patterns compactly. ([pytest][12])

## 14. Anti-goals and traps

### Anti-goals and traps

Do **not** build a simulator whose job is to make your strategy look good.

Do **not** choose the synthetic model based on which one gives the best strategy performance.

Do **not** fit the simulator on future data relative to the strategy test window.

Do **not** assume more synthetic paths means more truth. More paths only reduce Monte Carlo error under the same assumptions; they do not fix wrong assumptions.

Do **not** optimize too many strategy parameters. Synthetic data can make over-parameterized strategies look stable if the simulator is too narrow.

Do **not** ignore transaction costs, slippage, funding, borrow, spread, and turnover.

Do **not** treat candle data as sufficient for high-frequency execution realism. Candle OHLCV does not encode queue position, order book depth, adverse selection, liquidation cascades, or partial fills.

Do **not** use all BTC history as one stationary distribution unless the goal is explicitly “all-history stylized BTC.” It will mix incompatible regimes.

Do **not** evaluate a strategy repeatedly on the same real out-of-sample window while changing simulator, parameters, diagnostics, and costs. That becomes slow-motion leakage.

Do **not** make the first version a GAN. You will get beautiful fake data and weak interpretability. For this package, transparency beats glamour early.

Do **not** bury diagnostics in notebooks only. Diagnostics should be callable package functions.

Do **not** overfit the OHLCV reconstruction layer. If your high/low generator makes stop-loss strategies magically work, the simulator is dangerous.

Do **not** forget that the future path can disappoint even if your parameter is optimal across simulated paths. That point is central to Chan’s warning.

## 15. Final recommended MVP

Here is the exact MVP I recommend.

### Package name

Use the working title **Synthetic Data Generator**, but make the importable package:

```text
synthetic_data_generator
```

or, cleaner:

```text
synthetic_candles
```

I slightly prefer `synthetic_candles` because it says what the package actually generates.

### First model to implement

Implement:

```text
LinearReturnModel
```

with:

```text
AR order: 0 or 1 initially
fit target: close-to-close log returns
innovation mode: Gaussian or empirical residual bootstrap
OHLCV reconstruction: empirical candle-shape sampler
```

This satisfies your “simple linear model first” preference and keeps the model transparent.

### Second model to implement

Implement:

```text
GarchReturnModel
```

with:

```text
mean: constant initially, optional AR(1)
volatility: GARCH(1,1)
innovations: Student-t preferred, Normal as baseline
OHLCV reconstruction: same empirical shape sampler, conditioned on simulated volatility bucket
```

This gives you a direct before/after comparison: linear homoskedastic vs volatility-clustering model.

### Minimum diagnostics

Build these first:

```text
return ECDF + histogram
return QQ plot
tail quantile table
ACF returns
ACF absolute returns
ACF squared returns
rolling volatility distribution
max drawdown distribution
high-low range distribution
body/wick ratio distribution
log-volume distribution
return-volume relationship
simple trend and mean-reversion strategy probes
```

### Minimum Marimo notebooks

For the first few days, only build these:

```text
01_data_loading_and_cleaning.py
02_fit_linear_return_model.py
03_ohlcv_reconstruction.py
04_synthetic_vs_real_diagnostics.py
05_strategy_robustness_test.py
```

Add GARCH as notebook `06` once the linear model and diagnostics are stable.

### Minimum package structure

Start with this smaller structure:

```text
src/
  synthetic_candles/
    __init__.py
    schema.py
    transforms.py
    models/
      __init__.py
      base.py
      linear_returns.py
      iid_bootstrap.py
    simulation/
      __init__.py
      ohlcv.py
      timestamps.py
    diagnostics/
      __init__.py
      stats.py
      plots.py
      strategy_probes.py
    strategy/
      __init__.py
      evaluation.py
      metrics.py
    io.py
```

You can split modules later. Do not prematurely create 30 files before the first real path simulation works.

### First strategy test to run

Run a simple BTC 1h long/flat trend-following strategy:

```text
entry: close > moving_average(close, lookback)
exit: close <= moving_average(close, lookback)
parameters: lookback ∈ [12, 24, 48, 96, 168, 336]
costs: fixed bps per trade
evaluation: real train, synthetic train paths, untouched real OOS
```

Why this one? BTC has enough trend behavior historically that a trend probe is relevant, but simple enough that you can see whether the simulator is generating fake trendability. Also run a simple mean-reversion probe as a diagnostic, but make the first actual optimization test the trend model.

The MVP is done when you can say:

```text
I can fit a linear return model to BTC 1h training candles.
I can generate 1,000 valid synthetic OHLCV paths.
I can see exactly where the synthetic paths differ from real BTC.
I can evaluate a simple strategy across those paths.
I can select robust parameters without touching real OOS.
I can test once on real OOS and record the result.
```

That is small enough to build in a few days and serious enough to teach the right mental model. The key is not model sophistication. The key is building a workflow where every synthetic-data claim is forced through diagnostics, failure modes, and untouched out-of-sample reality.

[1]: https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima.model.ARIMA.html?utm_source=chatgpt.com "statsmodels.tsa.arima.model."
[2]: https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html?utm_source=chatgpt.com "ARCH Modeling - arch 7.2.0"
[3]: https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html?utm_source=chatgpt.com "Time-series Bootstraps - arch 8.0.0"
[4]: https://www.statsmodels.org/dev/tsa.html?utm_source=chatgpt.com "Time Series analysis tsa - statsmodels 0.15.0 (+989)"
[5]: https://www-stat.wharton.upenn.edu/~steele/Resources/FTSResources/StylizedFacts/Cont2001.pdf?utm_source=chatgpt.com "Empirical properties of asset returns: stylized facts and ..."
[6]: https://arxiv.org/html/2402.11930v2?utm_source=chatgpt.com "Stylized Facts of High-Frequency Bitcoin Time Series"
[7]: https://www.kraken.com/learn/what-makes-crypto-24-7-365?utm_source=chatgpt.com "What makes crypto 24/7/365?"
[8]: https://www.ecb.europa.eu/press/financial-stability-publications/macroprudential-bulletin/html/ecb.mpbu202207_2~836f682ed7.en.html?utm_source=chatgpt.com "Stablecoins' role in crypto and beyond: functions, risks and ..."
[9]: https://docs.marimo.io/?utm_source=chatgpt.com "marimo - Documentation for the Reactive Python Notebook ..."
[10]: https://pandera.readthedocs.io/en/stable/dataframe_schemas.html?utm_source=chatgpt.com "DataFrame Schemas - pandera documentation"
[11]: https://github.com/destenson/market-data-source "GitHub - destenson/market-data-source · GitHub"
[12]: https://docs.pytest.org/en/stable/example/parametrize.html?utm_source=chatgpt.com "Parametrizing tests"
