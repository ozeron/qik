"""qik.gauntlet on synthetic tapes: permutation invariants, gates, early failure, fail-fast."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

from qik.gauntlet import (
    ParamSpec,
    Subject,
    Thresholds,
    Variant,
    ambiguous_exits,
    futures_session,
    g4_route,
    permute_bars,
    pooled_rank,
    run_gauntlet,
    score,
    selector_null,
    spp_configs,
)

BARS = 390  # 09:30-16:00 NY, one-minute bars


def tape(sessions: int, edge: float, seed: int) -> pd.DataFrame:
    """RTH 1m tape; after minute 30 the drift follows the sign of the first 30 minutes (``edge``)."""

    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=sessions)
    stamps, closes = [], []
    price = 10_000.0
    for d in days:
        start = pd.Timestamp(d.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        r = rng.normal(0.0, 0.0006, BARS)
        r[0] += rng.normal(0.0, 0.002)  # overnight gap
        sign = np.sign(r[:30].sum())
        r[30:] += edge * sign
        path = price * np.exp(np.cumsum(r))
        price = float(path[-1])
        stamps.append(pd.date_range(start, periods=BARS, freq="1min"))
        closes.append(path)
    index = pd.DatetimeIndex(np.concatenate([s.asi8 for s in stamps])).tz_localize("UTC")
    close = np.concatenate(closes)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wiggle = np.abs(rng.normal(0, 0.0002, len(close))) * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wiggle,
            "low": np.minimum(open_, close) - wiggle,
            "close": close,
            "volume": 100.0,
        },
        index=index,
    )


@dataclass(frozen=True)
class Cfg:
    entry_minute: int = 30
    cost: float = 0.5
    lag: int = 0


def orb(cfg: Cfg, data: pd.DataFrame) -> pd.DataFrame:
    """Enter at ``entry_minute`` in the direction of the session so far; exit at the close."""

    c = data["close"].to_numpy(float)
    o = data["open"].to_numpy(float)
    n = len(c) // BARS
    c, o = c[: n * BARS].reshape(n, BARS), o[: n * BARS].reshape(n, BARS)
    k = min(cfg.entry_minute + cfg.lag, BARS - 2)
    side = np.sign(c[:, cfg.entry_minute - 1] - o[:, 0])
    pnl = side * (c[:, -1] - c[:, k]) - cfg.cost
    sessions = futures_session(pd.DatetimeIndex(data.index[::BARS][:n]))
    return pd.DataFrame({"session_date": sessions, "pnl": pnl})


def buy_open_sell_close(cfg: Cfg, data: pd.DataFrame) -> pd.DataFrame:
    """Session net move: invariant under within-session permutation (its null equals itself)."""

    o = data["open"].to_numpy(float).reshape(-1, BARS)
    c = data["close"].to_numpy(float).reshape(-1, BARS)
    sessions = futures_session(pd.DatetimeIndex(data.index[::BARS]))
    return pd.DataFrame({"session_date": sessions, "pnl": c[:, -1] - o[:, 0]})


def subject(raw: pd.DataFrame, run=orb, **kw) -> Subject:
    eligible = sorted(set(futures_session(pd.DatetimeIndex(raw.index))))
    return Subject(name="toy", run=run, cfg=Cfg(), raw=raw, eligible=eligible, **kw)


def test_permute_bars_keeps_bar_population_and_validity() -> None:
    raw = tape(20, 0.0, 1)
    out = permute_bars(raw, np.random.default_rng(3), group="all")
    assert out.index.equals(raw.index)
    assert (out.volume == raw.volume).all()
    assert (out.high >= out[["open", "close", "low"]].max(axis=1) - 1e-9).all()
    assert (out.low <= out[["open", "close", "high"]].min(axis=1) + 1e-9).all()
    shape = lambda f: np.sort(np.round(np.log(f.close / f.open).to_numpy(), 10))  # noqa: E731
    assert np.array_equal(shape(out), shape(raw))
    assert not np.allclose(out.close.to_numpy(), raw.close.to_numpy())


def test_session_group_keeps_each_session_net_move() -> None:
    raw = tape(15, 0.0, 2)
    out = permute_bars(raw, np.random.default_rng(4), group="session")
    a = buy_open_sell_close(Cfg(), raw)
    b = buy_open_sell_close(Cfg(), out)
    ra = np.log(raw.close.to_numpy()[BARS - 1 :: BARS] / raw.open.to_numpy()[::BARS])
    rb = np.log(out.close.to_numpy()[BARS - 1 :: BARS] / out.open.to_numpy()[::BARS])
    assert np.allclose(ra, rb)
    assert len(a) == len(b)


def test_null_passes_a_planted_edge() -> None:
    s = subject(tape(250, 0.00012, 5))
    t = Thresholds(null_draws=99, spp_draws=20)
    report = run_gauntlet(s, thresholds=t, seed=1, skip=("trade_mc",))
    null = next(c for c in report.checks if c.name == "null")
    assert null.status == "PASS", null.summary
    assert null.stats["K"] <= 4


def test_null_fails_early_and_exactly_when_failure_is_certain() -> None:
    # buy-open-sell-close is unchanged by a within-session permutation: every null Sharpe ties.
    raw = tape(120, 0.0, 6)
    raw["close"] = raw["close"] * np.linspace(1.0, 1.3, len(raw))  # trend so base passes
    raw["high"] = np.maximum(raw[["open", "close"]].max(axis=1), raw["high"])
    raw["low"] = np.minimum(raw[["open", "close"]].min(axis=1), raw["low"])
    s = subject(raw, run=buy_open_sell_close, null_group="session")
    t = Thresholds(null_draws=99)
    report = run_gauntlet(s, thresholds=t, seed=2, skip=("trade_mc",))
    null = next(c for c in report.checks if c.name == "null")
    assert null.status == "FAIL"
    assert null.stats["K"] >= 5
    assert null.stats["early_failure"] is True
    assert null.stats["completed"] < 99


def test_fail_fast_stops_after_failed_base() -> None:
    s = subject(tape(40, 0.0, 7))
    s.cfg = Cfg(cost=50.0)
    report = run_gauntlet(s)
    assert report.status == "FAIL"
    assert [c.status for c in report.checks[1:]] == ["NOT_RUN"] * 7


def test_stress_gate_uses_ratio_of_base_sharpe() -> None:
    s = subject(tape(200, 0.00015, 8), stress={"cost_big": Cfg(cost=500.0), "lag1": Cfg(lag=1)})
    report = run_gauntlet(s, skip=("trade_mc", "spp", "null"), fail_fast=False)
    stress = next(c for c in report.checks if c.name == "stress")
    assert stress.status == "FAIL"
    assert stress.stats["lag1"]["pass"] is True
    assert stress.stats["cost_big"]["pass"] is False


def test_variants_and_spp_and_parallel_determinism() -> None:
    raw = tape(150, 0.00012, 9)
    eligible = sorted(set(futures_session(pd.DatetimeIndex(raw.index))))
    s = subject(
        raw,
        params=(ParamSpec("cost", lo=0.0),),
        markets={"other": Variant(Cfg(), tape(150, 0.00012, 10), eligible)},
    )
    t = Thresholds(spp_draws=24, null_draws=19, null_alpha=0.1)
    one = run_gauntlet(s, thresholds=t, seed=3, jobs=1, fail_fast=False)
    two = run_gauntlet(s, thresholds=t, seed=3, jobs=2, fail_fast=False)
    for a, b in zip(one.checks, two.checks, strict=True):
        assert a.status == b.status
        assert a.stats == b.stats
    spp = next(c for c in one.checks if c.name == "spp")
    assert spp.runs == 24 and spp.status == "PASS"
    assert next(c for c in one.checks if c.name == "markets").status == "PASS"


def test_spp_configs_bounds_rounding_and_zero() -> None:
    rng = np.random.default_rng(0)
    out = spp_configs(Cfg(), [ParamSpec("entry_minute", lo=28, hi=31, integer=True)], 200, rng, 0.2)
    vals = {c.entry_minute for c in out}
    assert vals <= {28, 29, 30, 31} and len(vals) > 1
    with pytest.raises(ValueError, match="is 0"):
        spp_configs(Cfg(), [ParamSpec("lag")], 1, rng, 0.2)
    lag = spp_configs(Cfg(), [ParamSpec("lag", lo=0, hi=10, integer=True)], 50, rng, 0.2)
    assert {c.lag for c in lag} <= {0, 1, 2}
    with pytest.raises(ValueError, match="not a field"):
        spp_configs(Cfg(), [ParamSpec("nope")], 1, rng, 0.2)
    assert replace(Cfg(), lag=1).lag == 1


def test_score_counts_no_trade_sessions() -> None:
    trades = pd.DataFrame({"session_date": ["2021-01-04"], "pnl": [10.0]})
    s, daily = score(trades, ["2021-01-04", "2021-01-05", "2021-01-06"])
    assert s.trades == 1 and s.net == 10.0
    assert len(daily) == 3 and daily.sum() == 10.0


def test_ambiguous_exits_and_pooled_rank() -> None:
    high = np.array([10.0, 12.0, 11.0])
    low = np.array([9.0, 8.0, 10.5])
    amb = ambiguous_exits(high, low, np.array([1, 2]), np.array([8.5, 9.0]), np.array([11.5, 11.5]))
    assert amb.tolist() == [True, False]
    scores = pd.DataFrame({"NQ": [1.0, 0.5, 0.9], "ES": [0.2, 0.6, 0.8]}, index=["a", "b", "c"])
    ranked = pooled_rank(scores)
    assert ranked.index[0] == "c"
    assert ranked.loc["a", "worst_rank"] == 3


def test_errored_null_draws_count_against_and_report_their_text() -> None:
    raw = tape(150, 0.00012, 11)
    last = float(raw["close"].iloc[-1])

    def fragile(cfg: Cfg, data: pd.DataFrame) -> pd.DataFrame:
        if float(data["close"].iloc[-1]) != last:
            raise RuntimeError("cannot handle this tape")
        return orb(cfg, data)

    s = subject(raw, run=fragile)
    report = run_gauntlet(s, thresholds=Thresholds(null_draws=99), skip=("trade_mc",))
    null = next(c for c in report.checks if c.name == "null")
    assert null.status == "FAIL"
    assert null.stats["errors"] >= 5
    assert "cannot handle this tape" in null.stats["error_examples"][0]
    assert "ERRORED" in null.summary


GRID = [Cfg(entry_minute=m) for m in (10, 20, 30, 45, 60, 90)]


def best_sharpe(data: pd.DataFrame) -> float:
    eligible = sorted(set(futures_session(pd.DatetimeIndex(data.index))))
    return max(score(orb(c, data), eligible)[0].sharpe for c in GRID)


def test_selector_null_passes_a_strong_edge_and_stops_early_on_noise() -> None:
    strong = selector_null(tape(200, 0.0002, 12), best_sharpe, seed=1)
    assert strong.status == "PASS" and strong.runs == 19
    assert strong.stats["p"] == pytest.approx(0.05)
    noise = [selector_null(tape(120, 0.0, 20 + i), best_sharpe, seed=i, jobs=1) for i in range(6)]
    assert any(c.status == "FAIL" for c in noise)
    for c in noise:
        if c.status == "FAIL":
            assert c.runs < 19 or c.stats["K"] >= 1


def test_selector_null_errors_count_against() -> None:
    raw = tape(60, 0.0002, 13)
    last = float(raw["close"].iloc[-1])

    def fragile(data: pd.DataFrame) -> float:
        if float(data["close"].iloc[-1]) != last:
            raise ValueError("selector crashed")
        return best_sharpe(data)

    c = selector_null(raw, fragile)
    assert c.status == "FAIL" and c.runs == 1 and "selector crashed" in c.stats["error_examples"][0]


def test_g4_route() -> None:
    assert g4_route(0.9) == "PASS"
    assert g4_route(0.9, n_certain=False) == "SELECTOR_NULL"
    assert g4_route(0.7) == "SELECTOR_NULL"
    assert g4_route(None) == "SELECTOR_NULL"
    assert g4_route(0.5) == "FAIL"
    assert g4_route(0.5, n_certain=False) == "FAIL"
