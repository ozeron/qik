"""Vector engine: parameter grids, sizing, cash and the per-config summary.

Sizing is a pass over finished trades, so every test compares against the unsized run plus a
plain-Python restatement of the rule. The Nautilus side of the liquidity rule (A14) is tested against
a live engine in the repo-root ``tests/vector_engine/test_vector_engine_nautilus.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qik.metrics import summarize_trades
from qik.vector_engine import Bars, Costs, Instrument, Signals, Sizing, grid, simulate, stack

TICK = 0.25
INST = Instrument(tick=TICK, point_value=20.0)


def _tape(n: int = 3000, seed: int = 3, volume: float = 4000.0) -> Bars:
    rng = np.random.default_rng(seed)
    c = (16000 + np.cumsum(rng.integers(-3, 4, n))) * TICK
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.integers(0, 4, n) * TICK
    lo = np.minimum(o, c) - rng.integers(0, 4, n) * TICK
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="5min", tz="UTC")
    return Bars(o, h, lo, c, index=idx, volume=np.full(n, volume))


def _signals(bars: Bars, sl_ticks: int, tp_ticks: int, seed: int = 5) -> Signals:
    rng = np.random.default_rng(seed)
    n = len(bars)
    side = np.where(rng.random(n) < 0.03, rng.choice([-1, 1], n), 0).astype(np.int8)
    return Signals(
        side,
        np.zeros(n, bool),
        bars.close - side * sl_ticks * TICK,
        bars.close + side * tp_ticks * TICK,
    )


def _lv(levels: np.ndarray | None) -> np.ndarray:
    assert levels is not None
    return levels


def _index(bars: Bars) -> pd.DatetimeIndex:
    assert bars.index is not None
    return bars.index


# -- grid / stack ------------------------------------------------------------------------------------


def test_grid_is_the_cartesian_product_first_axis_slowest() -> None:
    g = grid(sl=[4, 8], tp=[8, 16, 24])
    assert g.index.name == "config"
    assert list(g.index) == list(range(6))
    assert g.to_dict("records")[:4] == [
        {"sl": 4, "tp": 8},
        {"sl": 4, "tp": 16},
        {"sl": 4, "tp": 24},
        {"sl": 8, "tp": 8},
    ]


@pytest.mark.parametrize("axes", [{}, {"sl": []}, {"sl": "48"}])
def test_grid_refuses_empty_or_string_axes(axes) -> None:
    with pytest.raises(ValueError):
        grid(**axes)


def test_stack_equals_hand_built_columns() -> None:
    bars = _tape(500)
    params = grid(sl=[4, 8], tp=[8, 16])
    sig = stack(params, lambda sl, tp: _signals(bars, sl, tp))
    assert sig.entry_side.shape == (500, 4)
    rows = params.to_dict("records")
    for j, row in enumerate(rows):
        one = _signals(bars, row["sl"], row["tp"])
        np.testing.assert_array_equal(sig.entry_side[:, j], one.entry_side)
        np.testing.assert_array_equal(_lv(sig.stop)[:, j], one.stop)
        np.testing.assert_array_equal(_lv(sig.target)[:, j], one.target)
    stacked = simulate(bars, sig).trades
    for j, row in enumerate(rows):
        alone = simulate(bars, _signals(bars, row["sl"], row["tp"])).trades.drop(columns="config")
        mine = stacked[stacked.config == j].drop(columns="config").reset_index(drop=True)
        pd.testing.assert_frame_equal(mine, alone)


def test_stack_fills_missing_levels_with_nan_and_checks_shapes() -> None:
    bars = _tape(50)
    params = grid(with_stop=[True, False])

    def build(with_stop: bool) -> Signals:
        s = _signals(bars, 4, 8)
        return Signals(s.entry_side, s.exit, s.stop if with_stop else None, None)

    sig = stack(params, build)
    assert sig.target is None
    assert np.isnan(_lv(sig.stop)[:, 1]).all() and not np.isnan(_lv(sig.stop)[:, 0]).all()

    def ragged(with_stop: bool) -> Signals:
        n = 10 if with_stop else 11
        return Signals(np.zeros(n, np.int8), np.zeros(n, bool))

    with pytest.raises(ValueError, match="config 1"):
        stack(params, ragged)


# -- sizing ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"contracts": 1, "risk_usd": 100.0},
        {"contracts": 0},
        {"contracts": 1.5},
        {"risk_usd": -1.0},
        {"risk_fraction": 1.5},
        {"risk_usd": 100.0, "max_contracts": 0},
    ],
)
def test_sizing_validation(bad) -> None:
    with pytest.raises(ValueError):
        Sizing(**bad)


def test_fixed_contracts_scale_pnl_and_costs_and_keep_fills() -> None:
    bars = _tape()
    sig = _signals(bars, 6, 10)
    costs = Costs(commission_per_side=2.0, slippage_ticks=1)
    one = simulate(bars, sig, instrument=INST, costs=costs)
    three = simulate(bars, sig, instrument=INST, costs=costs, sizing=Sizing(contracts=3))
    cols = ["config", "entry_i", "exit_i", "side", "entry_px", "exit_px", "reason"]
    pd.testing.assert_frame_equal(one.trades[cols], three.trades[cols])
    assert (one.trades.contracts == 1).all() and (three.trades.contracts == 3).all()
    np.testing.assert_allclose(three.trades.pnl_usd, 3 * one.trades.pnl_usd, rtol=0, atol=1e-9)
    np.testing.assert_allclose(three.trades.gross_usd, 3 * one.trades.gross_usd, rtol=0, atol=1e-9)
    assert (three.open_positions.contracts == 3).all()


def test_risk_usd_is_floor_of_budget_over_stop_risk_capped() -> None:
    bars = _tape()
    params = grid(sl=[3, 7, 13])
    sig = stack(params, lambda sl: _signals(bars, sl, 9))
    res = simulate(bars, sig, instrument=INST, sizing=Sizing(risk_usd=500.0, max_contracts=12))
    t = res.trades
    stop = _lv(sig.stop)[t.entry_i.to_numpy(), t.config.to_numpy()]
    risk = np.abs(t.entry_px.to_numpy() - stop) * INST.point_value
    want = np.minimum(np.floor(500.0 / risk + 1e-9), 12)
    np.testing.assert_array_equal(t.contracts.to_numpy(), want)
    # sl 3 ticks: $15/contract -> 33 -> cap 12; 7 ticks: $35 -> 14 -> cap 12; 13 ticks: $65 -> 7
    assert set(t[t.config < 2].contracts) == {12}
    assert set(t[t.config == 2].contracts) == {7}


def test_risk_fraction_compounds_on_the_equity_before_each_trade() -> None:
    bars = _tape()
    sig = stack(grid(sl=[5, 9]), lambda sl: _signals(bars, sl, 9))
    cash = 50_000.0
    costs = Costs(commission_per_side=1.5, slippage_ticks=1)
    res = simulate(
        bars,
        sig,
        instrument=INST,
        costs=costs,
        sizing=Sizing(risk_fraction=0.01),
        initial_cash=cash,
    )
    unsized = simulate(bars, sig, instrument=INST, costs=costs).trades
    per_contract_cost = 2 * (1.5 + 1 * TICK * INST.point_value)
    for j in (0, 1):
        equity = cash
        mine = res.trades[res.trades.config == j]
        base = unsized[unsized.config == j]
        for (_, t), (_, u) in zip(mine.iterrows(), base.iterrows(), strict=True):
            risk = abs(t.entry_px - _lv(sig.stop)[t.entry_i, j]) * INST.point_value
            n = math.floor(0.01 * equity / risk + 1e-9)
            assert t.contracts == n
            pnl = n * (u.gross_points * INST.point_value - per_contract_cost)
            assert t.pnl_usd == pytest.approx(pnl, abs=1e-9)
            equity += pnl
            assert t.equity_usd == pytest.approx(equity, abs=1e-6)


def test_zero_contracts_is_refused() -> None:
    bars = _tape()
    with pytest.raises(ValueError, match="0 contracts"):
        simulate(bars, _signals(bars, 40, 9), instrument=INST, sizing=Sizing(risk_usd=100.0))


def test_risk_sizing_needs_a_stop() -> None:
    bars = _tape()
    s = _signals(bars, 5, 9)
    with pytest.raises(ValueError, match="no stop"):
        simulate(bars, Signals(s.entry_side, s.exit, None, s.target), instrument=INST,
                 sizing=Sizing(risk_usd=1000.0))  # fmt: skip


def test_sizing_needs_instrument_and_fraction_needs_cash() -> None:
    bars = _tape(200)
    sig = _signals(bars, 5, 9)
    with pytest.raises(ValueError, match="instrument"):
        simulate(bars, sig, sizing=Sizing(contracts=2))
    with pytest.raises(ValueError, match="initial_cash"):
        simulate(bars, sig, instrument=INST, sizing=Sizing(risk_fraction=0.01))


def test_running_out_of_cash_is_refused() -> None:
    bars = _tape()
    sig = _signals(bars, 5, 9)
    with pytest.raises(ValueError, match="out of cash"):
        simulate(bars, sig, instrument=INST, costs=Costs(50.0), sizing=Sizing(contracts=5),
                 initial_cash=1_000.0)  # fmt: skip


def test_liquidity_limit_is_max_of_floor_volume_quarter_and_one() -> None:
    bars = _tape(volume=4000.0)
    sig = _signals(bars, 5, 9)
    n = len(bars)
    for volume, allowed in ((28.0, 7), (31.0, 7), (27.0, 6), (2.0, 1), (0.0, 1)):
        b = Bars(
            bars.open, bars.high, bars.low, bars.close, index=bars.index, volume=np.full(n, volume)
        )
        simulate(b, sig, instrument=INST, sizing=Sizing(contracts=allowed))
        with pytest.raises(ValueError, match="liquidity"):
            simulate(b, sig, instrument=INST, sizing=Sizing(contracts=allowed + 1))
    no_volume = Bars(bars.open, bars.high, bars.low, bars.close)
    simulate(no_volume, sig, instrument=INST, sizing=Sizing(contracts=1))
    with pytest.raises(ValueError, match="volume"):
        simulate(no_volume, sig, instrument=INST, sizing=Sizing(contracts=2))


# -- summary -----------------------------------------------------------------------------------------


def test_summary_is_qik_metrics_per_config() -> None:
    bars = _tape()
    params = grid(sl=[5, 9], tp=[9])
    sig = stack(params, lambda sl, tp: _signals(bars, sl, tp))
    res = simulate(bars, sig, instrument=INST, costs=Costs(2.0, 1), initial_cash=100_000.0)
    out = res.summary()
    assert list(out.index) == [0, 1]
    sessions = sorted(set(pd.Series(_index(bars)).dt.tz_convert("America/New_York").dt.date))
    for j in (0, 1):
        t = res.trades[res.trades.config == j]
        frame = pd.DataFrame(
            {
                "pnl_usd": t.pnl_usd.to_numpy(),
                "side": t.side.to_numpy(),
                "session_date": [ts.tz_convert("America/New_York").date() for ts in t.exit_time],
            }
        )
        want = summarize_trades(frame, eligible_sessions=sessions, initial_capital=100_000.0)
        got = out.loc[j].to_dict()
        assert got["n_trades"] == want["n_trades"] == len(t)
        for key in ("net_pnl_usd", "sharpe_ann", "profit_factor", "max_drawdown_pct"):
            assert got[key] == pytest.approx(want[key], nan_ok=True)
        assert got["n_eligible_sessions"] == len(sessions)


def test_summary_keeps_configs_without_trades() -> None:
    bars = _tape(300)
    s = _signals(bars, 5, 9)
    side = np.column_stack([s.entry_side, np.zeros(len(bars), np.int8)])
    stop, target = _lv(s.stop), _lv(s.target)
    sig = Signals(side, np.zeros(side.shape, bool), np.column_stack([stop, stop]),
                  np.column_stack([target, target]))  # fmt: skip
    out = simulate(bars, sig, instrument=INST, initial_cash=1e5).summary()
    assert len(out) == 2 and out.loc[1, "n_trades"] == 0 and out.loc[0, "n_trades"] > 0


def test_summary_needs_usd_index_and_cash() -> None:
    bars = _tape(300)
    sig = _signals(bars, 5, 9)
    with pytest.raises(ValueError, match="instrument"):
        simulate(bars, sig, initial_cash=1e5).summary()
    with pytest.raises(ValueError, match="initial_cash"):
        simulate(bars, sig, instrument=INST).summary()
    plain = Bars(bars.open, bars.high, bars.low, bars.close)
    with pytest.raises(ValueError, match="index"):
        simulate(plain, sig, instrument=INST, initial_cash=1e5).summary()
