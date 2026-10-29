"""Decision-bar quantity fixtures against an independent exact-rational interpreter."""

from __future__ import annotations

import ast
from pathlib import Path

import _qty_feature_oracle as oracle
import numpy as np
import pandas as pd
import pytest
from _qty_feature_strategies import Lanes, mutate, probe

from qik.vector_engine import (
    Bars,
    BracketAnchor,
    Costs,
    FillModel,
    FillTiming,
    Flatten,
    Instrument,
    Profile,
    Rules,
    Sizing,
    run,
)

ROWS = [
    (100.0, 101.0, 99.0, 100.0),
    (110.0, 112.0, 105.0, 110.0),
    (105.0, 109.0, 100.0, 105.0),
    (99.0, 106.0, 89.0, 100.0),
    (102.0, 105.0, 100.0, 103.0),
    (105.0, 107.0, 102.0, 106.0),
]


def execute(
    kind=0,
    level=90.0,
    rules=Rules(),
    lane=None,
    qty_feature="wanted",
    jit=False,
    leave=5,
    decisions=(0,),
    **sizing,
):
    data = np.array(ROWS)
    bars = Bars(
        *(data[:, i] for i in range(4)),
        index=pd.date_range("2023-03-06 10:00", periods=6, freq="min", tz="America/New_York"),
        volume=np.full(6, 1e6),
    )
    return run(
        probe,
        bars,
        features=Lanes(
            wanted=np.asarray(
                [3.9, 8.0, 9.0, 10.0, 11.0, 12.0] if lane is None else lane, dtype=np.float64
            ),
            other=np.full(6, 7.0),
        ),
        grid={"kind": [kind], "level": [level], "leave": [leave], "decision": list(decisions)},
        rules=rules,
        sizing=Sizing(qty_feature=qty_feature, **sizing),
        instrument=Instrument(
            0.25, 2.0, lot_step=sizing.get("lot_step", 1.0), min_qty=sizing.get("min_qty", 1.0)
        ),
        costs=Costs(commission_per_side=0.5),
        initial_cash=10000.0,
        jit=jit,
        signal_fields=(
            "entry_side",
            "exit",
            *{
                0: (),
                1: ("entry_limit", "stop_points"),
                2: ("entry_stop",),
                3: ("entry_stop_long", "entry_stop_short"),
                4: ("entry_limit", "stop_points"),
                5: ("entry_stop", "stop_points", "target_points"),
            }[kind],
        ),
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "kind,level,rules,order,delay,chase",
    [
        (0, 90.0, Rules(), "close", 0, 10),
        (
            0,
            90.0,
            Rules(
                profile=Profile.REPRO, entry_fill=FillTiming.NEXT_OPEN, exit_fill=FillTiming.CLOSE
            ),
            "next_open",
            0,
            10,
        ),
        (1, 90.0, Rules(limit_chase_bars=5), "limit", 0, 5),
        (1, 80.0, Rules(limit_chase_bars=2), "limit", 0, 2),
        (1, 80.0, Rules(limit_chase_bars=4), "limit", 0, 4),
        (1, 90.0, Rules(latency_bars=1, limit_chase_bars=5), "limit", 1, 5),
        (2, 108.0, Rules(), "stop", 0, 10),
        (3, 108.0, Rules(), "stop", 0, 10),
    ],
)
def test_hand_entry_identity(kind, level, rules, order, delay, chase, jit):
    lane = [3.9, 8.0, 9.0, 10.0, 11.0, 12.0]
    ei, px, qty = oracle.entry(ROWS, lane, 0, order, level, delay=delay, chase=chase)
    result = execute(kind, level, rules, jit=jit, leave=4 if rules.latency_bars else 5).results[0]
    t = result.trades.iloc[0]
    assert (t.entry_i, t.entry_px, t.contracts) == (ei, px, qty)
    assert t.pnl_usd == qty * ((106.0 - px) * 2 - 1)
    assert result.states.qty[0] == qty
    assert result.research_only


@pytest.mark.parametrize("jit", [False, True])
def test_lot_minimum_cap_open_and_column_isolation(jit):
    fractional = execute(jit=jit, lot_step=0.25, min_qty=0.5).results[0]
    assert fractional.trades.contracts.tolist() == [oracle.quantity(3.9, 0.25, 0.5)] == [3.75]
    result = execute(jit=jit, lot_step=0.25, min_qty=0.5, max_contracts=2.5, leave=20).results[0]
    assert result.open_positions.contracts.tolist() == [oracle.quantity(3.9, 0.25, 0.5, 2.5)]
    assert result.states.qty[0] == 2.5
    a = execute(jit=jit).results[0]
    b = execute(jit=jit, qty_feature="other").results[0]
    assert a.trades.contracts.tolist() == [3]
    assert b.trades.contracts.tolist() == [7]
    pd.testing.assert_frame_equal(a.trades, execute(jit=jit).results[0].trades, check_exact=True)
    with pytest.raises(ValueError, match="0 contracts.*bar 0, config 0"):
        execute(jit=jit, lane=[0.1] * 6)


@pytest.mark.parametrize("value", [np.nan, 0.0, -1.0, np.inf, -np.inf])
@pytest.mark.parametrize("jit", [False, True])
def test_invalid_submission_refuses(value, jit):
    with pytest.raises(ValueError, match="qty_feature.*finite.*bar 0, config 0"):
        execute(kind=1, jit=jit, lane=[value, 8.0, 9.0, 10.0, 11.0, 12.0])
    # Invalid values after submission must neither size nor refuse the resting entry.
    result = execute(kind=1, jit=jit, lane=[3.9] + [value] * 5).results[0]
    assert result.trades.contracts.tolist() == [3]


def test_declaration_and_exclusivity():
    for name, value in (
        ("contracts", 1),
        ("risk_usd", 10.0),
        ("risk_fraction", 0.1),
        ("notional_usd", 100.0),
    ):
        with pytest.raises(ValueError, match="exactly one"):
            Sizing(qty_feature="wanted", **{name: value})
    for name in ("", " ", 1):
        with pytest.raises(ValueError, match="feature lane"):
            Sizing(qty_feature=name)
    with pytest.raises(ValueError, match="clamp"):
        Sizing(qty_feature="wanted", min_qty_policy="clamp")
    with pytest.raises(ValueError, match="not a declared strategy feature"):
        execute(qty_feature="missing")


def test_oracle_has_no_engine_imports():
    tree = ast.parse(Path(oracle.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] in {
                "__future__",
                "fractions",
                "math",
                "typing",
                "collections",
            }
        assert not isinstance(node, ast.Import)


def test_python_equals_jit_frames_and_state_bytes():
    for kind, rules in (
        (0, Rules()),
        (1, Rules(limit_chase_bars=5)),
        (2, Rules()),
        (3, Rules()),
        (1, Rules(latency_bars=1, limit_chase_bars=5)),
    ):
        py = execute(kind=kind, level=108.0 if kind in (2, 3) else 90.0, rules=rules).results[0]
        native = execute(
            kind=kind, level=108.0 if kind in (2, 3) else 90.0, rules=rules, jit=True
        ).results[0]
        pd.testing.assert_frame_equal(py.trades, native.trades, check_exact=True)
        pd.testing.assert_frame_equal(py.open_positions, native.open_positions, check_exact=True)
        assert py.states.tobytes() == native.states.tobytes()


def test_grid_configs_and_named_feature_sources_do_not_leak():
    for jit in (False, True):
        a = execute(jit=jit, decisions=(0, 1))
        b = execute(jit=jit, decisions=(0, 1), qty_feature="other")
        assert [r.trades.contracts.iloc[0] for r in a.results] == [3, 8]
        assert [r.trades.contracts.iloc[0] for r in b.results] == [7, 7]


@pytest.mark.parametrize("jit", [False, True])
def test_replaced_entry_retains_new_decision_index(jit):
    rules = Rules(
        profile=Profile.REPRO,
        exit_and_rearm=True,
        flatten=Flatten.LAST_BAR_CLOSE,
        limit_chase_bars=5,
    )
    # The cancelled decision's NaN must never refuse the replacement order.
    lane = [np.nan, 8.0, 9.0, 10.0, 11.0, 12.0]
    got = execute(kind=4, rules=rules, jit=jit, lane=lane).results[0]
    ei, px, qty = oracle.entry(ROWS, lane, 2, "limit", 90.0, chase=5)
    assert (ei, px, qty) == (3, 90.0, 9.0)
    assert got.trades.entry_i.tolist() == [ei]
    assert got.trades.entry_px.tolist() == [px]
    assert got.trades.contracts.tolist() == [qty]
    assert got.states.qty[0] == qty


@pytest.mark.parametrize("jit", [False, True])
def test_run_start_quantity_snapshot_survives_callback_writes(jit):
    lane = np.array([3.9, 8.0, 9.0, 10.0, 11.0, 12.0])
    data = np.array(ROWS)
    bars = Bars(
        *(data[:, i] for i in range(4)),
        index=pd.date_range("2023-03-06 10:00", periods=6, freq="min", tz="America/New_York"),
        volume=np.full(6, 1e6),
    )
    got = run(
        mutate,
        bars,
        features=Lanes(lane, np.zeros(6)),
        rules=Rules(
            profile=Profile.REPRO, entry_fill=FillTiming.NEXT_OPEN, exit_fill=FillTiming.CLOSE
        ),
        instrument=Instrument(0.25, 1.0),
        costs=Costs(),
        sizing=Sizing(qty_feature="wanted"),
        jit=jit,
        signal_fields=("entry_side", "exit"),
    )
    assert lane[:2].tolist() == [100.0, 200.0]
    assert got.trades.contracts.tolist() == [3]
    assert got.states.qty[0] == 3.0


@pytest.mark.parametrize("jit", [False, True])
def test_strided_quantity_and_callback_use_the_same_normalized_values(jit):
    owner = np.array(
        [3.9, -999.0, 8.0, -999.0, 9.0, -999.0, 10.0, -999.0, 11.0, -999.0, 12.0, -999.0]
    )
    lane = owner[::2]
    assert not lane.flags.c_contiguous
    result = execute(lane=lane, jit=jit).results[0]
    assert result.trades.contracts.tolist() == [3]
    assert result.states.observed[0] == 12.0
    assert owner[1::2].tolist() == [-999.0] * 6


@pytest.mark.parametrize("jit", [False, True])
def test_one_tick_stop_gap_uses_quantity_not_submitted_risk(jit):
    rules = Rules(
        profile=Profile.REPRO,
        entry_fill=FillTiming.NEXT_OPEN,
        exit_fill=FillTiming.CLOSE,
        fill_model=FillModel.ONE_TICK,
    )
    got = execute(kind=2, level=108.0, rules=rules, jit=jit).results[0]
    assert got.trades.entry_i.tolist() == [1]
    assert got.trades.entry_px.tolist() == [110.25]
    assert got.trades.contracts.tolist() == [3]
    assert got.states.qty[0] == 3.0


@pytest.mark.parametrize("jit", [False, True])
def test_busy_fill_close_signal_does_not_replace_the_original_quantity(jit):
    got = execute(
        kind=5,
        jit=jit,
        rules=Rules(
            profile=Profile.REPRO,
            protection="oto",
            bracket_anchor=BracketAnchor.FILL,
            reenter_on_exit_bar=True,
        ),
    ).results[0]
    assert got.trades.entry_i.tolist() == [1]
    assert got.trades.exit_i.tolist() == [1]
    assert got.trades.contracts.tolist() == [3]


@pytest.mark.parametrize("jit", [False, True])
def test_large_finite_quantity_never_wraps_to_negative_contracts(jit):
    value = 1e20
    with pytest.raises(ValueError, match="unrepresentable"):
        oracle.quantity(value)
    with pytest.raises(ValueError, match="cannot be represented.*bar 0, config 0"):
        execute(jit=jit, lane=[value] * 6)
    result = execute(jit=jit, lane=[value] * 6, max_contracts=2).results[0]
    assert result.trades.contracts.tolist() == [oracle.quantity(value, cap=2)] == [2]
    assert result.states.qty[0] == 2.0


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", [1, 5])
def test_quantity_known_floor_tolerance(compiled, mode):
    """Pin the existing lot tolerance, including its upward boundary rounding.

    TODO: account.quantity's +1e-9 lot tolerance can round a wanted quantity
    slightly above its mathematical floor; changing it needs an engine fingerprint
    and acceptance review, independently of Q1.
    """
    from qik.vector_engine.account import quantity

    calculate = quantity if compiled else quantity.py_func
    wanted = 3.9999999995
    # Mode 1: budget/risk = wanted. Mode 5: submitted raw quantity = wanted.
    amount, risk = (wanted, 1.0) if mode == 1 else (0.0, wanted)
    assert np.floor(wanted) == 3.0
    assert calculate(mode, amount, 0.0, 10000.0, risk) == 4.0
    below = 3.999999998
    amount, risk = (below, 1.0) if mode == 1 else (0.0, below)
    assert calculate(mode, amount, 0.0, 10000.0, risk) == 3.0
