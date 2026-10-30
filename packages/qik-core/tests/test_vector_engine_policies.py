"""Extracted policies agree with independent Python arithmetic at their boundaries."""

from __future__ import annotations

import math

import numpy as np
import pytest
import ve_reference as R
from _typed_scripts import replay, script
from hypothesis import given, settings
from hypothesis import strategies as st

from qik.vector_engine import Bars, Costs, Instrument
from qik.vector_engine import account as A
from qik.vector_engine import protection as P
from scripts.engine_test_tiers import engine_examples

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
prices = st.one_of(finite, st.just(math.nan))
positive = st.floats(1e-4, 1e4, allow_nan=False, allow_infinity=False)


def exact(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert np.float64(actual).tobytes() == np.float64(expected).tobytes()


@settings(max_examples=engine_examples(400), deadline=None)
@given(
    side=st.sampled_from([-1, 1]),
    anchor=finite,
    stop=prices,
    target=prices,
    sp=prices,
    tp=prices,
    tick=st.sampled_from([0.00001, 0.01, 0.25, 1.0]),
    rounding=st.integers(0, 2),
)
def test_levels_independent(side, anchor, stop, target, sp, tp, tick, rounding):
    actual = P.levels(side, anchor, stop, target, sp, tp, tick, rounding)
    expected = R._levels(side, anchor, stop, target, sp, tp, tick, rounding)
    for left, right in zip(actual, expected, strict=True):
        exact(left, right)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("value", [-1.375, -0.125, -0.0, 0.125, 1.375, math.nan])
def test_rounding_ties_and_anchor_gaps(mode, side, value):
    exact(P._round_price(value, 0.25, mode, side), R._round_level(value, 0.25, mode, side))
    for anchor in range(3):
        exact(P._anchor(anchor, 100.0, 105.0, 99.0), R._anchor(anchor, 100.0, 105.0, 99.0))


@settings(max_examples=engine_examples(300), deadline=None)
@given(
    side=st.sampled_from([-1, 1]),
    stop=prices,
    target=prices,
    close=finite,
    flatten=st.booleans(),
    count_only=st.booleans(),
)
def test_fill_then_protect_independent(side, stop, target, close, flatten, count_only):
    arrays = [np.full(2, -999.0) for _ in range(7)]
    error, reason = P._fill_then_protect(
        side, 13, 104.0, stop, target, close, flatten, count_only, *arrays, 1
    )
    expected = R._protect_at_close(side, stop, target, close, flatten)
    assert error == (4 if expected == R.REFUSE else 0)
    assert reason == (0 if expected == R.REFUSE else expected)
    if reason and not count_only:
        expected_values = [13, 13, side, 104.0, close, reason, stop]
        for a, value in zip(arrays, expected_values, strict=True):
            exact(a[1], float(value))
            assert a[0] == -999.0
    else:
        assert all(np.all(a == -999.0) for a in arrays)


@settings(max_examples=engine_examples(400), deadline=None)
@given(
    mode=st.integers(0, 2),
    amount=positive,
    cap=st.one_of(st.just(0.0), positive),
    equity=st.one_of(positive, st.just(0.0), st.just(math.nan)),
    risk=st.one_of(positive, st.sampled_from([0.0, -1.0, math.nan, math.inf])),
    lot=st.sampled_from([0.01, 0.1, 1.0]),
    minimum=st.sampled_from([0.01, 0.1, 1.0]),
)
def test_quantity_independent(mode, amount, cap, equity, risk, lot, minimum):
    args = mode, amount, cap, equity, risk, lot, minimum
    exact(A.quantity(*args), R.quantity_reference(*args))


@settings(max_examples=engine_examples(400), deadline=None)
@given(
    entry=prices,
    exit=prices,
    pv=positive,
    bps=prices,
    commission=prices,
    slip=prices,
    tick=positive,
    gross=prices,
)
def test_account_arithmetic_independent(entry, exit, pv, bps, commission, slip, tick, gross):
    friction = A.fixed_cost(commission, slip, tick, pv)
    exact(friction, R.fixed_cost_reference(commission, slip, tick, pv))
    proportional = A.fill_cost(entry, exit, pv, bps)
    exact(proportional, R.fill_cost_reference(entry, exit, pv, bps))
    exact(
        A.settle(gross, pv, friction, proportional),
        R.settle_reference(gross, pv, friction, proportional),
    )
    entries, exits, points = (
        np.array([entry, -0.0, 100.0]),
        np.array([exit, 0.0, 105.0]),
        np.array([gross, 0.0, 5.0]),
    )
    costs = A.fill_cost(entries, exits, pv, bps)
    net = A.settle(points, pv, friction, costs)
    for i in range(3):
        cost = R.fill_cost_reference(entries[i], exits[i], pv, bps)
        exact(costs[i], cost)
        exact(net[i], R.settle_reference(points[i], pv, friction, cost))


@pytest.mark.parametrize("lot", [0.01, 0.1, 1.0])
def test_quantity_floor_and_minimum_boundaries(lot):
    for budget in [np.nextafter(3 * lot, 0), 3 * lot, np.nextafter(3 * lot, math.inf), 0.5 * lot]:
        for cap in [0.0, 2 * lot]:
            args = 1, budget, cap, 100.0, 1.0, lot, lot
            exact(A.quantity(*args), R.quantity_reference(*args))


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "stop_offset,target_offset", [(0.0, 1.0), (-1.0, 0.0), (-1.0, 1.0), (1.0, -1.0)]
)
def test_protection_close_ties_and_gaps(side, flatten, stop_offset, target_offset):
    stop, target = 100.0 + side * stop_offset, 100.0 + side * target_offset
    arrays = [np.zeros(1) for _ in range(7)]
    error, reason = P._fill_then_protect(
        side, 0, 105.0, stop, target, 100.0, flatten, False, *arrays, 0
    )
    expected = R._protect_at_close(side, stop, target, 100.0, flatten)
    assert (error, reason) == ((4, 0) if expected == R.REFUSE else (0, expected))


@pytest.mark.parametrize("handling", ["raise", "warn"])
def test_batch_cost_overflow_respects_numpy_errstate(handling: str) -> None:
    price = np.full(2, 1e308)
    bars = Bars(price, price, price, price)
    signals = script(np.array([1, 0]), np.array([False, True]))
    context = (
        pytest.raises(FloatingPointError, match="overflow encountered in add")
        if handling == "raise"
        else pytest.warns(RuntimeWarning, match="overflow encountered in add")
    )
    with np.errstate(over=handling), context:
        replay(bars, signals, instrument=Instrument(0.25, 1.0), costs=Costs(proportional_bps=1.0))


@pytest.mark.parametrize("jit", [False, True])
def test_host_fixed_cost_preserves_numpy_scalar_promotion(jit: bool) -> None:
    price = np.array([100.0, 101.0])
    bars = Bars(price, price, price, price)
    instrument = Instrument(np.float32(0.01), np.float32(1.1))
    costs = Costs(commission_per_side=0.1, slippage_ticks=0.3)
    signals = script(np.array([1, 0]), np.array([False, True]))
    result = replay(bars, signals, instrument=instrument, costs=costs, jit=jit)
    friction = 2.0 * (
        costs.commission_per_side + costs.slippage_ticks * instrument.tick * instrument.point_value
    )
    expected = np.array([1.0]) * instrument.point_value - friction
    exact(result.trades.pnl_usd.iloc[0], expected[0])


@pytest.mark.parametrize("round_trip", [2, 2.0])
def test_host_fixed_cost_preserves_original_factor_type(round_trip: float) -> None:
    fixed_cost = getattr(A.fixed_cost, "py_func", A.fixed_cost)
    args = np.int64(2**62), np.int64(0), np.int64(1), np.int64(1)
    if isinstance(round_trip, int):
        with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="scalar multiply"):
            fixed_cost(*args, round_trip)
    else:
        with np.errstate(over="raise"):
            exact(fixed_cost(*args, round_trip), round_trip * args[0])
