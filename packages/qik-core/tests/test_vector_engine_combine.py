"""Fill-anchored combined protection: independent arithmetic and scheduler parity."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qik.vector_engine import (
    HOLD,
    Bar,
    Bars,
    Compliance,
    Costs,
    Features,
    FillTiming,
    Instrument,
    Params,
    Position,
    Profile,
    Rules,
    Signals,
    State,
    TypedDecision,
    enter,
    run,
    simulate,
    strategy,
)
from qik.vector_engine import protection as P
from qik.vector_engine import reference as R


@settings(max_examples=500, deadline=None)
@given(
    side=st.sampled_from([-1, 1]),
    fill=st.floats(10, 1000, allow_nan=False),
    level=st.one_of(st.floats(-1000, 2000, allow_nan=False), st.just(math.nan)),
    points=st.one_of(st.floats(0.01, 100, allow_nan=False), st.just(math.nan)),
    rounding=st.integers(0, 2),
)
def test_combined_independent(side, fill, level, points, rounding):
    args = (side, fill + 20, level, level, points, points, 0.25, rounding)
    actual = P.levels(*args, combine=3, fill=fill)
    expected = R._levels(*args, combine=3, fill=fill)
    for a, b in zip(actual, expected, strict=True):
        assert (
            math.isnan(a) if math.isnan(b) else np.float64(a).tobytes() == np.float64(b).tobytes()
        )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("rounding", [0, 1, 2])
@pytest.mark.parametrize("offset", [-0.125000001, -0.125, -0.124999999, 0, 1, 2])
def test_ties_rounding_and_beyond_fill(side, rounding, offset):
    fill, points = 100.0, 1.0
    stop, target = fill + side * offset, fill - side * offset
    actual = P.levels(side, 200, stop, target, points, points, 0.25, rounding, 3, fill)
    expected = (
        P._round_price(max(stop, 99) if side == 1 else min(stop, 101), 0.25, rounding, -side),
        P._round_price(min(target, 101) if side == 1 else max(target, 99), 0.25, rounding, side),
    )
    assert actual == expected


def tape(side=1):
    # Decision close 100; actual next open 110. Combined long stop must be 108,
    # and it is touched on the fill bar, whose close recovered above the stop.
    o = np.array([100.0, 110.0, 110.0, 110.0])
    h = np.array([101.0, 112.0, 112.0, 112.0])
    lo = np.array([99.0, 107.0, 109.0, 109.0])
    c = np.array([100.0, 111.0, 111.0, 111.0])
    if side < 0:
        o, h, lo, c = 200 - o, 200 - lo, 200 - h, 200 - c
    return Bars(
        o, h, lo, c, index=pd.date_range("2023-01-03 15:00", periods=4, freq="min", tz="UTC")
    )


def signals(side=1):
    return Signals(
        entry_side=np.array([side, 0, 0, 0]),
        exit=np.zeros(4, bool),
        stop=np.array([105.0 if side > 0 else 95.0, math.nan, math.nan, math.nan]),
        stop_points=np.array([2.0, math.nan, math.nan, math.nan]),
    )


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("scheduler", ["simple", "slots", "venue"])
def test_schedulers_reference_and_existing_level_equivalence(side, scheduler):
    bars = tape(side)
    if scheduler == "venue":
        rules = Rules(stop_combine="tighter", latency_bars=1)
        # Delayed close fills use 111, not decision 100: channel stays the tighter stop.
        sig = replace(
            signals(side),
            stop=np.array([109.5 if side > 0 else 90.5, math.nan, math.nan, math.nan]),
        )
    else:
        rules = Rules(
            profile=Profile.REPRO,
            entry_fill=FillTiming.NEXT_OPEN,
            stop_combine="tighter",
            protection="oto" if scheduler == "slots" else "callback",
        )
        sig = signals(side)
    kwargs = dict(rules=rules, instrument=Instrument(tick=0.25, point_value=1), costs=Costs())
    actual = simulate(bars, sig, **kwargs)
    resolved = 109.5 if scheduler == "venue" else 108.0
    stop = np.array([resolved if side > 0 else 200 - resolved, math.nan, math.nan, math.nan])
    expected = simulate(bars, replace(sig, stop=stop, stop_points=None), **kwargs)
    pd.testing.assert_frame_equal(actual.trades, expected.trades, check_exact=True)
    ref = R.simulate_column_reference(
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        sig.entry_side,
        np.zeros(4, bool),
        sig.stop,
        np.full(4, math.nan),
        stop_pts=sig.stop_points,
        combine=1,
        next_open=scheduler != "venue",
        oto=scheduler == "slots",
        latency_bars=int(scheduler == "venue"),
    )
    assert ref.err == 0
    assert len(ref.trades) == len(actual.trades)
    for a, b in zip(actual.trades.itertuples(), ref.trades, strict=True):
        assert (a.entry_i, a.exit_i, a.entry_px, a.exit_px, a.stop) == (
            b.entry_i,
            b.exit_i,
            b.entry_px,
            b.exit_px,
            b.stop,
        )
    if scheduler != "venue":
        assert len(actual.trades) == 1
        assert actual.trades.iloc[0].exit_i == 1
        assert actual.trades.iloc[0].exit_px == (108 if side > 0 else 92)


def test_refusal_and_validation():
    with pytest.raises(ValueError, match="level and as points"):
        simulate(tape(), signals())
    for name in ["stop_combine", "target_combine"]:
        with pytest.raises(ValueError, match=name):
            Rules(**{name: "unknown"})
    with pytest.raises(ValueError, match="positive"):
        simulate(
            tape(),
            replace(signals(), stop_points=np.array([-1.0, math.nan, math.nan, math.nan])),
            rules=Rules(stop_combine="tighter"),
        )


class S(State):
    calls: int = 0


class C(Params):
    distance: float = 2.0


class F(Features):
    pass


@strategy(state=S, params=C, features=F)
def combined(s: S, p: C, f: F, bar: Bar, pos: Position) -> TypedDecision:
    s.calls += 1
    if bar.i == 0:
        return enter(1, stop=105.0, stop_points=p.distance)
    return HOLD


@pytest.mark.parametrize("scheduler", ["simple", "slots", "venue"])
def test_typed_python_jit(scheduler):
    rules = (
        Rules(stop_combine="tighter", latency_bars=1)
        if scheduler == "venue"
        else Rules(
            profile=Profile.REPRO,
            entry_fill=FillTiming.NEXT_OPEN,
            stop_combine="tighter",
            protection="oto" if scheduler == "slots" else "callback",
        )
    )
    kwargs = dict(rules=rules, features=F())
    # Default declaration exercises the slots path; explicit fields exercise simple/venue.
    if scheduler != "slots":
        kwargs["signal_fields"] = ("entry_side", "stop", "stop_points")
    j = run(combined, tape(), **kwargs)
    p = run(combined, tape(), jit=False, **kwargs)
    pd.testing.assert_frame_equal(j.trades, p.trades, check_exact=True)
    np.testing.assert_array_equal(j.states, p.states)


@pytest.mark.parametrize("side", [-1, 1])
def test_target_combines_and_defaults_refuse(side):
    bars = tape(side)
    sig = replace(
        signals(side),
        stop=None,
        stop_points=None,
        target=np.array([115.0 if side > 0 else 85.0, math.nan, math.nan, math.nan]),
        target_points=np.array([1.0, math.nan, math.nan, math.nan]),
    )
    rules = Rules(profile=Profile.REPRO, entry_fill=FillTiming.NEXT_OPEN)
    with pytest.raises(ValueError, match="target is given"):
        simulate(bars, sig, rules=rules)
    actual = simulate(bars, sig, rules=replace(rules, target_combine="tighter"))
    assert len(actual.trades) == 1
    assert actual.trades.iloc[0].exit_i == 1
    assert actual.trades.iloc[0].exit_px == (111 if side > 0 else 89)


def test_nautilus_oto_combined_admission_and_marketable_stop():
    sig = replace(signals(), stop=np.array([98.0, math.nan, math.nan, math.nan]))
    rules = Rules(protection="oto", stop_combine="tighter")
    result = simulate(tape(), sig, rules=rules)
    assert result.compliance is Compliance.NAUTILUS
    assert not result.deployable  # Preserve the existing V2 OTO admission boundary.
    assert result.open_positions.iloc[0].stop == 98.0
    # The chosen level cannot silently be clipped to the fill or ignored.
    with pytest.raises(ValueError, match="stop|protection"):
        simulate(tape(), signals(), rules=rules)
