"""A16/A17 API guards and dense differential execution checks."""

from __future__ import annotations

import numpy as np
import pytest

from qik.vector_engine import (
    Bars,
    Compliance,
    Costs,
    FillModel,
    Instrument,
    MarketFill,
    Profile,
    Rules,
    Signals,
    simulate,
)


def test_venue_rules_are_typed_and_exclude_next_open() -> None:
    for latency in (-1, 2, True, 1.0):
        with pytest.raises(ValueError, match="latency_bars"):
            Rules(latency_bars=latency)
    with pytest.raises(ValueError, match="FillModel"):
        Rules(fill_model="one_tick")
    for kwargs in (
        {"latency_bars": 1},
        {"fill_model": FillModel.ONE_TICK},
        {"latency_bars": 1, "fill_model": FillModel.ONE_TICK},
    ):
        with pytest.raises(ValueError, match="NEXT_OPEN cannot mix"):
            Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN, **kwargs)


def test_one_tick_needs_instrument_and_keeps_compliance() -> None:
    c = np.array([100.0, 101.0, 102.0, 103.0])
    bars = Bars(c, c + 1, c - 1, c)
    sig = Signals(np.array([1, 0, 0, 0]), np.array([False, False, True, False]))
    rules = Rules(latency_bars=1, fill_model=FillModel.ONE_TICK)
    with pytest.raises(ValueError, match="Instrument"):
        simulate(bars, sig, rules=rules)
    res = simulate(bars, sig, rules=rules, instrument=Instrument(0.25, 1.0), costs=Costs())
    assert res.compliance is Compliance.NAUTILUS and res.deployable
    assert res.trades[["entry_i", "exit_i", "entry_px", "exit_px"]].values.tolist() == [
        [1, 3, 101.25, 102.75]
    ]


@pytest.mark.parametrize("latency,one_tick", [(1, False), (0, True), (1, True)])
def test_venue_kernel_equals_independent_reference_dense(latency: int, one_tick: bool) -> None:
    from test_vector_engine import (
        _dense_limit_case,
        _kernel_final,
        _ref_final,
        _ref_trades,
        _run_kernel,
    )

    from qik.vector_engine.reference import simulate_column_reference

    rng = np.random.default_rng(6100 + latency + 10 * one_tick)
    errors: set[int] = set()
    reasons: set[int] = set()
    for _ in range(2000):
        a, side, exit_, stop, target, limit, sp, tp, chase, day = _dense_limit_case(rng)
        adaptive, reenter, flatten = (bool(rng.integers(2)) for _ in range(3))
        ref = simulate_column_reference(
            a["o"],
            a["h"],
            a["lo"],
            a["c"],
            side,
            exit_,
            stop,
            target,
            adaptive=adaptive,
            reenter=reenter,
            entry_limit=limit,
            stop_pts=sp,
            target_pts=tp,
            chase_after=chase,
            protect_flatten=flatten,
            new_day=day,
            latency_bars=latency,
            one_tick=one_tick,
            tick=0.25,
        )
        trades, err, eb, final, _ = _run_kernel(
            a,
            side,
            exit_,
            stop,
            target,
            adaptive,
            reenter,
            False,
            limit,
            sp,
            tp,
            chase,
            flatten,
            day,
            latency_bars=latency,
            one_tick=one_tick,
            tick=0.25,
        )
        assert (err, eb) == (ref.err, ref.err_bar)
        errors.add(err)
        if not err:
            assert trades == _ref_trades(ref)
            assert _kernel_final(final) == _ref_final(ref)
            reasons.update(t[5] for t in trades)
    assert {0, 4, 5} <= errors
    assert {1, 2, 3, 4, 5} <= reasons
