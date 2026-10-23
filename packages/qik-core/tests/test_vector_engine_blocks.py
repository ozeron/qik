"""Boundary contracts for blocks: readiness, signed risk, explicit clocks and amendments."""

from __future__ import annotations

import math
from datetime import time

import numpy as np
import pytest

from qik.vector_engine.blocks import (
    atr_stop,
    r_target,
    range_breakout,
    range_breakout_array,
    side_of,
    side_of_array,
    time_exit,
    trailing_stop,
)


@pytest.mark.parametrize(
    "value,reference,expected",
    [
        (9.0, 10.0, -1),
        (10.0, 10.0, 0),
        (11.0, 10.0, 1),
        (math.nan, 10.0, 0),
        (10.0, math.nan, 0),
    ],
)
def test_indicator_equality_and_unready_values(
    value: float, reference: float, expected: int
) -> None:
    assert side_of(value, reference) == expected
    np.testing.assert_array_equal(side_of_array([value], reference), [expected])


@pytest.mark.parametrize(
    "close,high,low,expected",
    [
        (11.0, 10.0, 9.0, 1),
        (8.0, 10.0, 9.0, -1),
        (10.0, 10.0, 9.0, 0),
        (9.0, 10.0, 9.0, 0),
        (8.0, math.nan, 9.0, -1),
        (11.0, 10.0, math.nan, 1),
        (10.0, math.inf, -math.inf, 0),
        (math.nan, 10.0, 9.0, 0),
    ],
)
def test_breakout_does_not_admit_touches_or_invent_missing_bounds(
    close: float,
    high: float,
    low: float,
    expected: int,
) -> None:
    assert range_breakout(close, high, low) == expected
    np.testing.assert_array_equal(range_breakout_array([close], high, low), [expected])


def test_array_broadcasting_preserves_independent_configs() -> None:
    close = np.array([9.0, 10.0, 11.0])[:, None]
    np.testing.assert_array_equal(side_of_array(close, [9.0, 11.0]), [[0, -1], [1, -1], [1, 0]])
    np.testing.assert_array_equal(
        range_breakout_array(close, [10.0, 11.0], [9.0, 10.0]),
        [[0, -1], [0, 0], [1, 0]],
    )


@pytest.mark.parametrize(
    "side,stop,target,trail", [(1, 94.0, 112.0, 96.75), (-1, 106.0, 88.0, 103.25)]
)
def test_risk_and_outward_amendment_both_sides(
    side: int,
    stop: float,
    target: float,
    trail: float,
) -> None:
    assert atr_stop(100.0, side, 2.0, 3.0) == stop
    assert r_target(100.0, side, 6.0, 2.0) == target
    d = trailing_stop(100.0, side, 3.1, tick=0.25)
    assert d.stop_update == trail and d.side == 0 and not d.exit
    # No local ratchet: less favorable candidates must still reach engine admission.
    worse = trailing_stop(99.0 if side == 1 else 101.0, side, 3.1, tick=0.25)
    assert side * worse.stop_update < side * d.stop_update


def test_atr_warmup_is_preserved() -> None:
    assert math.isnan(atr_stop(100, 1, math.nan, 3))


@pytest.mark.parametrize("side", [0, 2, -2])
def test_invalid_position_sides_are_refused(side: int) -> None:
    with pytest.raises(ValueError, match="side"):
        atr_stop(100, side, 2, 3)
    with pytest.raises(ValueError, match="side"):
        r_target(100, side, 2, 3)
    with pytest.raises(ValueError, match="side"):
        trailing_stop(100, side, 2, tick=0.25)


@pytest.mark.parametrize("bad", [-1.0, math.inf, -math.inf])
def test_bad_atr_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="ATR"):
        atr_stop(100, 1, bad, 3)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_bad_configured_scales_are_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="multiple"):
        atr_stop(100, 1, 2, bad)
    with pytest.raises(ValueError, match="reward"):
        r_target(100, 1, 2, bad)
    with pytest.raises(ValueError, match="tick"):
        trailing_stop(100, 1, 2, tick=bad)


def test_clock_policy_is_explicit_and_does_not_guess_a_missed_minute() -> None:
    assert time_exit(time(9, 30), time(9, 30))
    assert not time_exit(time(9, 31), time(9, 30))
    assert time_exit(time(9, 31), time(9, 30), at_or_after=True)
    assert not time_exit(388, 389, at_or_after=True)
    assert time_exit(389, 389)
    assert time_exit(390, 389, at_or_after=True)
