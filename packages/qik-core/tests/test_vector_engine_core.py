"""Strategy cores: the per-bar driver and its contract (golden checks: tests/vector_engine)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qik.vector_engine import Bars, BarView, Decision, Signals, check_causal, signals_from_core

N = 40


def _bars() -> Bars:
    c = 100.0 + np.arange(N) % 7 * 0.25
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2024-01-02 14:30", periods=N, freq="1min", tz="UTC")
    return Bars(o, np.maximum(o, c) + 0.25, np.minimum(o, c) - 0.25, c, index=idx,
                volume=np.arange(N, dtype=float))  # fmt: skip


class Every5:
    """Long every 5th bar with a stop one point below, exit two bars later."""

    def __init__(self) -> None:
        self.seen: list[BarView] = []

    def step(self, bar: BarView) -> Decision:
        self.seen.append(bar)
        if bar.i % 5 == 0:
            return Decision(side=1, stop=bar.close - 1.0)
        return Decision(exit=bar.i % 5 == 2)


def test_core_sees_each_bar_once_in_order_with_time_and_volume() -> None:
    bars = _bars()
    core = Every5()
    sig = signals_from_core(lambda: core, bars)
    assert [b.i for b in core.seen] == list(range(N))
    assert bars.index is not None and core.seen[3].ts == bars.index[3]
    assert core.seen[3].volume == 3.0
    assert core.seen[3].close == bars.close[3]
    assert (np.flatnonzero(sig.entry_side) == np.arange(0, N, 5)).all()
    assert (np.flatnonzero(sig.exit) == np.arange(2, N, 5)).all()
    assert sig.stop is not None
    np.testing.assert_array_equal(sig.stop[::5], bars.close[::5] - 1.0)
    assert sig.target is None and sig.entry_limit is None  # no decision sets them


def test_decision_and_driver_validation() -> None:
    with pytest.raises(ValueError, match="side"):
        Decision(side=2)

    class Bad:
        def step(self, bar: BarView) -> object:
            return 1

    with pytest.raises(TypeError, match="bar 0"):
        signals_from_core(Bad, _bars())  # ty: ignore[invalid-argument-type]


def test_check_causal_on_a_core_and_on_a_peek() -> None:
    bars = _bars()
    assert check_causal(lambda b: signals_from_core(Every5, b), bars) > 0

    def peek(b: Bars) -> Signals:
        sig = signals_from_core(Every5, b)
        sig.entry_side[:-1] = sig.entry_side[1:]  # tomorrow's decision today
        return sig

    with pytest.raises(ValueError, match="look-ahead"):
        check_causal(peek, bars)
    with pytest.raises(ValueError, match="inside"):
        check_causal(lambda b: signals_from_core(Every5, b), bars, [N])


class LimitEvery5:
    """Every5 as a limit entry one tick below the close, stop and target in points."""

    def step(self, bar: BarView) -> Decision:
        if bar.i % 5 == 0:
            return Decision(
                side=1, entry_limit=bar.close - 0.25, stop_points=1.0, target_points=0.5
            )
        return Decision(exit=bar.i % 5 == 2)


def test_core_carries_the_limit_fields_and_check_causal_sees_them() -> None:
    bars = _bars()
    sig = signals_from_core(LimitEvery5, bars)
    assert sig.entry_limit is not None and sig.stop_points is not None
    assert sig.stop is None and sig.target is None
    np.testing.assert_array_equal(sig.entry_limit[::5], bars.close[::5] - 0.25)
    assert check_causal(lambda b: signals_from_core(LimitEvery5, b), bars) > 0

    def limit_peek(b: Bars) -> Signals:
        sig = signals_from_core(LimitEvery5, b)
        assert sig.entry_limit is not None
        lim = sig.entry_limit.copy()
        nxt = np.r_[b.close[1:], np.nan]  # the next bar's close: future data
        lim[np.isfinite(lim)] = nxt[np.isfinite(lim)] - 0.25
        return Signals(sig.entry_side, sig.exit, entry_limit=lim, stop_points=sig.stop_points,
                       target_points=sig.target_points)  # fmt: skip

    with pytest.raises(ValueError, match="look-ahead: entry_limit"):
        check_causal(limit_peek, bars)
