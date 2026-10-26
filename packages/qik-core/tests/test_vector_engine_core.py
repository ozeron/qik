"""Typed strategies: the per-bar driver and its contract (golden checks: tests/vector_engine)."""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
import pytest
from _typed_scripts import Script, replay, script

from qik.vector_engine import (
    HOLD,
    Bar,
    Bars,
    Decision,
    Features,
    FloatArray,
    Params,
    Position,
    Rules,
    State,
    run,
    strategy,
)
from qik.vector_engine.testing import assert_causal, capture_decisions, replay_features

N = 40


def _bars() -> Bars:
    c = 100.0 + np.arange(N) % 7 * 0.25
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range("2024-01-02 14:30", periods=N, freq="1min", tz="UTC")
    return Bars(o, np.maximum(o, c) + 0.25, np.minimum(o, c) - 0.25, c, index=idx,
                volume=np.arange(N, dtype=float))  # fmt: skip


class Seen(State):
    calls: int = 0
    in_order: bool = True
    ts3: int = 0
    volume3: float = 0.0
    close3: float = 0.0


class NoParams(Params):
    pass


class NoFeatures(Features):
    pass


class NextClose(Features):
    next_close: FloatArray


@strategy(state=Seen, params=NoParams, features=NoFeatures)
def every5(s: Seen, p: NoParams, f: NoFeatures, bar: Bar, pos: Position) -> Decision:
    """Long every 5th bar with a stop one point below, exit two bars later."""
    s.in_order = s.in_order and bar.i == s.calls
    s.calls += 1
    if bar.i == 3:
        s.ts3, s.volume3, s.close3 = bar.ts, bar.volume, bar.close
    if bar.i % 5 == 0:
        return Decision(side=1, stop=bar.close - 1.0)
    return Decision(exit=bar.i % 5 == 2)


@strategy(state=Seen, params=NoParams, features=NoFeatures)
def limit_every5(s: Seen, p: NoParams, f: NoFeatures, bar: Bar, pos: Position) -> Decision:
    """every5 as a limit entry one tick below the close, stop and target in points."""
    if bar.i % 5 == 0:
        return Decision(side=1, entry_limit=bar.close - 0.25, stop_points=1.0, target_points=0.5)
    return Decision(exit=bar.i % 5 == 2)


@strategy(state=Seen, params=NoParams, features=NextClose)
def peek(s: Seen, p: NoParams, f: NextClose, bar: Bar, pos: Position) -> Decision:
    """Tomorrow's close decides today."""
    return Decision(side=1, stop=bar.close - 1.0) if f.next_close[bar.i] > bar.close else HOLD


@strategy(state=Seen, params=NoParams, features=NextClose)
def limit_peek(s: Seen, p: NoParams, f: NextClose, bar: Bar, pos: Position) -> Decision:
    """limit_every5 priced off the next bar's close: future data."""
    if bar.i % 5 == 0:
        limit = f.next_close[bar.i] - 0.25
        return Decision(side=1, entry_limit=limit, stop_points=1.0, target_points=0.5)
    return Decision(exit=bar.i % 5 == 2)


@strategy(state=Seen, params=NoParams, features=NoFeatures)
def not_a_decision(s: Seen, p: NoParams, f: NoFeatures, bar: Bar, pos: Position) -> Decision:
    return 1  # ty: ignore[invalid-return-type]


@strategy(state=Seen, params=NoParams, features=NoFeatures)
def every5_side2(s: Seen, p: NoParams, f: NoFeatures, bar: Bar, pos: Position) -> Decision:
    return Decision(side=2) if bar.i == 0 else HOLD


def next_close(bars: Bars) -> NextClose:
    return NextClose(np.r_[bars.close[1:], np.nan])


def emitted(decisions: tuple[Decision, ...], name: str) -> np.ndarray:
    """One field of the strategy's own emitted decision column."""
    return np.array([getattr(d, name) for d in decisions], dtype=float)


def every5_script(**entry: np.ndarray) -> Script:
    """The scripted column every5/limit_every5 should emit."""
    entries = np.arange(N) % 5 == 0
    levels = {name: np.where(entries, value, np.nan) for name, value in entry.items()}
    return script(entries.astype(np.int8), np.arange(N) % 5 == 2, **levels)


@pytest.mark.parametrize("jit", [False, True])
def test_strategy_sees_each_bar_once_in_order_with_time_and_volume(jit: bool) -> None:
    bars = _bars()
    fields = ("entry_side", "exit", "stop")
    result, (tape,) = capture_decisions(
        every5, bars, features=NoFeatures(), signal_fields=fields, jit=jit
    )
    seen = result.state()
    assert (seen.calls, seen.in_order) == (N, True)
    assert bars.index is not None and seen.ts3 == bars.index[3].value
    assert (seen.volume3, seen.close3) == (3.0, bars.close[3])
    # Emitted decisions: entries every 5th bar with stop = close - 1, exits two bars
    # later; no decision sets target or entry_limit.
    assert tape.signal_fields == fields
    d = tape.decisions
    assert len(d) == N
    np.testing.assert_array_equal(np.flatnonzero(emitted(d, "side")), np.arange(0, N, 5))
    np.testing.assert_array_equal(np.flatnonzero(emitted(d, "exit")), np.arange(2, N, 5))
    stop = emitted(d, "stop")
    np.testing.assert_array_equal(stop[::5], bars.close[::5] - 1.0)
    assert np.isnan(np.delete(stop, np.s_[::5])).all()
    assert np.isnan(emitted(d, "target")).all() and np.isnan(emitted(d, "entry_limit")).all()
    # And they execute exactly like the equivalent scripted column.
    expected = replay(bars, every5_script(stop=bars.close - 1.0), jit=jit)
    assert len(expected.trades) > 0
    pd.testing.assert_frame_equal(result.trades, expected.trades, check_exact=True)
    pd.testing.assert_frame_equal(result.open_positions, expected.open_positions, check_exact=True)


def test_decision_and_driver_validation() -> None:
    with pytest.raises(ValueError, match="side"):
        replay_features([Decision(side=2)])
    with pytest.raises(ValueError, match="side must be -1, 0, 1"):
        run(every5_side2, _bars(), jit=False)
    message = "not subscriptable" if nb.config.DISABLE_JIT else "compilation failed"
    with pytest.raises(TypeError, match=message):
        run(not_a_decision, _bars())


def test_assert_causal_on_a_strategy_and_on_a_peek() -> None:
    bars = _bars()
    assert_causal(every5, bars, N // 2)
    with pytest.raises(AssertionError, match="prefix-causal"):
        assert_causal(peek, bars, N // 2, features=next_close)
    with pytest.raises(ValueError, match="n must lie in"):
        assert_causal(every5, bars, N + 1)


def test_strategy_carries_the_limit_fields_and_assert_causal_sees_them() -> None:
    bars = _bars()
    fields = ("entry_side", "exit", "entry_limit", "stop_points", "target_points")
    # A limit filled at a close whose 1-point stop is marketable: V2 flattens (A15).
    rules = Rules(protect_rejected="flatten")
    result, (tape,) = capture_decisions(limit_every5, bars, signal_fields=fields, rules=rules)
    d = tape.decisions
    limit = emitted(d, "entry_limit")
    np.testing.assert_array_equal(limit[::5], bars.close[::5] - 0.25)
    assert np.isnan(np.delete(limit, np.s_[::5])).all()
    np.testing.assert_array_equal(emitted(d, "stop_points")[::5], 1.0)
    np.testing.assert_array_equal(emitted(d, "target_points")[::5], 0.5)
    assert np.isnan(emitted(d, "stop")).all() and np.isnan(emitted(d, "target")).all()
    np.testing.assert_array_equal(np.flatnonzero(emitted(d, "side")), np.arange(0, N, 5))
    np.testing.assert_array_equal(np.flatnonzero(emitted(d, "exit")), np.arange(2, N, 5))
    scripted = every5_script(
        entry_limit=bars.close - 0.25, stop_points=np.ones(N), target_points=np.full(N, 0.5)
    )
    pd.testing.assert_frame_equal(
        result.trades, replay(bars, scripted, rules=rules).trades, check_exact=True
    )
    assert_causal(limit_every5, bars, N // 2, rules=rules)
    with pytest.raises(AssertionError, match="prefix-causal"):
        # The prefix must end on an entry bar (every 5th) for the peek to change it.
        assert_causal(limit_peek, bars, N // 2 + 1, features=next_close, rules=rules)
