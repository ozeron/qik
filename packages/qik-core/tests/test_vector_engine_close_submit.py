"""Close submission and post-open-fill close decisions (P024 and P020)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import replay, script
from hypothesis import given, settings
from hypothesis import strategies as st

from qik.vector_engine import (
    HOLD,
    Bar,
    Bars,
    Features,
    FillTiming,
    Params,
    Position,
    Profile,
    Rules,
    State,
    TypedDecision,
    enter,
    exit,
    run,
    strategy,
    trail,
)
from qik.vector_engine import kernel as K
from qik.vector_engine import reference as R
from scripts.engine_test_tiers import engine_examples

RULES = Rules(
    profile=Profile.REPRO,
    entry_fill=FillTiming.NEXT_OPEN,
    exit_fill=FillTiming.CLOSE,
    amend_timing="close_submit",
    entry_bar_decision="completed_close",
)


def tape(side: int = 1, gap: float = 98.0) -> Bars:
    o = np.array([100.0, 100.0, gap, gap])
    c = np.array([100.0, 100.0, gap, gap])
    h, lo = np.maximum(o, c) + 0.25, np.minimum(o, c) - 0.25
    if side < 0:
        o, h, lo, c = 200 - o, 200 - lo, 200 - h, 200 - c
    return Bars(o, h, lo, c, index=pd.date_range("2023-01-03", periods=4, freq="min", tz="UTC"))


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("keep", [False, True])
def test_amend_validated_at_submission_and_gap_fills_at_open(side: int, keep: bool) -> None:
    bars = tape(side)
    sig = script(
        np.array([side, 0, 0, 0]),
        np.array([False, False, False, True]),
        stop=np.full(4, 95.0 if side > 0 else 105.0),
        stop_update=np.array([np.nan, 99.0 if side > 0 else 101.0, np.nan, np.nan]),
    )
    rules = replace(RULES, amend_rejected="keep" if keep else "error")
    result = replay(bars, sig, rules=rules)
    assert result.trades.exit_i.tolist() == [2]
    assert result.trades.exit_px.tolist() == [98.0 if side > 0 else 102.0]
    assert result.trades.reason.tolist() == ["stop_gap"]
    assert result.trades.stop.tolist() == [95.0 if side > 0 else 105.0]  # entry risk stays fixed
    if keep:
        old = replay(bars, sig, rules=replace(rules, amend_timing="next_open"))
        assert old.trades.exit_i.tolist() == [3]
    else:
        with pytest.raises(ValueError, match="amendment at bar 2"):
            replay(bars, sig, rules=replace(rules, amend_timing="next_open"))


@pytest.mark.parametrize("side", [-1, 1])
def test_close_submit_still_refuses_marketable_amend(side: int) -> None:
    sig = script(
        np.array([side, 0, 0, 0]),
        np.zeros(4, bool),
        stop=np.full(4, 95.0 if side > 0 else 105.0),
        stop_update=np.array([np.nan, 100.0, np.nan, np.nan]),
    )
    with pytest.raises(ValueError, match="amendment at bar 1"):
        replay(tape(side), sig, rules=RULES)


class S(State):
    calls: int = 0


class P(Params):
    pass


class F(Features):
    pass


@strategy(state=S, params=P, features=F)
def amend(s: S, p: P, f: F, bar: Bar, pos: Position) -> TypedDecision:
    s.calls += 1
    if bar.i == 0:
        return enter(1, stop=95.0)
    if bar.i == 1 and pos.side:
        return trail(99.0)
    if bar.i == 3:
        return exit()
    return HOLD


@strategy(state=S, params=P, features=F)
def reentry(s: S, p: P, f: F, bar: Bar, pos: Position) -> TypedDecision:
    s.calls += 1
    if bar.i == 0:
        return enter(1, stop=99.0)
    if bar.i == 1 and not pos.side:
        return enter(-1, stop=103.0)
    if bar.i == 2:
        return exit()
    return HOLD


@pytest.mark.parametrize("jit", [False, True])
def test_callback_observes_completed_close_once(jit: bool) -> None:
    bars = tape()
    result = run(amend, bars, features=F(), rules=RULES, jit=jit)
    assert result.trades.exit_i.tolist() == [2]
    assert result.trades.exit_px.tolist() == [98.0]
    assert result.state().calls == 4
    bars = replace(bars, low=np.array([99.75, 98.0, 97.75, 97.75]))
    for fields in (None, ("entry_side", "exit", "stop", "target")):
        kw = {} if fields is None else {"signal_fields": fields}
        result = run(reentry, bars, features=F(), rules=RULES, jit=jit, **kw)
        assert result.trades.entry_i.tolist() == [1, 2]
        assert result.trades.exit_i.tolist() == [1, 2]
        assert result.state().calls == 4
    legacy = run(
        reentry, bars, features=F(), rules=replace(RULES, entry_bar_decision="legacy"), jit=jit
    )
    assert legacy.trades.entry_i.tolist() == [1]
    blocked = run(
        reentry, bars, features=F(), rules=replace(RULES, reenter_on_exit_bar=False), jit=jit
    )
    assert blocked.trades.entry_i.tolist() == [1]


@settings(max_examples=engine_examples(200), deadline=None, derandomize=True)
@given(
    side=st.sampled_from([-1, 1]),
    gap=st.integers(96, 102),
    level=st.integers(96, 102),
    close_submit=st.booleans(),
    completed_close=st.booleans(),
    keep=st.booleans(),
    oto=st.booleans(),
)
def test_hypothesis_independent_reference(
    side, gap, level, close_submit, completed_close, keep, oto
):
    bars = tape(side, float(gap))
    a = np.array([side, -side, 0, 0], np.int8)
    x = np.array([False, False, False, True])
    stops = np.array([95.0, 105.0, 95.0, 95.0])
    updates = np.array([np.nan, float(level), np.nan, np.nan])
    if side < 0:
        stops, updates = 200 - stops, 200 - updates
    nan = np.full(4, np.nan)
    ref = R.simulate_column_reference(
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        a,
        x,
        stops,
        nan,
        next_open=True,
        exit_next_open=False,
        stop_update=updates,
        amend_close=close_submit,
        completed_close=completed_close,
        amend_keep=keep,
        oto=oto,
    )

    def col(z):
        return np.ascontiguousarray(z[None, :])

    out = K.run_columns(
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        col(a),
        col(x),
        col(stops),
        col(nan),
        col(nan),
        col(nan),
        col(nan),
        np.zeros(4, bool),
        1,
        False,
        True,
        False,
        False,
        next_open=True,
        exit_next_open=False,
        entry_trigger=col(nan),
        stop_updates=col(updates),
        target_updates=col(nan),
        be_after=col(nan),
        be_offset=col(nan),
        amend_close=close_submit,
        completed_close=completed_close,
        amend_keep=keep,
        oto=oto,
    )
    assert (int(out[2][0]), int(out[3][0])) == (ref.err, ref.err_bar)
    if not ref.err:
        assert list(zip(*out[8:], strict=True)) == [
            (t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px, t.reason, t.stop)
            for t in ref.trades
        ]


def test_validation() -> None:
    for field in ("amend_timing", "entry_bar_decision"):
        with pytest.raises(ValueError, match=field):
            Rules(**{field: "unknown"})
    with pytest.raises(ValueError, match="close_submit requires"):
        Rules(amend_timing="close_submit", latency_bars=1)


@pytest.mark.parametrize(
    "close_submit,completed_close", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("oto", [False, True])
@pytest.mark.parametrize("entry_next,exit_next", [(True, True), (True, False), (False, True)])
def test_new_rules_dense_reference(
    monkeypatch, close_submit, completed_close, oto, entry_next, exit_next
):
    """Exercise 18,000 existing dense cases with the new policies, without changing old goldens."""
    from functools import partial

    import test_vector_engine_slots as dense

    options = dict(amend_close=close_submit, completed_close=completed_close)
    monkeypatch.setattr(
        dense, "simulate_column_reference", partial(R.simulate_column_reference, **options)
    )
    monkeypatch.setattr(K, "run_columns", partial(K.run_columns, **options))
    dense.test_slots_kernel_equals_independent_reference_dense(0, False, oto, entry_next, exit_next)


def test_python_jit_full_results_and_state() -> None:
    from qik.vector_engine.testing import assert_python_jit_equal

    assert_python_jit_equal(amend, tape(), features=F(), rules=RULES)
    bars = replace(tape(), low=np.array([99.75, 98.0, 97.75, 97.75]))
    assert_python_jit_equal(reentry, bars, features=F(), rules=RULES)


def test_close_submit_terminal_amend_and_marketable_limit_target() -> None:
    bars = tape(gap=100.0)
    sig = script(
        np.array([1, 0, 0, 0]),
        np.zeros(4, bool),
        stop=np.full(4, 95.0),
        stop_update=np.array([np.nan, np.nan, np.nan, 99.0]),
    )
    assert len(replay(bars, sig, rules=RULES).open_positions) == 1
    with pytest.raises(ValueError, match="NEXT_OPEN"):
        replay(bars, sig, rules=replace(RULES, amend_timing="next_open"))
    sig = sig.replace(
        stop_update=None,
        target=np.full(4, 105.0),
        target_update=np.array([np.nan, 100.0, np.nan, np.nan]),
    )
    result = replay(bars, sig, rules=RULES)
    assert result.trades.exit_i.tolist() == [1]
    assert result.trades.exit_px.tolist() == [100.0]
    assert result.trades.reason.tolist() == ["target"]
