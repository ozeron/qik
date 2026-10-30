"""Opt-in first protection and NY civil-date DAY expiry contracts."""

from __future__ import annotations

from dataclasses import replace

import numba as nb
import numpy as np
import pandas as pd
import pytest
from _typed_scripts import Script, replay, script
from hypothesis import given, settings
from hypothesis import strategies as st
from ve_kernel_adapter import run_columns as kernel_columns
from ve_reference import simulate_column_reference
from vector_engine.lifecycle_fixtures import F, lifecycle_step

from qik.vector_engine import (
    Bars,
    Costs,
    FillModel,
    FillTiming,
    Instrument,
    Profile,
    Rules,
    run,
)
from qik.vector_engine.api import _day_order_dates
from qik.vector_engine.testing import assert_python_jit_equal


def tape(times: pd.DatetimeIndex) -> Bars:
    c = np.full(len(times), 100.0)
    return Bars(c, c + 1, c - 1, c, index=times)


def differential(b: Bars, sig: Script, rules: Rules) -> None:
    """Compare independently replayed price points, including refusal metadata."""
    n = len(b)
    nan = np.full(n, np.nan)

    def col(name: str) -> np.ndarray | None:
        x = sig.get(name)
        return None if x is None else x[:, 0]

    def arr(name: str) -> np.ndarray:
        x = col(name)
        return nan if x is None else x

    days = _day_order_dates(b)
    timing = dict(
        next_open=rules.effective_entry_fill is FillTiming.NEXT_OPEN,
        exit_next_open=rules.effective_exit_fill is FillTiming.NEXT_OPEN,
        amend_close=rules.amend_timing == "close_submit",
        completed_close=rules.entry_bar_decision == "completed_close",
    )
    slots = sig.get("stop_update") is not None or sig.get("entry_stop") is not None
    raw = kernel_columns(
        b.open,
        b.high,
        b.low,
        b.close,
        np.asarray(sig.entry_side[:, 0], np.int8)[None],
        np.asarray(sig.exit[:, 0], bool)[None],
        arr("stop")[None],
        arr("target")[None],
        arr("entry_limit")[None],
        nan[None],
        nan[None],
        days,
        rules.limit_chase_bars,
        rules.adaptive_high_low_ordering,
        True,
        False,
        rules.protect_rejected == "flatten",
        latency_bars=rules.latency_bars,
        one_tick=rules.fill_model is FillModel.ONE_TICK,
        tick=0.25,
        entry_trigger=arr("entry_stop")[None] if slots else None,
        stop_updates=arr("stop_update")[None] if slots else None,
        target_updates=arr("target_update")[None] if slots else None,
        be_after=nan[None] if slots else None,
        be_offset=nan[None] if slots else None,
        arm_stop=rules.arm_stop == "next_bar",
        day_cancel=rules.day_order_expiry == "cancel",
        expiry_days=days,
        **timing,
    )
    ref = simulate_column_reference(
        b.open,
        b.high,
        b.low,
        b.close,
        np.asarray(sig.entry_side[:, 0], np.int8),
        np.asarray(sig.exit[:, 0], bool),
        arr("stop"),
        arr("target"),
        entry_limit=col("entry_limit"),
        stop_entry=col("entry_stop"),
        stop_update=col("stop_update"),
        target_update=col("target_update"),
        new_day=days,
        latency_bars=rules.latency_bars,
        one_tick=rules.fill_model is FillModel.ONE_TICK,
        tick=0.25,
        adaptive=rules.adaptive_high_low_ordering,
        protect_flatten=rules.protect_rejected == "flatten",
        arm_stop=rules.arm_stop == "next_bar",
        day_cancel=rules.day_order_expiry == "cancel",
        expiry_days=days,
        **timing,
        chase_after=rules.limit_chase_bars,
    )
    assert (raw[2][0], raw[3][0]) == (ref.err, ref.err_bar)
    if not ref.err:
        got = np.array(list(zip(*raw[8:], strict=True)), float).reshape(-1, 7)
        want = np.array([tuple(t.__dict__.values()) for t in ref.trades], float).reshape(-1, 7)
        np.testing.assert_equal(got, want)
        assert raw[4][0] == ref.final_side


@pytest.mark.parametrize("field", ["arm_stop", "day_order_expiry"])
def test_rules_validate(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        Rules(**{field: "unknown"})


@pytest.mark.parametrize("amend_timing", ["next_open", "close_submit"])
@pytest.mark.parametrize("entry_bar_decision", ["legacy", "completed_close"])
@pytest.mark.parametrize("day_order_expiry", ["refuse", "cancel"])
def test_p019_causal_first_stop(amend_timing, entry_bar_decision, day_order_expiry) -> None:
    b = Bars(
        np.array([100.0, 106.0, 102.0, 106.0]),
        np.array([107.0, 108.0, 108.0, 109.0]),
        np.array([99.0, 99.0, 101.0, 105.0]),
        np.array([106.0, 107.0, 106.0, 108.0]),
        index=pd.date_range("2023-01-03 09:30", periods=4, freq="min", tz="America/New_York"),
    )
    rules = Rules(
        profile=Profile.REPRO,
        entry_fill=FillTiming.NEXT_OPEN,
        exit_fill=FillTiming.CLOSE,
        arm_stop="next_bar",
        amend_timing=amend_timing,
        entry_bar_decision=entry_bar_decision,
        day_order_expiry=day_order_expiry,
    )
    kw = dict(features=F(), rules=rules, instrument=Instrument(0.25, 1.0), costs=Costs())
    r = run(lifecycle_step, b, **kw)
    assert r.trades[["entry_i", "exit_i", "entry_px", "exit_px"]].values.tolist() == [
        [1, 2, 106.0, 102.0]
    ]
    if not nb.config.DISABLE_JIT:
        assert_python_jit_equal(lifecycle_step, b, **kw)
    with pytest.raises(ValueError, match="no working order"):
        run(lifecycle_step, b, **dict(kw, rules=replace(rules, arm_stop="refuse")), jit=False)


@pytest.mark.parametrize("latency", [0, 1])
@pytest.mark.parametrize("fill", [FillModel.DEFAULT, FillModel.ONE_TICK])
@settings(max_examples=15, deadline=None)
@given(
    side=st.sampled_from([-1, 1]),
    gap=st.integers(-6, 6),
    adaptive=st.booleans(),
    level=st.integers(96, 104),
    target=st.booleans(),
    amend_close=st.booleans(),
    completed_close=st.booleans(),
    cancel=st.booleans(),
    next_open=st.booleans(),
)
def test_arm_properties(
    latency,
    fill,
    side,
    gap,
    adaptive,
    level,
    target,
    amend_close,
    completed_close,
    cancel,
    next_open,
) -> None:
    b = tape(pd.date_range("2023-01-03 10:00", periods=7, freq="min", tz="America/New_York"))
    o, h, lo, c = (x.copy() for x in (b.open, b.high, b.low, b.close))
    o[4], h[4], lo[4], c[4] = 100 + gap, max(106, 100 + gap), min(94, 100 + gap), 100
    b = replace(b, open=o, high=h, low=lo, close=c)
    su, tu = np.full(7, np.nan), np.full(7, np.nan)
    su[2] = level
    if target:
        tu[2] = 100 + side * 3
    sig = script(
        np.array([side, 0, 0, 0, 0, 0, 0]), np.zeros(7, bool), stop_update=su, target_update=tu
    )
    differential(
        b,
        sig,
        Rules(
            arm_stop="next_bar",
            profile=Profile.REPRO,
            entry_fill=(
                FillTiming.NEXT_OPEN
                if next_open and not latency and fill is FillModel.DEFAULT
                else FillTiming.CLOSE
            ),
            amend_timing="close_submit" if amend_close and not latency else "next_open",
            entry_bar_decision="completed_close" if completed_close else "legacy",
            day_order_expiry="cancel" if cancel else "refuse",
            protect_rejected="flatten",
            adaptive_high_low_ordering=adaptive,
            latency_bars=latency,
            fill_model=fill,
        ),
    )


@pytest.mark.parametrize("kind", ["limit", "stop", "slots_limit"])
@pytest.mark.parametrize("latency", [0, 1])
@pytest.mark.parametrize("date", ["2023-03-12", "2023-11-05", "2023-11-24"])
def test_expiry_before_exact_boundary_match(kind, latency, date) -> None:
    midnight = pd.Timestamp(date, tz="America/New_York")
    b = tape(
        pd.DatetimeIndex(
            [
                midnight - pd.Timedelta(minutes=2),
                midnight - pd.Timedelta(minutes=1),
                midnight,
                midnight + pd.Timedelta(minutes=1),
                midnight + pd.Timedelta(minutes=2),
            ]
        )
    )
    h, lo = b.high.copy(), b.low.copy()
    h[2], lo[2] = 110, 90  # expired entry would fill here
    b = replace(b, high=h, low=lo)
    nan = np.full(5, np.nan)
    sig = script(
        np.array([1, 0, 0, 0, 0]),
        np.zeros(5, bool),
        stop=np.full(5, 80.0),
        entry_stop=np.full(5, 105.0) if kind == "stop" else None,
        entry_limit=np.full(5, 95.0) if kind != "stop" else None,
        stop_update=nan if kind == "slots_limit" else None,
    )
    rules = Rules(day_order_expiry="cancel", latency_bars=latency, limit_chase_bars=10)
    assert replay(b, sig, rules=rules).trades.empty
    assert replay(b, sig, rules=rules).open_positions.empty
    differential(b, sig, rules)
    with pytest.raises(ValueError, match="DAY limit/stop entry"):
        replay(b, sig, rules=replace(rules, day_order_expiry="refuse"))


def test_p111_daily_reissue_and_custom_sessions() -> None:
    b = tape(pd.date_range("2019-03-04 18:00", periods=4, freq="D", tz="America/New_York"))
    b = replace(b, session_labels=np.array([0, 0, 0, 0]))
    kw = dict(
        grid={"expiry": [True]},
        features=F(),
        rules=Rules(profile=Profile.REPRO, day_order_expiry="cancel"),
    )
    r = run(lifecycle_step, b, **kw)[0]
    assert r.trades.empty
    assert r.open_positions.empty
    assert r.state().seen_pending == 0
    if not nb.config.DISABLE_JIT:
        assert_python_jit_equal(lifecycle_step, b, **kw)
    with pytest.raises(ValueError, match="timezone-aware"):
        run(lifecycle_step, replace(b, index=None), **kw)


def test_day_expiry_is_research_only() -> None:
    b = tape(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="America/New_York"))
    sig = script(np.array([1, 0, 0]), np.zeros(3, bool), stop=np.array([95.0, np.nan, np.nan]))
    assert replay(b, sig).deployable
    assert not replay(b, sig, rules=Rules(day_order_expiry="cancel")).deployable


@pytest.mark.parametrize("protect", ["error", "flatten"])
def test_marketable_first_stop_uses_protection_policy(protect: str) -> None:
    b = tape(pd.date_range("2023-01-03 10:00", periods=4, freq="min", tz="America/New_York"))
    sig = script(
        np.array([1, 0, 0, 0]),
        np.zeros(4, bool),
        stop_update=np.array([np.nan, 100.0, np.nan, np.nan]),
    )
    rules = Rules(arm_stop="next_bar", protect_rejected=protect, amend_rejected="keep")
    if protect == "error":
        with pytest.raises(ValueError, match="protection goes out"):
            replay(b, sig, rules=rules)
    else:
        r = replay(b, sig, rules=rules)
        assert r.trades[["exit_i", "exit_px", "reason"]].values.tolist() == [
            [1, 100.0, "protect_fail"]
        ]
    differential(b, sig, rules)


@pytest.mark.parametrize("adaptive,expected", [(False, "target"), (True, "stop")])
def test_new_bracket_tie_uses_existing_ohlc_path(adaptive: bool, expected: str) -> None:
    b = tape(pd.date_range("2023-01-03 10:00", periods=4, freq="min", tz="America/New_York"))
    h, lo = b.high.copy(), b.low.copy()
    h[2], lo[2] = 104.0, 96.0
    b = replace(b, high=h, low=lo)
    sig = script(
        np.array([1, 0, 0, 0]),
        np.zeros(4, bool),
        stop_update=np.array([np.nan, 97.0, np.nan, np.nan]),
        target_update=np.array([np.nan, 103.0, np.nan, np.nan]),
    )
    rules = Rules(arm_stop="next_bar", adaptive_high_low_ordering=adaptive)
    r = replay(b, sig, rules=rules)
    assert r.trades.exit_i.tolist() == [2]
    assert r.trades.reason.tolist() == [expected]
    differential(b, sig, rules)


@pytest.mark.parametrize(
    "times",
    [
        ["2023-03-12 06:59Z", "2023-03-12 07:00Z"],
        ["2023-11-05 05:59Z", "2023-11-05 06:00Z"],
        ["2023-11-24 17:59Z", "2023-11-24 18:00Z"],
    ],
)
def test_dst_and_early_close_do_not_change_civil_date(times: list[str]) -> None:
    b = tape(pd.DatetimeIndex(times))
    np.testing.assert_array_equal(_day_order_dates(b), [False, False])
    sig = script(np.array([1, 0]), np.zeros(2, bool), entry_stop=np.full(2, 105.0))
    result = replay(b, sig, rules=Rules(day_order_expiry="cancel"))
    # A pending order remains; the absence of a fill alone cannot prove expiry.
    assert result.trades.empty
    h = b.high.copy()
    h[1] = 106.0
    result = replay(replace(b, high=h), sig, rules=Rules(day_order_expiry="cancel"))
    assert len(result.open_positions) == 1
