"""Independent feature fixtures, prefix checks and a compiled typed consumer."""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
import pytest

from qik.vector_engine import Bars, run
from qik.vector_engine.blocks import held_bars_exit, range_breakout
from qik.vector_engine.features import anchored_vwap, session_masks, session_range
from qik.vector_engine.schema import BoolArray, Features, FloatArray, Params, State, strategy
from qik.vector_engine.testing import assert_causal_feature, assert_python_jit_equal
from qik.vector_engine.types import HOLD, Bar, Decision, Position, enter, exit

ET = "America/New_York"
# Handwritten exchange dates: both DST changes, Thanksgiving closure/half day.
DAYS = ["2023-03-10", "2023-03-13", "2023-11-03", "2023-11-06", "2023-11-24"]
OPENS = ["14:30", "13:30", "13:30", "14:30", "14:30"]
CLOSES = ["21:00", "20:00", "20:00", "21:00", "18:00"]


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": pd.to_datetime(DAYS),
            "open_ts": pd.to_datetime([f"{d} {t}Z" for d, t in zip(DAYS, OPENS, strict=True)]),
            "close_ts": pd.to_datetime([f"{d} {t}Z" for d, t in zip(DAYS, CLOSES, strict=True)]),
            "early_close": [False, False, False, False, True],
        }
    )


def tape() -> pd.DataFrame:
    stamps = []
    for d in [*DAYS, "2023-11-23"]:
        # Missing 09:31 and final scheduled bar on the half day.
        times = ["09:29", "09:30", "09:32", "09:33", "12:58", "13:00", "15:58", "15:59", "16:00"]
        stamps.extend(f"{d} {t}" for t in times)
    ts = pd.DatetimeIndex(stamps).tz_localize(ET).sort_values()
    return pd.DataFrame(
        {
            "datetime": ts,
            "high": 102.0 + np.arange(len(ts)),
            "low": 98.0 + np.arange(len(ts)),
            "close": 100.0 + np.arange(len(ts)),
            "volume": np.resize([0.0, 0.0, 2.0, 0.0, 3.0], len(ts)),
        }
    )


def build(frame: pd.DataFrame):
    s = session_masks(pd.DatetimeIndex(frame.datetime), schedule(), cutoff_min=958)
    r = session_range(
        frame.high.to_numpy(),
        frame.low.to_numpy(),
        s.date,
        s.clock_min,
        start_min=570,
        end_min=573,
        min_bars=2,
    )
    reset = np.r_[True, s.date[1:] != s.date[:-1]] if len(frame) else np.zeros(0, dtype=bool)
    v = anchored_vwap(frame.close.to_numpy(), frame.volume.to_numpy(), reset, summation="naive")
    return s, r, v


def test_calendar_masks_handwritten_dst_holiday_and_missing_terminal() -> None:
    frame = tape()
    masks, _, _ = build(frame)
    for i, stamp in enumerate(pd.DatetimeIndex(frame.datetime).tz_convert(ET)):
        holiday = str(stamp.date()) == "2023-11-23"
        early = str(stamp.date()) == "2023-11-24"
        minute = stamp.hour * 60 + stamp.minute
        last = 779 if early else 959
        active = not holiday and 570 <= minute <= last
        assert masks.date[i] == int(stamp.strftime("%Y%m%d"))
        assert masks.clock_min[i] == minute
        assert masks.active[i] == active
        assert masks.start[i] == (active and minute == 570)
        assert masks.end[i] == (active and minute == last)
        assert masks.cutoff[i] == (active and minute >= min(958, last))
        assert masks.early_close[i] == early
    assert not masks.end[masks.early_close].any()  # no last-source-bar inference
    utc = pd.DatetimeIndex(schedule().open_ts)
    assert list(utc.hour) == [14, 13, 13, 14, 14]
    # Same instants in UTC must yield the same New York dates and masks.
    other = session_masks(
        pd.DatetimeIndex(frame.datetime).tz_convert("UTC"), schedule(), cutoff_min=958
    )
    for name in masks.__annotations__:
        np.testing.assert_array_equal(getattr(masks, name), getattr(other, name))


def test_range_matches_independent_streaming_reference() -> None:
    frame = tape()
    s, r, _ = build(frame)
    previous = -1
    hi, lo, count = -np.inf, np.inf, 0
    for i in range(len(frame)):
        if s.date[i] != previous:
            previous, hi, lo, count = s.date[i], -np.inf, np.inf, 0
        if 570 <= s.clock_min[i] < 573:
            hi, lo = max(hi, frame.high.iloc[i]), min(lo, frame.low.iloc[i])
            count += 1
        ready = s.clock_min[i] >= 573 and count >= 2
        assert r.ready[i] == ready
        assert r.high[i] == hi if ready else np.isnan(r.high[i])
        assert r.low[i] == lo if ready else np.isnan(r.low[i])
        assert range_breakout(1e9, r.high[i], r.low[i]) == int(ready)
    # Explicit completeness policy rejects the same missing minute.
    rejected = session_range(
        frame.high.to_numpy(),
        frame.low.to_numpy(),
        s.date,
        s.clock_min,
        start_min=570,
        end_min=573,
        min_bars=3,
    )
    assert not rejected.ready.any()
    # A session starting after the window cannot inherit yesterday's extrema.
    only_late = frame.loc[frame.datetime.dt.hour >= 13].reset_index(drop=True)
    assert not build(only_late)[1].ready.any()


@pytest.mark.parametrize("summation", ["naive", "kahan"])
def test_vwap_reference_zero_volume_and_explicit_intraday_reset(summation) -> None:
    px = np.array([10.0, 20.0, 30.0, 99.0, 50.0, 60.0, 80.0])
    vol = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0])
    reset = np.array([True, False, False, False, True, False, False])
    actual = anchored_vwap(px, vol, reset, summation=summation)
    np.testing.assert_array_equal(actual, [np.nan, np.nan, 30.0, 30.0, np.nan, 60.0, 75.0])
    pv = weight = 0.0
    expected = []
    for p, v, rst in zip(px, vol, reset, strict=True):
        if rst:
            pv = weight = 0.0
        pv += p * v
        weight += v
        expected.append(pv / weight if weight else np.nan)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 9, 10, 18, 27, 40, 53])
def test_all_features_and_held_bar_decisions_are_prefix_causal(n) -> None:
    frame = tape()
    for field in ("date", "clock_min", "active", "start", "end", "cutoff", "early_close"):
        assert_causal_feature(lambda f: getattr(build(f)[0], field), frame, n)
    for field in ("high", "low", "ready"):
        assert_causal_feature(lambda f: getattr(build(f)[1], field), frame, n)
    for summation in ("naive", "kahan"):

        def vwap(f):
            reset = f.datetime.dt.date.ne(f.datetime.dt.date.shift()).to_numpy()
            return anchored_vwap(
                f.close.to_numpy(), f.volume.to_numpy(), reset, summation=summation
            )

        assert_causal_feature(vwap, frame, n)
    assert_causal_feature(
        lambda f: np.array([held_bars_exit(i, 3) for i in range(len(f))]), frame, n
    )


@nb.njit(cache=True)
def compiled_holding(count, limit):
    return held_bars_exit(count, limit)


@pytest.mark.parametrize("count,expected", [(0, False), (2, False), (3, True), (4, True)])
def test_holding_python_equals_jit(count, expected) -> None:
    assert held_bars_exit(count, 3) == expected
    assert compiled_holding(count, 3) == expected


@pytest.mark.parametrize("count,limit", [(-1, 3), (0, 0), (1, -1)])
def test_holding_rejects_invalid_counts(count, limit) -> None:
    for fn in (held_bars_exit, compiled_holding):
        with pytest.raises(ValueError):
            fn(count, limit)


class S(State):
    seen: int = 0


class P(Params):
    hold: int = 2


class F(Features):
    high: FloatArray
    low: FloatArray
    vwap: FloatArray
    active: BoolArray
    start: BoolArray
    end: BoolArray
    cutoff: BoolArray
    early: BoolArray
    ready: BoolArray


@strategy(state=S, params=P, features=F)
def template_step(s: S, p: P, f: F, bar: Bar, pos: Position) -> Decision:
    i = bar.i
    if f.start[i]:
        s.seen = 0
    if f.active[i]:
        s.seen += 1
    if pos.side:
        if f.end[i] or f.cutoff[i] or held_bars_exit(i - pos.entry_i, p.hold):
            return exit()
    elif f.active[i] and f.ready[i] and not f.early[i] and bar.close >= f.vwap[i]:
        side = range_breakout(bar.close, f.high[i], f.low[i])
        if side:
            return enter(side, stop_points=20.0)
    return HOLD


def test_typed_feature_consumer_python_jit_subprocess() -> None:
    frame = tape()
    s, r, v = build(frame)
    features = F(
        high=r.high,
        low=r.low,
        vwap=v,
        active=s.active,
        start=s.start,
        end=s.end,
        cutoff=s.cutoff,
        early=s.early_close,
        ready=r.ready,
    )
    c = frame.close.to_numpy()
    bars = Bars(
        c,
        frame.high.to_numpy(),
        frame.low.to_numpy(),
        c,
        index=pd.DatetimeIndex(frame.datetime),
        volume=frame.volume.to_numpy(),
    )
    if nb.config.DISABLE_JIT:
        assert len(run(template_step, bars, features=features).trades) > 0
    else:
        assert_python_jit_equal(template_step, bars, features=features)
        assert len(run(template_step, bars, features=features).trades) > 0


@pytest.mark.parametrize(
    "kind",
    ["naive_clock", "duplicates", "unordered", "nonminute", "overnight", "bad_date", "bad_flag"],
)
def test_masks_refuse_malformed_inputs(kind) -> None:
    ts, sched = pd.DatetimeIndex(tape().datetime), schedule()
    if kind == "naive_clock":
        ts = ts.tz_localize(None)
    elif kind == "duplicates":
        ts = ts.insert(0, ts[0])
    elif kind == "unordered":
        ts = ts[::-1]
    elif kind == "nonminute":
        ts += pd.Timedelta(seconds=1)
    elif kind == "overnight":
        sched.loc[0, "open_ts"] -= pd.Timedelta(days=1)
    elif kind == "bad_date":
        sched.loc[0, "session_date"] += pd.Timedelta(hours=1)
    else:
        sched["early_close"] = 0
    with pytest.raises((ValueError, TypeError)):
        session_masks(ts, sched, cutoff_min=955)


def test_empty_inputs_and_bad_feature_inputs() -> None:
    empty = np.zeros(0, dtype=np.float64)
    assert len(anchored_vwap(empty, empty, np.zeros(0, dtype=bool), summation="naive")) == 0
    assert not session_range(
        empty,
        empty,
        empty.astype(np.int64),
        empty.astype(np.int64),
        start_min=570,
        end_min=585,
        min_bars=1,
    ).ready.any()
    for px, vol, reset in [
        ([1.0], [-1.0], [True]),
        ([np.nan], [1.0], [True]),
        ([1.0], [1.0], [False]),
        ([1.0], [1.0], [1]),
        ([1.0], [], [True]),
    ]:
        with pytest.raises(ValueError):
            anchored_vwap(np.array(px), np.array(vol), np.array(reset), summation="naive")
    with pytest.raises(ValueError, match="unique increasing"):
        session_range(
            np.ones(2),
            np.ones(2),
            np.ones(2, dtype=np.int64),
            np.array([570, 570]),
            start_min=570,
            end_min=585,
            min_bars=1,
        )
