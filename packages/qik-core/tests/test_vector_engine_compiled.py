"""Compiled close decisions replay the same hypothesis/dense rule axes as batch matching."""

from __future__ import annotations

import inspect
from collections import namedtuple

import numba as nb
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numba.experimental import jitclass
from test_vector_engine import _case, _dense_case

from qik.vector_engine import Bars, Rules, Signals, run, simulate
from qik.vector_engine import kernel as K
from qik.vector_engine.api import SIGNAL_FIELDS
from qik.vector_engine.compiled import _jitclass_step, _run

_BATCH = K.run_columns


@nb.njit(cache=True)
def replay(state, params, i, feats, o, h, lo, c, v, view, out):
    # The saved pre-IOC fixture retains its original fifteen numeric fields.
    out[:15] = feats[int(params[0]), i]


def compare(*args, **kwargs):
    class_core = kwargs.pop("_class_core", False)
    batch = _BATCH(*args, **kwargs)
    sig = inspect.signature(_BATCH if nb.config.DISABLE_JIT else _BATCH.py_func).bind(
        *args, **kwargs
    )
    sig.apply_defaults()
    d = sig.arguments
    k, n = d["side_sig"].shape
    feats = np.full((k, n, 15), np.nan)
    for q, name in enumerate(
        (
            "side_sig",
            "exit_sig",
            "entry_stop",
            "entry_target",
            "entry_limit",
            "stop_pts",
            "target_pts",
            "entry_trigger",
            "stop_updates",
            "target_updates",
            "be_after",
            "be_offset",
            "unused",
            "exit_long",
            "exit_short",
        )
    ):
        value = d.get(name)
        feats[:, :, q] = (0 if q in (0, 1, 12, 13, 14) else np.nan) if value is None else value
    rc = np.array(
        [
            d["chase"],
            d["adaptive"],
            d["reenter"],
            d["stale_drop"],
            d["protect_flatten"],
            d["next_open"],
            d["next_open"] if d["exit_next_open"] is None else d["exit_next_open"],
            d["latency_bars"],
            d["one_tick"],
            d["tick"],
            d["adverse_first"],
            d["bracket_anchor"],
            d["target_gap"],
            d["tick_round"],
            d["cross_session"],
            d["mit"],
            d["oto"],
            d["amend_keep"],
            d["ratchet"],
            d["stop_skip"],
            0,
            0,
            # Numba does not bounds-check: the replay vector must cover every
            # index _run reads (combine, amend_close, completed_close,
            # arm_stop, day_cancel).
            d["combine"],
            d["amend_close"],
            d["completed_close"],
            d["arm_stop"],
            d["day_cancel"],
        ],
        float,
    )
    masks = np.zeros((k, 2, n), np.int8)
    for q, name in enumerate(("force_close", "force_open")):
        if d[name] is not None:
            masks[:, q] = d[name]
    cores = None
    clock = feats
    if class_core:
        from numba.typed import List

        cores = List([ReplayCore(np.ascontiguousarray(row)) for row in feats])
        clock = np.full((n, 1), -1, np.int64)
    raw = _run(
        _jitclass_step if class_core else replay,
        np.zeros(0),
        np.arange(k, dtype=float)[:, None],
        clock,
        d["o"],
        d["h"],
        d["lo"],
        d["c"],
        np.zeros(n),
        d["new_day"],
        np.zeros(n, bool),
        rc,
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        np.nan,
        d["entry_trigger"] is not None,
        cores,
        masks,
        expiry_days=d["expiry_days"],
    )
    counts, meta, *arrays = raw
    np.testing.assert_array_equal(meta[:, 0], batch[2])
    np.testing.assert_array_equal(meta[:, 1], batch[3])
    for j in range(k):
        if batch[2][j]:
            continue
        assert counts[j] == batch[0][j]
        for q in range(4):
            np.testing.assert_array_equal(meta[j, q + 2], batch[q + 4][j])
        start, stop = batch[1][j : j + 2]
        for q in range(7):
            np.testing.assert_array_equal(arrays[q][j, : counts[j]], batch[q + 8][start:stop])
    return batch


@given(
    case=_case(),
    slots=st.booleans(),
    next_open=st.booleans(),
    exit_next_open=st.booleans(),
    latency=st.integers(0, 1),
    one_tick=st.booleans(),
    adverse=st.booleans(),
    rounding=st.integers(0, 2),
    anchor=st.integers(0, 2),
    gap=st.booleans(),
)
@settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_stateless_replay_hypothesis(
    case, slots, next_open, exit_next_open, latency, one_tick, adverse, rounding, anchor, gap
):
    arrays, side, exits, stop, target, adaptive, reenter, stale = case
    n = len(side)
    nan = np.full((1, n), np.nan)
    compare(
        arrays["o"],
        arrays["h"],
        arrays["lo"],
        arrays["c"],
        side[None],
        exits[None],
        stop[None],
        target[None],
        nan,
        nan,
        nan,
        np.arange(n) % 7 == 0,
        1,
        adaptive,
        reenter,
        stale,
        False,
        next_open,
        latency,
        one_tick,
        0.25,
        exit_next_open=exit_next_open,
        entry_trigger=nan if slots else None,
        stop_updates=nan if slots else None,
        target_updates=nan if slots else None,
        be_after=nan if slots else None,
        be_offset=nan if slots else None,
        adverse_first=adverse,
        bracket_anchor=anchor,
        target_gap=gap,
        tick_round=rounding,
    )


def test_stateless_replay_dense():
    rng = np.random.default_rng(7911)
    for slots in (False, True):
        for _ in range(500):
            case = _dense_case(rng)
            arrays, side, exits, stop, target = case[:5]
            n = len(side)
            nan = np.full((1, n), np.nan)
            compare(
                arrays["o"],
                arrays["h"],
                arrays["lo"],
                arrays["c"],
                side[None],
                exits[None],
                stop[None],
                target[None],
                nan,
                nan,
                nan,
                np.zeros(n, bool),
                1,
                False,
                True,
                True,
                False,
                entry_trigger=nan if slots else None,
                stop_updates=nan if slots else None,
                target_updates=nan if slots else None,
                be_after=nan if slots else None,
                be_offset=nan if slots else None,
            )


# The legacy 16-value record (through IOC); the adapter fills absent paired fields.
NumericDecision = namedtuple("NumericDecision", ("side", *SIGNAL_FIELDS[1:16]))


@jitclass([("signals", nb.float64[:, ::1])])
class ReplayCore:
    def __init__(self, signals):
        self.signals = signals

    def step(self, bar, pos):
        r = self.signals[bar.i]
        return NumericDecision(
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            r[9],
            r[10],
            r[11],
            r[12],
            r[13],
            r[14],
            np.nan,  # IOC is absent in the saved pre-IOC corpus.
        )


def test_public_jitclass_runner_is_field_exact():
    c = np.array([100.0, 100.0, 101.0])
    bars = Bars(c, c + 1, c - 1, c)
    feats = np.full((3, 15), np.nan)
    feats[:, 0] = [1, 0, 0]
    feats[:, 1] = [0, 0, 1]
    feats[:, 5] = 5.0
    feats[:, 12:] = 0
    result = run(
        ReplayCore,
        bars,
        features={"signals": feats},
        signal_fields=("entry_side", "exit", "stop_points"),
    )[0]
    expected = simulate(
        bars,
        Signals(feats[:, 0].astype(np.int8), feats[:, 1].astype(bool), stop_points=feats[:, 5]),
    )
    pd.testing.assert_frame_equal(result.trades, expected.trades, check_exact=True)


def test_jitclass_replays_hypothesis_and_dense_corpora(monkeypatch):
    import sys

    numeric_compare = compare

    def class_compare(*args, **kwargs):
        return numeric_compare(*args, **kwargs, _class_core=True)

    monkeypatch.setattr(sys.modules[__name__], "compare", class_compare)
    test_stateless_replay_hypothesis()
    test_stateless_replay_dense()


def test_public_last_close_flatten_needs_no_calendar():
    from qik.vector_engine import Flatten, Profile

    c = np.full(4, 100.0)
    bars = Bars(c, c + 1, c - 1, c)
    matrix = np.full((4, 15), np.nan)
    matrix[:, (0, 1, 12, 13, 14)] = 0
    matrix[0, 0] = 1
    matrix[2, 12] = 1
    rules = Rules(profile=Profile.REPRO, flatten=Flatten.LAST_BAR_CLOSE)
    result = run(
        ReplayCore,
        bars,
        features={"signals": matrix},
        signal_fields=("entry_side", "exit", "flatten"),
        rules=rules,
    )[0]
    expected = simulate(
        bars,
        Signals(
            matrix[:, 0].astype(np.int8),
            matrix[:, 1].astype(bool),
            flatten=matrix[:, 12].astype(bool),
        ),
        rules=rules,
    )
    pd.testing.assert_frame_equal(result.trades, expected.trades, check_exact=True)
    assert result.trades.exit_i.tolist() == [2]


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize(
    "entry_clock,exit_clock", [(False, False), (True, True), (True, False), (False, True)]
)
def test_replay_original_hypothesis_corpus(monkeypatch, slots, entry_clock, exit_clock):
    from test_vector_engine import test_kernel_equals_reference

    monkeypatch.setattr(K, "run_columns", compare)
    test_kernel_equals_reference(next_open=entry_clock, exit_next_open=exit_clock, slots=slots)


def test_replay_original_dense_corpora(monkeypatch):
    from test_vector_engine import (
        test_kernel_equals_reference_dense_both_stale_modes,
        test_kernel_equals_reference_dense_limit_entries,
    )
    from test_vector_engine_slots import test_slots_kernel_equals_independent_reference_dense
    from test_vector_engine_venue import test_venue_kernel_equals_independent_reference_dense

    monkeypatch.setattr(K, "run_columns", compare)
    for entry_clock, exit_clock in ((False, False), (True, True), (True, False), (False, True)):
        test_kernel_equals_reference_dense_both_stale_modes(entry_clock, exit_clock)
    test_kernel_equals_reference_dense_limit_entries()
    for latency in (0, 1):
        for one_tick in (False, True):
            if latency or one_tick:
                test_venue_kernel_equals_independent_reference_dense(latency, one_tick)
            for oto in (False, True):
                test_slots_kernel_equals_independent_reference_dense(
                    latency, one_tick, oto, False, False
                )
    for oto in (False, True):
        for entry_clock, exit_clock in ((True, True), (True, False), (False, True)):
            test_slots_kernel_equals_independent_reference_dense(
                0, False, oto, entry_clock, exit_clock
            )


@jitclass(
    [("signals", nb.float64[:, ::1]), ("observed", nb.float64[:, ::1]), ("clock", nb.int64[::1])]
)
class CaptureCore:
    def __init__(self, signals, observed, clock):
        self.signals, self.observed, self.clock = signals, observed, clock

    def step(self, bar, pos):
        self.clock[bar.i] = bar.ts
        self.observed[bar.i, 0] = pos.side
        self.observed[bar.i, 1] = pos.qty
        self.observed[bar.i, 2] = pos.entry_i
        self.observed[bar.i, 3] = pos.entry_px
        self.observed[bar.i, 4] = pos.stop
        self.observed[bar.i, 5] = pos.target
        self.observed[bar.i, 6] = pos.working_side
        self.observed[bar.i, 7] = pos.working_level
        self.observed[bar.i, 8] = pos.last_exit_i
        self.observed[bar.i, 9] = pos.last_exit_px
        self.observed[bar.i, 10] = pos.last_exit_reason
        self.observed[bar.i, 11] = pos.equity
        r = self.signals[bar.i]
        return NumericDecision(
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            r[9],
            r[10],
            r[11],
            r[12],
            r[13],
            r[14],
            np.nan,  # IOC is absent in the saved pre-IOC corpus.
        )


def test_view_is_after_resting_fills_and_reports_actual_amended_stop():
    from qik.vector_engine import Costs, Instrument, Sizing

    o = np.array([100.0, 102.0, 104.0, 99.0, 100.0])
    c = o.copy()
    bars = Bars(
        o, o + 1, o - 1, c, index=pd.date_range("2023-01-03", periods=5, freq="1ns", tz="UTC")
    )
    signals = np.full((5, 15), np.nan)
    signals[:, 0] = 0
    signals[:, 1] = 0
    signals[:, 12:] = 0
    signals[0, 0] = 1
    signals[0, 2] = 95
    signals[0, 3] = 110
    signals[1, 8] = 101
    signals[2, 8] = 103
    observed = np.full((5, 12), np.nan)
    clock = np.zeros(5, np.int64)
    result = run(
        CaptureCore,
        bars,
        features={"signals": signals, "observed": observed, "clock": clock},
        rules=Rules(
            stop_update_mode="ratchet",
            amend_rejected="keep",
            reenter_on_exit_bar=False,
            target_order="mit",
        ),
        instrument=Instrument(0.25, 2.0),
        costs=Costs(commission_per_side=1.0),
        sizing=Sizing(contracts=1),
        initial_cash=1000.0,
        signal_fields=("entry_side", "exit", "stop", "target", "stop_update"),
    )[0]
    assert len(result.trades) == 1
    np.testing.assert_array_equal(observed[:, 0], [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(observed[1:3, 4], [95, 101])
    np.testing.assert_array_equal(observed[3, 8:12], [3, 99, 4, 996])
    np.testing.assert_array_equal(clock, bars.index.asi8)


@pytest.mark.parametrize("slots", [False, True])
def test_view_has_no_working_protection_until_callback_order_settles(slots):
    from qik.vector_engine import Costs, Instrument, Sizing

    c = np.full(7, 100.0)
    bars = Bars(c, c + 1, c - 1, c)
    signals = np.full((7, 15), np.nan)
    signals[:, 0] = 0
    signals[:, 1] = 0
    signals[:, 12:] = 0
    signals[0, 0] = 1
    signals[0, 2] = 95
    signals[0, 3] = 110
    signals[4, 1] = 1
    observed = np.full((7, 12), np.nan)
    clock = np.zeros(7, np.int64)
    fields = ("entry_side", "exit", "stop", "target") + (("stop_update",) if slots else ())
    result = run(
        CaptureCore,
        bars,
        features={"signals": signals, "observed": observed, "clock": clock},
        rules=Rules(latency_bars=1),
        instrument=Instrument(0.25, 2.0),
        costs=Costs(),
        sizing=Sizing(contracts=1),
        initial_cash=1000.0,
        signal_fields=fields,
    )[0]
    assert len(result.trades) == 1
    np.testing.assert_array_equal(observed[:, 0], [0, 0, 1, 1, 1, 1, 0])
    assert np.isnan(observed[2, 4:6]).all()
    np.testing.assert_array_equal(observed[3:6, 4:6], [[95, 110]] * 3)


@pytest.mark.parametrize("target", [110.0, 99.5])
@pytest.mark.parametrize("slots", [False, True])
def test_resting_limit_view_precedes_callback_protection(slots, target):
    c = np.full(4, 100.0)
    bars = Bars(
        c,
        c + 1,
        np.array([100.0, 98.0, 100.0, 100.0]),
        c,
        index=pd.date_range("2023-01-03 15:00", periods=4, freq="min", tz="UTC"),
    )
    signals = np.full((4, 15), np.nan)
    signals[:, (0, 1, 12, 13, 14)] = 0
    signals[0, [0, 2, 3, 4]] = [1, 95, target, 99]
    signals[2, 1] = 1
    observed, clock = np.full((4, 12), np.nan), np.zeros(4, np.int64)
    fields = ("entry_side", "exit", "stop", "target", "entry_limit")
    result = run(
        CaptureCore,
        bars,
        features={"signals": signals, "observed": observed, "clock": clock},
        signal_fields=fields + (("stop_update",) if slots else ()),
    )[0]
    assert observed[1, 0] == 1
    assert observed[1, 3] == 99
    assert np.isnan(observed[1, 4:6]).all()
    assert observed[1, 8] == -1
    # The callback may fill a newly accepted target at the same close. The decision
    # must precede that settlement, but the batch trade remains field-exact.
    expected = simulate(
        bars,
        Signals(
            signals[:, 0].astype(np.int8),
            signals[:, 1].astype(bool),
            stop=signals[:, 2],
            target=signals[:, 3],
            entry_limit=signals[:, 4],
        ),
    )
    pd.testing.assert_frame_equal(result.trades, expected.trades, check_exact=True)


def test_declared_inactive_calendar_fields_allow_naive_last_close():
    from qik.vector_engine import Flatten, Profile

    c = np.full(4, 100.0)
    bars = Bars(c, c + 1, c - 1, c, index=pd.date_range("2023-01-03", periods=4, freq="min"))
    signals = np.full((4, 15), np.nan)
    signals[:, (0, 1, 12, 13, 14)] = 0
    signals[0, 0], signals[2, 12] = 1, 1
    rules = Rules(profile=Profile.REPRO, flatten=Flatten.LAST_BAR_CLOSE)
    actual = run(ReplayCore, bars, features={"signals": signals}, rules=rules)[0]
    expected = simulate(
        bars,
        Signals(
            signals[:, 0].astype(np.int8),
            signals[:, 1].astype(bool),
            flatten=signals[:, 12].astype(bool),
        ),
        rules=rules,
    )
    pd.testing.assert_frame_equal(actual.trades, expected.trades, check_exact=True)


def test_pending_limit_refuses_explicit_session_boundary_without_index():
    from qik.vector_engine import Profile

    c = np.full(3, 100.0)
    bars = Bars(c, c + 1, np.array([100.0, 98.0, 100.0]), c, session_labels=np.array([0, 1, 1]))
    signals = np.full((3, 15), np.nan)
    signals[:, (0, 1, 12, 13, 14)] = 0
    signals[0, [0, 2, 4]], signals[2, 1] = [1, 95, 99], 1
    rules = Rules(profile=Profile.REPRO)
    with pytest.raises(ValueError, match="pending|New York|DAY"):
        run(ReplayCore, bars, features={"signals": signals}, rules=rules)
    with pytest.raises(ValueError, match="pending|New York|DAY"):
        simulate(
            bars,
            Signals(
                signals[:, 0].astype(np.int8),
                signals[:, 1].astype(bool),
                stop=signals[:, 2],
                entry_limit=signals[:, 4],
            ),
            rules=rules,
        )


def test_sequential_cash_is_opt_in_and_default_is_batch_exact():
    from itertools import accumulate

    from qik.vector_engine import Costs, Instrument, Sizing

    c = np.tile([100.0, 100.25], 10)
    bars = Bars(c, c + 0.25, c - 0.25, c)
    signals = np.full((len(c), 15), np.nan)
    signals[:, (0, 1, 12, 13, 14)] = 0
    signals[::2, 0], signals[1::2, 1] = 1, 1
    kw = dict(
        instrument=Instrument(0.25, 2.0),
        costs=Costs(commission_per_side=0.1),
        sizing=Sizing(contracts=1),
        initial_cash=100000.0,
    )
    fields = ("entry_side", "exit")
    default = run(ReplayCore, bars, features={"signals": signals}, signal_fields=fields, **kw)[0]
    sequential = run(
        ReplayCore,
        bars,
        features={"signals": signals},
        signal_fields=fields,
        cash_accumulation="sequential",
        **kw,
    )[0]
    batch = simulate(bars, Signals(signals[:, 0].astype(np.int8), signals[:, 1].astype(bool)), **kw)
    pd.testing.assert_frame_equal(default.trades, batch.trades, check_exact=True)
    pd.testing.assert_frame_equal(default.open_positions, batch.open_positions, check_exact=True)
    expected = batch.trades.copy()
    expected["equity_usd"] = np.fromiter(
        accumulate(expected.pnl_usd, initial=kw["initial_cash"]), float
    )[1:]
    pd.testing.assert_frame_equal(sequential.trades, expected, check_exact=True)
    assert not np.array_equal(default.trades.equity_usd, sequential.trades.equity_usd)


@pytest.mark.parametrize("amend_close", [False, True])
@pytest.mark.parametrize("completed_close", [False, True])
@pytest.mark.parametrize("day_cancel", [False, True])
def test_replay_combined_lifecycle_rules(amend_close, completed_close, day_cancel):
    o = np.array([100.0, 106.0, 102.0, 106.0])
    h = np.array([107.0, 108.0, 108.0, 109.0])
    lo = np.array([99.0, 99.0, 101.0, 105.0])
    c = np.array([106.0, 107.0, 106.0, 108.0])
    nan = np.full((1, 4), np.nan)
    compare(
        o,
        h,
        lo,
        c,
        np.array([[1, 0, 0, 0]], np.int8),
        np.zeros((1, 4), bool),
        nan,
        nan,
        nan,
        nan,
        nan,
        np.zeros(4, bool),
        1,
        False,
        True,
        False,
        False,
        next_open=True,
        entry_trigger=nan,
        stop_updates=np.array([[np.nan, 103.0, np.nan, np.nan]]),
        target_updates=nan,
        be_after=nan,
        be_offset=nan,
        amend_close=amend_close,
        completed_close=completed_close,
        arm_stop=True,
        day_cancel=day_cancel,
        expiry_days=np.array([False, False, True, False]),
    )
