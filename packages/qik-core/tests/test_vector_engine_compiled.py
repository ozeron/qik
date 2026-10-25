"""Compiled close decisions replay the same hypothesis/dense rule axes as batch matching."""

from __future__ import annotations

import inspect

import numba as nb
import numpy as np
import pandas as pd
import pytest
from _typed_scripts import Script, replay, script
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_vector_engine import _case, _dense_case

from qik.vector_engine import (
    Bar,
    Bars,
    BoolArray,
    Decision,
    Features,
    FloatArray,
    IntArray,
    Position,
    Rules,
    run,
    strategy,
)
from qik.vector_engine import kernel as K
from qik.vector_engine.compiled import RULES_WIDTH, _bound
from qik.vector_engine.testing import (
    ReplayParams,
    ReplayState,
    replay_decisions,
    replay_features,
    replay_step,
)

_BATCH = K.run_columns
# Stateless typed records: numba refuses zero-width records (same byte as run).
_EMPTY = np.zeros(1, dtype=[("_empty", "i1")]).view(np.recarray)


def rows(matrix: np.ndarray) -> list[tuple[float, ...]]:
    """The saved pre-IOC fixture retains its original fifteen numeric fields."""
    return [tuple(row) for row in matrix]


def replay_column(decisions, o, h, lo, c, days, rc, slots, masks, expiry_days):
    """One recorded column through the strategy-bound runner of typed replay_step."""
    features = replay_features(decisions)
    clock = np.full(len(c), -1, np.int64)
    raw = _bound(replay_step).run(
        _EMPTY,
        _EMPTY,
        (clock, *features),
        o,
        h,
        lo,
        c,
        np.zeros(len(c)),
        days,
        np.zeros(len(c), bool),
        rc,
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
        np.nan,
        slots,
        None,
        masks,
        None,
        expiry_days,
        False,
    )
    return raw[:-1]  # drop the final (empty) state rows


def compare(*args, **kwargs):
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
            # index the runner reads (combine, amend_close, completed_close,
            # arm_stop, day_cancel, next_bar, exit_and_rearm).
            d["combine"],
            d["amend_close"],
            d["completed_close"],
            d["arm_stop"],
            d["day_cancel"],
            d["next_bar"],
            d["exit_and_rearm"],
        ],
        float,
    )
    assert len(rc) == RULES_WIDTH
    masks = np.zeros((k, 2, n), np.int8)
    for q, name in enumerate(("force_close", "force_open")):
        if d[name] is not None:
            masks[:, q] = d[name]
    for j in range(k):
        counts, meta, *arrays = replay_column(
            rows(feats[j]),
            d["o"],
            d["h"],
            d["lo"],
            d["c"],
            d["new_day"],
            rc,
            d["entry_trigger"] is not None,
            np.ascontiguousarray(masks[j : j + 1]),
            d["expiry_days"],
        )
        np.testing.assert_array_equal(meta[0, 0], batch[2][j])
        np.testing.assert_array_equal(meta[0, 1], batch[3][j])
        if batch[2][j]:
            continue
        assert counts[0] == batch[0][j]
        for q in range(4):
            np.testing.assert_array_equal(meta[0, q + 2], batch[q + 4][j])
        start, stop = batch[1][j : j + 2]
        for q in range(7):
            np.testing.assert_array_equal(arrays[q][0, : counts[0]], batch[q + 8][start:stop])
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


def test_public_replay_runner_is_field_exact():
    c = np.array([100.0, 100.0, 101.0])
    bars = Bars(c, c + 1, c - 1, c)
    feats = np.full((3, 15), np.nan)
    feats[:, 0] = [1, 0, 0]
    feats[:, 1] = [0, 0, 1]
    feats[:, 5] = 5.0
    feats[:, 12:] = 0
    fields = ("entry_side", "exit", "stop_points")
    python, compiled = (
        replay_decisions(rows(feats), bars, signal_fields=fields, jit=jit) for jit in (False, True)
    )
    pd.testing.assert_frame_equal(python.trades, compiled.trades, check_exact=True)
    trades = compiled.trades[["entry_i", "exit_i", "side", "entry_px", "exit_px", "stop", "reason"]]
    assert trades.values.tolist() == [[0, 2, 1, 100.0, 101.0, 95.0, "signal"]]


def test_public_last_close_flatten_needs_no_calendar():
    from qik.vector_engine import Flatten, Profile

    c = np.full(4, 100.0)
    bars = Bars(c, c + 1, c - 1, c)
    matrix = np.full((4, 15), np.nan)
    matrix[:, (0, 1, 12, 13, 14)] = 0
    matrix[0, 0] = 1
    matrix[2, 12] = 1
    rules = Rules(profile=Profile.REPRO, flatten=Flatten.LAST_BAR_CLOSE)
    fields = ("entry_side", "exit", "flatten")
    python, compiled = (
        replay_decisions(rows(matrix), bars, signal_fields=fields, rules=rules, jit=jit)
        for jit in (False, True)
    )
    pd.testing.assert_frame_equal(python.trades, compiled.trades, check_exact=True)
    assert compiled.trades.exit_i.tolist() == [2]


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


class CaptureFeatures(Features):
    """Scripted inputs plus writable per-bar observation outputs (test capture only)."""

    side: IntArray
    exit: BoolArray
    stop: FloatArray
    target: FloatArray
    entry_limit: FloatArray
    stop_update: FloatArray
    seen_side: FloatArray
    seen_entry_px: FloatArray
    seen_stop: FloatArray
    seen_target: FloatArray
    seen_last_exit_i: FloatArray
    seen_last_exit_px: FloatArray
    seen_last_exit_reason: FloatArray
    seen_equity: FloatArray
    seen_ts: IntArray


@strategy(state=ReplayState, params=ReplayParams, features=CaptureFeatures)
def capture_step(
    s: ReplayState, p: ReplayParams, f: CaptureFeatures, bar: Bar, pos: Position
) -> Decision:
    f.seen_ts[bar.i] = bar.ts
    f.seen_side[bar.i] = pos.side
    f.seen_entry_px[bar.i] = pos.entry_px
    f.seen_stop[bar.i] = pos.stop
    f.seen_target[bar.i] = pos.target
    f.seen_last_exit_i[bar.i] = pos.last_exit_i
    f.seen_last_exit_px[bar.i] = pos.last_exit_px
    f.seen_last_exit_reason[bar.i] = pos.last_exit_reason
    f.seen_equity[bar.i] = pos.equity
    i = bar.i
    return Decision(
        side=f.side[i],
        exit=f.exit[i],
        stop=f.stop[i],
        target=f.target[i],
        entry_limit=f.entry_limit[i],
        stop_update=f.stop_update[i],
    )


def capture(scripted: Script) -> CaptureFeatures:
    """Fresh NaN observation arrays (zero clock) beside the scripted inputs."""
    n = len(scripted.entry_side)

    def level(name: str) -> np.ndarray:
        value = scripted.get(name)
        return np.full(n, np.nan) if value is None else value[:, 0]

    seen = {name: np.full(n, np.nan) for name in CaptureFeatures._fields if name[:5] == "seen_"}
    return CaptureFeatures(
        side=scripted.entry_side[:, 0].copy(),
        exit=scripted.exit[:, 0].copy(),
        **{name: level(name) for name in ("stop", "target", "entry_limit", "stop_update")},
        **dict(seen, seen_ts=np.zeros(n, np.int64)),
    )


def first(n: int, *values: float) -> np.ndarray:
    """NaN column with the given leading values (decision bars 0, 1, ...)."""
    return np.r_[values, np.full(n - len(values), np.nan)]


def test_view_is_after_resting_fills_and_reports_actual_amended_stop():
    from qik.vector_engine import Costs, Instrument, Sizing

    o = np.array([100.0, 102.0, 104.0, 99.0, 100.0])
    c = o.copy()
    bars = Bars(
        o, o + 1, o - 1, c, index=pd.date_range("2023-01-03", periods=5, freq="1ns", tz="UTC")
    )
    scripted = script(
        np.array([1, 0, 0, 0, 0]),
        np.zeros(5, bool),
        stop=first(5, 95),
        target=first(5, 110),
        stop_update=first(5, np.nan, 101, 103),
    )
    f = capture(scripted)
    result = run(
        capture_step,
        bars,
        features=f,
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
    )
    assert len(result.trades) == 1
    np.testing.assert_array_equal(f.seen_side, [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(f.seen_stop[1:3], [95, 101])
    last_exit = (f.seen_last_exit_i, f.seen_last_exit_px, f.seen_last_exit_reason, f.seen_equity)
    np.testing.assert_array_equal([x[3] for x in last_exit], [3, 99, 4, 996])
    np.testing.assert_array_equal(f.seen_ts, bars.index.asi8)


@pytest.mark.parametrize("slots", [False, True])
def test_view_has_no_working_protection_until_callback_order_settles(slots):
    from qik.vector_engine import Costs, Instrument, Sizing

    c = np.full(7, 100.0)
    bars = Bars(c, c + 1, c - 1, c)
    exits = np.zeros(7, bool)
    exits[4] = True
    scripted = script(
        np.array([1, 0, 0, 0, 0, 0, 0]), exits, stop=first(7, 95), target=first(7, 110)
    )
    f = capture(scripted)
    fields = ("entry_side", "exit", "stop", "target") + (("stop_update",) if slots else ())
    result = run(
        capture_step,
        bars,
        features=f,
        rules=Rules(latency_bars=1),
        instrument=Instrument(0.25, 2.0),
        costs=Costs(),
        sizing=Sizing(contracts=1),
        initial_cash=1000.0,
        signal_fields=fields,
    )
    assert len(result.trades) == 1
    np.testing.assert_array_equal(f.seen_side, [0, 0, 1, 1, 1, 1, 0])
    assert np.isnan([f.seen_stop[2], f.seen_target[2]]).all()
    np.testing.assert_array_equal(np.c_[f.seen_stop, f.seen_target][3:6], [[95, 110]] * 3)


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
    scripted = script(
        np.array([1, 0, 0, 0]),
        np.array([False, False, True, False]),
        stop=first(4, 95),
        target=first(4, target),
        entry_limit=first(4, 99),
    )
    f = capture(scripted)
    fields = ("entry_side", "exit", "stop", "target", "entry_limit")
    result = run(
        capture_step,
        bars,
        features=f,
        signal_fields=fields + (("stop_update",) if slots else ()),
    )
    assert f.seen_side[1] == 1
    assert f.seen_entry_px[1] == 99
    assert np.isnan([f.seen_stop[1], f.seen_target[1]]).all()
    assert f.seen_last_exit_i[1] == -1
    # The callback may fill a newly accepted target at the same close. The decision
    # must precede that settlement, but the scripted replay trade remains field-exact.
    expected = replay(bars, scripted)
    pd.testing.assert_frame_equal(result.trades, expected.trades, check_exact=True)


def test_declared_inactive_calendar_fields_allow_naive_last_close():
    from qik.vector_engine import Flatten, Profile

    c = np.full(4, 100.0)
    bars = Bars(c, c + 1, c - 1, c, index=pd.date_range("2023-01-03", periods=4, freq="min"))
    signals = np.full((4, 15), np.nan)
    signals[:, (0, 1, 12, 13, 14)] = 0
    signals[0, 0], signals[2, 12] = 1, 1
    rules = Rules(profile=Profile.REPRO, flatten=Flatten.LAST_BAR_CLOSE)
    # Every field declared (the replay default), all but three inactive.
    actual = replay_decisions(rows(signals), bars, rules=rules)
    expected = replay(
        bars,
        script(
            signals[:, 0].astype(np.int8),
            signals[:, 1].astype(bool),
            flatten=signals[:, 12].astype(bool),
        ),
        rules=rules,
    )
    pd.testing.assert_frame_equal(actual.trades, expected.trades, check_exact=True)
    assert actual.trades.exit_i.tolist() == [2]


def test_pending_limit_refuses_explicit_session_boundary_without_index():
    from qik.vector_engine import Profile

    c = np.full(3, 100.0)
    bars = Bars(c, c + 1, np.array([100.0, 98.0, 100.0]), c, session_labels=np.array([0, 1, 1]))
    signals = np.full((3, 15), np.nan)
    signals[:, (0, 1, 12, 13, 14)] = 0
    signals[0, [0, 2, 4]], signals[2, 1] = [1, 95, 99], 1
    rules = Rules(profile=Profile.REPRO)
    with pytest.raises(ValueError, match="pending|New York|DAY"):
        replay_decisions(rows(signals), bars, rules=rules)
    with pytest.raises(ValueError, match="pending|New York|DAY"):
        replay(
            bars,
            script(
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
        signal_fields=("entry_side", "exit"),
    )
    default = replay_decisions(rows(signals), bars, **kw)
    sequential = replay_decisions(rows(signals), bars, cash_accumulation="sequential", **kw)
    # Pinned from the legacy array-path engine at main 7a9b562f2 on these inputs:
    # ten long round trips 100.0 -> 100.25 at $2/point less 2 x $0.1, batch equity
    # initial_cash + k * 0.3 (each value the float nearest that decimal).
    trades = default.trades
    assert trades.entry_i.tolist() == list(range(0, 20, 2))
    assert trades.exit_i.tolist() == list(range(1, 20, 2))
    assert trades.side.tolist() == [1] * 10
    assert trades.reason.tolist() == ["signal"] * 10
    assert trades.entry_px.tolist() == [100.0] * 10
    assert trades.exit_px.tolist() == [100.25] * 10
    assert trades.contracts.tolist() == [1] * 10
    assert trades.gross_usd.tolist() == [0.5] * 10
    assert trades.pnl_usd.tolist() == [0.3] * 10
    assert trades.equity_usd.tolist() == [
        100000.3, 100000.6, 100000.9, 100001.2, 100001.5,
        100001.8, 100002.1, 100002.4, 100002.7, 100003.0,
    ]  # fmt: skip
    assert default.open_positions.empty
    # Sequential accumulates the pinned per-trade PnL in trade order.
    expected = trades.copy()
    expected["equity_usd"] = list(accumulate([0.3] * 10, initial=kw["initial_cash"]))[1:]
    pd.testing.assert_frame_equal(sequential.trades, expected, check_exact=True)
    pd.testing.assert_frame_equal(sequential.open_positions, default.open_positions)
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
