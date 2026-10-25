"""A18/A19/A20 API contracts and independent execution differential coverage."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import infinite_stop_update_step, replay, script, stop_update_step

from qik.vector_engine import (
    Bars,
    Compliance,
    Costs,
    FillModel,
    Instrument,
    MarketFill,
    Profile,
    Rules,
    Sizing,
    run,
)
from qik.vector_engine import kernel as K
from qik.vector_engine.reference import simulate_column_reference
from qik.vector_engine.testing import assert_causal


def _bars(n: int = 5) -> Bars:
    c = np.full(n, 100.0)
    return Bars(
        c,
        c + 0.25,
        c - 0.25,
        c,
        index=pd.date_range("2023-01-03 14:31", periods=n, freq="min", tz="UTC"),
        volume=np.full(n, 100.0),
    )


def test_slot_rules_validate_and_next_open_accepts_slots() -> None:
    for field in (
        "protection",
        "target_order",
        "amend_rejected",
        "stop_update_mode",
        "stop_entry_in_market",
    ):
        with pytest.raises(ValueError, match=field):
            Rules(**{field: "unknown"})
    sig = script(np.array([1, 0, 0, 0, 0]), np.zeros(5, bool), entry_stop=np.full(5, 101.0))
    assert replay(
        _bars(), sig, rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN)
    ).trades.empty
    with pytest.raises(ValueError, match="both entry_stop and entry_limit"):
        replay(_bars(), sig.replace(entry_limit=np.full(5, 99.0)))
    with pytest.raises(ValueError, match="timezone-aware"):
        replay(replace(_bars(), index=None), sig)
    with pytest.raises(ValueError, match="positive"):
        replay(_bars(), sig.replace(stop_update=np.full(5, 0.0)))
    with pytest.raises(ValueError, match="finite or NaN"):
        replay(_bars(), sig.replace(target_update=np.full(5, np.inf)))
    with pytest.raises(ValueError, match="off the.*tick grid"):
        replay(
            _bars(),
            sig.replace(stop_update=np.full(5, 99.1)),
            instrument=Instrument(0.25, 1),
            costs=Costs(),
        )


def test_slot_decisions_and_deployability_include_empty_results() -> None:
    every_bar = np.full(5, 1.0)
    sig = script(
        np.array([1, 0, 0, 0, 0]),
        np.zeros(5, bool),
        stop=98.0 * every_bar,
        entry_stop=101.0 * every_bar,
        stop_update=99.0 * every_bar,
        target_update=103.0 * every_bar,
        breakeven_after_points=2.0 * every_bar,
        breakeven_offset_points=0.25 * every_bar,
    )
    res = replay(_bars(), sig, instrument=Instrument(0.25, 1), costs=Costs(), initial_cash=10000)
    assert res.compliance is Compliance.NAUTILUS and res.deployable
    assert res.trades.empty
    assert res.summary().deployable.tolist() == [True]
    assert replay(
        _bars(), script(np.zeros(5), np.zeros(5)), rules=Rules(target_order="mit")
    ).deployable


@pytest.mark.parametrize("latency", [0, 1])
@pytest.mark.parametrize("fill", [FillModel.DEFAULT, FillModel.ONE_TICK])
def test_proven_callback_models_deploy_but_absolute_oto_does_not(latency, fill):
    signals = script(np.zeros(5), np.zeros(5))
    for protection, expected in (("callback", True), ("oto", False)):
        result = replay(
            _bars(),
            signals,
            rules=Rules(
                target_order="mit", protection=protection, latency_bars=latency, fill_model=fill
            ),
            instrument=Instrument(0.25, 1),
            costs=Costs(),
            sizing=Sizing(contracts=1),
        )
        assert result.deployable is expected


def test_unproved_slot_contracts_and_repro_remain_research_only():
    side = np.array([1, 0, 0, 0, 0])
    sig = script(side, np.array([False, False, False, True, False]), stop=np.full(5, 90.0))
    for rules, signals in (
        (Rules(target_order="mit", adaptive_high_low_ordering=True), sig),
        (Rules(target_order="mit", reenter_on_exit_bar=False), sig),
        (Rules(target_order="mit"), sig.replace(stop=None)),
        (Rules(target_order="mit", profile=Profile.TRADINGVIEW), sig),
        (Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN), sig),
    ):
        assert not replay(_bars(), signals, rules=rules).deployable


def test_amendments_preserve_entry_risk_for_sizing() -> None:
    b = _bars()
    su = np.array([np.nan, 99.0, np.nan, np.nan, np.nan])
    sig = script(
        np.array([1, 0, 0, 0, 0]),
        np.array([False, False, True, False, False]),
        stop=np.full(5, 98.0),
        stop_update=su,
    )
    result = replay(
        b, sig, instrument=Instrument(0.25, 1), costs=Costs(), sizing=Sizing(risk_usd=10)
    )
    assert result.trades.contracts.tolist() == [5]
    assert result.trades.stop.tolist() == [98.0]
    assert not result.deployable


def test_slot_refusals_and_stop_entry_skip() -> None:
    side, exit_ = np.array([1, 0, 0, 0, 0]), np.zeros(5, bool)
    sig = script(side, exit_, stop=np.full(5, 98.0), entry_stop=np.full(5, 100.0))
    with pytest.raises(ValueError, match="stop entry.*in the market"):
        replay(_bars(), sig)
    skipped = replay(_bars(), sig, rules=Rules(stop_entry_in_market="skip"))
    assert skipped.trades.empty and skipped.open_positions.empty and skipped.deployable
    with pytest.raises(ValueError, match="new New York date"):
        replay(
            replace(_bars(), index=pd.date_range("2023-01-03", periods=5, freq="D", tz="UTC")),
            sig.replace(entry_stop=np.full(5, 101.0)),
        )
    for field, value, message in (
        ("entry_limit", np.full(5, 99.0), "entry_limit / chase"),
        ("stop_points", np.full(5, 2.0), "absolute protection"),
    ):
        values = {field: value, "entry_stop": None, "stop_update": np.full(5, 99.0)}
        if field == "stop_points":
            values["stop"] = None
        with pytest.raises(ValueError, match=message):
            replay(_bars(), sig.replace(**values), rules=Rules(protection="oto"))
    with pytest.raises(ValueError, match="no working order"):
        replay(_bars(), script(side, exit_, target_update=np.full(5, 103.0)))
    with pytest.raises(ValueError, match="conflicts with breakeven"):
        replay(
            _bars(),
            script(
                side,
                exit_,
                stop=np.full(5, 98.0),
                stop_update=np.full(5, 99.0),
                breakeven_after_points=np.full(5, 0.25),
            ),
        )
    with pytest.raises(ValueError, match="needs an entry stop"):
        replay(_bars(), script(side, exit_, breakeven_after_points=np.full(5, 1.0)))
    with pytest.raises(ValueError, match="needs breakeven_after_points"):
        replay(_bars(), script(side, exit_, breakeven_offset_points=np.full(5, 0.25)))


def test_causality_cuts_on_amendments_without_entry_signals() -> None:
    def peek(b: Bars) -> dict[str, np.ndarray]:
        update = np.full(len(b), np.nan)
        update[1] = float(len(b))
        return {"stop_update": update}

    with pytest.raises(AssertionError, match="not prefix-causal"):
        assert_causal(
            stop_update_step,
            _bars(),
            2,
            features=peek,
            signal_fields=("entry_side", "exit", "stop_update"),
        )


def test_invalid_core_slot_is_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="finite or NaN"):
        run(
            infinite_stop_update_step,
            _bars(),
            features={"stop_update": np.full(5, np.nan)},
            signal_fields=("entry_side", "exit", "stop_update"),
        )


@pytest.mark.parametrize(
    "latency,one_tick,oto,next_open,exit_next_open",
    [(a, b, c, False, False) for a in (0, 1) for b in (False, True) for c in (False, True)]
    + [
        (0, False, c, e, x)
        for c in (False, True)
        for e, x in ((True, True), (True, False), (False, True))
    ],
)
def test_slots_kernel_equals_independent_reference_dense(
    latency: int, one_tick: bool, oto: bool, next_open: bool, exit_next_open: bool
) -> None:
    from test_vector_engine import _nan_free

    rng = np.random.default_rng(18000 + latency + 10 * one_tick + 100 * oto)
    errors: set[int] = set()
    reasons: set[int] = set()
    for case_i in range(1000):
        n = int(rng.integers(6, 65))
        o = 100 + rng.integers(-12, 13, n) * 0.25
        c = o + rng.integers(-4, 5, n) * 0.25
        h = np.maximum(o, c) + rng.integers(0, 13, n) * 0.25
        lo = np.minimum(o, c) - rng.integers(0, 13, n) * 0.25
        side = rng.choice(np.array([-1, 0, 1], np.int8), n, p=[0.15, 0.7, 0.15])
        exit_ = rng.random(n) < 0.15
        st = c - side * rng.integers(1, 13, n) * 0.25
        tg = c + side * rng.integers(1, 13, n) * 0.25
        trig = np.where(rng.random(n) < 0.5, c + side * rng.integers(-1, 9, n) * 0.25, np.nan)
        if case_i % 2:
            trig[:] = np.nan
        limits = (
            np.full(n, np.nan)
            if oto
            else np.where((rng.random(n) < 0.25) & np.isnan(trig), c - side * 0.5, np.nan)
        )
        sp = np.full(n, np.nan) if oto else np.where(rng.random(n) < 0.2, 0.75, np.nan)
        tp = np.full(n, np.nan) if oto else np.where(rng.random(n) < 0.2, 1.25, np.nan)
        su = np.where(rng.random(n) < 0.25, c + rng.integers(-8, 9, n) * 0.25, np.nan)
        tu = np.where(rng.random(n) < 0.25, c + rng.integers(-8, 9, n) * 0.25, np.nan)
        ba = np.where(rng.random(n) < 0.25, 0.5, np.nan)
        bo = np.where(np.isfinite(ba), rng.integers(-1, 2, n) * 0.25, np.nan)
        days = rng.random(n) < 0.04
        adaptive, reenter, keep, ratchet, mit, flatten, stale, skip = (
            bool(rng.integers(2)) for _ in range(8)
        )
        adverse_first, bracket_anchor = bool(rng.integers(2)), int(rng.integers(3))
        chase = int(rng.integers(1, 4))
        rounding = case_i % 3
        if rounding:
            st -= side * 0.09
            tg += side * 0.09
            trig += side * 0.09
            limits -= side * 0.09
            sp += 0.09
            tp += 0.09
            su += 0.09
            tu += 0.09
            bo += 0.09
        xl, xs = rng.random(n) < 0.1, rng.random(n) < 0.1
        fc, fo = (rng.random(n) < 0.03).astype(np.int8), rng.random(n) < 0.03
        ref = simulate_column_reference(
            o,
            h,
            lo,
            c,
            side,
            exit_,
            st,
            tg,
            entry_limit=limits,
            stop_pts=sp,
            target_pts=tp,
            new_day=days,
            adaptive=adaptive,
            reenter=reenter,
            stale_drop=stale,
            chase_after=chase,
            protect_flatten=flatten,
            adverse_first=adverse_first,
            bracket_anchor=bracket_anchor,
            target_gap=bool(case_i % 2),
            tick_round=rounding,
            exit_long=xl,
            exit_short=xs,
            force_close=fc,
            force_open=fo,
            next_open=next_open,
            exit_next_open=exit_next_open,
            latency_bars=latency,
            one_tick=one_tick,
            tick=0.25,
            stop_entry=trig,
            stop_update=su,
            target_update=tu,
            be_after=ba,
            be_offset=bo,
            mit=mit,
            oto=oto,
            amend_keep=keep,
            ratchet=ratchet,
            stop_skip=skip,
        )

        def col(a):
            return np.ascontiguousarray(a[None, :])

        out = K.run_columns(
            o,
            h,
            lo,
            c,
            col(side),
            col(exit_),
            col(st),
            col(tg),
            col(limits),
            col(sp),
            col(tp),
            days,
            chase,
            adaptive,
            reenter,
            stale,
            flatten,
            next_open,
            latency,
            one_tick,
            0.25,
            col(trig),
            col(su),
            col(tu),
            col(ba),
            col(bo),
            mit,
            oto,
            keep,
            ratchet,
            skip,
            exit_next_open=exit_next_open,
            adverse_first=adverse_first,
            bracket_anchor=bracket_anchor,
            target_gap=bool(case_i % 2),
            tick_round=rounding,
            exit_long=col(xl),
            exit_short=col(xs),
            force_close=col(fc),
            force_open=col(fo),
        )
        assert (int(out[2][0]), int(out[3][0])) == (ref.err, ref.err_bar), dict(
            o=o,
            c=c,
            h=h,
            lo=lo,
            side=side,
            exits=exit_,
            st=st,
            tg=tg,
            trig=trig,
            limits=limits,
            sp=sp,
            tp=tp,
            su=su,
            tu=tu,
            ba=ba,
            bo=bo,
            flatten=flatten,
            mit=mit,
            oto=oto,
            adaptive=adaptive,
            reenter=reenter,
            keep=keep,
            ratchet=ratchet,
            days=days,
            chase=chase,
        )
        errors.add(ref.err)
        if ref.err:
            continue
        got = [tuple(_nan_free(v) for v in row) for row in zip(*out[8:], strict=True)]
        want = [
            (t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px, t.reason, _nan_free(t.stop))
            for t in ref.trades
        ]
        assert got == want
        assert (int(out[4][0]), int(out[5][0]), _nan_free(out[6][0]), _nan_free(out[7][0])) == (
            ref.final_side,
            ref.final_entry_i,
            _nan_free(ref.final_entry_px),
            _nan_free(ref.final_stop),
        )
        reasons.update(t.reason for t in ref.trades)
    if not (next_open or exit_next_open):
        assert {K.ERR_NONE, K.ERR_STOP_ENTRY_IN_MARKET, K.ERR_AMEND_REJECTED} <= errors
    else:
        assert K.ERR_NEXT_OPEN_SESSION_END in errors
    assert reasons
