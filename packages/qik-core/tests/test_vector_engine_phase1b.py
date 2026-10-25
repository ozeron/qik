"""R7-R12 admission, boundary and quantity/cost arithmetic contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import replay, script

from qik.vector_engine import (
    Bars,
    Costs,
    Decision,
    FillTiming,
    Flatten,
    Instrument,
    Profile,
    Rules,
    SessionBoundary,
    Sizing,
    TargetGap,
    TickRound,
    require_nautilus,
)


def tape(n: int = 4, *, labels=None, volume=None) -> Bars:
    c = np.arange(n, dtype=float) + 100
    return Bars(
        c,
        c + 0.25,
        c - 0.25,
        c,
        index=pd.date_range("2023-01-03 15:58", periods=n, freq="min", tz="America/New_York"),
        session_labels=labels,
        volume=volume,
    )


@pytest.mark.parametrize(
    "field,enum",
    [
        ("target_gap", TargetGap),
        ("tick_round", TickRound),
        ("flatten", Flatten),
        ("session_boundary", SessionBoundary),
    ],
)
def test_repro_admission_all_values(field, enum) -> None:
    with pytest.raises(ValueError, match="enum"):
        Rules(**{field: list(enum)[0].value})
    for value in list(enum)[1:]:
        with pytest.raises(ValueError, match="repro"):
            Rules(**{field: value})
        for profile in (Profile.REPRO, Profile.TRADINGVIEW):
            result = replay(
                tape(),
                script([0] * 4, [False] * 4),
                rules=Rules(profile=profile, **{field: value}),
                instrument=Instrument(0.25, 1),
                costs=Costs(),
            )
            assert not result.deployable
            with pytest.raises(ValueError):
                require_nautilus(result)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("slots", [False, True])
def test_session_labels_boundary_exit_and_terminal_entry(side: int, slots: bool) -> None:
    b = tape(labels=np.array(["day1", "day1", "day2", "day2"]))
    mask = np.array([False, True, False, False])
    sig = script(
        [side, 0, 0, 0],
        [False] * 4,
        exit_long=mask if side > 0 else None,
        exit_short=mask if side < 0 else None,
        stop_update=np.full(4, np.nan) if slots else None,
    )
    rules = Rules(profile=Profile.REPRO, exit_fill=FillTiming.NEXT_OPEN)
    with pytest.raises(ValueError, match="session end"):
        replay(b, sig, rules=rules)
    fallback = replay(b, sig, rules=replace(rules, session_boundary=SessionBoundary.LAST_BAR_CLOSE))
    assert fallback.trades.exit_i.tolist() == [1]
    assert fallback.trades.exit_px.tolist() == [101.0]
    across = replay(b, sig, rules=replace(rules, session_boundary=SessionBoundary.NEXT_OBSERVED))
    assert across.trades.exit_i.tolist() == [2]
    assert across.trades.exit_px.tolist() == [102.0]
    # Entries at a terminal decision are refused, never silently dropped.
    terminal = script([0, 0, 0, side], [False] * 4, stop_update=sig.get("stop_update"))
    with pytest.raises(ValueError, match="session end"):
        replay(
            b,
            terminal,
            rules=replace(
                rules,
                entry_fill=FillTiming.NEXT_OPEN,
                session_boundary=SessionBoundary.LAST_BAR_CLOSE,
            ),
        )
    ended = replay(
        b,
        terminal,
        rules=replace(rules, entry_fill=FillTiming.NEXT_OPEN, flatten=Flatten.SESSION_END),
    )
    assert ended.open_positions.empty and ended.trades.empty


@pytest.mark.parametrize("mode", list(Flatten))
def test_flatten_intent_and_boundary(mode: Flatten) -> None:
    b = tape()
    sig = script([1, 0, 0, 0], [False] * 4, flatten=[False, False, False, True])
    rules = Rules(profile=Profile.REPRO, flatten=mode)
    if mode is Flatten.NEXT_OPEN:
        with pytest.raises(ValueError, match="session end"):
            replay(b, sig, rules=rules)
        rules = replace(rules, session_boundary=SessionBoundary.LAST_BAR_CLOSE)
    r = replay(b, sig, rules=rules)
    assert r.trades.exit_i.tolist() == [3]
    assert r.trades.exit_px.tolist() == [103.0]
    with pytest.raises(ValueError, match="repro"):
        replay(b, sig)


def test_flatten_cancels_resting_entry_and_suppresses_new_entry() -> None:
    b = tape()
    sig = script(
        [1, 0, 0, 1],
        [False] * 4,
        entry_stop=[105, np.nan, np.nan, 106],
        stop=[99, np.nan, np.nan, 99],
    )
    r = replay(b, sig, rules=Rules(profile=Profile.REPRO, flatten=Flatten.SESSION_END))
    assert r.trades.empty and r.open_positions.empty


@pytest.mark.parametrize("side", [1, -1])
def test_proportional_costs_and_fractional_compounding(side: int) -> None:
    b = tape(6, volume=np.full(6, 100.0))
    sig = script(
        [side, 0, side, 0, side, 0],
        [False, True, False, True, False, True],
        stop=np.full(6, 90 if side > 0 else 110),
    )
    instr = Instrument(0.25, 10, lot_step=0.01, min_qty=0.01)
    costs = Costs(commission_per_side=0.2, slippage_ticks=0.5, proportional_bps=7)
    r = replay(
        b,
        sig,
        instrument=instr,
        costs=costs,
        sizing=Sizing(risk_fraction=0.01, lot_step=0.01, min_qty=0.01),
        initial_cash=10_000,
    )
    equity = 10_000.0
    for t in r.trades.itertuples():
        qty = np.floor((0.01 * equity / (abs(t.entry_px - t.stop) * 10)) / 0.01 + 1e-9) * 0.01
        fee = 2 * (0.2 + 0.5 * 0.25 * 10) + (abs(t.entry_px) + abs(t.exit_px)) * 10 * 7 / 10_000
        assert t.contracts == pytest.approx(qty)
        assert t.pnl_usd == pytest.approx(qty * (side * (t.exit_px - t.entry_px) * 10 - fee))
        equity += t.pnl_usd
        assert t.equity_usd == pytest.approx(equity)
    assert r.deployable
    require_nautilus(r)


def test_fractional_grid_minimum_open_and_liquidity() -> None:
    b = tape(volume=np.full(4, 6.0))
    sig = script([1, 0, 0, 0], [False] * 4)
    instr = Instrument(0.25, 1, lot_step=0.1, min_qty=0.1)
    r = replay(
        b,
        sig,
        instrument=instr,
        costs=Costs(),
        sizing=Sizing(contracts=1.5, lot_step=0.1, min_qty=0.1),
    )
    assert r.open_positions.contracts.tolist() == [1.5]
    with pytest.raises(ValueError, match="liquidity"):
        replay(
            b,
            sig,
            instrument=instr,
            costs=Costs(),
            sizing=Sizing(contracts=1.6, lot_step=0.1, min_qty=0.1),
        )
    with pytest.raises(ValueError, match="quantity grid"):
        replay(
            b,
            sig,
            instrument=Instrument(0.25, 1),
            costs=Costs(),
            sizing=Sizing(contracts=0.1, lot_step=0.1, min_qty=0.1),
        )
    with pytest.raises(ValueError, match="0 contracts"):
        replay(
            b,
            sig.replace(stop=[99, 99, 99, 99]),
            instrument=instr,
            costs=Costs(),
            sizing=Sizing(risk_usd=0.01, lot_step=0.1, min_qty=0.1),
        )
    for kw in (
        {"contracts": 0.15, "lot_step": 0.1},
        {"contracts": 0.1, "lot_step": 0},
        {"contracts": 0.1, "min_qty": 0},
        {"risk_usd": float("inf")},
    ):
        with pytest.raises(ValueError):
            Sizing(**kw)


def test_directional_exits_are_optional_per_config() -> None:
    b = tape()
    s = script(
        [1, 0, 0, 0], [False] * 4, exit_short=[True] * 4, exit_long=[False, False, True, False]
    )
    assert replay(b, s).trades.exit_i.tolist() == [2]
    # A config without directional exits holds; the other's exits stay its own.
    assert replay(b, [s, script([1, 0, 0, 0], [False] * 4)]).open_positions.config.tolist() == [1]


def test_tick_round_requires_instrument_and_none_keeps_refusal() -> None:
    b = tape()
    s = script([1, 0, 0, 0], [False, True, False, False], stop=[99.89, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="off the"):
        replay(b, s, instrument=Instrument(0.25, 1), costs=Costs())
    with pytest.raises(ValueError, match="Instrument"):
        replay(b, s, rules=Rules(profile=Profile.REPRO, tick_round=TickRound.UP))


def test_last_close_flatten_needs_no_calendar_and_default_quantity_is_admitted() -> None:
    b = Bars([100, 101], [100, 101], [100, 101], [100, 101])
    sig = script([1, 0], [False, False], flatten=[False, True])
    r = replay(b, sig, rules=Rules(profile=Profile.REPRO, flatten=Flatten.LAST_BAR_CLOSE))
    assert r.trades.exit_px.tolist() == [101.0]
    with pytest.raises(ValueError, match="quantity grid"):
        replay(
            b,
            script([1, 0], [False, True]),
            instrument=Instrument(0.25, 1, lot_step=0.3, min_qty=0.3),
            costs=Costs(),
        )


def test_decision_positional_fields_remain_compatible() -> None:
    d = Decision(1, False, 99, 102)
    assert d.stop == 99 and d.target == 102
    assert not (d.flatten or d.exit_long or d.exit_short)


@pytest.mark.parametrize("kind", ["limit", "stop"])
def test_next_observed_does_not_extend_day_order_lifetime(kind: str) -> None:
    b = tape(labels=np.array(["a", "a", "b", "b"]))
    kwargs = (
        {"entry_limit": [99, np.nan, np.nan, np.nan]}
        if kind == "limit"
        else {"entry_stop": [105, np.nan, np.nan, np.nan]}
    )
    sig = script([1, 0, 0, 0], [False] * 4, stop=[95, np.nan, np.nan, np.nan], **kwargs)
    with pytest.raises(ValueError, match="New York|calendar day"):
        replay(
            b,
            sig,
            rules=Rules(
                profile=Profile.REPRO,
                session_boundary=SessionBoundary.NEXT_OBSERVED,
                limit_chase_bars=5,
            ),
        )


@pytest.mark.parametrize("kind", ["limit", "stop"])
def test_cross_session_flatten_cannot_bypass_day_admission(kind: str) -> None:
    b = tape(labels=np.array(["a", "a", "b", "b"]))
    entry = (
        {"entry_limit": [99, np.nan, np.nan, np.nan]}
        if kind == "limit"
        else {"entry_stop": [105, np.nan, np.nan, np.nan]}
    )
    sig = script(
        [1, 0, 0, 0],
        [False] * 4,
        stop=[95, np.nan, np.nan, np.nan],
        flatten=[False, True, False, False],
        **entry,
    )
    with pytest.raises(ValueError, match="New York|calendar day"):
        replay(
            b,
            sig,
            rules=Rules(
                profile=Profile.REPRO,
                session_boundary=SessionBoundary.NEXT_OBSERVED,
                limit_chase_bars=5,
            ),
        )


def test_custom_sessions_cannot_change_nautilus_day_contract() -> None:
    b = tape(labels=np.array(["a", "a", "a", "a"]))
    with pytest.raises(ValueError, match="repro"):
        replay(b, script([1, 0, 0, 0], [False, False, False, True]))
