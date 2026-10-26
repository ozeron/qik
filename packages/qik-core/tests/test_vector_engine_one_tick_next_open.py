"""R15: ONE_TICK with next-open entries (P025, P024): hand cases, refusals, kernel == reference."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import Script, replay, script
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from test_vector_engine import _case, _dense_case, _kernel_final, _nan_free, _ref_final, _ref_trades

from qik.vector_engine import (
    HOLD,
    Bar,
    Bars,
    BracketAnchor,
    Costs,
    Features,
    FillModel,
    FillTiming,
    Flatten,
    Instrument,
    Params,
    Position,
    Profile,
    Rules,
    SessionBoundary,
    Sizing,
    State,
    TickRound,
    TypedDecision,
    enter,
    enter_limit_ioc,
    run,
    strategy,
)
from qik.vector_engine import kernel as K
from qik.vector_engine.reference import simulate_column_reference
from scripts.engine_test_tiers import engine_examples

TICK = 0.25
NAN = math.nan
INSTRUMENT = Instrument(tick=TICK, point_value=2.0)
RULES = Rules(
    profile=Profile.REPRO,
    author_platform="mt5",
    entry_fill=FillTiming.NEXT_OPEN,
    exit_fill=FillTiming.NEXT_OPEN,
    bracket_anchor=BracketAnchor.FILL,
    fill_model=FillModel.ONE_TICK,
)

# Decision close 100; entry open 101 fills at 101.25; bar 2 touches 99 (bar 3 gaps to 98.5).
ROWS = [
    (100.0, 100.5, 99.5, 100.0),
    (101.0, 102.0, 100.5, 101.5),
    (101.0, 101.5, 99.0, 99.5),
    (98.5, 99.0, 98.0, 98.75),
]


def tape(side: int = 1, rows: list[tuple[float, float, float, float]] = ROWS) -> Bars:
    o, h, lo, c = (np.array([r[k] for r in rows]) for k in range(4))
    if side < 0:
        o, h, lo, c = 200 - o, 200 - lo, 200 - h, 200 - c
    index = pd.date_range("2023-01-03 15:00", periods=len(rows), freq="min", tz="UTC")
    return Bars(o, h, lo, c, index=index)


def entry(side: int = 1, n: int = 4, exit: np.ndarray | None = None, **levels: float) -> Script:
    def first(value: float) -> np.ndarray:
        return np.r_[value, np.full(n - 1, NAN)]

    return script(
        entry_side=np.r_[side, np.zeros(n - 1)].astype(np.int8),
        exit=np.zeros(n, bool) if exit is None else exit,
        **{name: first(value) for name, value in levels.items()},
    )


def one_tick(sig: Script, rules: Rules = RULES, side: int = 1) -> pd.DataFrame:
    return replay(tape(side), sig, rules=rules, instrument=INSTRUMENT, costs=Costs()).trades


def trade(rules: Rules, side: int, sig: Script) -> pd.Series:
    result = replay(tape(side), sig, rules=rules, instrument=INSTRUMENT, costs=Costs())
    assert len(result.trades) == 1
    return result.trades.iloc[0]


def mirror(side: int, price: float) -> float:
    return price if side > 0 else 200 - price


@pytest.mark.parametrize("side", [1, -1])
def test_entry_slips_and_protection_anchors_on_the_slipped_fill(side: int) -> None:
    t = trade(RULES, side, entry(side, stop_points=2.0))
    assert (t.entry_i, t.exit_i, t.reason) == (1, 2, "stop")
    assert t.entry_px == mirror(side, 101.25)
    assert t.stop == mirror(side, 99.25)  # slipped fill minus the distance (legacy P025/P024)
    assert t.exit_px == mirror(side, 99.0)  # the stop price, one tick worse


@pytest.mark.parametrize("side", [1, -1])
def test_gap_stop_fills_one_tick_through_the_open(side: int) -> None:
    rows = [*ROWS[:3], (98.0, 98.5, 97.5, 98.0)]
    t = replay(
        tape(side, rows), entry(side, stop_points=3.0),
        rules=RULES, instrument=INSTRUMENT, costs=Costs(),
    ).trades.iloc[0]  # fmt: skip
    assert (t.exit_i, t.reason, t.stop) == (3, "stop_gap", mirror(side, 98.25))
    assert t.exit_px == mirror(side, 97.75)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize(
    "anchor,expected", [(None, 99.25), (BracketAnchor.FILL, 99.25), (BracketAnchor.SIGNAL, 98.0)]
)
def test_anchor_modes(side: int, anchor: BracketAnchor | None, expected: float) -> None:
    t = trade(replace(RULES, bracket_anchor=anchor), side, entry(side, stop_points=2.0))
    assert t.stop == mirror(side, expected)


@pytest.mark.parametrize("side", [1, -1])
def test_tick_round_and_combined_stop_use_the_slipped_fill(side: int) -> None:
    rounded = trade(
        replace(RULES, tick_round=TickRound.NEAREST), side, entry(side, stop_points=1.9)
    )
    assert rounded.stop == mirror(side, 99.25)  # 101.25 - 1.9 = 99.35 -> 99.25
    tighter = replace(RULES, stop_combine="tighter")
    t = trade(tighter, side, entry(side, stop=mirror(side, 99.5), stop_points=2.0))
    assert t.stop == mirror(side, 99.5)  # max(level 99.5, slipped fill 101.25 - 2)
    t = trade(tighter, side, entry(side, stop=mirror(side, 99.0), stop_points=2.0))
    assert t.stop == mirror(side, 99.25)


@pytest.mark.parametrize("side", [1, -1])
def test_market_exits_slip_on_both_clocks(side: int) -> None:
    sig = entry(side, exit=np.array([False, True, False, False]))
    t = trade(RULES, side, sig)
    assert (t.exit_i, t.exit_px) == (2, mirror(side, 100.75))  # bar 2 open 101 - tick
    t = trade(replace(RULES, exit_fill=FillTiming.CLOSE), side, sig)
    assert (t.exit_i, t.exit_px) == (1, mirror(side, 101.25))  # bar 1 close 101.5 - tick


@pytest.mark.parametrize("side", [1, -1])
def test_flatten_and_boundary_exits_slip(side: int) -> None:
    session = trade(replace(RULES, flatten=Flatten.SESSION_END), side, entry(side))
    assert (session.exit_i, session.exit_px) == (3, mirror(side, 98.5))
    flat = entry(side).replace(flatten=np.array([False, True, False, False]))
    t = trade(RULES, side, flat)
    assert (t.exit_i, t.exit_px) == (2, mirror(side, 100.75))
    t = trade(replace(RULES, flatten=Flatten.LAST_BAR_CLOSE), side, flat)
    assert (t.exit_i, t.exit_px) == (1, mirror(side, 101.25))
    boundary = replace(RULES, session_boundary=SessionBoundary.LAST_BAR_CLOSE)
    last = entry(side, exit=np.array([False, False, False, True]))
    t = trade(boundary, side, last)
    assert (t.exit_i, t.exit_px) == (3, mirror(side, 98.5))


@pytest.mark.parametrize("side", [1, -1])
def test_risk_sizing_clamp_and_costs_measure_the_slipped_fill(side: int) -> None:
    sizing = Sizing(risk_usd=1.0, max_contracts=5, min_qty_policy="clamp")
    costs = Costs(commission_per_side=1.0, slippage_ticks=1.0)
    result = replay(
        tape(side), entry(side, stop_points=2.0),
        rules=RULES, instrument=INSTRUMENT, costs=costs, sizing=sizing, initial_cash=1000.0,
    )  # fmt: skip
    t = result.trades.iloc[0]
    assert t.contracts == 1  # $1 budget < 2 points x $2: clamped up to the minimum
    assert t.gross_points == -2.25  # 101.25 -> 99.0, the slipped prices
    # Costs.slippage_ticks remains a separate post-fill cost on top of the fill adversity.
    assert t.pnl_usd == -2.25 * 2.0 - 2 * (1.0 + TICK * 2.0)


def test_one_tick_stop_is_in_the_market_after_slippage() -> None:
    with pytest.raises(ValueError, match="stop"):
        one_tick(entry(1, stop_points=TICK))
    default = replace(RULES, fill_model=FillModel.DEFAULT)
    t = replay(tape(), entry(1, stop_points=TICK), rules=default).trades.iloc[0]
    assert t.stop == 100.75


def test_refusals() -> None:
    for kwargs in (
        {"latency_bars": 1},
        {"latency_bars": 1, "fill_model": FillModel.ONE_TICK},
        {"entry_fill": FillTiming.CLOSE},
    ):
        with pytest.raises(ValueError, match="NEXT_OPEN cannot mix"):
            replace(RULES, **kwargs)
    for sig in (entry(1, target=105.0), entry(1, target_points=2.0)):
        with pytest.raises(ValueError, match="does not support targets"):
            one_tick(sig)
    for sig, rules in (
        (entry(1, stop_points=2.0), replace(RULES, target_order="mit")),
        (entry(1, stop=99.0), replace(RULES, protection="oto")),
    ):
        with pytest.raises(ValueError, match="does not support order slots"):
            one_tick(sig, rules)
    with pytest.raises(ValueError, match="entry_limit_ioc ONE_TICK"):
        one_tick(entry(1, entry_limit_ioc=102.0))
    with pytest.raises(ValueError, match="Instrument"):
        replay(tape(), entry(1), rules=RULES)


class S(State):
    calls: int = 0


class P(Params):
    distance: float = 2.0


class F(Features):
    pass


@strategy(state=S, params=P, features=F)
def breakout(s: S, p: P, f: F, bar: Bar, pos: Position) -> TypedDecision:
    s.calls += 1
    if bar.i == 0:
        return enter(1, stop_points=p.distance)
    return HOLD


def test_typed_python_equals_jit_and_slots_refuse_before_matching() -> None:
    def typed(fields: tuple[str, ...], jit: bool = True, rules: Rules = RULES):
        return run(
            breakout, tape(), rules=rules, features=F(), instrument=INSTRUMENT, costs=Costs(),
            signal_fields=fields, jit=jit,
        )  # fmt: skip

    j = typed(("entry_side", "stop_points"))
    p = typed(("entry_side", "stop_points"), jit=False)
    pd.testing.assert_frame_equal(j.trades, p.trades, check_exact=True)
    np.testing.assert_array_equal(j.states, p.states)
    assert j.trades.iloc[0].entry_px == 101.25 and j.trades.iloc[0].exit_px == 99.0
    pd.testing.assert_frame_equal(
        j.trades, typed(("entry_side", "stop_points", "stop_update")).trades, check_exact=True
    )
    with pytest.raises(ValueError, match="does not support order slots"):
        typed(("entry_side", "stop_points"), rules=replace(RULES, protection="oto"))


def _compare(
    a: dict[str, np.ndarray],
    side: np.ndarray,
    exit_: np.ndarray,
    stop: np.ndarray,
    stop_pts: np.ndarray | None,
    *,
    exit_next_open: bool,
    bracket_anchor: int,
    tick_round: int,
    cross_session: bool,
    new_day: np.ndarray,
    adaptive: bool,
    reenter: bool,
    force_close: np.ndarray,
    force_open: np.ndarray,
    combine: int,
    exit_long: np.ndarray | None,
    exit_short: np.ndarray | None,
) -> list[tuple]:
    """Kernel and independent reference on one R15 column; equal trades, errors and finals."""
    n = len(side)
    nan = np.full(n, np.nan)
    ref = simulate_column_reference(
        a["o"], a["h"], a["lo"], a["c"], side, exit_, stop, nan, stop_pts=stop_pts,
        next_open=True, exit_next_open=exit_next_open, one_tick=True, tick=TICK,
        bracket_anchor=bracket_anchor, tick_round=tick_round, cross_session=cross_session,
        new_day=new_day, adaptive=adaptive, reenter=reenter, force_close=force_close,
        force_open=force_open, combine=combine, exit_long=exit_long, exit_short=exit_short,
    )  # fmt: skip
    out = K.run_columns(
        a["o"], a["h"], a["lo"], a["c"], side[None, :], exit_[None, :], stop[None, :],
        nan[None, :], nan[None, :], (nan if stop_pts is None else stop_pts)[None, :], nan[None, :],
        new_day, 1, adaptive, reenter, False, False, True, 0, True, TICK,
        exit_next_open=exit_next_open, bracket_anchor=bracket_anchor, tick_round=tick_round,
        cross_session=cross_session, force_close=force_close.astype(np.int8)[None, :],
        force_open=force_open[None, :], combine=combine,
        exit_long=None if exit_long is None else exit_long[None, :],
        exit_short=None if exit_short is None else exit_short[None, :],
    )  # fmt: skip
    _, _, errs, err_bar, fs, fe, fpx, fst, e_i, x_i, sd, e_px, x_px, why, t_stop = out
    if errs[0]:
        assert (int(errs[0]), int(err_bar[0])) == (ref.err, ref.err_bar)
        return []
    trades = list(
        zip(e_i.tolist(), x_i.tolist(), sd.tolist(), e_px.tolist(), x_px.tolist(), why.tolist(),
            [_nan_free(x) for x in t_stop], strict=True)
    )  # fmt: skip
    final = (int(fs[0]), int(fe[0]), _nan_free(fpx[0]), _nan_free(fst[0]))
    assert ref.err == 0
    assert trades == _ref_trades(ref)
    assert _kernel_final(final) == _ref_final(ref)
    return trades


@pytest.mark.parametrize("exit_next_open", [False, True])
@settings(
    max_examples=engine_examples(1500), deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
@given(
    case=_case(),
    bracket_anchor=st.integers(0, 2),
    points=st.sampled_from(["none", "points", "combined"]),
    tick_round=st.integers(0, 2),
    cross_session=st.booleans(),
    side_exits=st.booleans(),
    flatten_at=st.integers(0, 29),
)
def test_kernel_equals_reference(
    case: tuple,
    exit_next_open: bool,
    bracket_anchor: int,
    points: str,
    tick_round: int,
    cross_session: bool,
    side_exits: bool,
    flatten_at: int,
) -> None:
    a, side, exit_, stop, _, adaptive, reenter, _ = case
    n = len(side)
    stop_pts = None if points == "none" else np.full(n, 1.0 + (0.09 if tick_round else 0.0))
    if points == "points":
        stop = np.full(n, np.nan)
    elif tick_round:
        stop = stop - side * 0.09
    odd = np.arange(n) % 2 == 1
    _compare(
        a, side, np.zeros(n, bool) if side_exits else exit_, stop, stop_pts,
        exit_next_open=exit_next_open, bracket_anchor=bracket_anchor, tick_round=tick_round,
        cross_session=cross_session, new_day=np.arange(n) % 7 == 0, adaptive=adaptive,
        reenter=reenter, force_close=np.arange(n) == flatten_at,
        force_open=np.arange(n) == flatten_at + 1, combine=int(points == "combined"),
        exit_long=exit_ & ~odd if side_exits else None, exit_short=exit_ & odd if side_exits else None,
    )  # fmt: skip


@pytest.mark.parametrize("exit_next_open", [False, True])
def test_kernel_equals_reference_dense(exit_next_open: bool) -> None:
    rng = np.random.default_rng(1313)
    reasons = set()
    for j in range(1000):
        a, side, exits, stop, _ = _dense_case(rng)
        n = len(side)
        points = j % 3
        stop_pts = None if points == 0 else np.full(n, 0.84)
        if points == 1:
            stop[:] = np.nan
        else:
            stop -= side * 0.09
        trades = _compare(
            a, side, exits, stop, stop_pts,
            exit_next_open=exit_next_open, bracket_anchor=j % 3, tick_round=1 + j % 2,
            cross_session=bool(j % 2), new_day=np.arange(n) % 13 == 0, adaptive=bool(j % 5 == 0),
            reenter=bool(j % 4), force_close=(rng.random(n) < 0.05).astype(np.int8) * (1 + 6 * (j % 2)),
            force_open=rng.random(n) < 0.05, combine=int(points == 2),
            exit_long=rng.random(n) < 0.08, exit_short=rng.random(n) < 0.08,
        )  # fmt: skip
        reasons.update(t[5] for t in trades)
    assert {K.REASON_SIGNAL, K.REASON_STOP, K.REASON_STOP_GAP} <= reasons


@pytest.mark.parametrize(
    "overrides",
    [
        {"amend_timing": "close_submit"},
        {"entry_bar_decision": "completed_close"},
        {"amend_timing": "close_submit", "entry_bar_decision": "completed_close"},
    ],
)
def test_slot_clock_overrides_depend_on_config_not_data(overrides: dict[str, str]) -> None:
    # R15b: clock overrides always select the slots scheduler, so a run without any
    # amendment and one with a no-op amendment (same level) are both accepted and equal.
    rules = replace(RULES, **overrides)
    plain = one_tick(entry(1, stop_points=2.0), rules)
    amended = entry(1, stop_points=2.0).replace(stop_update=np.array([NAN, 99.25, NAN, NAN]))
    pd.testing.assert_frame_equal(plain, one_tick(amended, rules), check_exact=True)
    assert plain.entry_px.tolist() == [101.25] and plain.stop.tolist() == [99.25]


@strategy(state=S, params=P, features=F)
def ioc_breakout(s: S, p: P, f: F, bar: Bar, pos: Position) -> TypedDecision:
    if bar.i == 0:
        return enter_limit_ioc(1, 102.0, stop=99.0)
    return HOLD


def test_typed_ioc_one_tick_refused() -> None:
    with pytest.raises(ValueError, match="entry_limit_ioc ONE_TICK"):
        run(
            ioc_breakout,
            tape(),
            rules=RULES,
            features=F(),
            instrument=INSTRUMENT,
            signal_fields=("entry_side", "entry_limit_ioc", "stop"),
        )


@pytest.mark.parametrize("side", [1, -1])
def test_combined_stop_clamp_and_proportional_costs(side: int) -> None:
    result = replay(
        tape(side),
        entry(side, stop=mirror(side, 99.5), stop_points=2.0),
        rules=replace(RULES, stop_combine="tighter", tick_round=TickRound.NEAREST),
        instrument=INSTRUMENT,
        sizing=Sizing(risk_usd=1.0, max_contracts=5, min_qty_policy="clamp"),
        costs=Costs(proportional_bps=3.0),
        initial_cash=1000.0,
    )
    t = result.trades.iloc[0]
    assert t.contracts == 1
    assert t.entry_px == mirror(side, 101.25)
    assert t.stop == mirror(side, 99.5)
    assert t.exit_px == mirror(side, 99.25)
    expected_fee = (abs(t.entry_px) + abs(t.exit_px)) * 2.0 * 3.0 / 10000
    assert t.pnl_usd == -4.0 - expected_fee
