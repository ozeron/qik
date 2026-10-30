"""Vector engine: kernel == independent reference on generated inputs, plus API contract tests.

Golden-data checks against Nautilus live in the repo-root ``tests/vector_engine/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import Script, replay, script
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from ve_kernel_adapter import run_columns as kernel_columns

from qik.vector_engine import Bars, Costs, Instrument, MarketFill, Profile, Rules
from qik.vector_engine import kernel as K
from qik.vector_engine.reference import simulate_column_reference
from scripts.engine_test_tiers import engine_examples

TICK = 0.25


def _bars(rows: list[tuple[float, float, float, float]]) -> Bars:
    o, h, lo, c = (np.array([r[k] for r in rows]) for k in range(4))
    return Bars(o, h, lo, c)


@st.composite
def _case(draw):
    n = draw(st.integers(2, 30))
    base = draw(st.integers(380, 420))  # in ticks
    closes = [base]
    opens, highs, lows = [], [], []
    for i in range(n):
        step = draw(st.integers(-8, 8))
        gap = draw(st.integers(-6, 6)) if draw(st.booleans()) else 0
        o = closes[-1] + gap
        c = max(o + step, 4)
        up = draw(st.integers(0, 8))
        dn = draw(st.integers(0, 8))
        opens.append(o)
        closes.append(c)
        highs.append(max(o, c) + up)
        lows.append(max(min(o, c) - dn, 1))
    closes = closes[1:]
    side = [draw(st.sampled_from([0, 0, 0, 1, -1])) for _ in range(n)]
    exit_ = [draw(st.sampled_from([False, False, False, True])) for _ in range(n)]
    stop_off = [draw(st.one_of(st.none(), st.integers(-1, 8))) for _ in range(n)]
    tgt_off = [draw(st.one_of(st.none(), st.integers(-1, 8))) for _ in range(n)]
    arrays = {k: np.array(v, dtype=np.float64) * TICK for k, v in
              dict(o=opens, h=highs, lo=lows, c=closes).items()}  # fmt: skip
    stop = np.full(n, np.nan)
    target = np.full(n, np.nan)
    for i in range(n):
        s = side[i]
        if s == 0:
            continue
        if stop_off[i] is not None:
            stop[i] = arrays["c"][i] - s * stop_off[i] * TICK
        if tgt_off[i] is not None:
            target[i] = arrays["c"][i] + s * tgt_off[i] * TICK
    return (
        arrays,
        np.array(side, np.int8),
        np.array(exit_, bool),
        stop,
        target,
        draw(st.booleans()),
        draw(st.booleans()),
        draw(st.booleans()),
    )


def _nan_free(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


def _run_kernel(
    arrays, side, exit_, stop, target, adaptive, reenter, stale_drop,
    limit=None, stop_pts=None, target_pts=None, chase=1, protect_flatten=False, new_day=None,
    next_open=False, latency_bars=0, one_tick=False, tick=0.0, exit_next_open=None, slots=False, adverse_first=False, bracket_anchor=0,
    target_gap=False, tick_round=0, exit_long=None, exit_short=None, force_close=None, force_open=None, cross_session=False,
):  # fmt: skip
    n = len(side)
    nan = np.full(n, np.nan)
    extra = [nan if a is None else a for a in (limit, stop_pts, target_pts)]
    day = np.zeros(n, np.bool_) if new_day is None else new_day
    out = kernel_columns(
        arrays["o"], arrays["h"], arrays["lo"], arrays["c"],
        side[None, :], exit_[None, :], stop[None, :], target[None, :],
        extra[0][None, :], extra[1][None, :], extra[2][None, :],
        day, chase, adaptive, reenter, stale_drop, protect_flatten, next_open,
        latency_bars, one_tick, tick, exit_next_open=exit_next_open,
        adverse_first=adverse_first, bracket_anchor=bracket_anchor,
        target_gap=target_gap, tick_round=tick_round, cross_session=cross_session,
        exit_long=None if exit_long is None else exit_long[None, :],
        exit_short=None if exit_short is None else exit_short[None, :],
        force_close=None if force_close is None else force_close.astype(np.int8)[None, :],
        force_open=None if force_open is None else force_open[None, :],
        entry_trigger=nan[None, :] if slots else None,
        stop_updates=nan[None, :] if slots else None, target_updates=nan[None, :] if slots else None,
        be_after=nan[None, :] if slots else None, be_offset=nan[None, :] if slots else None,
    )  # fmt: skip
    (counts, _, errs, err_bar, final_side, final_entry, final_px, final_stop,
     e_i, x_i, sd, e_px, x_px, why, t_stop) = out  # fmt: skip
    trades = list(
        zip(
            e_i.tolist(),
            x_i.tolist(),
            sd.tolist(),
            e_px.tolist(),
            x_px.tolist(),
            why.tolist(),
            [_nan_free(x) for x in t_stop],
            strict=True,
        )
    )
    final = (
        int(final_side[0]),
        int(final_entry[0]),
        _nan_free(final_px[0]),
        _nan_free(final_stop[0]),
    )
    return trades, int(errs[0]), int(err_bar[0]), final, int(counts[0])


def _ref_trades(ref) -> list[tuple]:
    return [
        (t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px, t.reason, _nan_free(t.stop))
        for t in ref.trades
    ]


def _ref_final(ref) -> tuple:
    held = ref.final_side != 0
    return (
        ref.final_side,
        ref.final_entry_i,
        _nan_free(ref.final_entry_px) if held else None,
        _nan_free(ref.final_stop) if held else None,
    )


def _kernel_final(final: tuple) -> tuple:
    side, entry, px, stop = final
    return (side, entry, px, stop) if side != 0 else (side, entry, None, None)


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("exit_next_open", [False, True])
@pytest.mark.parametrize("next_open", [False, True])
@settings(
    max_examples=engine_examples(1500), deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
@given(
    case=_case(),
    adverse_first=st.booleans(),
    bracket_anchor=st.integers(0, 2),
    point_brackets=st.booleans(),
    target_gap=st.booleans(),
    tick_round=st.integers(0, 2),
    cross_session=st.booleans(),
    side_exits=st.booleans(),
    flatten_at=st.integers(0, 29),
)
def test_kernel_equals_reference(
    case,
    next_open: bool,
    exit_next_open: bool,
    slots: bool,
    adverse_first: bool,
    bracket_anchor: int,
    point_brackets: bool,
    target_gap: bool,
    tick_round: int,
    side_exits: bool,
    cross_session: bool,
    flatten_at: int,
) -> None:
    arrays, side, exit_, stop, target, adaptive, reenter, stale_drop = case
    # Anchor changes have no effect on absolute levels. Include distances so
    # SIGNAL versus FILL really diverges on next-open gaps.
    stop_pts = target_pts = None
    if point_brackets:
        stop_pts = np.full(len(side), 1.0)
        target_pts = np.full(len(side), 1.5)
        stop = np.full(len(side), np.nan)
        target = np.full(len(side), np.nan)
    n = len(side)
    # Off-grid values make NEAREST / UP observably different, also with points.
    if tick_round:
        stop = stop - side * 0.09
        target = target + side * 0.09
        if stop_pts is not None:
            stop_pts = stop_pts + 0.09
            target_pts = target_pts + 0.09
    long_exit = exit_ & (np.arange(n) % 2 == 0) if side_exits else np.zeros(n, bool)
    short_exit = exit_ & (np.arange(n) % 2 == 1) if side_exits else np.zeros(n, bool)
    exit_ = np.zeros(n, bool) if side_exits else exit_
    close_force = np.arange(n) == flatten_at
    open_force = np.arange(n) == flatten_at + 1
    phase1b = dict(
        target_gap=target_gap,
        cross_session=cross_session,
        new_day=np.arange(n) % 7 == 0,
        tick_round=tick_round,
        tick=TICK,
        exit_long=long_exit,
        exit_short=short_exit,
        force_close=close_force,
        force_open=open_force,
    )
    ref = simulate_column_reference(
        arrays["o"], arrays["h"], arrays["lo"], arrays["c"], side, exit_, stop, target,
        adaptive=adaptive, reenter=reenter, stale_drop=stale_drop, next_open=next_open, exit_next_open=exit_next_open,
        stop_update=np.full(len(side), np.nan) if slots else None,
        stop_pts=stop_pts, target_pts=target_pts,
        adverse_first=adverse_first, bracket_anchor=bracket_anchor, **phase1b,
    )  # fmt: skip
    trades, err, err_bar, final, cnt = _run_kernel(
        arrays,
        side,
        exit_,
        stop,
        target,
        adaptive,
        reenter,
        stale_drop,
        stop_pts=stop_pts,
        target_pts=target_pts,
        next_open=next_open,
        exit_next_open=exit_next_open,
        slots=slots,
        adverse_first=adverse_first,
        bracket_anchor=bracket_anchor,
        **phase1b,
    )
    if err:
        # on error the kernel stops; the table is not built (api raises), so compare the verdict
        assert (err, err_bar) == (ref.err, ref.err_bar)
        return
    assert ref.err == 0
    assert cnt == len(ref.trades)
    assert trades == _ref_trades(ref)
    assert _kernel_final(final) == _ref_final(ref)


def _dense_case(rng: np.random.Generator):
    """Tight bars, dense bracket entries: stale-book, stop-gap and same-bar re-entry all occur."""

    n = 40
    c = (400 + np.cumsum(rng.integers(-4, 5, n))) * TICK
    o = np.r_[c[0], c[:-1]] + rng.integers(-2, 3, n) * TICK
    h = np.maximum(o, c) + rng.integers(0, 6, n) * TICK
    lo = np.minimum(o, c) - rng.integers(0, 6, n) * TICK
    side = rng.choice([0, 1, -1], n, p=[0.5, 0.25, 0.25]).astype(np.int8)
    exit_ = rng.random(n) < 0.05
    stop = np.where(side != 0, c - side * rng.integers(1, 9, n) * TICK, np.nan)
    target = np.where(side != 0, c + side * rng.integers(1, 5, n) * TICK, np.nan)
    return dict(o=o, h=h, lo=lo, c=c), side, exit_, stop, target


@settings(max_examples=engine_examples(300), deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(90, 110), st.sampled_from([-1, 0, 1]), st.booleans()),
        min_size=2,
        max_size=30,
    )
)
def test_next_open_equals_shifted_close_for_unprotected_markets(
    rows: list[tuple[int, int, bool]],
) -> None:
    """With no intrabar protection, shift decisions one bar and set the fill close to that open.

    Protection and reentry suppression after intrabar fills are deliberately outside this
    equivalence: a CLOSE entry cannot activate protection during its own bar.
    """

    price, side, exit_ = (np.array(values) for values in zip(*rows, strict=True))
    side[-1], exit_[-1] = 0, False  # no order without a following fill bar
    index = pd.date_range("2023-01-03 10:00", periods=len(rows), freq="min", tz="UTC")
    # Decision closes differ from fill opens, to catch accidentally filling at the decision price.
    bars = Bars(price, price + 2, price - 2, price + 1, index=index)
    repro = replay(
        bars,
        script(side, exit_),
        rules=Rules(
            profile=Profile.TRADINGVIEW,
            market_fill=MarketFill.NEXT_OPEN,
        ),
    )
    close = replay(
        Bars(price, price + 2, price - 2, price, index=index),
        script(np.r_[0, side[:-1]], np.r_[False, exit_[:-1]]),
    )
    pd.testing.assert_frame_equal(repro.trades, close.trades)
    pd.testing.assert_frame_equal(repro.open_positions, close.open_positions)


@pytest.mark.parametrize("exit_next_open", [False, True])
@pytest.mark.parametrize("next_open", [False, True])
def test_kernel_equals_reference_dense_both_stale_modes(
    next_open: bool, exit_next_open: bool, cases: int = 2500
) -> None:
    rng = np.random.default_rng(11)
    stale_hits = 0
    for stale_drop in (False, True):
        for _ in range(cases):
            arrays, side, exit_, stop, target = _dense_case(rng)
            adverse_first, bracket_anchor = bool(rng.integers(2)), int(rng.integers(3))
            adaptive, reenter = bool(rng.random() < 0.5), bool(rng.random() < 0.8)
            new_day = rng.random(len(side)) < 0.04 if next_open else None
            # Mixed absolute/point protection, with occasional no levels, in next-open mode.
            stop_pts = target_pts = None
            if next_open:
                points = rng.random(len(side)) < 0.5
                stop_pts = np.where(points, rng.integers(1, 9, len(side)) * TICK, np.nan)
                target_pts = np.where(points, rng.integers(1, 5, len(side)) * TICK, np.nan)
                stop, target = np.where(points, np.nan, stop), np.where(points, np.nan, target)
                naked = rng.random(len(side)) < 0.2
                stop[naked] = target[naked] = stop_pts[naked] = target_pts[naked] = np.nan
            ref = simulate_column_reference(
                arrays["o"], arrays["h"], arrays["lo"], arrays["c"], side, exit_, stop, target,
                adaptive=adaptive, reenter=reenter, stale_drop=stale_drop,
                next_open=next_open, exit_next_open=exit_next_open, new_day=new_day, stop_pts=stop_pts, target_pts=target_pts,
                adverse_first=adverse_first, bracket_anchor=bracket_anchor,
            )  # fmt: skip
            trades, err, err_bar, final, _ = _run_kernel(
                arrays,
                side,
                exit_,
                stop,
                target,
                adaptive,
                reenter,
                stale_drop,
                next_open=next_open,
                exit_next_open=exit_next_open,
                adverse_first=adverse_first,
                bracket_anchor=bracket_anchor,
                new_day=new_day,
                stop_pts=stop_pts,
                target_pts=target_pts,
            )
            assert (err, err_bar if err else -1) == (ref.err, ref.err_bar if ref.err else -1)
            stale_hits += err == K.ERR_STALE_BOOK
            if not err:
                assert trades == _ref_trades(ref)
                assert _kernel_final(final) == _ref_final(ref)
    if not next_open:
        # the stale-book branch is really exercised in error mode (> 10 in the full 2500)
        assert stale_hits > cases // 250
    else:
        assert stale_hits == 0  # next-open entries never use the preceding close's stale book


def test_dense_sampler_reaches_every_outcome() -> None:
    """A differential test is only as good as its inputs: every exit reason and error must occur."""

    seen = dict.fromkeys(
        ("stop", "target", "gap", "signal", "err_stop", "err_target", "err_stale"), 0
    )
    names = {
        K.REASON_STOP: "stop",
        K.REASON_TARGET: "target",
        K.REASON_STOP_GAP: "gap",
        K.REASON_SIGNAL: "signal",
    }
    rng = np.random.default_rng(5)
    for _ in range(3000):
        arrays, side, exit_, stop, target = _dense_case(rng)
        # a fraction of columns gets inside-the-market levels so the entry-validation errors occur too
        if rng.random() < 0.3:
            stop = np.where(
                side != 0, arrays["c"] - side * rng.integers(-1, 2, len(side)) * TICK, np.nan
            )
        if rng.random() < 0.3:
            target = np.where(
                side != 0, arrays["c"] + side * rng.integers(-1, 2, len(side)) * TICK, np.nan
            )
        trades, err, *_ = _run_kernel(
            arrays, side, exit_, stop, target, bool(rng.random() < 0.5), True, False
        )
        seen["err_stop"] += err == K.ERR_STOP_IN_MARKET
        seen["err_target"] += err == K.ERR_TARGET_IN_MARKET
        seen["err_stale"] += err == K.ERR_STALE_BOOK
        for t in trades if err == 0 else []:  # an errored column leaves its table unfilled
            seen[names[t[5]]] += 1
    assert all(v > 0 for v in seen.values()), seen


def _dense_limit_case(rng: np.random.Generator):
    """Dense limit-then-market entries with absolute and point levels, chase 1-3 bars."""

    arrays, side, exit_, stop, target = _dense_case(rng)
    n = len(side)
    c = arrays["c"]
    use_limit = rng.random(n) < 0.7
    # mostly resting (better than the close), sometimes marketable (at or through it)
    off = rng.integers(-2, 6, n)
    limit = np.where((side != 0) & use_limit, c - side * off * TICK, np.nan)
    pts = rng.random(n) < 0.5
    stop_pts = np.where((side != 0) & pts, rng.integers(1, 9, n) * TICK, np.nan)
    target_pts = np.where((side != 0) & pts & (rng.random(n) < 0.7), rng.integers(1, 6, n) * TICK,
                          np.nan)  # fmt: skip
    stop = np.where(pts, np.nan, stop)
    target = np.where(pts, np.nan, target)
    # absolute levels for resting limits are set from the limit, as a strategy would
    rest = np.isfinite(limit) & ~pts
    stop = np.where(rest, limit - side * rng.integers(1, 9, n) * TICK, stop)
    target = np.where(
        rest & np.isfinite(target), limit + side * rng.integers(1, 6, n) * TICK, target
    )
    new_day = rng.random(n) < 0.04
    return (arrays, side, exit_, stop, target, limit, stop_pts, target_pts,
            int(rng.integers(1, 4)), new_day)  # fmt: skip


LIMIT_OUTCOMES = ("limit_fill", "chase", "marketable", "same_bar_target", "err_protect",
                  "err_new_day", "target_beats_exit", "fill_bar_exit", "points",
                  "protect_fail")  # fmt: skip


def test_kernel_equals_reference_dense_limit_entries(cases: int = 6000) -> None:
    rng = np.random.default_rng(23)
    seen = dict.fromkeys(LIMIT_OUTCOMES, 0)
    for _ in range(cases):
        (arrays, side, exit_, stop, target, limit, stop_pts, target_pts, chase,
         new_day) = _dense_limit_case(rng)  # fmt: skip
        adaptive, reenter = bool(rng.random() < 0.5), bool(rng.random() < 0.8)
        stale_drop = bool(rng.random() < 0.5)
        flat = bool(rng.random() < 0.5)
        adverse_first, bracket_anchor = bool(rng.integers(2)), int(rng.integers(3))
        ref = simulate_column_reference(
            arrays["o"], arrays["h"], arrays["lo"], arrays["c"], side, exit_, stop, target,
            adaptive=adaptive, reenter=reenter, stale_drop=stale_drop, entry_limit=limit,
            stop_pts=stop_pts, target_pts=target_pts, chase_after=chase, protect_flatten=flat,
            new_day=new_day,
            adverse_first=adverse_first, bracket_anchor=bracket_anchor,
        )  # fmt: skip
        trades, err, err_bar, final, _ = _run_kernel(
            arrays, side, exit_, stop, target, adaptive, reenter, stale_drop,
            limit, stop_pts, target_pts, chase, flat, new_day,
            adverse_first=adverse_first, bracket_anchor=bracket_anchor,
        )  # fmt: skip
        assert (err, err_bar if err else -1) == (ref.err, ref.err_bar if ref.err else -1)
        seen["err_protect"] += err == K.ERR_PROTECT_REJECTED
        seen["err_new_day"] += err == K.ERR_PENDING_NEW_DAY
        if err:
            continue
        assert trades == _ref_trades(ref)
        assert _kernel_final(final) == _ref_final(ref)
        c = arrays["c"]
        for e, x, sd, epx, _xpx, why, _st in trades:
            if epx != c[e]:  # only a resting limit fills away from its bar's close
                seen["limit_fill"] += 1
            j = e - chase
            if epx == c[e] and j >= 0 and side[j] == sd and np.isfinite(limit[j]) and side[e] == 0:
                seen["chase"] += 1
            if side[e] == sd and np.isfinite(limit[e]) and epx == c[e]:
                seen["marketable"] += (limit[e] >= c[e]) if sd > 0 else (limit[e] <= c[e])
            seen["same_bar_target"] += e == x and why == K.REASON_TARGET
            seen["target_beats_exit"] += e == x and why == K.REASON_TARGET and bool(exit_[e])
            seen["fill_bar_exit"] += e == x and why == K.REASON_SIGNAL and epx != c[e]
            seen["protect_fail"] += why == K.REASON_PROTECT_FAIL
            seen["points"] += bool(np.isfinite(stop_pts[e]))
    assert all(v > 0 for v in seen.values()), seen


# --- API contract -----------------------------------------------------------------------------------
def test_simulate_basic_long_target_and_costs() -> None:
    bars = _bars(
        [(100.0, 100.5, 99.5, 100.0), (100.0, 101.0, 99.75, 100.75), (100.75, 102.0, 100.5, 101.5)]
    )
    side = np.array([1, 0, 0], np.int8)
    sig = script(side, np.zeros(3, bool), stop=np.array([99.0, np.nan, np.nan]),
                 target=np.array([101.5, np.nan, np.nan]))  # fmt: skip
    res = replay(
        bars,
        sig,
        instrument=Instrument(TICK, 50.0),
        costs=Costs(commission_per_side=2.0, slippage_ticks=1),
    )
    t = res.trades.iloc[0]
    assert (t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px, t.reason) == (
        0,
        2,
        1,
        100.0,
        101.5,
        "target",
    )
    assert t.gross_points == 1.5 and t.gross_usd == 75.0
    assert t.pnl_usd == 75.0 - 2 * (2.0 + 1 * TICK * 50.0)
    assert res.open_positions.empty


def test_simulate_many_configs_independent() -> None:
    bars = _bars([(100.0, 100.5, 99.5, 100.0)] + [(100.0, 101.0, 99.0, 100.0)] * 4)
    side = np.zeros((5, 3), np.int8)
    side[0, :] = 1
    stop = np.full((5, 3), np.nan)
    stop[0] = [99.75, 99.0, 98.0]
    res = replay(bars, script(side, np.zeros((5, 3), bool), stop=stop))
    assert res.n_configs == 3
    by = res.trades.set_index("config")
    assert (
        by.loc[0, "exit_i"] == 1 and by.loc[1, "exit_i"] == 1 and 2 not in by.index
    )  # config 2 never stopped


def test_index_adds_times() -> None:
    idx = pd.date_range("2024-01-02 14:30", periods=2, freq="1min", tz="UTC")
    bars = Bars(np.array([100.0, 100.0]), np.array([101.0, 101.0]), np.array([99.0, 99.0]),
                np.array([100.0, 100.0]), index=idx)  # fmt: skip
    res = replay(bars, script(np.array([1, 0]), np.array([False, True])))
    assert res.trades.iloc[0].entry_time == idx[0] and res.trades.iloc[0].exit_time == idx[1]


@pytest.mark.parametrize(
    "bad",
    [
        (
            np.array([100.0]),
            np.array([99.0]),
            np.array([98.0]),
            np.array([99.5]),
        ),  # high below open
        (np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([np.nan])),
        (np.array([100.0]), np.array([101.0]), np.array([99.0]), np.array([-1.0])),
    ],
)
def test_bars_validation(bad) -> None:
    with pytest.raises(ValueError):
        Bars(*bad)


def test_signal_validation() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)] * 3)
    ok_exit = np.zeros(3, bool)
    with pytest.raises(ValueError, match="side must be -1, 0 or"):
        replay(bars, script(np.array([2, 0, 0]), ok_exit))
    with pytest.raises(ValueError, match="expected length 3"):
        replay(bars, script(np.zeros(2), ok_exit[:2]))
    with pytest.raises(ValueError, match="finite or NaN"):
        replay(bars, script(np.zeros(3), ok_exit, stop=np.array([np.inf, 1.0, 1.0])))
    with pytest.raises(ValueError, match="costs need an instrument"):
        replay(bars, script(np.zeros(3), ok_exit), costs=Costs(1.0))
    with pytest.raises(ValueError, match="stale_book"):
        Rules(stale_book="yes")  # ty: ignore[invalid-argument-type]


def test_off_tick_level_refused() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)] * 2)
    sig = script(np.array([1, 0]), np.zeros(2, bool), stop=np.array([99.9, np.nan]))
    with pytest.raises(ValueError, match="tick grid"):
        replay(bars, sig, instrument=Instrument(TICK, 50.0), costs=Costs())


def test_stop_at_close_refused() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)] * 2)
    sig = script(np.array([1, 0]), np.zeros(2, bool), stop=np.array([100.0, np.nan]))
    with pytest.raises(ValueError, match="not beyond the entry close"):
        replay(bars, sig)


def _stale_case() -> tuple[Bars, Script]:
    bars = _bars([(3956.5, 3957.0, 3952.5, 3954.25), (3954.25, 3954.5, 3943.0, 3943.0)])
    sig = script(
        np.array([-1, -1]),
        np.zeros(2, bool),
        stop=np.array([3954.75, 3950.0]),
        target=np.array([3953.75, np.nan]),
    )
    return bars, sig


def test_stale_book_error_then_nautilus_mode() -> None:
    bars, sig = _stale_case()
    with pytest.raises(ValueError, match="stale"):
        replay(bars, sig)
    res = replay(bars, sig, rules=Rules(stale_book="nautilus"))
    assert list(res.trades.reason) == ["target"]
    held = res.open_positions.iloc[0]
    assert (held.entry_i, held.entry_px, held.side) == (
        1,
        3943.0,
        -1,
    )  # entered, left without levels


def test_stale_book_avoided_without_same_bar_reentry() -> None:
    bars, sig = _stale_case()
    res = replay(bars, sig, rules=Rules(reenter_on_exit_bar=False))
    assert list(res.trades.reason) == ["target"] and res.open_positions.empty


# --- limit entries (A15) at the API: what V2 does not do is refused --------------------------------
def _limit_bars(n_days: int = 1) -> Bars:
    rows = [(100.0, 100.5, 99.5, 100.0), (100.0, 100.25, 99.75, 100.0),
            (100.0, 100.25, 99.75, 100.0)]  # fmt: skip
    o, h, lo, c = (np.array([r[k] for r in rows]) for k in range(4))
    idx = pd.DatetimeIndex(
        ["2024-01-02 15:00", "2024-01-02 15:01", "2024-01-03 15:00" if n_days > 1 else
         "2024-01-02 15:02"], tz="UTC")  # fmt: skip
    return Bars(o, h, lo, c, index=idx)


def _limit_sig(**levels: np.ndarray) -> Script:
    nan = np.full(3, np.nan)
    return script(np.array([1, 0, 0]), np.zeros(3, bool),
                  entry_limit=np.array([99.0, np.nan, np.nan]),
                  **{k: levels.get(k, nan) for k in ("stop", "target")})  # fmt: skip


def test_limit_entry_needs_a_stop_and_dates() -> None:
    bars = _limit_bars()
    tgt = np.array([101.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="needs a stop"):
        replay(bars, _limit_sig(target=tgt))  # V2 sends no protection without a stop
    stop = np.array([98.0, np.nan, np.nan])
    no_index = Bars(bars.open, bars.high, bars.low, bars.close)
    with pytest.raises(ValueError, match="index"):
        replay(no_index, _limit_sig(stop=stop))
    assert bars.index is not None
    naive = Bars(bars.open, bars.high, bars.low, bars.close, index=bars.index.tz_localize(None))
    with pytest.raises(ValueError, match="timezone-aware"):
        replay(naive, _limit_sig(stop=stop))
    res = replay(bars, _limit_sig(stop=stop), rules=Rules(limit_chase_bars=5))
    assert res.trades.empty and res.open_positions.empty  # never filled, never chased


def test_limit_entry_working_into_a_new_day_is_refused() -> None:
    stop = np.array([98.0, np.nan, np.nan])
    with pytest.raises(ValueError, match="new New York date"):
        replay(_limit_bars(n_days=2), _limit_sig(stop=stop), rules=Rules(limit_chase_bars=5))
    exit_ = np.array([False, True, False])  # cancelled before the session ends: fine
    cancelled = script(
        np.array([1, 0, 0]), exit_, stop=stop, entry_limit=np.array([99.0, np.nan, np.nan])
    )
    replay(_limit_bars(n_days=2), cancelled, rules=Rules(limit_chase_bars=5))


def test_explicit_inputs_are_required() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)] * 2)
    sig = script(np.array([1, 0]), np.array([False, True]))
    with pytest.raises(ValueError, match="costs"):
        replay(bars, sig, instrument=Instrument(TICK, 50.0))  # no silent zero costs
    for bad in ({"commission_per_side": -1.0}, {"slippage_ticks": float("nan")}):
        with pytest.raises(ValueError, match="non-negative"):
            Costs(**bad)
    with pytest.raises(ValueError, match="limit_chase_bars"):
        Rules(limit_chase_bars=2.0)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "slots,latency,one_tick",
    [(False, 0, False), (True, 0, False), (False, 1, False), (False, 0, True), (True, 1, True)],
)
def test_phase1b_dense_active_axes(slots: bool, latency: int, one_tick: bool) -> None:
    """All execution bodies with gaps, off-grid distances, directional exits and flatten events."""
    rng = np.random.default_rng(713)
    changed = set()
    for j in range(1000):
        a, side, exits, stop, target = _dense_case(rng)
        n = len(side)
        rounding = j % 3
        points = j % 2 == 0
        if points:
            stop[:] = target[:] = np.nan
        else:
            stop -= side * 0.09
            target += side * 0.09
        sp = np.full(n, 0.84) if points else None
        tp = np.full(n, 0.59) if points else None
        fl = (rng.random(n) < 0.05).astype(np.int8)
        fo = rng.random(n) < 0.05
        xl, xs = rng.random(n) < 0.12, rng.random(n) < 0.12
        opts = dict(
            target_gap=bool(j % 2),
            tick_round=rounding,
            tick=TICK,
            exit_long=xl,
            exit_short=xs,
            force_close=fl,
            force_open=fo,
            latency_bars=latency,
            one_tick=one_tick,
            stop_pts=sp,
            target_pts=tp,
            adverse_first=bool(j % 3),
            bracket_anchor=j % 3,
        )
        ref = simulate_column_reference(
            a["o"],
            a["h"],
            a["lo"],
            a["c"],
            side,
            exits,
            stop,
            target,
            stop_update=np.full(n, np.nan) if slots else None,
            **opts,
        )
        trades, err, eb, final, _ = _run_kernel(
            a, side, exits, stop, target, False, True, False, slots=slots, **opts
        )
        assert (err, eb if err else -1) == (ref.err, ref.err_bar if ref.err else -1)
        if not err:
            assert trades == _ref_trades(ref)
            assert _kernel_final(final) == _ref_final(ref)
            changed.update(t[5] for t in trades)
    assert {K.REASON_SIGNAL, K.REASON_STOP, K.REASON_TARGET}.issubset(changed)
