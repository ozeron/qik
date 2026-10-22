"""Vector engine: kernel == independent reference on generated inputs, plus API contract tests.

Golden-data checks against Nautilus live in the repo-root ``tests/vector_engine/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qik.vector_engine import Bars, Costs, Instrument, Rules, Signals, simulate
from qik.vector_engine import kernel as K
from qik.vector_engine.reference import simulate_column_reference

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


def _run_kernel(arrays, side, exit_, stop, target, adaptive, reenter, stale_drop):
    out = K.run_columns(
        arrays["o"], arrays["h"], arrays["lo"], arrays["c"],
        side[None, :], exit_[None, :], stop[None, :], target[None, :],
        adaptive, reenter, stale_drop,
    )  # fmt: skip
    counts, _, errs, err_bar, final_side, final_entry, e_i, x_i, sd, e_px, x_px, why = out
    trades = list(
        zip(
            e_i.tolist(),
            x_i.tolist(),
            sd.tolist(),
            e_px.tolist(),
            x_px.tolist(),
            why.tolist(),
            strict=True,
        )
    )
    return (
        trades,
        int(errs[0]),
        int(err_bar[0]),
        int(final_side[0]),
        int(final_entry[0]),
        int(counts[0]),
    )


@settings(max_examples=1500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_case())
def test_kernel_equals_reference(case) -> None:
    arrays, side, exit_, stop, target, adaptive, reenter, stale_drop = case
    ref = simulate_column_reference(
        arrays["o"], arrays["h"], arrays["lo"], arrays["c"], side, exit_, stop, target,
        adaptive=adaptive, reenter=reenter, stale_drop=stale_drop,
    )  # fmt: skip
    trades, err, err_bar, final_side, final_entry, cnt = _run_kernel(*case)
    if err:
        # on error the kernel stops; the table is not built (api raises), so compare the verdict
        assert (err, err_bar) == (ref.err, ref.err_bar)
        return
    assert ref.err == 0
    assert cnt == len(ref.trades)
    expected = [(t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px, t.reason) for t in ref.trades]
    assert trades == expected
    assert (final_side, final_entry) == (ref.final_side, ref.final_entry_i)


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


def test_kernel_equals_reference_dense_both_stale_modes() -> None:
    rng = np.random.default_rng(11)
    stale_hits = 0
    for stale_drop in (False, True):
        for _ in range(2500):
            arrays, side, exit_, stop, target = _dense_case(rng)
            adaptive, reenter = bool(rng.random() < 0.5), bool(rng.random() < 0.8)
            ref = simulate_column_reference(
                arrays["o"], arrays["h"], arrays["lo"], arrays["c"], side, exit_, stop, target,
                adaptive=adaptive, reenter=reenter, stale_drop=stale_drop,
            )  # fmt: skip
            trades, err, err_bar, final_side, final_entry, _ = _run_kernel(
                arrays, side, exit_, stop, target, adaptive, reenter, stale_drop
            )
            assert (err, err_bar if err else -1) == (ref.err, ref.err_bar if ref.err else -1)
            stale_hits += err == K.ERR_STALE_BOOK
            if not err:
                assert trades == [
                    (t.entry_i, t.exit_i, t.side, t.entry_px, t.exit_px, t.reason)
                    for t in ref.trades
                ]
                assert (final_side, final_entry) == (ref.final_side, ref.final_entry_i)
    assert stale_hits > 10  # the stale-book branch is really exercised in error mode


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


# --- API contract -----------------------------------------------------------------------------------
def test_simulate_basic_long_target_and_costs() -> None:
    bars = _bars(
        [(100.0, 100.5, 99.5, 100.0), (100.0, 101.0, 99.75, 100.75), (100.75, 102.0, 100.5, 101.5)]
    )
    side = np.array([1, 0, 0], np.int8)
    sig = Signals(entry_side=side, exit=np.zeros(3, bool), stop=np.array([99.0, np.nan, np.nan]),
                  target=np.array([101.5, np.nan, np.nan]))  # fmt: skip
    res = simulate(
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
    res = simulate(bars, Signals(entry_side=side, exit=np.zeros((5, 3), bool), stop=stop))
    assert res.n_configs == 3
    by = res.trades.set_index("config")
    assert (
        by.loc[0, "exit_i"] == 1 and by.loc[1, "exit_i"] == 1 and 2 not in by.index
    )  # config 2 never stopped


def test_index_adds_times() -> None:
    idx = pd.date_range("2024-01-02 14:30", periods=2, freq="1min", tz="UTC")
    bars = Bars(np.array([100.0, 100.0]), np.array([101.0, 101.0]), np.array([99.0, 99.0]),
                np.array([100.0, 100.0]), index=idx)  # fmt: skip
    sig = Signals(entry_side=np.array([1, 0], np.int8), exit=np.array([False, True]))
    res = simulate(bars, sig)
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
    with pytest.raises(ValueError, match="entry_side must be"):
        simulate(bars, Signals(entry_side=np.array([2, 0, 0], np.int8), exit=ok_exit))
    with pytest.raises(ValueError, match="shape"):
        simulate(bars, Signals(entry_side=np.zeros(2, np.int8), exit=ok_exit))
    with pytest.raises(ValueError, match="finite or NaN"):
        simulate(
            bars,
            Signals(
                entry_side=np.zeros(3, np.int8), exit=ok_exit, stop=np.array([np.inf, 1.0, 1.0])
            ),
        )
    with pytest.raises(ValueError, match="costs need an instrument"):
        simulate(bars, Signals(entry_side=np.zeros(3, np.int8), exit=ok_exit), costs=Costs(1.0))
    with pytest.raises(ValueError, match="stale_book"):
        Rules(stale_book="yes")


def test_off_tick_level_refused() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)] * 2)
    sig = Signals(
        entry_side=np.array([1, 0], np.int8), exit=np.zeros(2, bool), stop=np.array([99.9, np.nan])
    )
    with pytest.raises(ValueError, match="tick grid"):
        simulate(bars, sig, instrument=Instrument(TICK, 50.0))


def test_stop_at_close_refused() -> None:
    bars = _bars([(100.0, 101.0, 99.0, 100.0)] * 2)
    sig = Signals(
        entry_side=np.array([1, 0], np.int8), exit=np.zeros(2, bool), stop=np.array([100.0, np.nan])
    )
    with pytest.raises(ValueError, match="not beyond the entry close"):
        simulate(bars, sig)


def _stale_case():
    bars = _bars([(3956.5, 3957.0, 3952.5, 3954.25), (3954.25, 3954.5, 3943.0, 3943.0)])
    sig = Signals(
        entry_side=np.array([-1, -1], np.int8),
        exit=np.zeros(2, bool),
        stop=np.array([3954.75, 3950.0]),
        target=np.array([3953.75, np.nan]),
    )
    return bars, sig


def test_stale_book_error_then_nautilus_mode() -> None:
    bars, sig = _stale_case()
    with pytest.raises(ValueError, match="stale"):
        simulate(bars, sig)
    res = simulate(bars, sig, rules=Rules(stale_book="nautilus"))
    assert list(res.trades.reason) == ["target"]
    held = res.open_positions.iloc[0]
    assert (held.entry_i, held.entry_px, held.side) == (
        1,
        3943.0,
        -1,
    )  # entered, left without levels


def test_stale_book_avoided_without_same_bar_reentry() -> None:
    bars, sig = _stale_case()
    res = simulate(bars, sig, rules=Rules(reenter_on_exit_bar=False))
    assert list(res.trades.reason) == ["target"] and res.open_positions.empty
