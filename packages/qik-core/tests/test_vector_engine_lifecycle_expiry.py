"""DAY expiry preserves exposure and admits fresh orders, across venue schedulers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import replay, script
from hypothesis import given, settings
from hypothesis import strategies as st
from test_vector_engine_lifecycle import differential, tape

from qik.vector_engine import Bars, Costs, FillModel, Instrument, Rules


@pytest.mark.parametrize("fill", [FillModel.DEFAULT, FillModel.ONE_TICK])
@settings(max_examples=15, deadline=None)
@given(
    side=st.sampled_from([-1, 1]),
    latency=st.integers(0, 1),
    date=st.sampled_from(["2023-03-12", "2023-11-05", "2023-11-24"]),
    gap_minutes=st.integers(0, 240),
)
def test_fresh_stop_after_expiry(fill, side, latency, date, gap_minutes) -> None:
    midnight = pd.Timestamp(date, tz="America/New_York")
    index = pd.DatetimeIndex(
        [
            midnight - pd.Timedelta(minutes=2),
            midnight - pd.Timedelta(minutes=1),
            midnight + pd.Timedelta(minutes=gap_minutes),
            midnight + pd.Timedelta(minutes=gap_minutes + 1),
            midnight + pd.Timedelta(minutes=gap_minutes + 2),
        ]
    )
    b = tape(index)
    h, lo = b.high.copy(), b.low.copy()
    h[2:], lo[2:] = 110.0, 90.0
    b = replace(b, high=h, low=lo)
    sig = script(
        np.array([side, 0, side, 0, 0]),
        np.array([False] * 4 + [True]),
        entry_stop=np.full(5, 100.0 + side * 5.0),
        stop=np.full(5, 100.0 - side * 20.0),
    )
    rules = Rules(day_order_expiry="cancel", latency_bars=latency, fill_model=fill)
    # Exit with latency may remain in flight on the final bar; the old DAY order
    # must never become the entry, irrespective of the eventual new-order fill.
    r = replay(b, sig, rules=rules, instrument=Instrument(0.25, 1.0), costs=Costs())
    rows = r.trades if not r.trades.empty else r.open_positions
    assert len(rows) == 1
    assert int(rows.entry_i.iloc[0]) == 3 + latency
    differential(b, sig, rules)


def test_expiry_preserves_held_position_and_protection() -> None:
    index = pd.DatetimeIndex(["2023-01-03 23:59", "2023-01-04 00:00"], tz="America/New_York")
    b = Bars(
        np.array([100.0, 95.0]),
        np.array([101.0, 101.0]),
        np.array([99.0, 94.0]),
        np.array([100.0, 100.0]),
        index=index,
    )
    sig = script(np.array([1, 0]), np.zeros(2, bool), stop=np.full(2, 97.0))
    r = replay(b, sig, rules=Rules(day_order_expiry="cancel"))
    assert r.trades[["entry_i", "exit_i", "exit_px"]].values.tolist() == [[0, 1, 95.0]]


def test_rejected_first_stop_cancels_new_target() -> None:
    b = tape(pd.date_range("2023-01-03 10:00", periods=4, freq="min", tz="America/New_York"))
    sig = script(
        np.array([1, 0, 0, 0]),
        np.zeros(4, bool),
        stop_update=np.array([np.nan, 100.0, np.nan, np.nan]),
        target_update=np.array([np.nan, 99.0, np.nan, np.nan]),
    )
    rules = Rules(arm_stop="next_bar", protect_rejected="flatten")
    r = replay(b, sig, rules=rules)
    assert r.trades.reason.tolist() == ["protect_fail"]
    differential(b, sig, rules)
