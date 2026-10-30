"""The qualified contract uses explicit clocks; reproduction conventions stay refused."""

from __future__ import annotations

import pytest

from qik.vector_engine import (
    BracketAnchor,
    BracketPolicy,
    FillModel,
    FillTiming,
    Flatten,
    MarketFill,
    Rules,
    SessionBoundary,
    TargetGap,
    TickRound,
)


@pytest.mark.parametrize(
    "clocks",
    [
        {"entry_fill": FillTiming.NEXT_OPEN},
        {"exit_fill": FillTiming.NEXT_OPEN},
        {"entry_fill": FillTiming.NEXT_OPEN, "exit_fill": FillTiming.NEXT_OPEN},
    ],
)
def test_explicit_scheduled_clocks_admitted(clocks: dict) -> None:
    Rules(**clocks, bracket_anchor=BracketAnchor.FILL, flatten=Flatten.LAST_BAR_CLOSE)


@pytest.mark.parametrize(
    "extra",
    [
        {"fill_model": FillModel.ONE_TICK},
        {"latency_bars": 1},
        {"protection": "oto"},
        {"target_order": "mit"},
        {"target_gap": TargetGap.GAP_IMPROVED},
        {"tick_round": TickRound.NEAREST},
        {"session_boundary": SessionBoundary.NEXT_OBSERVED},
        {"bracket_anchor": BracketAnchor.SIGNAL},
        {"bracket_policy": BracketPolicy.ADVERSE_FIRST},
    ],
)
def test_unqualified_scheduled_compositions_refuse(extra: dict) -> None:
    with pytest.raises(ValueError):
        Rules(entry_fill=FillTiming.NEXT_OPEN, **extra)


def test_alias_stays_repro_only() -> None:
    with pytest.raises(ValueError, match="repro only"):
        Rules(market_fill=MarketFill.NEXT_OPEN)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_native_priority_and_repro_compatibility(
    side: int, slots: bool, jit: bool, native: bool
) -> None:
    import numpy as np
    import pandas as pd
    from _typed_scripts import replay, script

    from qik.vector_engine import Bars, Costs, Instrument, Profile, require_nautilus

    end = 120.0 if side == 1 else 90.0
    bars = Bars(
        np.array([100.0, 105.0, end]),
        np.array([101.0, 107.0, end + 2]),
        np.array([99.0, 103.0, end - 1]),
        np.array([100.0, 106.0, end + 1]),
        index=pd.date_range("2023-01-03 09:30", periods=3, freq="min", tz="America/New_York"),
    )
    fields = {
        "stop_points": np.array([20.0, np.nan, np.nan]),
        "target_points": np.array([10.0, np.nan, np.nan]),
    }
    if slots:
        fields["stop_update"] = np.full(3, np.nan)
    rules = Rules(
        profile=Profile.NAUTILUS if native else Profile.REPRO,
        entry_fill=FillTiming.NEXT_OPEN,
        exit_fill=FillTiming.NEXT_OPEN,
        bracket_anchor=BracketAnchor.FILL,
    )
    result = replay(
        bars,
        script(np.array([side, 0, 0]), np.array([False, True, False]), **fields),
        rules=rules,
        instrument=Instrument(tick=0.25, point_value=20),
        costs=Costs(),
        jit=jit,
    )
    assert result.trades.exit_px.tolist() == [105.0 + side * 10 if native else end]
    assert not result.deployable
    if native:
        require_nautilus(result)
        assert result.research_only
