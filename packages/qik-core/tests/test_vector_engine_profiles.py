"""Typed execution profiles and the explicit research/selection boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qik.vector_engine import (
    Bars,
    Compliance,
    Costs,
    Instrument,
    MarketFill,
    Profile,
    Rules,
    Signals,
    require_nautilus,
    simulate,
)


def _bars(index: pd.DatetimeIndex | None = None) -> Bars:
    return Bars(*[np.array([100.0, 101.0, 102.0])] * 4, index=index)


def test_typed_defaults_and_refusals() -> None:
    assert Rules().profile is Profile.NAUTILUS
    assert Rules().market_fill is MarketFill.CLOSE
    with pytest.raises(ValueError, match="TRADINGVIEW"):
        Rules(market_fill=MarketFill.NEXT_OPEN)
    with pytest.raises(ValueError, match="Profile"):
        Rules(profile="nautilus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MarketFill"):
        Rules(market_fill="close")  # type: ignore[arg-type]


@pytest.mark.parametrize("profile", list(Profile))
@pytest.mark.parametrize("empty", [False, True])
def test_result_tags_guard_and_summary(profile: Profile, empty: bool) -> None:
    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="US/Eastern"))
    result = simulate(
        bars,
        Signals(np.array([0 if empty else 1, 0, 0]), np.array([False, True, False])),
        rules=Rules(profile=profile),
        instrument=Instrument(tick=1, point_value=1),
        costs=Costs(),
        initial_cash=1000,
    )
    compliant = profile is Profile.NAUTILUS
    expected = Compliance.NAUTILUS if compliant else Compliance.REPRO_TRADINGVIEW
    assert result.compliance is expected
    assert result.deployable is compliant
    assert result.summary()["compliance"].tolist() == [expected.value]
    if compliant:
        require_nautilus(result)
    else:
        with pytest.raises(ValueError, match="repro:tradingview"):
            require_nautilus(result)


@pytest.mark.parametrize("index", [None, pd.date_range("2023-01-03", periods=3, freq="min")])
def test_next_open_requires_timezone(index: pd.DatetimeIndex | None) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        simulate(
            _bars(index),
            Signals(np.array([1, 0, 0]), np.zeros(3, bool)),
            rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN),
        )


def test_next_open_refuses_limit_entries() -> None:
    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="UTC"))
    with pytest.raises(ValueError, match="entry_limit"):
        simulate(
            bars,
            Signals(
                np.array([1, 0, 0]), np.zeros(3, bool), entry_limit=np.array([99, np.nan, np.nan])
            ),
            rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN),
        )


def test_next_open_tags_are_repro() -> None:
    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="UTC"))
    result = simulate(
        bars,
        Signals(np.array([1, 0, 0]), np.array([False, True, False])),
        rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN),
    )
    assert result.compliance is Compliance.REPRO_TRADINGVIEW
    assert result.deployable is False
    assert result.trades[["entry_i", "exit_i", "entry_px", "exit_px"]].values.tolist() == [
        [1, 2, 101, 102]
    ]
