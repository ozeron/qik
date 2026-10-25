"""Typed execution profiles and the explicit research/selection boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import replay, script

from qik.vector_engine import (
    Bars,
    Compliance,
    Costs,
    Instrument,
    MarketFill,
    Profile,
    Rules,
    require_nautilus,
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
    result = replay(
        bars,
        script(np.array([0 if empty else 1, 0, 0]), np.array([False, True, False])),
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
        replay(
            _bars(index),
            script(np.array([1, 0, 0]), np.zeros(3, bool)),
            rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN),
        )


def test_next_open_refuses_limit_entries() -> None:
    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="UTC"))
    with pytest.raises(ValueError, match="entry_limit"):
        replay(
            bars,
            script(
                np.array([1, 0, 0]), np.zeros(3, bool), entry_limit=np.array([99, np.nan, np.nan])
            ),
            rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN),
        )


@pytest.mark.parametrize("profile", [Profile.TRADINGVIEW, Profile.REPRO])
@pytest.mark.parametrize("exit_fill", list(MarketFill))
def test_exit_clock_preserves_limit_entry_stop_requirement(
    profile: Profile, exit_fill: MarketFill
) -> None:
    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="UTC"))
    signals = script(
        np.array([1, 0, 0]),
        np.array([False, True, False]),
        entry_limit=np.array([101.0, np.nan, np.nan]),
    )
    with pytest.raises(ValueError, match="limit entry needs a stop"):
        replay(bars, signals, rules=Rules(profile=profile, exit_fill=exit_fill))


def test_next_open_tags_are_repro() -> None:
    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="UTC"))
    result = replay(
        bars,
        script(np.array([1, 0, 0]), np.array([False, True, False])),
        rules=Rules(profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN),
    )
    assert result.compliance is Compliance.REPRO_TRADINGVIEW
    assert result.deployable is False
    assert result.trades[["entry_i", "exit_i", "entry_px", "exit_px"]].values.tolist() == [
        [1, 2, 101, 102]
    ]


def test_market_fill_alias_survives_dataclass_replace() -> None:
    from dataclasses import replace

    from qik.vector_engine import FillTiming

    rules = replace(Rules(), profile=Profile.TRADINGVIEW, market_fill=MarketFill.NEXT_OPEN)
    assert rules.effective_entry_fill is FillTiming.NEXT_OPEN
    assert rules.effective_exit_fill is FillTiming.NEXT_OPEN
    mixed = replace(rules, exit_fill=FillTiming.CLOSE)
    assert mixed.effective_entry_fill is FillTiming.NEXT_OPEN
    assert mixed.effective_exit_fill is FillTiming.CLOSE


def test_repro_bracket_rules_refuse_nautilus() -> None:
    from qik.vector_engine import BracketAnchor, BracketPolicy

    for values in (
        {"bracket_policy": BracketPolicy.ADVERSE_FIRST},
        {"bracket_anchor": BracketAnchor.SIGNAL},
        {"bracket_anchor": BracketAnchor.FILL},
    ):
        with pytest.raises(ValueError, match="repro profile"):
            Rules(**values)
        Rules(profile=Profile.TRADINGVIEW, **values)
    with pytest.raises(ValueError, match="BracketPolicy"):
        Rules(profile=Profile.TRADINGVIEW, bracket_policy="target_first")


@pytest.mark.parametrize("platform", ["tradingview", "nt8", "mt5", "custom_author"])
@pytest.mark.parametrize("empty", [False, True])
def test_generic_repro_tags_and_boundary(platform: str, empty: bool) -> None:
    from qik.vector_engine import BracketAnchor, BracketPolicy, FillTiming

    bars = _bars(pd.date_range("2023-01-03 10:00", periods=3, freq="min", tz="UTC"))
    result = replay(
        bars,
        script(np.array([0 if empty else 1, 0, 0]), np.array([False, True, False])),
        rules=Rules(
            profile=Profile.REPRO,
            author_platform=platform,
            entry_fill=FillTiming.NEXT_OPEN,
            exit_fill=FillTiming.CLOSE,
            bracket_policy=BracketPolicy.ADVERSE_FIRST,
            bracket_anchor=BracketAnchor.FILL,
        ),
        instrument=Instrument(1, 1),
        costs=Costs(),
        initial_cash=1000,
    )
    assert result.compliance == f"repro:{platform}"
    assert result.compliance.value == f"repro:{platform}"
    assert result.summary().compliance.tolist() == [f"repro:{platform}"]
    assert not result.deployable
    with pytest.raises(ValueError, match=f"repro:{platform}"):
        require_nautilus(result)


@pytest.mark.parametrize("platform", ["", "NT8", "white space", "nt8:live", None, 1])
def test_invalid_platform_is_refused(platform: object) -> None:
    with pytest.raises(ValueError, match="author_platform"):
        Rules(profile=Profile.REPRO, author_platform=platform)


def test_custom_platform_requires_generic_repro() -> None:
    for profile in (Profile.NAUTILUS, Profile.TRADINGVIEW):
        with pytest.raises(ValueError, match="Profile.REPRO"):
            Rules(profile=profile, author_platform="nt8")


def test_legacy_rules_positional_arguments_are_unchanged() -> None:
    from qik.vector_engine import FillModel

    rules = Rules(
        False,
        True,
        "error",
        1,
        "error",
        Profile.TRADINGVIEW,
        MarketFill.NEXT_OPEN,
        0,
        FillModel.DEFAULT,
        "callback",
        "limit",
        "error",
        "any",
        "error",
    )
    assert rules.market_fill is MarketFill.NEXT_OPEN
    assert rules.effective_entry_fill is MarketFill.NEXT_OPEN
    assert rules.latency_bars == 0


def test_nautilus_refuses_nondefault_alias_even_if_both_clocks_override_it() -> None:
    from qik.vector_engine import FillTiming

    with pytest.raises(ValueError, match="repro only"):
        Rules(
            market_fill=MarketFill.NEXT_OPEN,
            entry_fill=FillTiming.CLOSE,
            exit_fill=FillTiming.CLOSE,
        )
