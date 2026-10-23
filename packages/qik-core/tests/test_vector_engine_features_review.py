"""Adversarial independent boundary and every-prefix checks for shared features."""

import numpy as np
import pandas as pd
import pytest

from qik.vector_engine.features import anchored_vwap, session_masks, session_range


@pytest.mark.parametrize("day,cutoff", [("2023-03-12", 150), ("2023-11-05", 90)])
def test_inactive_dst_sunday_cutoff_is_not_localized(day: str, cutoff: int) -> None:
    schedule = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2023-03-13"]),
            "open_ts": pd.to_datetime(["2023-03-13 13:30Z"]),
            "close_ts": pd.to_datetime(["2023-03-13 20:00Z"]),
            "early_close": [False],
        }
    )
    stamps = pd.date_range(day, periods=24, freq="h", tz="America/New_York")
    for n in range(len(stamps) + 1):
        masks = session_masks(stamps[:n], schedule, cutoff_min=cutoff)
        assert not masks.active.any()
        assert not masks.start.any()
        assert not masks.end.any()
        assert not masks.cutoff.any()
        assert not masks.early_close.any()


def test_gap_missing_first_half_day_and_zero_volume_every_prefix() -> None:
    stamps = pd.DatetimeIndex(
        [
            "2023-11-24 09:31",
            "2023-11-24 09:33",
            "2023-11-24 09:40",
            "2023-11-24 12:58",
            "2023-11-24 12:59",
            "2023-11-24 13:00",
            "2023-11-27 09:32",
            "2023-11-27 09:40",
        ]
    ).tz_localize("America/New_York")
    schedule = pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2023-11-24", "2023-11-27"]),
            "open_ts": pd.to_datetime(["2023-11-24 14:30Z", "2023-11-27 14:30Z"]),
            "close_ts": pd.to_datetime(["2023-11-24 18:00Z", "2023-11-27 21:00Z"]),
            "early_close": [True, False],
        }
    )
    high = np.array([11.0, 14.0, 999.0, 998.0, 997.0, 996.0, 21.0, 888.0])
    low = np.array([9.0, 8.0, 1.0, 2.0, 3.0, 4.0, 19.0, 0.0])
    # The 09:35 boundary falls inside a gap; future extrema must stay invisible.
    expected_high = np.array([np.nan, np.nan, 14.0, 14.0, 14.0, 14.0, np.nan, np.nan])
    expected_low = np.array([np.nan, np.nan, 8.0, 8.0, 8.0, 8.0, np.nan, np.nan])
    for n in range(len(stamps) + 1):
        masks = session_masks(stamps[:n], schedule, cutoff_min=955)
        result = session_range(
            high[:n], low[:n], masks.date, masks.clock_min, start_min=570, end_min=575, min_bars=2
        )
        np.testing.assert_array_equal(result.high, expected_high[:n])
        np.testing.assert_array_equal(result.low, expected_low[:n])
        np.testing.assert_array_equal(result.ready, np.isfinite(expected_high[:n]))
        np.testing.assert_array_equal(masks.start, np.zeros(n, dtype=bool))
        np.testing.assert_array_equal(masks.end, np.array([False] * 4 + [True] + [False] * 3)[:n])
        np.testing.assert_array_equal(
            masks.cutoff, np.array([False] * 4 + [True] + [False] * 3)[:n]
        )
        reset = np.array([True, False, False, False, False, False, True, False])[:n]
        for summation in ("naive", "kahan"):
            actual = anchored_vwap(high[:n], np.zeros(n), reset, summation=summation)
            assert np.isnan(actual).all()
