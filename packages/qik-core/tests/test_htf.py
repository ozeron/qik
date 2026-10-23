"""Hand-calculated close alignment, DST/session boundaries and lake equivalence."""

import numpy as np
import pandas as pd
import pytest

from qik.vector_engine.htf import htf, htf_align
from qik.vector_engine.testing import assert_causal_feature, minute_bars


def close_values(frame):
    return frame.close.to_numpy(dtype=np.float64)


def test_fifteen_minute_warmup_and_prefixes():
    bars = minute_bars(periods=47, tz="America/New_York", step=1.0)
    out = htf(bars, "15min", close_values)
    assert np.isnan(out[:14]).all()
    np.testing.assert_array_equal(out[14:29], 114.0)
    np.testing.assert_array_equal(out[29:44], 129.0)
    np.testing.assert_array_equal(out[44:], 144.0)
    for n in (1, 14, 15, 16, 29, 30, 46):
        assert_causal_feature(lambda b: htf(b, "15min", close_values), bars, n)
    rolling = htf(bars, "15min", lambda f: f.close.rolling(2).mean())
    assert np.isnan(rolling[:29]).all()
    assert rolling[29] == 121.5


def test_daily_cme_boundary_dst_and_warmup():
    # Thursday/Friday before spring DST; Sunday/Monday after the transition.
    times = pd.DatetimeIndex(
        [
            "2023-03-09 18:00",
            "2023-03-10 16:58",
            "2023-03-10 16:59",
            "2023-03-12 18:00",
            "2023-03-13 16:58",
            "2023-03-13 16:59",
            "2023-03-13 18:00",
        ],
        tz="America/New_York",
    )
    prices = np.arange(len(times), dtype=float) + 100.0
    bars = pd.DataFrame(
        {
            "datetime": times,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1.0,
        }
    )
    out = htf(bars, "1D", close_values)
    np.testing.assert_array_equal(out, [np.nan, np.nan, 102.0, 102.0, 102.0, 105.0, 105.0])
    rolling = htf(bars, "1D", lambda f: f.close.rolling(2).mean())
    np.testing.assert_array_equal(rolling, [np.nan] * 5 + [103.5, 103.5])
    for n in range(1, len(bars) + 1):
        assert_causal_feature(lambda b: htf(b, "1D", close_values), bars, n)
    closes = []
    htf(bars, "1D", lambda f: closes.extend(f.datetime) or close_values(f))
    assert closes[0].tz_convert("UTC").hour == 22
    assert closes[1].tz_convert("UTC").hour == 21
    labels = ["2023-03-10"] * 3 + ["2023-03-13"] * 3 + ["2023-03-14"]
    np.testing.assert_array_equal(htf(bars, "1D", close_values, session=labels), out)


def test_alignment_and_deliberate_leak():
    np.testing.assert_array_equal(
        htf_align(np.array([3.0, 4.0]), np.array([15, 30]), np.array([14, 15, 29, 30])),
        [np.nan, 3.0, 3.0, 4.0],
    )
    with pytest.raises(ValueError, match="ordered"):
        htf_align(np.array([1.0, 2.0]), np.array([30, 15]), np.array([20]))
    bars = minute_bars(periods=30, tz="America/New_York", step=1.0)

    def leaky(b):
        # Reveals the final close of a still-forming 15-minute bucket.
        return b.close.groupby(np.arange(len(b)) // 15).transform("last").to_numpy()

    with pytest.raises(AssertionError, match="looks ahead"):
        assert_causal_feature(leaky, bars, 7)


@pytest.mark.parametrize("every,minutes", [("15min", 15), ("1h", 60)])
def test_lake_resample_equivalence(every, minutes):
    from lake.bars import resample

    bars = minute_bars(periods=133, tz="America/New_York", step=0.1, wick=0.5)
    captured = []

    def capture(frame):
        captured.append(frame)
        return close_values(frame)

    out = htf(bars, every, capture)
    expected = resample(bars, bar_minutes=minutes)
    columns = ["open", "high", "low", "close", "volume", "count"]
    pd.testing.assert_frame_equal(captured[0][columns], expected[columns], check_dtype=False)
    closes = pd.DatetimeIndex(expected.datetime) + pd.Timedelta(minutes=minutes)
    np.testing.assert_array_equal(
        out,
        htf_align(
            close_values(expected),
            closes,
            pd.DatetimeIndex(bars.datetime) + pd.Timedelta(minutes=1),
        ),
    )


def test_refuse_malformed_and_incomplete_inputs():
    bars = minute_bars(periods=30, tz="America/New_York")
    incomplete = bars.drop(index=[4, 20]).reset_index(drop=True)
    assert np.isnan(htf(incomplete, "15min", close_values)).all()
    with pytest.raises(ValueError, match="every"):
        htf(bars, "bad", close_values)
    with pytest.raises(ValueError, match="timezone"):
        htf(bars.assign(datetime=bars.datetime.dt.tz_localize(None)), "15min", close_values)
