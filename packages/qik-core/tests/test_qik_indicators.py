"""qik.indicators vs independent references + synthetic edge cases.

References are written as plain loops here, independent of the pandas tricks
used in the implementation. Real-bar checks live in
``libs/book/book/tests/indicators/test_indicators_real_bars.py`` (qik tests stay
free of private packages such as ``lake``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import qik.indicators as qi


def _walk(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    spread = rng.uniform(0.1, 2.0, n)
    high = close + spread * rng.uniform(0.0, 1.0, n)
    low = close - spread * rng.uniform(0.0, 1.0, n)
    gap = rng.uniform(0, 1, n) < 0.05  # occasional opening gaps
    shift = np.where(gap, rng.normal(0, 5, n), 0.0).cumsum()
    return pd.DataFrame(
        {
            "open": close - rng.normal(0, 0.3, n) + shift,
            "high": high + shift,
            "low": low + shift,
            "close": close + shift,
            "volume": rng.integers(0, 50, n).astype(float),
        }
    )


def _ref_tr(h: list[float], l: list[float], c: list[float]) -> list[float]:
    out = []
    for i in range(len(h)):
        if i == 0:
            out.append(h[i] - l[i])
        else:
            out.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return out


def _ref_wilder_sma_seed(x: list[float], n: int) -> list[float]:
    out = [math.nan] * len(x)
    if len(x) >= n:
        prev = sum(x[:n]) / n
        out[n - 1] = prev
        for i in range(n, len(x)):
            prev = (prev * (n - 1) + x[i]) / n
            out[i] = prev
    return out


def _ref_ewm_first(x: list[float], alpha: float, n: int) -> list[float]:
    out, prev = [], None
    for i, v in enumerate(x):
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev if i >= n - 1 else math.nan)
    return out


def _close(a: pd.Series | list[float], b: list[float], rtol: float = 1e-12) -> None:
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b), rtol=rtol, atol=1e-9)


# ---------------------------------------------------------------- ATR / TR


@pytest.mark.parametrize("n", [1, 3, 14])
def test_true_range_and_wilder_atr_match_loop_reference(n: int) -> None:
    df = _walk()
    h, l, c = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
    tr = _ref_tr(h, l, c)
    _close(qi.true_range(df["high"], df["low"], df["close"]), tr)
    _close(qi.atr(df["high"], df["low"], df["close"], n), _ref_wilder_sma_seed(tr, n))
    _close(
        qi.atr(df["high"], df["low"], df["close"], n, seed="first"),
        _ref_ewm_first(tr, 1.0 / n, n),
    )
    sma = [math.nan] * (n - 1) + [sum(tr[i - n + 1 : i + 1]) / n for i in range(n - 1, len(tr))]
    _close(qi.atr(df["high"], df["low"], df["close"], n, smoothing="sma"), sma)


def test_atr_seeding_variants_differ_then_converge() -> None:
    df = _walk(400)
    a = qi.atr(df["high"], df["low"], df["close"], 14, seed="sma")
    b = qi.atr(df["high"], df["low"], df["close"], 14, seed="first")
    assert a.iloc[:13].isna().all() and b.iloc[:13].isna().all()
    assert abs(a.iloc[13] - b.iloc[13]) > 1e-6  # warm-up disagreement is real
    assert abs(a.iloc[-1] - b.iloc[-1]) < 1e-3 * a.iloc[-1]


def test_true_range_first_bar_and_custom_prev_close() -> None:
    h = pd.Series([10.0, 12.0, 11.0])
    l = pd.Series([9.0, 10.0, 8.0])
    c = pd.Series([9.5, 11.0, 9.0])
    assert qi.true_range(h, l, c).tolist() == [1.0, 2.5, 3.0]
    assert math.isnan(qi.true_range(h, l, c, first="nan").iloc[0])
    # roll guard: blank prev close → that bar is H-L
    pc = c.shift(1).where([False, False, True])
    assert qi.true_range(h, l, c, prev_close=pc).tolist() == [1.0, 2.0, 3.0]


def test_atr_constant_prices_is_the_range() -> None:
    n = 30
    h, l, c = (pd.Series([v] * n) for v in (101.0, 99.0, 100.0))
    for kw in ({}, {"seed": "first"}, {"smoothing": "sma"}, {"smoothing": "ema"}):
        out = qi.atr(h, l, c, 5, **kw).dropna()  # type: ignore[arg-type]
        assert len(out) == n - 4
        assert np.allclose(out, 2.0)


def test_atr_short_input_is_all_nan() -> None:
    df = _walk(3)
    assert qi.atr(df["high"], df["low"], df["close"], 5).isna().all()
    with pytest.raises(ValueError):
        qi.atr(df["high"], df["low"], df["close"], 0)


# ------------------------------------------------------------- averages


def test_sma_ema_rma_references() -> None:
    x = _walk()["close"]
    xs = x.tolist()
    n = 10
    _close(qi.sma(x, n), [math.nan] * 9 + [sum(xs[i - 9 : i + 1]) / 10 for i in range(9, len(xs))])
    _close(qi.ema(x, n, min_periods=n), _ref_ewm_first(xs, 2 / (n + 1), n))
    _close(qi.ema(x, n, min_periods=0).iloc[:3], _ref_ewm_first(xs, 2 / (n + 1), 1)[:3])
    pine = [math.nan] * len(xs)
    prev = sum(xs[:n]) / n
    pine[n - 1] = prev
    for i in range(n, len(xs)):
        prev = 2 / (n + 1) * xs[i] + (1 - 2 / (n + 1)) * prev
        pine[i] = prev
    _close(qi.ema(x, n, seed="sma"), pine)
    _close(qi.rma(x, n), _ref_wilder_sma_seed(xs, n))
    with pytest.raises(ValueError):
        qi.ema(x, n, seed="sma", adjust=True)


def test_wma_hma_reference_and_constant() -> None:
    x = _walk(60)["close"]
    xs = x.tolist()

    def ref_wma(v: list[float], n: int) -> list[float]:
        w = list(range(1, n + 1))
        out = [math.nan] * len(v)
        for i in range(n - 1, len(v)):
            if any(math.isnan(t) for t in v[i - n + 1 : i + 1]):
                continue
            out[i] = sum(a * b for a, b in zip(v[i - n + 1 : i + 1], w, strict=True)) / sum(w)
        return out

    _close(qi.wma(x, 7), ref_wma(xs, 7))
    half, full = ref_wma(xs, 4), ref_wma(xs, 9)  # n=9: half 4, root 3
    _close(qi.hma(x, 9), ref_wma([2 * a - b for a, b in zip(half, full, strict=True)], 3))
    assert np.allclose(qi.hma(pd.Series([5.0] * 40), 16).dropna(), 5.0)


# ------------------------------------------------------------------ RSI


def test_rsi_wilder_and_cutler_reference() -> None:
    x = _walk()["close"]
    xs = x.tolist()
    n = 3
    d = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    g = _ref_ewm_first([max(v, 0.0) for v in d], 1 / n, n)
    lo = _ref_ewm_first([max(-v, 0.0) for v in d], 1 / n, n)
    ref = [math.nan] + [
        (100.0 if a > 0 else 50.0) if b == 0 else 100 - 100 / (1 + a / b)
        for a, b in zip(g, lo, strict=True)
    ]
    _close(qi.rsi(x, n), ref)
    gc = [sum(max(v, 0.0) for v in d[i - n + 1 : i + 1]) / n for i in range(n - 1, len(d))]
    lc = [sum(max(-v, 0.0) for v in d[i - n + 1 : i + 1]) / n for i in range(n - 1, len(d))]
    ref_c = [math.nan] * n + [
        (100.0 if a > 0 else 50.0) if b == 0 else 100 - 100 / (1 + a / b)
        for a, b in zip(gc, lc, strict=True)
    ]
    _close(qi.rsi(x, n, smoothing="sma"), ref_c)


def test_rsi_zero_loss_and_first_diff_variants() -> None:
    up = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    flat = pd.Series([3.0] * 5)
    assert qi.rsi(up, 2).dropna().tolist() == [100.0] * 3
    assert qi.rsi(flat, 2).dropna().tolist() == [50.0] * 3
    assert qi.rsi(up, 2, zero_loss="nan").isna().all()
    # first_diff="zero" counts bar 0 as an observation → one bar earlier
    assert qi.rsi(up, 2).first_valid_index() == 2
    assert qi.rsi(up, 2, first_diff="zero").first_valid_index() == 1


def test_streak_and_connors_rsi() -> None:
    c = pd.Series([10.0, 11.0, 12.0, 12.0, 11.0, 10.0, 11.0])
    assert qi.streak(c).tolist() == [0.0, 1.0, 2.0, 0.0, -1.0, -2.0, 1.0]
    x = _walk(200)["close"]
    crsi = qi.connors_rsi(x, pct_period=20)
    assert crsi.dropna().between(0.0, 100.0).all()
    pr = qi.percent_rank(x.pct_change(), 20)
    manual = (x.pct_change().iloc[-20:] <= x.pct_change().iloc[-1]).mean() * 100.0
    assert pr.iloc[-1] == pytest.approx(manual)  # no ties on a random walk


# ------------------------------------------------------------ Bollinger


def test_bollinger_ddof_and_mid() -> None:
    x = _walk(80)["close"]
    for ddof in (0, 1):
        bb = qi.bollinger(x, 20, 2.0, ddof=ddof)
        w = x.iloc[-20:].to_numpy()
        sd = math.sqrt(sum((v - w.mean()) ** 2 for v in w) / (20 - ddof))
        assert bb["mid"].iloc[-1] == pytest.approx(w.mean())
        assert bb["upper"].iloc[-1] == pytest.approx(w.mean() + 2 * sd)
        assert bb["lower"].iloc[-1] == pytest.approx(w.mean() - 2 * sd)
    assert (
        qi.bollinger(x, 20, ddof=0)["width"].iloc[-1]
        < qi.bollinger(x, 20, ddof=1)["width"].iloc[-1]
    )
    flat = qi.bollinger(pd.Series([4.0] * 30), 10)
    assert (flat["width"].dropna() == 0.0).all() and flat["pct_b"].isna().all()
    ema_mid = qi.bollinger(x, 20, mid="ema")["mid"]
    assert ema_mid.iloc[0] == x.iloc[0]


# ----------------------------------------------------------------- VWAP


def test_session_vwap_resets_holds_on_zero_volume_and_clips() -> None:
    df = pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 20.0, 21.0, 22.0],
            "low": [9.0, 10.0, 11.0, 18.0, 19.0, 20.0],
            "close": [10.0, 11.0, 12.0, 19.0, 20.0, 21.0],
            "volume": [0.0, 2.0, 0.0, 1.0, -5.0, 3.0],
            "session": ["a", "a", "a", "b", "b", "b"],
        }
    )
    v = qi.session_vwap(df["high"], df["low"], df["close"], df["volume"], df["session"])
    assert math.isnan(v.iloc[0])  # no volume yet
    assert v.iloc[1] == 11.0 and v.iloc[2] == 11.0  # zero volume holds
    assert v.iloc[3] == 19.0 and v.iloc[4] == 19.0  # reset; negative clipped to 0
    assert v.iloc[5] == pytest.approx((19.0 + 3 * 21.0) / 4)
    raw = qi.session_vwap(
        df["high"], df["low"], df["close"], df["volume"], df["session"], clip_negative_volume=False
    )
    assert raw.iloc[4] == pytest.approx((19.0 - 5 * 20.0) / -4)
    close_px = qi.session_vwap(
        df["high"], df["low"], df["close"], df["volume"], df["session"], price="close"
    )
    assert close_px.iloc[1] == 11.0


def test_session_vwap_summation_variants_agree_closely() -> None:
    df = _walk(2000)
    df["volume"] = df["volume"] + 1.0
    key = pd.Series(np.repeat(np.arange(5), 400))
    a = qi.session_vwap(df["high"], df["low"], df["close"], df["volume"], key)
    b = qi.session_vwap(df["high"], df["low"], df["close"], df["volume"], key, summation="kahan")
    np.testing.assert_allclose(a, b, rtol=1e-12)
    ref = []
    for _, g in df.groupby(key):
        tp = (g["high"] + g["low"] + g["close"]) / 3.0
        ref.extend(((tp * g["volume"]).cumsum() / g["volume"].cumsum()).tolist())
    assert a.tolist() == ref  # naive == Series.cumsum, bit for bit


# --------------------------------------------------------- opening range


def _or_frame() -> pd.DataFrame:
    rows = []
    for sess, clocks in (("d1", range(565, 600)), ("d2", [570, 571, 573, 590])):  # d2: gaps
        for m in clocks:
            px = 100.0 + (m - 570) + (0 if sess == "d1" else 50)
            rows.append(
                {
                    "session_date": sess,
                    "clock_min": m,
                    "datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=m),
                    "open": px,
                    "high": px + 0.5,
                    "low": px - 0.5,
                    "close": px + 0.25,
                }
            )
    return pd.DataFrame(rows)


def test_opening_range_clock_window_is_half_open() -> None:
    orr = qi.opening_range(_or_frame(), 15)
    d1 = orr.loc["d1"]
    assert d1["or_bars"] == 15  # 09:30..09:44; premarket and 09:45 excluded
    assert d1["or_high"] == 100.0 + 14 + 0.5 and d1["or_low"] == 99.5
    assert d1["or_open"] == 100.0 and d1["or_close"] == 114.25
    assert d1["or_width"] == 15.0
    assert d1["or_width_bps"] == pytest.approx(15.0 / d1["or_mid"] * 1e4)
    assert orr.loc["d2", "or_bars"] == 3  # missing bars: window does not stretch


def test_opening_range_min_bars_and_bar_count_mode() -> None:
    f = _or_frame()
    assert list(qi.opening_range(f, 15, min_bars=12).index) == ["d1"]
    by_bars = qi.opening_range(f[f["clock_min"] >= 570], 4, by="bars")
    assert by_bars.loc["d2", "or_high"] == 50 + 100 + 20 + 0.5  # stretches to 09:50
    assert by_bars.loc["d1", "or_bars"] == 4
    assert qi.opening_range(f, 15, start_min=900).empty
