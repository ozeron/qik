from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from qik.synthetic import (
    DefaultMarketContextProvider,
    SyntheticContext,
    SyntheticConversionError,
    SyntheticEngine,
    SyntheticContextError,
    SyntheticValidationError,
    as_request,
    bars_from_span,
    calibrate_ohlcv_params,
    compare_ohlcv,
    fingerprint_candle_table,
    to_backtesting_frame,
    to_backtesting_payload,
    validate_backtesting_payload,
)
from qik.synthetic.kernels import simulate_returns_ar1

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic" / "request_btc.json"


def _load_fixture() -> dict:
    with FIXTURE.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    data["start_ts"] = datetime.fromisoformat(data["start_ts"])
    return data


def test_generate_deterministic():
    payload = _load_fixture()
    request = as_request(**payload)

    engine = SyntheticEngine(model="gaussian_rw")
    batch_a = engine.generate(request)
    batch_b = engine.generate(request)

    assert batch_a.n_rows == payload["n_bars"]
    assert batch_a.n_rows == batch_b.n_rows
    assert batch_a.bars == batch_b.bars


def test_chunk_iteration_matches_materialized_generation():
    payload = _load_fixture()
    request = as_request(**payload)
    engine = SyntheticEngine(model="gaussian_rw")

    batch = engine.generate(request)
    chunked = tuple(bar for chunk in engine.iter_generate(request, chunk_size=5) for bar in chunk.bars)

    assert chunked == batch.bars


def test_generate_frame_matches_legacy_payload():
    payload = _load_fixture()
    request = as_request(**payload)
    engine = SyntheticEngine(model="gaussian_rw")

    legacy = to_backtesting_payload(engine.generate(request))["candle_table"]
    frame = engine.generate_frame(request)
    direct_payload = engine.generate_backtesting_payload(request)

    assert fingerprint_candle_table(frame) == fingerprint_candle_table(legacy)
    assert fingerprint_candle_table(direct_payload["candle_table"]) == fingerprint_candle_table(legacy)


def test_generate_frame_can_compact_identity_columns():
    payload = _load_fixture()
    request = as_request(**payload)

    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request, compact_identity=True)

    assert str(frame["symbol"].dtype) == "category"
    assert str(frame["venue"].dtype) == "category"
    assert str(frame["dataset"].dtype) == "category"
    assert str(frame["interval"].dtype) == "category"


def test_horizon_rejects_ambiguous_inputs():
    with pytest.raises(Exception):
        bars_from_span(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            "1h",
            n_bars=10,
            duration=timedelta(hours=10),
        )
    with pytest.raises(Exception):
        bars_from_span(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            "1h",
            duration=timedelta(hours=10),
            end_ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )


def test_payload_bridge_requires_identity():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=8,
        seed=42,
    )
    engine = SyntheticEngine(model="ar1")
    batch = engine.generate(request)

    payload = to_backtesting_payload(batch)
    validate_backtesting_payload(payload)

    assert payload["metadata"]["identity"]["symbol"] == "BTCUSDT"
    assert payload["metadata"]["first_timestamp"] == batch.bars[0]["timestamp"]
    assert payload["metadata"]["last_timestamp"] == batch.bars[-1]["timestamp"]
    assert payload["metadata"]["first_close"] == batch.bars[0]["close"]


def test_backtesting_frame_sorts_shuffled_bars():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=8,
        seed=42,
    )
    batch = SyntheticEngine(model="gaussian_rw").generate(request)
    shuffled = type(batch)(
        bars=tuple(reversed(batch.bars)),
        context=batch.context,
        request=batch.request,
        metadata=batch.metadata,
    )

    frame = to_backtesting_frame(shuffled)

    assert frame["timestamp"].is_monotonic_increasing


def test_payload_bridge_requires_complete_identity():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        n_bars=8,
        seed=42,
    )
    engine = SyntheticEngine(model="gaussian_rw")
    batch = engine.generate(request)

    with pytest.raises(SyntheticConversionError):
        to_backtesting_payload(batch)


def test_identity_override_applies_to_table_and_metadata():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        n_bars=4,
        seed=42,
    )
    batch = SyntheticEngine(model="gaussian_rw").generate(request)

    payload = to_backtesting_payload(
        batch,
        identity_override={
            "symbol": "BTCUSDT",
            "venue": "binance",
            "dataset": "trade_bars",
            "interval": "1h",
        },
    )

    assert payload["metadata"]["identity"]["symbol"] == "BTCUSDT"
    assert set(payload["candle_table"]["symbol"]) == {"BTCUSDT"}
    assert set(payload["candle_table"]["venue"]) == {"binance"}
    assert set(payload["candle_table"]["dataset"]) == {"trade_bars"}


def test_model_params_affect_metadata_fingerprints():
    base = {
        "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "interval": "1h",
        "start_price": 1000.0,
        "symbol": "BTCUSDT",
        "n_bars": 4,
        "seed": 42,
    }
    engine = SyntheticEngine(model="gaussian_rw")

    low_vol = engine.generate(as_request(**base, model_params={"volatility": 0.01}))
    high_vol = engine.generate(as_request(**base, model_params={"volatility": 0.02}))

    assert low_vol.metadata.request_id != high_vol.metadata.request_id
    assert low_vol.metadata.run_id != high_vol.metadata.run_id
    assert low_vol.metadata.seed != high_vol.metadata.seed


def test_context_mismatch_guard():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1d",
        start_price=100.0,
        symbol="SPY",
        n_bars=2,
        seed=1,
        model="gaussian_rw",
        model_params={"trading_hours": "24x7"},
    )
    engine = SyntheticEngine(model="gaussian_rw")
    with pytest.raises(SyntheticContextError):
        engine.generate(request)


def test_default_context_provider_resolves_btc_and_spy():
    provider = DefaultMarketContextProvider()

    btc = provider.resolve(symbol="BTC", interval="1h")
    spy = provider.resolve(symbol="SPY", interval="1d")

    assert (btc.symbol, btc.venue, btc.dataset, btc.asset_class, btc.trading_hours) == (
        "BTCUSDT",
        "binance",
        "trade_bars",
        "crypto",
        "24x7",
    )
    assert (spy.symbol, spy.venue, spy.dataset, spy.asset_class, spy.trading_hours) == (
        "SPY",
        "alpaca",
        "stock_bars",
        "equity",
        "equity_sessions",
    )


def test_callable_context_provider_is_supported():
    def resolve_context(**kwargs):
        return SyntheticContext(
            symbol=kwargs["symbol"],
            venue="test",
            dataset="bars",
            interval=kwargs["interval"] or "1h",
            asset_class="test",
            trading_hours="24x7",
        )

    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=100.0,
        symbol="TEST",
        n_bars=2,
        seed=1,
    )
    batch = SyntheticEngine(model="gaussian_rw", context_provider=resolve_context).generate(request)

    assert batch.context.venue == "test"


def test_fingerprint_candle_table_detects_upload_mutation():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=8,
        seed=42,
    )
    payload = to_backtesting_payload(SyntheticEngine(model="gaussian_rw").generate(request))
    uploaded = payload["candle_table"].copy()

    assert fingerprint_candle_table(payload["candle_table"]) == fingerprint_candle_table(uploaded)

    uploaded.loc[0, "close"] = uploaded.loc[0, "close"] + 1.0
    assert fingerprint_candle_table(payload["candle_table"]) != fingerprint_candle_table(uploaded)


def test_compare_ohlcv_reports_similarity_and_drift():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=64,
        seed=42,
    )
    frame = to_backtesting_payload(SyntheticEngine(model="gaussian_rw").generate(request))["candle_table"]

    same_report = compare_ohlcv(frame, frame)
    assert same_report.passed
    assert same_report.row_counts == {"real": 64, "synthetic": 64}

    shifted = frame.copy()
    shifted["close"] = shifted["close"] * 1.2
    shifted["open"] = shifted["open"] * 1.2
    shifted["high"] = shifted["high"] * 1.2
    shifted["low"] = shifted["low"] * 1.2
    drift_report = compare_ohlcv(frame, shifted)

    assert not drift_report.passed
    assert any("close_median_rel_diff" in warning for warning in drift_report.warnings)


def test_compare_ohlcv_rejects_non_finite_values():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=8,
        seed=42,
    )
    frame = to_backtesting_payload(SyntheticEngine(model="gaussian_rw").generate(request))["candle_table"]
    corrupted = frame.copy()
    corrupted.loc[0, "close"] = float("inf")

    with pytest.raises(Exception):
        compare_ohlcv(frame, corrupted)


def test_compare_ohlcv_treats_row_count_mismatch_as_failure():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=8,
        seed=42,
    )
    frame = to_backtesting_payload(SyntheticEngine(model="gaussian_rw").generate(request))["candle_table"]

    report = compare_ohlcv(frame, frame.iloc[:-1].copy())

    assert not report.passed
    assert report.row_counts == {"real": 8, "synthetic": 7}
    assert any("row_count_mismatch" in warning for warning in report.warnings)


def test_compare_ohlcv_rejects_multi_identity_frames():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=8,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)
    mixed = frame.copy()
    mixed.loc[mixed.index[-1], "symbol"] = "ETHUSDT"

    with pytest.raises(SyntheticValidationError):
        compare_ohlcv(mixed, mixed)


def test_calibrated_params_reduce_range_and_volume_drift():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=256,
        seed=42,
    )
    default_frame = to_backtesting_payload(SyntheticEngine(model="gaussian_rw").generate(request))["candle_table"]
    real_like = default_frame.copy()
    real_like["high"] = real_like[["open", "close"]].max(axis=1) * 1.0002
    real_like["low"] = real_like[["open", "close"]].min(axis=1) * 0.9998
    real_like["volume"] = 750.0

    calibrated_params = calibrate_ohlcv_params(real_like)
    calibrated_request = as_request(
        start_ts=request.start_ts,
        interval=request.interval,
        start_price=request.start_price,
        symbol=request.symbol,
        venue=request.venue,
        dataset=request.dataset,
        n_bars=request.n_bars,
        seed=request.seed,
        model_params=calibrated_params,
    )
    calibrated_frame = to_backtesting_payload(
        SyntheticEngine(model="gaussian_rw").generate(calibrated_request)
    )["candle_table"]

    default_report = compare_ohlcv(real_like, default_frame)
    calibrated_report = compare_ohlcv(real_like, calibrated_frame)

    assert calibrated_params["volume_loc"] == pytest.approx(6.620073, rel=1e-6)
    assert calibrated_report.metrics["volume_median_rel_diff"] < default_report.metrics["volume_median_rel_diff"]
    assert calibrated_report.metrics["range_median_rel_diff"] < default_report.metrics["range_median_rel_diff"]


def test_calibration_sorts_timestamped_frames_before_estimating_returns():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=32,
        seed=42,
    )
    frame = to_backtesting_payload(SyntheticEngine(model="ar1").generate(request))["candle_table"]

    sorted_params = calibrate_ohlcv_params(frame)
    shuffled_params = calibrate_ohlcv_params(frame.sample(frac=1.0, random_state=7))

    assert shuffled_params["drift"] == pytest.approx(sorted_params["drift"])
    assert shuffled_params["volatility"] == pytest.approx(sorted_params["volatility"])
    assert shuffled_params["phi"] == pytest.approx(sorted_params["phi"])


def test_calibration_returns_ar1_innovation_volatility():
    rows = []
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 1000.0
    ret = 0.001
    for idx in range(64):
        ret = 0.8 * ret + 0.001
        open_ = price
        close = price * (1.0 + ret)
        rows.append(
            {
                "timestamp": start + timedelta(hours=idx),
                "open": open_,
                "high": max(open_, close) * 1.001,
                "low": min(open_, close) * 0.999,
                "close": close,
                "volume": 100.0 + idx,
            }
        )
        price = close

    params = calibrate_ohlcv_params(__import__("pandas").DataFrame(rows))

    assert params["phi"] > 0.5
    assert params["ar1_volatility"] < params["volatility"]


def test_calibration_rejects_all_zero_volume():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=8,
        seed=42,
    )
    frame = to_backtesting_payload(SyntheticEngine(model="gaussian_rw").generate(request))["candle_table"]
    frame["volume"] = 0.0

    with pytest.raises(SyntheticValidationError):
        calibrate_ohlcv_params(frame)


def test_calibration_rejects_multi_identity_frames():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=8,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)
    frame.loc[frame.index[-1], "symbol"] = "ETHUSDT"

    with pytest.raises(SyntheticValidationError):
        calibrate_ohlcv_params(frame)


def test_ar1_recurrence_matches_convolution_definition():
    n = 128
    seed = 123
    phi = 0.4
    volatility = 0.02
    drift = 0.001
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    kernels = np.power(phi, np.arange(n, dtype=float))
    expected = np.convolve(volatility * eps, kernels, mode="full")[:n] + drift

    actual = simulate_returns_ar1(n, seed=seed, phi=phi, drift=drift, volatility=volatility)

    np.testing.assert_allclose(actual, expected)
