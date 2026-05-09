from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qik.synthetic import (
    DefaultMarketContextProvider,
    SyntheticContext,
    SyntheticConversionError,
    SyntheticEngine,
    SyntheticContextError,
    SyntheticDependencyError,
    SyntheticValidationError,
    as_request,
    bars_from_span,
    OhlcvCalibrationProfile,
    calibrate_diagnostic_thresholds,
    calibrate_many_ohlcv_params,
    calibrate_ohlcv_params,
    compare_ohlcv,
    compare_ohlcv_by_identity,
    extract_ohlcv_features,
    fit_many_ohlcv_profiles,
    fit_ohlcv_profile,
    list_models,
    fingerprint_candle_table,
    to_backtesting_frame,
    to_backtesting_payload,
    validate_backtesting_payload,
)
from qik.synthetic.kernels import build_ohlcv_from_features, simulate_returns_ar1, stationary_bootstrap_indices

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
    drift_report = compare_ohlcv(frame, shifted, price_mode="anchored")

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


def test_compare_ohlcv_categories_and_price_modes():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=64,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)
    shifted = frame.copy()
    for column in ("open", "high", "low", "close"):
        shifted[column] = shifted[column] * 1.2

    normalized = compare_ohlcv(frame, shifted, price_mode="normalized")
    anchored = compare_ohlcv(frame, shifted, price_mode="anchored")

    assert normalized.categories
    assert set(normalized.categories) >= {"data_quality", "return_distribution", "temporal_dependence", "candle_shape", "volume", "price_path"}
    assert "close_median_rel_diff" not in normalized.metrics
    assert "close_median_rel_diff" in anchored.metrics
    assert any("close_median_rel_diff" in warning for warning in anchored.warnings)


def test_compare_ohlcv_by_identity_preserves_null_identity_group():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        n_bars=8,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)

    reports = compare_ohlcv_by_identity(frame, frame)

    assert len(reports) == 1
    assert next(iter(reports.values())).passed


def test_calibrate_diagnostic_thresholds_returns_metric_thresholds():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=96,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)

    thresholds = calibrate_diagnostic_thresholds(frame, horizon_n=24, n_replicates=4, block_size=8)

    assert "return_std_rel_diff" in thresholds
    assert thresholds["return_std_rel_diff"] >= 0


def test_fit_ohlcv_profile_extracts_features_and_preserves_scalar_params():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=64,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)
    original = frame.copy(deep=True)

    profile = fit_ohlcv_profile(frame)
    params = calibrate_ohlcv_params(frame)
    features = extract_ohlcv_features(frame)

    assert isinstance(profile, OhlcvCalibrationProfile)
    assert profile.row_count == len(frame)
    assert profile.identity["symbol"] == "BTCUSDT"
    assert "close_to_close_return" in features
    assert "ohlcv_features" in params
    pd.testing.assert_frame_equal(frame, original)


def test_fit_many_profiles_and_params_preserve_null_identity_group():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        n_bars=16,
        seed=42,
    )
    frame = SyntheticEngine(model="gaussian_rw").generate_frame(request)

    profiles = fit_many_ohlcv_profiles(frame)
    params = calibrate_many_ohlcv_params(frame)

    assert len(profiles) == 1
    assert len(params) == 1
    assert next(iter(profiles.values())).row_count == 16


def test_build_ohlcv_from_features_preserves_bounds():
    out = build_ohlcv_from_features(
        100.0,
        open_gap_returns=np.zeros(4),
        intrabar_returns=np.array([0.01, -0.02, 0.0, 0.03]),
        upper_wick_rel=np.array([0.01, 0.02, 0.05, 0.01]),
        lower_wick_rel=np.array([0.02, 0.01, 0.03, 0.01]),
        volume=np.array([10.0, 20.0, 30.0, 40.0]),
    )

    assert len(out["close"]) == 4
    assert np.all(out["high"] >= np.maximum(out["open"], out["close"]))
    assert np.all(out["low"] <= np.minimum(out["open"], out["close"]))
    assert np.all(out["volume"] == np.array([10.0, 20.0, 30.0, 40.0]))


def test_stationary_bootstrap_model_generates_seeded_valid_ohlcv():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=64,
        seed=42,
    )
    training = SyntheticEngine(model="gaussian_rw").generate_frame(request)
    profile = fit_ohlcv_profile(training)
    generation_request = as_request(
        start_ts=datetime(2024, 2, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        venue="binance",
        dataset="trade_bars",
        n_bars=32,
        seed=7,
        model="stationary_bootstrap",
    )

    engine = SyntheticEngine(model="stationary_bootstrap").fit_profile(profile)
    first = engine.generate_frame(generation_request)
    second = SyntheticEngine(model="stationary_bootstrap").fit_profile(profile).generate_frame(generation_request)

    assert len(first) == 32
    assert fingerprint_candle_table(first) == fingerprint_candle_table(second)
    assert np.all(first["high"] >= first[["open", "close"]].max(axis=1))
    assert np.all(first["low"] <= first[["open", "close"]].min(axis=1))


def test_stationary_bootstrap_indices_are_seeded_and_bounded():
    first = stationary_bootstrap_indices(10, 20, mean_block_size=4, seed=1)
    second = stationary_bootstrap_indices(10, 20, mean_block_size=4, seed=1)

    assert np.array_equal(first, second)
    assert np.all(first >= 0)
    assert np.all(first < 10)


def test_model_registry_includes_realism_models():
    models = list_models()

    assert "stationary_bootstrap" in models
    assert "student_t_rw" in models


def test_student_t_model_generates_deterministically():
    request = as_request(
        start_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        interval="1h",
        start_price=1000.0,
        symbol="BTCUSDT",
        n_bars=16,
        seed=42,
        model="student_t_rw",
    )
    engine = SyntheticEngine(model="student_t_rw")

    first = engine.generate(request)
    second = engine.generate(request)

    assert first.bars == second.bars


def test_btc_1h_realism_cache_smoke_skips_without_env(monkeypatch):
    monkeypatch.delenv("QIK_SYNTHETIC_OHLCV_CACHE_PATH", raising=False)

    if not __import__("os").environ.get("QIK_SYNTHETIC_OHLCV_CACHE_PATH"):
        pytest.skip("QIK_SYNTHETIC_OHLCV_CACHE_PATH is not set")


def test_garch_preserves_optional_dependency_error_path():
    try:
        SyntheticEngine(model="garch")
    except SyntheticDependencyError:
        return
    pytest.skip("arch optional dependency is installed in this environment")


def test_garch_fit_raises_when_arch_fit_fails(monkeypatch):
    import qik.synthetic.models.garch as garch_module

    class BrokenArchModel:
        def fit(self, *, disp):
            raise RuntimeError("optimizer failed")

    monkeypatch.setattr(garch_module, "arch_model", lambda *args, **kwargs: BrokenArchModel())
    frame = pd.DataFrame({"close": [100.0, 101.0, 100.5, 102.0]})

    with pytest.raises(SyntheticValidationError, match="GARCH fit failed"):
        garch_module.GARCHModel().fit(frame)


def test_garch_simulation_raises_when_arch_simulation_fails(monkeypatch):
    import qik.synthetic.models.garch as garch_module

    class BrokenSimulationModel:
        def simulate(self, params, *, nobs, burn, rng):
            raise RuntimeError("simulation failed")

    class FakeResult:
        params = np.array([1.0])
        model = BrokenSimulationModel()

    monkeypatch.setattr(garch_module, "arch_model", lambda *args, **kwargs: WorkingSimulationModel())
    context = SyntheticContext(
        symbol="BTCUSDT",
        venue="BINANCE",
        dataset="trade_bars",
        interval="1h",
        asset_class="crypto",
        trading_hours="24x7",
    )

    with pytest.raises(SyntheticValidationError, match="GARCH simulation failed"):
        garch_module.GARCHModel(_result=FakeResult()).simulate_returns(context, 4, seed=1)


def test_garch_simulation_uses_supported_arch_signature(monkeypatch):
    import qik.synthetic.models.garch as garch_module

    class WorkingSimulationModel:
        def simulate(self, params, *, nobs, burn):
            return {"data": np.full(nobs, 2.0)}

    class FakeResult:
        params = np.array([1.0])
        model = WorkingSimulationModel()

    monkeypatch.setattr(garch_module, "arch_model", lambda *args, **kwargs: WorkingSimulationModel())
    context = SyntheticContext(
        symbol="BTCUSDT",
        venue="BINANCE",
        dataset="trade_bars",
        interval="1h",
        asset_class="crypto",
        trading_hours="24x7",
    )

    returns = garch_module.GARCHModel(_result=FakeResult()).simulate_returns(context, 4, seed=1)

    assert returns.tolist() == [0.02, 0.02, 0.02, 0.02]


def test_garch_fit_rejects_non_stationary_parameters(monkeypatch):
    import qik.synthetic.models.garch as garch_module

    class NonStationaryArchModel:
        def fit(self, *, disp):
            return type(
                "FakeResult",
                (),
                {"params": pd.Series({"omega": 0.1, "alpha[1]": 0.7, "beta[1]": 0.4})},
            )()

    monkeypatch.setattr(garch_module, "arch_model", lambda *args, **kwargs: NonStationaryArchModel())
    frame = pd.DataFrame({"close": [100.0, 101.0, 100.5, 102.0, 103.0]})

    with pytest.raises(SyntheticValidationError, match="GARCH fit is non-stationary"):
        garch_module.GARCHModel().fit(frame)


def test_garch_uses_feature_candles_when_calibrated(monkeypatch):
    import qik.synthetic.models.garch as garch_module

    class WorkingSimulationModel:
        def simulate(self, params, *, nobs, burn):
            return {"data": np.full(nobs, 1.0)}

    class FakeResult:
        params = pd.Series({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8})
        model = WorkingSimulationModel()

    monkeypatch.setattr(garch_module, "arch_model", lambda *args, **kwargs: WorkingSimulationModel())
    context = SyntheticContext(
        symbol="BTCUSDT",
        venue="BINANCE",
        dataset="trade_bars",
        interval="1h",
        asset_class="crypto",
        trading_hours="24x7",
    )
    features = {
        "upper_wick_rel": np.array([0.01, 0.02, 0.03, 0.04]),
        "lower_wick_rel": np.array([0.01, 0.02, 0.03, 0.04]),
        "volume": np.array([10.0, 20.0, 30.0, 40.0]),
    }

    out = garch_module.GARCHModel(_result=FakeResult()).simulate_batch(
        context,
        4,
        seed=7,
        params={"start_price": 100.0, "ohlcv_features": features},
    )

    assert set(out["volume"].tolist()) <= {10.0, 20.0, 30.0, 40.0}
    assert np.all(out["high"] >= np.maximum(out["open"], out["close"]))
    assert np.all(out["low"] <= np.minimum(out["open"], out["close"]))


def test_garch_fit_excludes_corrupt_extreme_returns(monkeypatch):
    import qik.synthetic.models.garch as garch_module

    captured = {}

    class StationaryArchModel:
        def __init__(self, returns):
            captured["returns"] = np.asarray(returns, dtype=float)

        def fit(self, *, disp):
            return type(
                "FakeResult",
                (),
                {"params": pd.Series({"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8})},
            )()

    monkeypatch.setattr(garch_module, "arch_model", lambda returns, *args, **kwargs: StationaryArchModel(returns))
    frame = pd.DataFrame({"close": [100.0, 101.0, 100_000.0, 102.0, 103.0]})

    model = garch_module.GARCHModel().fit(frame)

    assert captured["returns"].size == 2
    assert model._dropped_return_count == 2


def test_calibrate_ohlcv_params_accepts_timestamp_free_ohlcv():
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )

    params = calibrate_ohlcv_params(frame)

    assert "volatility" in params
    assert "ohlcv_features" in params


def test_realism_benchmark_skips_garch_without_arch(monkeypatch):
    import importlib.util

    path = ROOT / "packages" / "qik-synthetic" / "benchmarks" / "evaluate_btc_1h_realism.py"
    spec = importlib.util.spec_from_file_location("btc_realism_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    monkeypatch.setattr(benchmark, "SyntheticEngine", lambda model, seed=None: (_ for _ in ()).throw(SyntheticDependencyError("missing arch")) if model == "garch" else SyntheticEngine(model=model, seed=seed))

    assert tuple(benchmark.available_models(("gaussian_rw", "garch", "ar1"))) == ("gaussian_rw", "ar1")


def test_large_array_model_params_affect_metadata_without_repr_truncation():
    base = {
        "start_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "interval": "1h",
        "start_price": 1000.0,
        "symbol": "BTCUSDT",
        "n_bars": 2,
        "seed": 42,
    }
    first_array = np.zeros(2000, dtype=float)
    second_array = np.zeros(2000, dtype=float)
    second_array[1000] = 1.0
    assert repr(first_array) == repr(second_array)

    engine = SyntheticEngine(model="gaussian_rw")
    first = engine.generate(as_request(**base, model_params={"calibration_values": first_array}))
    second = engine.generate(as_request(**base, model_params={"calibration_values": second_array}))

    assert first.metadata.request_id != second.metadata.request_id
    assert first.metadata.run_id != second.metadata.run_id


def test_threshold_calibration_preserves_bootstrap_sample_order(monkeypatch):
    import qik.synthetic.diagnostics as diagnostics_module

    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "open": [100.0, 110.0, 90.0, 120.0],
            "high": [101.0, 111.0, 91.0, 121.0],
            "low": [99.0, 109.0, 89.0, 119.0],
            "close": [100.0, 110.0, 90.0, 120.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )
    sample = base.iloc[[2, 0, 1]].copy()
    seen = []

    monkeypatch.setattr(diagnostics_module, "_block_resample", lambda frame, horizon_n, block_size, rng: sample.copy())

    def capture_compare(real, synthetic, **kwargs):
        seen.append(tuple(pd.to_datetime(real["timestamp"], utc=True)))
        return diagnostics_module.OhlcvComparisonReport(
            passed=True,
            metrics={"return_mean_abs_diff": 0.0},
            warnings=(),
            thresholds={},
            row_counts={"real": len(real), "synthetic": len(synthetic)},
        )

    monkeypatch.setattr(diagnostics_module, "compare_ohlcv", capture_compare)

    diagnostics_module.calibrate_diagnostic_thresholds(base, horizon_n=3, n_replicates=1, block_size=2)

    assert seen[0] == tuple(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
