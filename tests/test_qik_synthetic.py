from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qik.synthetic import (
    DefaultMarketContextProvider,
    SyntheticContext,
    SyntheticConversionError,
    SyntheticEngine,
    SyntheticContextError,
    as_request,
    bars_from_span,
    to_backtesting_payload,
    validate_backtesting_payload,
)

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
