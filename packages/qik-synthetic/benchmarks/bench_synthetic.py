"""Local performance probes for qik-synthetic.

These benchmarks are intentionally not part of normal CI because wall-clock and
RSS numbers vary by machine. Run from the repo root, for example:

    uv run --with pandas --with pyarrow python packages/qik-synthetic/benchmarks/bench_synthetic.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from qik.synthetic import (
    SyntheticEngine,
    as_request,
    calibrate_ohlcv_params,
    compare_ohlcv,
    fingerprint_candle_table,
    to_backtesting_payload,
)


def timed(label: str, fn):
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    print(f"{label}_seconds {elapsed:.6f}")
    return result, elapsed


def main() -> None:
    n_year_1m = 365 * 24 * 60
    request = as_request(
        symbol="BTCUSDT",
        venue="BINANCE",
        dataset="trade_bars",
        interval="1m",
        start_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
        start_price=100_000.0,
        n_bars=n_year_1m,
        seed=20260509,
        model="gaussian_rw",
    )

    engine = SyntheticEngine(model="gaussian_rw", seed=17)
    frame, _ = timed("generate_frame_one_year_1m", lambda: engine.generate_frame(request, compact_identity=True))
    print("frame_shape", tuple(frame.shape))
    print("frame_memory_bytes", int(frame.memory_usage(deep=True).sum()))

    payload, _ = timed("generate_payload_one_year_1m", lambda: engine.generate_backtesting_payload(request))
    print("payload_rows", len(payload["candle_table"]))

    timed("fingerprint_one_year_1m", lambda: fingerprint_candle_table(frame))
    timed("compare_self_one_year_1m", lambda: compare_ohlcv(frame, frame))
    timed("calibrate_one_year_1m", lambda: calibrate_ohlcv_params(frame))

    ar1_request = as_request(
        symbol="BTCUSDT",
        venue="BINANCE",
        dataset="trade_bars",
        interval="1m",
        start_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
        start_price=100_000.0,
        n_bars=n_year_1m,
        seed=20260509,
        model="ar1",
    )
    timed("ar1_generate_frame_one_year_1m", lambda: SyntheticEngine(model="ar1", seed=17).generate_frame(ar1_request))

    batch_request = as_request(
        symbol="BTCUSDT",
        venue="BINANCE",
        dataset="trade_bars",
        interval="1m",
        start_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
        start_price=100_000.0,
        n_bars=100_000,
        seed=20260509,
        model="gaussian_rw",
    )
    batch, _ = timed("legacy_generate_100k", lambda: engine.generate(batch_request))
    timed("legacy_to_backtesting_payload_100k", lambda: to_backtesting_payload(batch))


if __name__ == "__main__":
    main()
