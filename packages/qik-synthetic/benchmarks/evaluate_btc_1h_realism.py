"""Opt-in BTCUSDT 1h realism benchmark.

Run from repo root with:

    QIK_SYNTHETIC_OHLCV_CACHE_PATH=/path/to/ohlcv_catalog/ohlcv \
      uv run --with pandas --with pyarrow python packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from qik.synthetic import (
    SyntheticDependencyError,
    SyntheticEngine,
    as_request,
    calibrate_ohlcv_params,
    compare_ohlcv,
    fit_ohlcv_profile,
)

CACHE_ENV = "QIK_SYNTHETIC_OHLCV_CACHE_PATH"
DEFAULT_MODELS = ("gaussian_rw", "ar1", "garch", "stationary_bootstrap", "student_t_rw")


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_btcusdt_1h_cache(cache_root: Path) -> pd.DataFrame:
    data_dir = cache_root / "BTCUSDT.BINANCE" / "1h"
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No BTCUSDT.BINANCE/1h parquet files under {cache_root}")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    validate_contiguous(frame, "1h")
    return frame


def validate_contiguous(frame: pd.DataFrame, interval: str) -> None:
    expected = pd.Timedelta(interval)
    gaps = frame["timestamp"].diff().dropna()
    if not (gaps == expected).all():
        raise SystemExit(f"Non-contiguous {interval} cache; largest gap={gaps.max()}")


def chronological_splits(frame: pd.DataFrame, train_frac: float = 0.70, validation_frac: float = 0.15) -> SplitFrames:
    n = len(frame)
    train_end = int(n * train_frac)
    validation_end = int(n * (train_frac + validation_frac))
    if train_end < 2 or validation_end <= train_end or n <= validation_end:
        raise SystemExit("Not enough rows for train/validation/test split")
    return SplitFrames(
        train=frame.iloc[:train_end].reset_index(drop=True),
        validation=frame.iloc[train_end:validation_end].reset_index(drop=True),
        test=frame.iloc[validation_end:].reset_index(drop=True),
    )


def available_models(models: tuple[str, ...] = DEFAULT_MODELS) -> tuple[str, ...]:
    out: list[str] = []
    for model in models:
        try:
            engine = SyntheticEngine(model=model)
            if model == "garch":
                engine.fit([])
        except SyntheticDependencyError:
            print(f"skipping {model}: optional dependency is not installed")
            continue
        out.append(model)
    return tuple(out)


def evaluate_model(model: str, train: pd.DataFrame, reference: pd.DataFrame, seeds: tuple[int, ...]) -> dict[str, object]:
    params = calibrate_ohlcv_params(train)
    profile = fit_ohlcv_profile(train)
    metrics_by_name: dict[str, list[float]] = {}
    warning_counts = []
    category_failures: dict[str, int] = {}
    start = perf_counter()
    engine = SyntheticEngine(model=model, seed=17)
    if model == "stationary_bootstrap":
        engine.fit_profile(profile)
    if model == "garch":
        engine.fit_frame(train)
    for seed in seeds:
        request = as_request(
            symbol=str(reference["symbol"].iloc[0]) if "symbol" in reference.columns else "BTCUSDT",
            venue=str(reference["venue"].iloc[0]) if "venue" in reference.columns else "BINANCE",
            dataset="trade_bars",
            interval="1h",
            start_ts=reference["timestamp"].iloc[0].to_pydatetime(),
            start_price=float(reference["open"].iloc[0]),
            n_bars=len(reference),
            seed=seed,
            model=model,
            model_params=params,
        )
        generated = engine.generate_frame(request, compact_identity=True)
        report = compare_ohlcv(reference, generated, price_mode="normalized")
        warning_counts.append(len(report.warnings))
        for category, category_report in (report.categories or {}).items():
            if not category_report.passed:
                category_failures[category] = category_failures.get(category, 0) + 1
        for name, value in report.metrics.items():
            metrics_by_name.setdefault(name, []).append(value)
    elapsed = perf_counter() - start
    return {
        "model": model,
        "seeds": len(seeds),
        "rows": len(reference),
        "seconds": elapsed,
        "pass_rate": float(np.mean(np.asarray(warning_counts) == 0)),
        "warning_count_median": float(np.median(warning_counts)),
        "category_failures": category_failures,
        "metrics": summarize_metrics(metrics_by_name),
    }


def real_vs_real(train: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    horizon = len(reference)
    first = train.iloc[:horizon].copy() if len(train) >= horizon else train.copy()
    second = reference.iloc[: len(first)].copy()
    report = compare_ohlcv(first, second, price_mode="normalized")
    return {
        "rows": len(first),
        "warnings": len(report.warnings),
        "category_failures": [name for name, item in (report.categories or {}).items() if not item.passed],
    }


def summarize_metrics(metrics_by_name: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
        for name, values in sorted(metrics_by_name.items())
    }


def main() -> None:
    cache = os.environ.get(CACHE_ENV)
    if not cache:
        raise SystemExit(f"Set {CACHE_ENV} to run cache-backed realism evaluation")
    frame = load_btcusdt_1h_cache(Path(cache))
    splits = chronological_splits(frame)
    seeds = tuple(range(101, 111))
    print("rows", len(frame))
    print("train", len(splits.train), splits.train["timestamp"].iloc[0], splits.train["timestamp"].iloc[-1])
    print("validation", len(splits.validation), splits.validation["timestamp"].iloc[0], splits.validation["timestamp"].iloc[-1])
    print("test", len(splits.test), splits.test["timestamp"].iloc[0], splits.test["timestamp"].iloc[-1])
    print("real_vs_real_validation", real_vs_real(splits.train, splits.validation))
    print("real_vs_real_test", real_vs_real(splits.train, splits.test))
    for window_name, reference in (("validation", splits.validation), ("test", splits.test)):
        for model in available_models():
            print(window_name, evaluate_model(model, splits.train, reference, seeds))


if __name__ == "__main__":
    main()
