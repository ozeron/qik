"""Opt-in BTCUSDT 1h realism benchmark.

Run from repo root with:

    QIK_SYNTHETIC_OHLCV_CACHE_PATH=/path/to/ohlcv_catalog/ohlcv \
      uv run --with pandas --with pyarrow python packages/qik-synthetic/benchmarks/evaluate_btc_1h_realism.py
"""

from __future__ import annotations

import os
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from qik.synthetic import (
    SyntheticDependencyError,
    SyntheticEngine,
    RegimeLabelConfig,
    as_request,
    calibrate_ohlcv_params,
    compare_ohlcv,
    fit_ohlcv_profile,
)
from qik.synthetic.regimes import summarize_regime_state

CACHE_ENV = "QIK_SYNTHETIC_OHLCV_CACHE_PATH"
DEFAULT_MODELS = (
    "gaussian_rw",
    "ar1",
    "garch",
    "stationary_bootstrap",
    "student_t_rw",
    "empirical_sv",
    "regime_conditioned_bootstrap",
)
PROFILE_FIT_MODELS = {
    "stationary_bootstrap",
    "empirical_sv",
    "regime_bootstrap",
    "seasonal_bootstrap",
    "regime_conditioned_bootstrap",
}
TARGETS_V1 = {
    "return_std_rel_diff": 0.25,
    "return_robust_std_rel_diff": 0.15,
    "return_iqr_rel_diff": 0.10,
    "return_tail_ratio_rel_diff": 0.20,
    "return_q01_abs_diff": 0.0030,
    "return_q99_abs_diff": 0.0030,
    "return_acf_lag_1_abs_diff": 0.07,
    "abs_return_acf_lag_1_abs_diff": 0.15,
    "squared_return_acf_lag_1_abs_diff": 0.15,
    "rolling_24h_vol_median_rel_diff": 0.08,
    "true_range_median_rel_diff": 0.10,
    "range_q95_rel_diff": 0.20,
    "volume_median_rel_diff": 0.25,
    "volume_acf_lag_1_abs_diff": 0.10,
    "volume_acf_lag_24_abs_diff": 0.12,
    "hourly_volume_profile_distance": 0.75,
    "corr_abs_return_log_volume_abs_diff": 0.05,
    "corr_range_log_volume_abs_diff": 0.08,
}
PRICE_GUARDRAILS = {
    "endpoint_return_abs_diff": 1.50,
    "normalized_close_q05_rel_diff": 0.75,
    "normalized_close_q95_rel_diff": 0.75,
}


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class BenchmarkWindow:
    def __init__(
        self,
        frame: pd.DataFrame,
        symbol: str,
        venue: str,
        dataset: str,
        interval: str,
    ) -> None:
        self.frame = frame
        self.symbol = symbol
        self.venue = venue
        self.dataset = dataset
        self.interval = interval

    @classmethod
    def from_frame(cls, frame: pd.DataFrame):
        return cls(
            frame=frame,
            symbol=_first_identity(frame, "symbol", "BTCUSDT"),
            venue=_first_identity(frame, "venue", "BINANCE"),
            dataset=_first_identity(frame, "dataset", "trade_bars"),
            interval=_first_identity(frame, "interval", "1h"),
        )

    def request_for(self, *, model: str, seed: int, model_params: dict[str, object]):
        return as_request(
            symbol=self.symbol,
            venue=self.venue,
            dataset=self.dataset,
            interval=self.interval,
            start_ts=self.frame["timestamp"].iloc[0].to_pydatetime(),
            start_price=float(self.frame["open"].iloc[0]),
            n_bars=len(self.frame),
            seed=seed,
            model=model,
            model_params=model_params,
        )


class CacheLoadResult:
    def __init__(
        self,
        frame: pd.DataFrame | None,
        status: str,
        reason: str,
        symbol_path: str,
        interval: str,
    ) -> None:
        self.frame = frame
        self.status = status
        self.reason = reason
        self.symbol_path = symbol_path
        self.interval = interval


def ensure_parquet_backend() -> None:
    """Fail early with an actionable message if pandas cannot read parquet."""

    try:
        import pyarrow  # noqa: F401
        return
    except ModuleNotFoundError:
        try:
            import fastparquet  # noqa: F401
            return
        except ModuleNotFoundError as fastparquet_exc:
            raise SystemExit("Reading parquet cache data requires pyarrow or fastparquet") from fastparquet_exc


def load_ohlcv_interval_cache(
    cache_root: Path,
    *,
    symbol_path: str,
    interval: str,
    unavailable: str = "fail",
) -> CacheLoadResult:
    """Load one symbol/interval cache, with explicit skip/fail behavior."""

    if unavailable not in {"fail", "skip"}:
        raise SystemExit("unavailable must be fail or skip")
    data_dir = cache_root / symbol_path / interval
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        reason = f"No {symbol_path}/{interval} parquet files under {cache_root}"
        if unavailable == "skip":
            return CacheLoadResult(None, "skipped", reason, symbol_path, interval)
        raise SystemExit(reason)
    ensure_parquet_backend()
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    try:
        validate_contiguous(frame, interval)
    except SystemExit as exc:
        reason = str(exc)
        if unavailable == "skip":
            return CacheLoadResult(None, "skipped", reason, symbol_path, interval)
        raise
    return CacheLoadResult(frame, "loaded", "", symbol_path, interval)


def load_btcusdt_1h_cache(cache_root: Path) -> pd.DataFrame:
    result = load_ohlcv_interval_cache(cache_root, symbol_path="BTCUSDT.BINANCE", interval="1h")
    if result.frame is None:
        raise SystemExit(result.reason)
    return result.frame


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


def _first_identity(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns:
        return default
    return str(frame[column].iloc[0])


def evaluate_model(
    model: str,
    train: pd.DataFrame,
    reference: pd.DataFrame,
    seeds: tuple[int, ...],
    *,
    price_mode: str = "normalized",
    compact_output: bool = False,
) -> dict[str, object]:
    window = BenchmarkWindow.from_frame(reference)
    params = calibrate_ohlcv_params(train)
    profile = fit_ohlcv_profile(train, regime_config=RegimeLabelConfig(calendar_conditioning="timestamp" in train.columns))
    metrics_by_name: dict[str, list[float]] = {}
    warning_counts = []
    per_seed = []
    regime_diagnostics = []
    category_failures: dict[str, int] = {}
    start = perf_counter()
    engine = SyntheticEngine(model=model, seed=17)
    if model in PROFILE_FIT_MODELS:
        engine.fit_profile(profile)
    if model == "garch":
        engine.fit_frame(train)
    for seed in seeds:
        request = window.request_for(model=model, seed=seed, model_params=params)
        generated = engine.generate_frame(request, compact_identity=True)
        _validate_generated_frame(generated, expected_rows=len(reference))
        report = compare_ohlcv(reference, generated, price_mode=price_mode)
        diagnostics = getattr(engine.model, "generation_diagnostics", None)
        if diagnostics is not None:
            regime_diagnostics.append(
                {
                    "seed": seed,
                    "sampled_regimes": len(getattr(diagnostics, "sampled_regimes", ())),
                    "source_indices": int(getattr(diagnostics, "source_indices", np.zeros(0)).size),
                    "block_starts": tuple(getattr(diagnostics, "block_starts", ())),
                    "relaxation_counts": dict(getattr(diagnostics, "relaxation_counts", {})),
                    "repeated_block_concentration": float(getattr(diagnostics, "repeated_block_concentration", 0.0)),
                }
            )
        seed_summary = {
            "seed": seed,
            "warnings": len(report.warnings),
            "score": score_metrics(report.metrics),
        }
        if not compact_output:
            seed_summary["metrics"] = {name: float(value) for name, value in sorted(report.metrics.items())}
        per_seed.append(seed_summary)
        warning_counts.append(len(report.warnings))
        for category, category_report in (report.categories or {}).items():
            if not category_report.passed:
                category_failures[category] = category_failures.get(category, 0) + 1
        for name, value in report.metrics.items():
            metrics_by_name.setdefault(name, []).append(value)
    elapsed = perf_counter() - start
    metrics = summarize_metrics(metrics_by_name)
    return {
        "model": model,
        "seeds": len(seeds),
        "rows": len(reference),
        "seconds": elapsed,
        "pass_rate": float(np.mean(np.asarray(warning_counts) == 0)),
        "warning_count_median": float(np.median(warning_counts)),
        "category_failures": category_failures,
        "objective_score": score_summary(metrics),
        "target_passes": {
            name: bool(metrics[name]["median"] <= target)
            for name, target in TARGETS_V1.items()
            if name in metrics
        },
        "guardrail_violations": [
            name for name, threshold in PRICE_GUARDRAILS.items() if name in metrics and metrics[name]["median"] > threshold
        ],
        "metrics": metrics,
        "per_seed": per_seed,
        "regime_diagnostics": regime_diagnostics,
        "regime_state": summarize_regime_state(profile.regime_state) if model in PROFILE_FIT_MODELS and profile.regime_state is not None else {},
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


def score_metrics(metrics: dict[str, float]) -> float:
    score = 0.0
    for name, target in TARGETS_V1.items():
        value = float(metrics.get(name, target * 4.0))
        score += min(value / target, 10.0)
    for name, threshold in PRICE_GUARDRAILS.items():
        value = float(metrics.get(name, 0.0))
        if value > threshold:
            score += 5.0 * min((value - threshold) / threshold, 10.0)
    return float(score)


def score_summary(metrics: dict[str, dict[str, float]]) -> float:
    score = 0.0
    for name, target in TARGETS_V1.items():
        summary = metrics.get(name)
        if summary is None:
            score += 10.0
            continue
        median = float(summary["median"])
        instability = max(float(summary["p95"]) - median, 0.0)
        score += min((median + 0.25 * instability) / target, 10.0)
    for name, threshold in PRICE_GUARDRAILS.items():
        summary = metrics.get(name)
        if summary is not None and float(summary["median"]) > threshold:
            score += 5.0 * min((float(summary["median"]) - threshold) / threshold, 10.0)
    return float(score)


def _validate_generated_frame(frame: pd.DataFrame, *, expected_rows: int) -> None:
    if len(frame) != expected_rows:
        raise SystemExit(f"generated row-count mismatch: expected={expected_rows} actual={len(frame)}")
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SystemExit("generated OHLCV contains non-finite values")
    if (numeric[:, :4] <= 0).any() or (numeric[:, 4] < 0).any():
        raise SystemExit("generated OHLCV contains invalid price or volume values")
    if (frame["high"].to_numpy(dtype=float) < frame[["open", "close"]].max(axis=1).to_numpy(dtype=float)).any():
        raise SystemExit("generated OHLCV violates high bound")
    if (frame["low"].to_numpy(dtype=float) > frame[["open", "close"]].min(axis=1).to_numpy(dtype=float)).any():
        raise SystemExit("generated OHLCV violates low bound")


def parse_seeds(raw: str) -> tuple[int, ...]:
    if "-" in raw and "," not in raw:
        start, end = raw.split("-", 1)
        return tuple(range(int(start), int(end) + 1))
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model names")
    parser.add_argument("--seeds", default="101-110", help="Comma-separated seeds or inclusive range like 101-110")
    parser.add_argument("--json-out", default=None, help="Optional path for full JSON benchmark output")
    parser.add_argument("--price-mode", default="normalized", choices=("ignore", "normalized", "anchored"))
    parser.add_argument("--optimize-validation", action="store_true", help="Evaluate all models on validation and only the best validation score on test")
    parser.add_argument("--intervals", default="1h", help="Comma-separated cache intervals to try in order")
    parser.add_argument("--unavailable", default="fail", choices=("fail", "skip"), help="Whether unavailable intervals fail or are reported as skipped")
    parser.add_argument("--max-models", type=int, default=None, help="Limit evaluated models for bounded research sweeps")
    parser.add_argument("--max-seeds", type=int, default=None, help="Limit evaluated seeds for bounded research sweeps")
    parser.add_argument("--compact-output", action="store_true", help="Avoid storing full per-seed metric maps in JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = os.environ.get(CACHE_ENV)
    if not cache:
        raise SystemExit(f"Set {CACHE_ENV} to run cache-backed realism evaluation")
    interval_coverage = []
    frame = None
    selected_interval = None
    for interval in tuple(item.strip() for item in args.intervals.split(",") if item.strip()):
        result = load_ohlcv_interval_cache(
            Path(cache),
            symbol_path="BTCUSDT.BINANCE",
            interval=interval,
            unavailable=args.unavailable,
        )
        interval_coverage.append(
            {
                "symbol_path": result.symbol_path,
                "interval": result.interval,
                "status": result.status,
                "reason": result.reason,
                "rows": 0 if result.frame is None else len(result.frame),
            }
        )
        if result.frame is not None and frame is None:
            frame = result.frame
            selected_interval = interval
    if frame is None:
        raise SystemExit(f"No configured intervals loaded: {interval_coverage}")
    splits = chronological_splits(frame)
    seeds = parse_seeds(args.seeds)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if args.max_seeds is not None:
        seeds = seeds[: max(args.max_seeds, 0)]
    if args.max_models is not None:
        models = models[: max(args.max_models, 0)]
    print("rows", len(frame))
    print("train", len(splits.train), splits.train["timestamp"].iloc[0], splits.train["timestamp"].iloc[-1])
    print("validation", len(splits.validation), splits.validation["timestamp"].iloc[0], splits.validation["timestamp"].iloc[-1])
    print("test", len(splits.test), splits.test["timestamp"].iloc[0], splits.test["timestamp"].iloc[-1])
    real_validation = real_vs_real(splits.train, splits.validation)
    real_test = real_vs_real(splits.train, splits.test)
    print("real_vs_real_validation", real_validation)
    print("real_vs_real_test", real_test)
    results: dict[str, object] = {
        "rows": len(frame),
        "splits": {"train": len(splits.train), "validation": len(splits.validation), "test": len(splits.test)},
        "selected_interval": selected_interval,
        "interval_coverage": interval_coverage,
        "seeds": seeds,
        "target_profile": "v1",
        "targets": TARGETS_V1,
        "price_guardrails": PRICE_GUARDRAILS,
        "real_vs_real_validation": real_validation,
        "real_vs_real_test": real_test,
        "evaluations": {},
    }
    selected_models = available_models(models)
    validation_results = []
    for model in selected_models:
        result = evaluate_model(
            model,
            splits.train,
            splits.validation,
            seeds,
            price_mode=args.price_mode,
            compact_output=args.compact_output,
        )
        validation_results.append(result)
        print("validation", result)
    best_model = min(validation_results, key=lambda item: float(item["objective_score"]))["model"] if validation_results else None
    if args.optimize_validation and best_model is not None:
        test_models = (str(best_model),)
    else:
        test_models = selected_models
    test_results = []
    for model in test_models:
        result = evaluate_model(
            model,
            splits.train,
            splits.test,
            seeds,
            price_mode=args.price_mode,
            compact_output=args.compact_output,
        )
        test_results.append(result)
        print("test", result)
    results["selected_config"] = {"model": best_model, "selection_window": "validation", "score": float(min((float(item["objective_score"]) for item in validation_results), default=float("nan")))}
    results["evaluations"] = {"validation": validation_results, "test": test_results}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
