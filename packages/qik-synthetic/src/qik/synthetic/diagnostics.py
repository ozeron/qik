"""Diagnostics for upload fidelity and OHLCV similarity."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Literal

import numpy as np
import pandas as pd

from .backtesting import REQUIRED_COLUMNS
from .errors import SyntheticValidationError


PriceMode = Literal["ignore", "normalized", "anchored"]


DEFAULT_THRESHOLDS: dict[str, float] = {
    "return_mean_abs_diff": 0.002,
    "return_std_rel_diff": 0.50,
    "return_robust_std_rel_diff": 0.50,
    "return_iqr_rel_diff": 0.50,
    "return_q01_abs_diff": 0.03,
    "return_q05_abs_diff": 0.02,
    "return_q50_abs_diff": 0.01,
    "return_q95_abs_diff": 0.02,
    "return_q99_abs_diff": 0.03,
    "return_tail_ratio_rel_diff": 1.00,
    "return_acf_lag_1_abs_diff": 0.25,
    "return_acf_lag_6_abs_diff": 0.25,
    "return_acf_lag_24_abs_diff": 0.25,
    "abs_return_acf_lag_1_abs_diff": 0.30,
    "abs_return_acf_lag_6_abs_diff": 0.30,
    "abs_return_acf_lag_24_abs_diff": 0.30,
    "squared_return_acf_lag_1_abs_diff": 0.30,
    "rolling_24h_vol_median_rel_diff": 0.75,
    "rolling_168h_vol_median_rel_diff": 0.75,
    "open_median_rel_diff": 0.10,
    "close_median_rel_diff": 0.10,
    "close_q05_rel_diff": 0.15,
    "close_q95_rel_diff": 0.15,
    "normalized_close_q05_rel_diff": 0.25,
    "normalized_close_q95_rel_diff": 0.25,
    "endpoint_return_abs_diff": 0.25,
    "range_median_rel_diff": 0.75,
    "range_q95_rel_diff": 1.00,
    "true_range_median_rel_diff": 0.75,
    "upper_wick_median_rel_diff": 1.00,
    "lower_wick_median_rel_diff": 1.00,
    "body_to_range_median_abs_diff": 0.40,
    "volume_median_rel_diff": 1.00,
    "log_volume_iqr_rel_diff": 1.00,
    "log_volume_q95_rel_diff": 1.00,
    "zero_volume_rate_abs_diff": 0.05,
    "volume_acf_lag_1_abs_diff": 0.30,
    "volume_acf_lag_24_abs_diff": 0.30,
    "corr_abs_return_log_volume_abs_diff": 0.35,
    "corr_range_log_volume_abs_diff": 0.35,
    "hourly_volume_profile_distance": 1.00,
}


CATEGORY_METRICS: dict[str, tuple[str, ...]] = {
    "data_quality": ("row_count_mismatch",),
    "return_distribution": (
        "return_mean_abs_diff",
        "return_std_rel_diff",
        "return_robust_std_rel_diff",
        "return_iqr_rel_diff",
        "return_q01_abs_diff",
        "return_q05_abs_diff",
        "return_q50_abs_diff",
        "return_q95_abs_diff",
        "return_q99_abs_diff",
        "return_tail_ratio_rel_diff",
    ),
    "temporal_dependence": (
        "return_acf_lag_1_abs_diff",
        "return_acf_lag_6_abs_diff",
        "return_acf_lag_24_abs_diff",
        "abs_return_acf_lag_1_abs_diff",
        "abs_return_acf_lag_6_abs_diff",
        "abs_return_acf_lag_24_abs_diff",
        "squared_return_acf_lag_1_abs_diff",
        "rolling_24h_vol_median_rel_diff",
        "rolling_168h_vol_median_rel_diff",
    ),
    "candle_shape": (
        "range_median_rel_diff",
        "range_q95_rel_diff",
        "true_range_median_rel_diff",
        "upper_wick_median_rel_diff",
        "lower_wick_median_rel_diff",
        "body_to_range_median_abs_diff",
    ),
    "volume": (
        "volume_median_rel_diff",
        "log_volume_iqr_rel_diff",
        "log_volume_q95_rel_diff",
        "zero_volume_rate_abs_diff",
        "volume_acf_lag_1_abs_diff",
        "volume_acf_lag_24_abs_diff",
        "corr_abs_return_log_volume_abs_diff",
        "corr_range_log_volume_abs_diff",
        "hourly_volume_profile_distance",
    ),
    "price_path": (
        "open_median_rel_diff",
        "close_median_rel_diff",
        "close_q05_rel_diff",
        "close_q95_rel_diff",
        "normalized_close_q05_rel_diff",
        "normalized_close_q95_rel_diff",
        "endpoint_return_abs_diff",
    ),
}


@dataclass(frozen=True)
class OhlcvCategoryReport:
    """Per-category diagnostic result."""

    passed: bool
    metrics: dict[str, float]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class OhlcvComparisonReport:
    """Summary of reference-vs-synthetic candle similarity checks."""

    passed: bool
    metrics: dict[str, float]
    warnings: tuple[str, ...]
    thresholds: dict[str, float]
    row_counts: dict[str, int]
    categories: dict[str, OhlcvCategoryReport] | None = None
    price_mode: str = "normalized"


def fingerprint_candle_table(
    table: pd.DataFrame,
    *,
    columns: tuple[str, ...] = REQUIRED_COLUMNS,
    identity_columns: tuple[str, ...] = ("symbol", "venue", "dataset", "interval"),
) -> str:
    """Return a stable hash for exact candle-table handoff verification."""

    selected = _canonical_frame(table, columns=columns, identity_columns=identity_columns)
    digest = blake2b(digest_size=16)
    for column in selected.columns:
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        if column == "timestamp":
            values = pd.to_datetime(selected[column], utc=True).astype("int64").to_numpy()
            digest.update(values.astype("<i8", copy=False).tobytes())
        elif column in {"open", "high", "low", "close", "volume"}:
            values = selected[column].to_numpy(dtype=float)
            digest.update(values.astype("<f8", copy=False).tobytes())
        else:
            for value in selected[column].astype(str):
                encoded = str(value).encode("utf-8")
                digest.update(len(encoded).to_bytes(4, "little"))
                digest.update(encoded)
    return digest.hexdigest()


def compare_ohlcv(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    price_mode: PriceMode = "normalized",
) -> OhlcvComparisonReport:
    """Compare real/reference and synthetic OHLCV distributions."""

    if price_mode not in {"ignore", "normalized", "anchored"}:
        raise SyntheticValidationError("price_mode must be ignore, normalized, or anchored")
    _reject_multi_identity(real)
    _reject_multi_identity(synthetic)
    reference = _prepare_ohlcv(real)
    candidate = _prepare_ohlcv(synthetic)
    if len(reference) < 2 or len(candidate) < 2:
        raise SyntheticValidationError("compare_ohlcv requires at least two rows in each table")

    effective_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    real_stats = _distribution_stats(reference)
    synthetic_stats = _distribution_stats(candidate)
    metrics = _comparison_metrics(real_stats, synthetic_stats, price_mode=price_mode)
    warnings = tuple(
        f"{name}={value:.6g} exceeds threshold {effective_thresholds[name]:.6g}"
        for name, value in metrics.items()
        if name in effective_thresholds and value > effective_thresholds[name]
    )
    if len(reference) != len(candidate):
        metrics["row_count_mismatch"] = float(abs(len(reference) - len(candidate)))
        warnings = (
            f"row_count_mismatch real={len(reference)} synthetic={len(candidate)}",
            *warnings,
        )
    categories = _category_reports(metrics, warnings, effective_thresholds)

    return OhlcvComparisonReport(
        passed=not warnings,
        metrics=metrics,
        warnings=warnings,
        thresholds=effective_thresholds,
        row_counts={"real": int(len(reference)), "synthetic": int(len(candidate))},
        categories=categories,
        price_mode=price_mode,
    )


def compare_ohlcv_by_identity(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("symbol", "venue", "dataset", "interval"),
    thresholds: dict[str, float] | None = None,
    price_mode: PriceMode = "normalized",
) -> dict[tuple[object, ...], OhlcvComparisonReport]:
    """Compare OHLCV frames per identity group."""

    real_groups = _groups_by_identity(real, group_columns)
    synthetic_groups = _groups_by_identity(synthetic, group_columns)
    reports: dict[tuple[object, ...], OhlcvComparisonReport] = {}
    for key in sorted(set(real_groups) | set(synthetic_groups), key=repr):
        if key not in real_groups:
            reports[key] = _missing_group_report("real", len(synthetic_groups[key]), thresholds, price_mode)
            continue
        if key not in synthetic_groups:
            reports[key] = _missing_group_report("synthetic", len(real_groups[key]), thresholds, price_mode)
            continue
        reports[key] = compare_ohlcv(real_groups[key], synthetic_groups[key], thresholds=thresholds, price_mode=price_mode)
    return reports


def calibrate_diagnostic_thresholds(
    reference: pd.DataFrame,
    *,
    horizon_n: int,
    n_replicates: int = 32,
    block_size: int = 48,
    quantile: float = 0.95,
    seed: int | None = 1,
    price_mode: PriceMode = "normalized",
) -> dict[str, float]:
    """Estimate metric thresholds from real-vs-real block-resampled comparisons."""

    prepared = _prepare_ohlcv(reference)
    if horizon_n < 2:
        raise SyntheticValidationError("horizon_n must be at least 2")
    if len(prepared) < horizon_n:
        raise SyntheticValidationError("reference must contain at least horizon_n rows")
    rng = np.random.default_rng(seed)
    metrics_by_name: dict[str, list[float]] = {}
    template_timestamps = prepared["timestamp"].iloc[:horizon_n].reset_index(drop=True)
    for _ in range(max(n_replicates, 1)):
        first = _with_sample_order_timestamps(_block_resample(prepared, horizon_n, block_size, rng), template_timestamps)
        second = _with_sample_order_timestamps(_block_resample(prepared, horizon_n, block_size, rng), template_timestamps)
        report = compare_ohlcv(first, second, thresholds={}, price_mode=price_mode)
        for name, value in report.metrics.items():
            metrics_by_name.setdefault(name, []).append(value)
    return {
        name: max(float(np.quantile(values, quantile)), DEFAULT_THRESHOLDS.get(name, 0.0))
        for name, values in metrics_by_name.items()
    }


def _canonical_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    identity_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SyntheticValidationError("missing required columns: " + ", ".join(missing))

    selected_columns = list(columns) + [column for column in identity_columns if column in frame.columns]
    selected = frame[selected_columns].copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], utc=True)
    selected = selected.sort_values(["timestamp", *[column for column in identity_columns if column in selected.columns]])
    selected = selected.reset_index(drop=True)

    for column in ("open", "high", "low", "close", "volume"):
        selected[column] = selected[column].astype(float).round(12)
    return selected


def _prepare_ohlcv(table: pd.DataFrame) -> pd.DataFrame:
    frame = _canonical_frame(table, columns=REQUIRED_COLUMNS, identity_columns=tuple())
    prices = frame[["open", "high", "low", "close"]]
    numeric = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise SyntheticValidationError("OHLCV values must be finite")
    if not (prices > 0).all().all():
        raise SyntheticValidationError("OHLC prices must be positive")
    if not (frame["volume"] >= 0).all():
        raise SyntheticValidationError("volume must be non-negative")
    if not (frame["low"] <= frame[["open", "close"]].min(axis=1)).all():
        raise SyntheticValidationError("low must be <= min(open, close)")
    if not (frame["high"] >= frame[["open", "close"]].max(axis=1)).all():
        raise SyntheticValidationError("high must be >= max(open, close)")
    return frame


def _reject_multi_identity(table: pd.DataFrame) -> None:
    identity_columns = [column for column in ("symbol", "venue", "dataset", "interval") if column in table.columns]
    if not identity_columns:
        return
    if table[identity_columns].drop_duplicates().shape[0] > 1:
        raise SyntheticValidationError("compare_ohlcv expects a single identity; group multi-symbol frames first")


def _groups_by_identity(table: pd.DataFrame, group_columns: tuple[str, ...]) -> dict[tuple[object, ...], pd.DataFrame]:
    available = [column for column in group_columns if column in table.columns]
    if not available:
        return {tuple(): table}
    out: dict[tuple[object, ...], pd.DataFrame] = {}
    for key, group in table.groupby(available, sort=True, observed=True, dropna=False):
        out[_normalize_group_key(key)] = group
    return out


def _normalize_group_key(key: object) -> tuple[object, ...]:
    raw = key if isinstance(key, tuple) else (key,)
    return tuple(None if pd.isna(value) else value for value in raw)


def _missing_group_report(missing_side: str, present_count: int, thresholds: dict[str, float] | None, price_mode: str):
    effective_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if missing_side == "real":
        row_counts = {"real": 0, "synthetic": int(present_count)}
    else:
        row_counts = {"real": int(present_count), "synthetic": 0}
    categories = {
        "data_quality": OhlcvCategoryReport(
            passed=False,
            metrics={},
            warnings=(f"missing_identity_group side={missing_side}",),
        )
    }
    return OhlcvComparisonReport(
        passed=False,
        metrics={},
        warnings=(f"missing_identity_group side={missing_side}",),
        thresholds=effective_thresholds,
        row_counts=row_counts,
        categories=categories,
        price_mode=price_mode,
    )


def _distribution_stats(frame: pd.DataFrame) -> dict[str, float]:
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    returns = np.diff(np.log(close))
    abs_returns = np.abs(returns)
    squared_returns = returns * returns
    candle_range = (high - low) / close
    true_range = _true_range(open_, high, low, close) / close
    upper_wick = (high - np.maximum(open_, close)) / close
    lower_wick = (np.minimum(open_, close) - low) / close
    body = np.abs(close - open_) / close
    body_to_range = body / np.maximum(candle_range, 1e-12)
    positive_volume = np.maximum(volume, 1e-12)
    log_volume = np.log(positive_volume)
    normalized_close = close / close[0]
    return_quantiles = np.quantile(returns, [0.01, 0.05, 0.50, 0.95, 0.99])
    close_quantiles = np.quantile(close, [0.05, 0.95])
    normalized_close_quantiles = np.quantile(normalized_close, [0.05, 0.95])
    log_volume_quantiles = np.quantile(log_volume, [0.25, 0.75, 0.95])
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_robust_std": _robust_std(returns),
        "return_iqr": float(return_quantiles[3] - return_quantiles[1]),
        "return_q01": float(return_quantiles[0]),
        "return_q05": float(return_quantiles[1]),
        "return_q50": float(return_quantiles[2]),
        "return_q95": float(return_quantiles[3]),
        "return_q99": float(return_quantiles[4]),
        "return_tail_ratio": _tail_ratio(returns),
        "return_acf_lag_1": _acf(returns, 1),
        "return_acf_lag_6": _acf(returns, 6),
        "return_acf_lag_24": _acf(returns, 24),
        "abs_return_acf_lag_1": _acf(abs_returns, 1),
        "abs_return_acf_lag_6": _acf(abs_returns, 6),
        "abs_return_acf_lag_24": _acf(abs_returns, 24),
        "squared_return_acf_lag_1": _acf(squared_returns, 1),
        "rolling_24h_vol_median": _rolling_vol_median(returns, 24),
        "rolling_168h_vol_median": _rolling_vol_median(returns, 168),
        "open_median": float(np.median(open_)),
        "close_median": float(np.median(close)),
        "close_q05": float(close_quantiles[0]),
        "close_q95": float(close_quantiles[1]),
        "normalized_close_q05": float(normalized_close_quantiles[0]),
        "normalized_close_q95": float(normalized_close_quantiles[1]),
        "endpoint_return": float(np.log(close[-1] / close[0])),
        "range_median": float(np.median(candle_range)),
        "range_q95": float(np.quantile(candle_range, 0.95)),
        "true_range_median": float(np.median(true_range)),
        "upper_wick_median": float(np.median(upper_wick)),
        "lower_wick_median": float(np.median(lower_wick)),
        "body_to_range_median": float(np.median(body_to_range)),
        "volume_median": float(np.median(volume)),
        "log_volume_iqr": float(log_volume_quantiles[1] - log_volume_quantiles[0]),
        "log_volume_q95": float(log_volume_quantiles[2]),
        "zero_volume_rate": float(np.mean(volume == 0.0)),
        "volume_acf_lag_1": _acf(log_volume, 1),
        "volume_acf_lag_24": _acf(log_volume, 24),
        "corr_abs_return_log_volume": _corr(_pad_to_len(abs_returns, len(log_volume)), log_volume),
        "corr_range_log_volume": _corr(candle_range, log_volume),
        "hourly_volume_profile": _hourly_profile(frame["timestamp"], log_volume),
    }


def _comparison_metrics(real: dict[str, float], synthetic: dict[str, float], *, price_mode: str) -> dict[str, float]:
    metrics = {
        "return_mean_abs_diff": abs(synthetic["return_mean"] - real["return_mean"]),
        "return_std_rel_diff": _relative_diff(synthetic["return_std"], real["return_std"]),
        "return_robust_std_rel_diff": _relative_diff(synthetic["return_robust_std"], real["return_robust_std"]),
        "return_iqr_rel_diff": _relative_diff(synthetic["return_iqr"], real["return_iqr"]),
        "return_q01_abs_diff": abs(synthetic["return_q01"] - real["return_q01"]),
        "return_q05_abs_diff": abs(synthetic["return_q05"] - real["return_q05"]),
        "return_q50_abs_diff": abs(synthetic["return_q50"] - real["return_q50"]),
        "return_q95_abs_diff": abs(synthetic["return_q95"] - real["return_q95"]),
        "return_q99_abs_diff": abs(synthetic["return_q99"] - real["return_q99"]),
        "return_tail_ratio_rel_diff": _relative_diff(synthetic["return_tail_ratio"], real["return_tail_ratio"]),
        "return_acf_lag_1_abs_diff": abs(synthetic["return_acf_lag_1"] - real["return_acf_lag_1"]),
        "return_acf_lag_6_abs_diff": abs(synthetic["return_acf_lag_6"] - real["return_acf_lag_6"]),
        "return_acf_lag_24_abs_diff": abs(synthetic["return_acf_lag_24"] - real["return_acf_lag_24"]),
        "abs_return_acf_lag_1_abs_diff": abs(synthetic["abs_return_acf_lag_1"] - real["abs_return_acf_lag_1"]),
        "abs_return_acf_lag_6_abs_diff": abs(synthetic["abs_return_acf_lag_6"] - real["abs_return_acf_lag_6"]),
        "abs_return_acf_lag_24_abs_diff": abs(synthetic["abs_return_acf_lag_24"] - real["abs_return_acf_lag_24"]),
        "squared_return_acf_lag_1_abs_diff": abs(synthetic["squared_return_acf_lag_1"] - real["squared_return_acf_lag_1"]),
        "rolling_24h_vol_median_rel_diff": _relative_diff(synthetic["rolling_24h_vol_median"], real["rolling_24h_vol_median"]),
        "rolling_168h_vol_median_rel_diff": _relative_diff(synthetic["rolling_168h_vol_median"], real["rolling_168h_vol_median"]),
        "range_median_rel_diff": _relative_diff(synthetic["range_median"], real["range_median"]),
        "range_q95_rel_diff": _relative_diff(synthetic["range_q95"], real["range_q95"]),
        "true_range_median_rel_diff": _relative_diff(synthetic["true_range_median"], real["true_range_median"]),
        "upper_wick_median_rel_diff": _relative_diff(synthetic["upper_wick_median"], real["upper_wick_median"]),
        "lower_wick_median_rel_diff": _relative_diff(synthetic["lower_wick_median"], real["lower_wick_median"]),
        "body_to_range_median_abs_diff": abs(synthetic["body_to_range_median"] - real["body_to_range_median"]),
        "volume_median_rel_diff": _relative_diff(synthetic["volume_median"], real["volume_median"]),
        "log_volume_iqr_rel_diff": _relative_diff(synthetic["log_volume_iqr"], real["log_volume_iqr"]),
        "log_volume_q95_rel_diff": _relative_diff(synthetic["log_volume_q95"], real["log_volume_q95"]),
        "zero_volume_rate_abs_diff": abs(synthetic["zero_volume_rate"] - real["zero_volume_rate"]),
        "volume_acf_lag_1_abs_diff": abs(synthetic["volume_acf_lag_1"] - real["volume_acf_lag_1"]),
        "volume_acf_lag_24_abs_diff": abs(synthetic["volume_acf_lag_24"] - real["volume_acf_lag_24"]),
        "corr_abs_return_log_volume_abs_diff": abs(synthetic["corr_abs_return_log_volume"] - real["corr_abs_return_log_volume"]),
        "corr_range_log_volume_abs_diff": abs(synthetic["corr_range_log_volume"] - real["corr_range_log_volume"]),
        "hourly_volume_profile_distance": float(np.linalg.norm(synthetic["hourly_volume_profile"] - real["hourly_volume_profile"])),
    }
    if price_mode == "normalized":
        metrics.update(
            {
                "normalized_close_q05_rel_diff": _relative_diff(synthetic["normalized_close_q05"], real["normalized_close_q05"]),
                "normalized_close_q95_rel_diff": _relative_diff(synthetic["normalized_close_q95"], real["normalized_close_q95"]),
                "endpoint_return_abs_diff": abs(synthetic["endpoint_return"] - real["endpoint_return"]),
            }
        )
    elif price_mode == "anchored":
        metrics.update(
            {
                "open_median_rel_diff": _relative_diff(synthetic["open_median"], real["open_median"]),
                "close_median_rel_diff": _relative_diff(synthetic["close_median"], real["close_median"]),
                "close_q05_rel_diff": _relative_diff(synthetic["close_q05"], real["close_q05"]),
                "close_q95_rel_diff": _relative_diff(synthetic["close_q95"], real["close_q95"]),
                "endpoint_return_abs_diff": abs(synthetic["endpoint_return"] - real["endpoint_return"]),
            }
        )
    return metrics


def _category_reports(metrics: dict[str, float], warnings: tuple[str, ...], thresholds: dict[str, float]) -> dict[str, OhlcvCategoryReport]:
    out: dict[str, OhlcvCategoryReport] = {}
    for category, names in CATEGORY_METRICS.items():
        category_metrics = {name: metrics[name] for name in names if name in metrics}
        category_warnings = tuple(warning for warning in warnings if any(warning.startswith(name) for name in names))
        if category == "data_quality":
            category_warnings = tuple(warning for warning in warnings if "row_count_mismatch" in warning or "missing_identity_group" in warning)
        out[category] = OhlcvCategoryReport(
            passed=not category_warnings,
            metrics=category_metrics,
            warnings=category_warnings,
        )
    return out


def _relative_diff(candidate: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-12)
    return abs(candidate - reference) / denominator


def _robust_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float((q75 - q25) / 1.349)


def _tail_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    q95 = np.quantile(np.abs(values), 0.95)
    q50 = np.quantile(np.abs(values), 0.50)
    return float(q95 / max(q50, 1e-12))


def _acf(values: np.ndarray, lag: int) -> float:
    if values.size <= lag or lag <= 0:
        return 0.0
    x = values[:-lag]
    y = values[lag:]
    return _corr(x, y)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0 or y_std <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rolling_vol_median(returns: np.ndarray, window: int) -> float:
    if returns.size < 2:
        return 0.0
    if returns.size < window:
        return float(np.std(returns))
    values = pd.Series(returns).rolling(window=window).std().dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float(np.std(returns))
    return float(np.median(values))


def _true_range(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.empty_like(close)
    prev_close[0] = open_[0]
    prev_close[1:] = close[:-1]
    return np.maximum(high, prev_close) - np.minimum(low, prev_close)


def _pad_to_len(values: np.ndarray, n: int) -> np.ndarray:
    if values.size == n:
        return values
    if values.size == n - 1:
        return np.concatenate([[values[0] if values.size else 0.0], values])
    if values.size > n:
        return values[:n]
    return np.pad(values, (0, n - values.size))


def _hourly_profile(timestamps: pd.Series, values: np.ndarray) -> np.ndarray:
    hours = pd.to_datetime(timestamps, utc=True).dt.hour.to_numpy()
    out = np.zeros(24, dtype=float)
    for hour in range(24):
        mask = hours == hour
        out[hour] = float(np.median(values[mask])) if mask.any() else 0.0
    center = np.median(out)
    return out - center


def _block_resample(frame: pd.DataFrame, horizon_n: int, block_size: int, rng: np.random.Generator) -> pd.DataFrame:
    blocks = []
    while sum(len(block) for block in blocks) < horizon_n:
        start = int(rng.integers(0, max(len(frame) - 1, 1)))
        end = min(start + max(block_size, 1), len(frame))
        blocks.append(frame.iloc[start:end])
    return pd.concat(blocks, ignore_index=True).iloc[:horizon_n].copy()


def _with_sample_order_timestamps(frame: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    out["timestamp"] = pd.to_datetime(timestamps.iloc[: len(out)], utc=True).to_numpy()
    return out
