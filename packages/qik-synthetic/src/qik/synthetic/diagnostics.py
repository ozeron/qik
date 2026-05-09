"""Diagnostics for upload fidelity and OHLCV similarity."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
import pandas as pd

from .backtesting import REQUIRED_COLUMNS
from .errors import SyntheticValidationError


DEFAULT_THRESHOLDS: dict[str, float] = {
    "return_mean_abs_diff": 0.002,
    "return_std_rel_diff": 0.50,
    "return_q01_abs_diff": 0.03,
    "return_q05_abs_diff": 0.02,
    "return_q95_abs_diff": 0.02,
    "return_q99_abs_diff": 0.03,
    "open_median_rel_diff": 0.10,
    "close_median_rel_diff": 0.10,
    "close_q05_rel_diff": 0.15,
    "close_q95_rel_diff": 0.15,
    "range_median_rel_diff": 0.75,
    "volume_median_rel_diff": 1.00,
}


@dataclass(frozen=True)
class OhlcvComparisonReport:
    """Summary of reference-vs-synthetic candle similarity checks."""

    passed: bool
    metrics: dict[str, float]
    warnings: tuple[str, ...]
    thresholds: dict[str, float]
    row_counts: dict[str, int]


def fingerprint_candle_table(
    table: pd.DataFrame,
    *,
    columns: tuple[str, ...] = REQUIRED_COLUMNS,
    identity_columns: tuple[str, ...] = ("symbol", "venue", "dataset", "interval"),
) -> str:
    """Return a stable hash for exact candle-table handoff verification."""

    selected = _canonical_frame(table, columns=columns, identity_columns=identity_columns)
    payload = selected.to_json(
        orient="split",
        date_format="iso",
        double_precision=12,
        index=False,
    )
    return blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def compare_ohlcv(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
) -> OhlcvComparisonReport:
    """Compare real/reference and synthetic OHLCV distributions."""

    reference = _prepare_ohlcv(real)
    candidate = _prepare_ohlcv(synthetic)
    if len(reference) < 2 or len(candidate) < 2:
        raise SyntheticValidationError("compare_ohlcv requires at least two rows in each table")

    effective_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    real_stats = _distribution_stats(reference)
    synthetic_stats = _distribution_stats(candidate)
    metrics = _comparison_metrics(real_stats, synthetic_stats)
    warnings = tuple(
        f"{name}={value:.6g} exceeds threshold {effective_thresholds[name]:.6g}"
        for name, value in metrics.items()
        if name in effective_thresholds and value > effective_thresholds[name]
    )
    if len(reference) != len(candidate):
        warnings = (
            f"row_count_mismatch real={len(reference)} synthetic={len(candidate)}",
            *warnings,
        )

    return OhlcvComparisonReport(
        passed=not warnings,
        metrics=metrics,
        warnings=warnings,
        thresholds=effective_thresholds,
        row_counts={"real": int(len(reference)), "synthetic": int(len(candidate))},
    )


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


def _distribution_stats(frame: pd.DataFrame) -> dict[str, float]:
    close = frame["close"].to_numpy(dtype=float)
    returns = np.diff(np.log(close))
    candle_range = ((frame["high"] - frame["low"]) / frame["close"]).to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_q01": float(np.quantile(returns, 0.01)),
        "return_q05": float(np.quantile(returns, 0.05)),
        "return_q95": float(np.quantile(returns, 0.95)),
        "return_q99": float(np.quantile(returns, 0.99)),
        "open_median": float(np.median(frame["open"].to_numpy(dtype=float))),
        "close_median": float(np.median(close)),
        "close_q05": float(np.quantile(close, 0.05)),
        "close_q95": float(np.quantile(close, 0.95)),
        "range_median": float(np.median(candle_range)),
        "volume_median": float(np.median(volume)),
    }


def _comparison_metrics(real: dict[str, float], synthetic: dict[str, float]) -> dict[str, float]:
    return {
        "return_mean_abs_diff": abs(synthetic["return_mean"] - real["return_mean"]),
        "return_std_rel_diff": _relative_diff(synthetic["return_std"], real["return_std"]),
        "return_q01_abs_diff": abs(synthetic["return_q01"] - real["return_q01"]),
        "return_q05_abs_diff": abs(synthetic["return_q05"] - real["return_q05"]),
        "return_q95_abs_diff": abs(synthetic["return_q95"] - real["return_q95"]),
        "return_q99_abs_diff": abs(synthetic["return_q99"] - real["return_q99"]),
        "open_median_rel_diff": _relative_diff(synthetic["open_median"], real["open_median"]),
        "close_median_rel_diff": _relative_diff(synthetic["close_median"], real["close_median"]),
        "close_q05_rel_diff": _relative_diff(synthetic["close_q05"], real["close_q05"]),
        "close_q95_rel_diff": _relative_diff(synthetic["close_q95"], real["close_q95"]),
        "range_median_rel_diff": _relative_diff(synthetic["range_median"], real["range_median"]),
        "volume_median_rel_diff": _relative_diff(synthetic["volume_median"], real["volume_median"]),
    }


def _relative_diff(candidate: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-12)
    return abs(candidate - reference) / denominator
