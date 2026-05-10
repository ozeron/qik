"""Regime classification and sampling helpers for OHLCV realism models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .errors import SyntheticValidationError
from .horizon import parse_interval


@dataclass(frozen=True)
class RegimeLabelConfig:
    """Configuration for deterministic, interval-aware regime labels."""

    volatility_window: str = "24h"
    trend_window: str = "7d"
    drawdown_window: str = "30d"
    volume_window: str = "7d"
    block_length: str = "2d"
    min_bucket_count: int = 2
    calendar_conditioning: bool = False
    sparse_policy: str = "strict"


@dataclass(frozen=True)
class RegimeState:
    """Fitted regime labels and compact support metadata."""

    labels: dict[str, np.ndarray]
    composite_labels: np.ndarray
    bucket_support: dict[str, int]
    transition_counts: dict[str, int]
    duration_summary: dict[str, float]
    source_block_starts: tuple[int, ...]
    config: RegimeLabelConfig
    interval: str
    block_bars: int


@dataclass(frozen=True)
class RegimeGenerationDiagnostics:
    """Diagnostics from a regime-conditioned generation run."""

    sampled_regimes: tuple[str, ...] = ()
    source_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    block_starts: tuple[int, ...] = ()
    relaxation_counts: dict[str, int] = field(default_factory=dict)
    repeated_block_concentration: float = 0.0


def bars_for_duration(duration: str, interval: str) -> int:
    """Convert duration strings into at least one bar for a candle interval."""

    duration_step = parse_interval(duration)
    interval_step = parse_interval(interval)
    if interval_step.total_seconds() <= 0:
        raise SyntheticValidationError("interval must be positive")
    return max(1, int(np.ceil(duration_step.total_seconds() / interval_step.total_seconds())))


def fit_regime_state(
    features: dict[str, np.ndarray],
    *,
    interval: str | None,
    timestamps_present: bool,
    config: RegimeLabelConfig | None = None,
) -> RegimeState:
    """Fit deterministic regime labels from extracted OHLCV features."""

    cfg = config or RegimeLabelConfig(calendar_conditioning=timestamps_present)
    if cfg.calendar_conditioning and not timestamps_present:
        raise SyntheticValidationError("calendar conditioning requires timestamp column")
    if interval is None:
        interval = "1h"
    n = _feature_length(features)
    returns = _required_feature(features, "close_to_close_return", n)
    log_volume = np.asarray(features.get("log_volume", np.zeros(n)), dtype=float)
    hours = np.asarray(features.get("hour_of_day", np.zeros(n)), dtype=int)
    days = np.asarray(features.get("day_of_week", np.zeros(n)), dtype=int)

    vol_bars = bars_for_duration(cfg.volatility_window, interval)
    trend_bars = bars_for_duration(cfg.trend_window, interval)
    drawdown_bars = bars_for_duration(cfg.drawdown_window, interval)
    volume_bars = bars_for_duration(cfg.volume_window, interval)
    block_bars = bars_for_duration(cfg.block_length, interval)

    volatility_basis = np.maximum(_rolling_std(returns, vol_bars), np.abs(returns))
    volatility = _quantile_bucket(volatility_basis, (0.30, 0.75, 0.95), prefix="vol")
    trend_signal = _rolling_sum(returns, trend_bars)
    trend = _trend_labels(trend_signal)
    drawdown = _drawdown_labels(returns, drawdown_bars)
    volume = _quantile_bucket(_rolling_median(log_volume, volume_bars), (0.30, 0.75), prefix="volume")
    if cfg.calendar_conditioning:
        calendar = np.asarray([f"d{int(day)}h{int(hour)}" for day, hour in zip(days, hours, strict=True)], dtype=object)
    else:
        calendar = np.full(n, "all", dtype=object)

    labels = {
        "volatility": volatility,
        "trend": trend,
        "drawdown": drawdown,
        "volume": volume,
        "calendar": calendar,
        "hour_of_day": hours.astype(int),
        "day_of_week": days.astype(int),
    }
    composite = np.asarray(
        [
            "|".join((str(volatility[i]), str(trend[i]), str(drawdown[i]), str(volume[i]), str(calendar[i])))
            for i in range(n)
        ],
        dtype=object,
    )
    support = dict(Counter(str(item) for item in composite))
    transitions = _transition_counts(composite)
    durations = _duration_summary(composite)
    starts = tuple(range(0, n, block_bars))
    return RegimeState(
        labels=labels,
        composite_labels=composite,
        bucket_support=support,
        transition_counts=transitions,
        duration_summary=durations,
        source_block_starts=starts,
        config=cfg,
        interval=interval,
        block_bars=block_bars,
    )


def sample_regime_indices(
    state: RegimeState,
    output_n: int,
    *,
    seed: int | None,
    target_timestamps: tuple[datetime, ...] | None,
    sparse_policy: str | None = None,
    offset: int = 0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample source row indices under strict or named relaxed calendar policy."""

    if output_n <= 0:
        return np.zeros(0, dtype=int), {}
    policy = sparse_policy or state.config.sparse_policy
    if policy not in {"strict", "relax_calendar"}:
        raise SyntheticValidationError(f"unsupported regime sparse policy: {policy}")
    source_n = len(state.composite_labels)
    if source_n <= 0:
        raise SyntheticValidationError("regime state has no source rows")
    if target_timestamps is None or not state.config.calendar_conditioning:
        return _stationary_indices(source_n, output_n, state, seed=seed, offset=offset), {}

    target_timestamps = tuple(target_timestamps)
    if len(target_timestamps) != output_n:
        raise SyntheticValidationError("target_timestamps length must match output_n for calendar-conditioned sampling")
    if any(ts is None for ts in target_timestamps):
        raise SyntheticValidationError("target_timestamps must not contain missing values")

    source_hours = np.asarray(state.labels["hour_of_day"], dtype=int)
    source_days = np.asarray(state.labels["day_of_week"], dtype=int)
    by_day_hour = {
        (day, hour): np.flatnonzero((source_days == day) & (source_hours == hour))
        for day in range(7)
        for hour in range(24)
    }
    by_hour = {hour: np.flatnonzero(source_hours == hour) for hour in range(24)}
    out = np.empty(output_n, dtype=int)
    relaxations: dict[str, int] = {}
    for idx, ts in enumerate(target_timestamps):
        day_hour = (int(ts.weekday()), int(ts.hour))
        candidates = by_day_hour.get(day_hour, np.zeros(0, dtype=int))
        if candidates.size == 0:
            if policy == "strict":
                raise SyntheticValidationError(
                    f"sparse regime bucket day={day_hour[0]} hour={day_hour[1]}; "
                    "set regime_sparse_policy='relax_calendar' to allow hour-only sampling"
                )
            candidates = by_hour.get(day_hour[1], np.zeros(0, dtype=int))
            if candidates.size == 0:
                raise SyntheticValidationError(f"sparse regime bucket hour={day_hour[1]}")
            relaxations["relax_calendar"] = relaxations.get("relax_calendar", 0) + 1
        row_rng = np.random.default_rng(_row_seed(seed, offset + idx))
        out[idx] = int(candidates[int(row_rng.integers(0, candidates.size))])
    return out, relaxations


def merge_generation_diagnostics(
    previous: RegimeGenerationDiagnostics | None,
    *,
    state: RegimeState,
    indices: np.ndarray,
    block_start: int,
    relaxation_counts: dict[str, int],
) -> RegimeGenerationDiagnostics:
    """Append one generated block to regime diagnostics."""

    sampled = tuple(str(item) for item in state.composite_labels[indices])
    if previous is None or block_start == 0:
        all_sampled = sampled
        all_indices = np.asarray(indices, dtype=int)
        block_starts = (block_start,)
        counts = dict(relaxation_counts)
    else:
        all_sampled = (*previous.sampled_regimes, *sampled)
        all_indices = np.concatenate([previous.source_indices, np.asarray(indices, dtype=int)])
        block_starts = (*previous.block_starts, block_start)
        counts = dict(previous.relaxation_counts)
        for name, value in relaxation_counts.items():
            counts[name] = counts.get(name, 0) + value
    concentration = _repeated_concentration(all_indices)
    return RegimeGenerationDiagnostics(
        sampled_regimes=all_sampled,
        source_indices=all_indices,
        block_starts=block_starts,
        relaxation_counts=counts,
        repeated_block_concentration=concentration,
    )


def summarize_regime_state(state: RegimeState) -> dict[str, object]:
    """Return compact research metadata for fitted regime support."""

    sparse = {
        bucket: count
        for bucket, count in sorted(state.bucket_support.items())
        if count < state.config.min_bucket_count
    }
    top = dict(Counter(state.bucket_support).most_common(10))
    return {
        "rows": int(len(state.composite_labels)),
        "bucket_count": int(len(state.bucket_support)),
        "top_bucket_support": top,
        "sparse_bucket_count": int(len(sparse)),
        "sparse_buckets": sparse,
        "transition_count": int(sum(state.transition_counts.values())),
        "duration_summary": dict(state.duration_summary),
        "source_block_count": int(len(state.source_block_starts)),
        "interval": state.interval,
        "block_bars": int(state.block_bars),
        "calendar_conditioning": bool(state.config.calendar_conditioning),
        "sparse_policy": state.config.sparse_policy,
    }


def _feature_length(features: dict[str, np.ndarray]) -> int:
    for name in ("close_to_close_return", "intrabar_return", "volume"):
        if name in features:
            return int(len(features[name]))
    raise SyntheticValidationError("regime classification requires fitted OHLCV features")


def _required_feature(features: dict[str, np.ndarray], name: str, n: int) -> np.ndarray:
    if name not in features:
        raise SyntheticValidationError(f"regime classification requires feature '{name}'")
    values = np.asarray(features[name], dtype=float)
    if values.size != n:
        raise SyntheticValidationError(f"regime feature '{name}' length does not match fitted rows")
    return values


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    min_periods = min(2, max(int(window), 1))
    return pd.Series(values).rolling(window=window, min_periods=min_periods).std().bfill().fillna(0.0).to_numpy(dtype=float)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).median().to_numpy(dtype=float)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).sum().to_numpy(dtype=float)


def _quantile_bucket(values: np.ndarray, quantiles: tuple[float, ...], *, prefix: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(values.size, f"{prefix}0", dtype=object)
    thresholds = np.quantile(finite, quantiles)
    buckets = np.digitize(values, thresholds, right=True)
    return np.asarray([f"{prefix}{int(item)}" for item in buckets], dtype=object)


def _trend_labels(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    threshold = float(np.quantile(np.abs(finite), 0.60)) if finite.size else 0.0
    threshold = max(threshold, 1e-12)
    return np.where(values > threshold, "up", np.where(values < -threshold, "down", "flat")).astype(object)


def _drawdown_labels(returns: np.ndarray, window: int) -> np.ndarray:
    path = np.exp(np.cumsum(returns, dtype=float))
    peaks = pd.Series(path).rolling(window=window, min_periods=1).max().to_numpy(dtype=float)
    drawdown = (path / np.maximum(peaks, 1e-12)) - 1.0
    improving = np.r_[False, np.diff(drawdown) > 0]
    out = np.full(returns.size, "shallow", dtype=object)
    out[drawdown < -0.03] = "drawdown"
    out[drawdown < -0.10] = "deep"
    out[(drawdown < -0.03) & improving] = "recovery"
    return out


def _transition_counts(labels: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for left, right in zip(labels[:-1], labels[1:], strict=False):
        key = f"{left}->{right}"
        out[key] = out.get(key, 0) + 1
    return out


def _duration_summary(labels: np.ndarray) -> dict[str, float]:
    if labels.size == 0:
        return {"median": 0.0, "max": 0.0}
    durations: list[int] = []
    current = labels[0]
    run = 1
    for value in labels[1:]:
        if value == current:
            run += 1
        else:
            durations.append(run)
            current = value
            run = 1
    durations.append(run)
    return {"median": float(np.median(durations)), "max": float(max(durations))}


def _stationary_indices(
    source_n: int,
    output_n: int,
    state: RegimeState,
    *,
    seed: int | None,
    offset: int,
) -> np.ndarray:
    if offset < 0:
        raise SyntheticValidationError("regime sample offset must be non-negative")
    restart_probability = 1.0 / max(float(state.block_bars), 1.0)
    total_n = output_n + offset
    rng = np.random.default_rng(seed)
    cursor = int(rng.integers(0, source_n))
    out = np.empty(total_n, dtype=int)
    for idx in range(total_n):
        if idx == 0 or rng.random() < restart_probability:
            cursor = int(rng.integers(0, source_n))
        else:
            cursor = (cursor + 1) % source_n
        out[idx] = cursor
    return out[offset:]


def _repeated_concentration(indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    counts = np.bincount(indices.astype(int))
    return float(counts.max() / indices.size)


def _row_seed(seed: int | None, absolute_index: int) -> int:
    base = 0 if seed is None else int(seed)
    return (base + (absolute_index * 1_000_003)) % (2**63 - 1)
