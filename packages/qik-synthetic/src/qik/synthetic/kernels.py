"""Vectorized numerical kernels for synthetic generation."""

from __future__ import annotations

from datetime import datetime
import numpy as np

from .errors import SyntheticValidationError


def simulate_log_returns(
    n: int,
    *,
    seed: int | None,
    drift: float,
    volatility: float,
    noise_scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return drift + (volatility * noise_scale * rng.normal(size=n))


def simulate_returns_ar1(
    n: int,
    *,
    seed: int | None,
    phi: float,
    drift: float,
    volatility: float,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)

    if abs(phi) >= 1:
        raise ValueError("phi must satisfy |phi| < 1")

    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    out = np.empty(n, dtype=float)
    state = 0.0
    for idx, shock in enumerate(eps):
        state = (phi * state) + (volatility * shock)
        out[idx] = state
    if drift != 0.0:
        out = out + drift
    return out


def build_ohlcv(
    start_price: float,
    log_returns: np.ndarray,
    *,
    seed: int | None = None,
    wick_scale: float = 1.0,
    volume_loc: float = 12.0,
    volume_scale: float = 0.35,
) -> dict[str, np.ndarray]:
    close = np.cumsum(log_returns, dtype=np.float64)
    np.exp(close, out=close)
    close *= start_price
    if close.size == 0:
        empty = np.array([], dtype=float)
        return {"open": empty, "high": empty, "low": empty, "close": empty, "volume": empty}

    open_price = np.empty_like(close)
    open_price[0] = start_price
    open_price[1:] = close[:-1]

    rng = np.random.default_rng(seed)
    raw_width = rng.normal(loc=0.0, scale=0.35, size=close.size)
    np.abs(raw_width, out=raw_width)
    wick = close - open_price
    np.abs(wick, out=wick)
    np.maximum(wick, 1e-12, out=wick)
    wick *= 1.0 + raw_width
    wick *= max(float(wick_scale), 0.0)
    high = np.maximum(open_price, close)
    high += wick
    low = np.minimum(open_price, close)
    low -= wick
    np.maximum(low, 1e-12, out=low)
    volume = rng.normal(loc=float(volume_loc), scale=max(float(volume_scale), 0.0), size=close.size)
    np.exp(volume, out=volume)

    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def build_ohlcv_from_features(
    start_price: float,
    *,
    open_gap_returns: np.ndarray,
    intrabar_returns: np.ndarray,
    upper_wick_rel: np.ndarray,
    lower_wick_rel: np.ndarray,
    volume: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build OHLCV arrays from explicit candle feature vectors."""

    n = len(intrabar_returns)
    if not all(len(values) == n for values in (open_gap_returns, upper_wick_rel, lower_wick_rel, volume)):
        raise ValueError("feature arrays must have identical length")
    if n == 0:
        empty = np.array([], dtype=float)
        return {"open": empty, "high": empty, "low": empty, "close": empty, "volume": empty}
    open_price = np.empty(n, dtype=float)
    close = np.empty(n, dtype=float)
    previous_close = float(start_price)
    for idx in range(n):
        open_price[idx] = previous_close * np.exp(float(open_gap_returns[idx]))
        close[idx] = open_price[idx] * np.exp(float(intrabar_returns[idx]))
        previous_close = close[idx]
    high = np.maximum(open_price, close) * (1.0 + np.maximum(upper_wick_rel, 0.0))
    low = np.minimum(open_price, close) / (1.0 + np.maximum(lower_wick_rel, 0.0))
    low = np.maximum(low, 1e-12)
    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.maximum(volume.astype(float), 0.0),
    }


def stationary_bootstrap_indices(
    source_n: int,
    output_n: int,
    *,
    mean_block_size: float = 48.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return stationary-bootstrap sample indices for dependent feature rows."""

    if source_n <= 0:
        raise ValueError("source_n must be positive")
    if output_n <= 0:
        return np.zeros(0, dtype=int)
    rng = np.random.default_rng(seed)
    indices = np.empty(output_n, dtype=int)
    restart_probability = 1.0 / max(float(mean_block_size), 1.0)
    cursor = int(rng.integers(0, source_n))
    for idx in range(output_n):
        if idx == 0 or rng.random() < restart_probability:
            cursor = int(rng.integers(0, source_n))
        else:
            cursor = (cursor + 1) % source_n
        indices[idx] = cursor
    return indices


def stationary_bootstrap_indices_by_time(
    source_hours: np.ndarray,
    target_hours: np.ndarray,
    *,
    source_days: np.ndarray | None = None,
    target_days: np.ndarray | None = None,
    mean_block_size: float = 48.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return stationary-bootstrap indices whose restarts match target hour/day.

    Continuing a block advances the source cursor to preserve local dependence.
    On block restarts, the source row is chosen from matching day/hour rows when
    possible, then matching-hour rows, then the full source history.
    """

    source_hours = np.asarray(source_hours, dtype=int)
    target_hours = np.asarray(target_hours, dtype=int)
    if source_hours.size <= 0:
        raise ValueError("source_hours must be non-empty")
    if target_hours.size <= 0:
        return np.zeros(0, dtype=int)
    source_n = int(source_hours.size)
    if source_days is not None:
        source_days = np.asarray(source_days, dtype=int)
        if source_days.size != source_n:
            raise ValueError("source_days must match source_hours length")
    if target_days is not None:
        target_days = np.asarray(target_days, dtype=int)
        if target_days.size != target_hours.size:
            raise ValueError("target_days must match target_hours length")

    rng = np.random.default_rng(seed)
    indices = np.empty(target_hours.size, dtype=int)
    restart_probability = 1.0 / max(float(mean_block_size), 1.0)
    by_hour = {hour: np.flatnonzero(source_hours == hour) for hour in range(24)}
    by_day_hour: dict[tuple[int, int], np.ndarray] = {}
    if source_days is not None:
        for day in range(7):
            for hour in range(24):
                by_day_hour[(day, hour)] = np.flatnonzero((source_days == day) & (source_hours == hour))

    def matching_cursor(idx: int, hour: int) -> int:
        if source_days is not None and target_days is not None:
            candidates = by_day_hour.get((int(target_days[idx]), int(hour)), np.array([], dtype=int))
            bucket = f"day={int(target_days[idx])} hour={int(hour)}"
        else:
            candidates = by_hour.get(int(hour), np.array([], dtype=int))
            bucket = f"hour={int(hour)}"
        if candidates.size == 0:
            raise SyntheticValidationError(
                f"time-aligned bootstrap has no fitted rows for target {bucket}; "
                "pass time_align_bootstrap=False to use unaligned stationary bootstrap"
            )
        return int(candidates[int(rng.integers(0, candidates.size))])

    cursor = 0
    for idx, hour in enumerate(target_hours):
        if idx == 0 or rng.random() < restart_probability:
            cursor = matching_cursor(idx, int(hour))
        else:
            next_cursor = (cursor + 1) % source_n
            hour_matches = int(source_hours[next_cursor]) == int(hour)
            day_matches = True
            if source_days is not None and target_days is not None:
                day_matches = int(source_days[next_cursor]) == int(target_days[idx])
            cursor = next_cursor if hour_matches and day_matches else matching_cursor(idx, int(hour))
        indices[idx] = cursor
    return indices


def bootstrap_feature_indices(
    features: dict[str, np.ndarray],
    output_n: int,
    *,
    params: dict | None = None,
    mean_block_size: float = 48.0,
    seed: int | None = None,
) -> np.ndarray:
    """Bootstrap feature rows, matching target timestamps when available."""

    params = params or {}
    target_timestamps = params.get("_timestamps")
    if target_timestamps is not None and params.get("time_align_bootstrap", True):
        if _calendar_features_present(features) is False:
            return stationary_bootstrap_indices(
                _feature_source_length(features),
                output_n,
                mean_block_size=float(params.get("bootstrap_mean_block_size", mean_block_size)),
                seed=seed,
            )
        missing = [name for name in ("hour_of_day", "day_of_week") if name not in features]
        if missing:
            raise SyntheticValidationError(
                "time-aligned feature sampling requires fitted feature columns: " + ", ".join(missing)
            )
        target_tuple = tuple(target_timestamps)
        target_hours = np.fromiter((int(getattr(item, "hour")) for item in target_tuple), dtype=int, count=len(target_tuple))
        target_days = np.fromiter((int(item.weekday()) for item in target_tuple), dtype=int, count=len(target_tuple))
        return stationary_bootstrap_indices_by_time(
            np.asarray(features["hour_of_day"], dtype=int),
            target_hours,
            source_days=np.asarray(features["day_of_week"], dtype=int),
            target_days=target_days,
            mean_block_size=float(params.get("bootstrap_mean_block_size", mean_block_size)),
            seed=seed,
        )
    return stationary_bootstrap_indices(
        _feature_source_length(features),
        output_n,
        mean_block_size=float(params.get("bootstrap_mean_block_size", mean_block_size)),
        seed=seed,
    )


def _feature_source_length(features: dict[str, np.ndarray]) -> int:
    for name in ("volume", "intrabar_return", "close_to_close_return"):
        if name in features:
            return len(features[name])
    raise SyntheticValidationError("feature sampling requires at least one source-length feature array")


def _calendar_features_present(features: dict[str, np.ndarray]) -> bool | None:
    marker = features.get("calendar_features_present")
    if marker is None:
        return None
    values = np.asarray(marker, dtype=bool)
    return bool(values.size and values.ravel()[0])


def bars_to_rows(
    timestamps: tuple[datetime, ...],
    ohlcv: dict[str, np.ndarray],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "timestamp": ts,
            "open": float(ohlcv["open"][i]),
            "high": float(ohlcv["high"][i]),
            "low": float(ohlcv["low"][i]),
            "close": float(ohlcv["close"][i]),
            "volume": float(ohlcv["volume"][i]),
        }
        for i, ts in enumerate(timestamps)
    )
