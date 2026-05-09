"""Horizon helpers for request-to-bar translation and timestamp generation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
import math
import re

from .errors import SyntheticValidationError


_INTERVAL_RE = re.compile(r"^(\d+)([smhdw])$")


@lru_cache(maxsize=128)
def parse_interval(interval: str) -> timedelta:
    match = _INTERVAL_RE.match(interval.strip().lower())
    if not match:
        raise SyntheticValidationError(f"Unsupported interval format: {interval}")

    magnitude = int(match.group(1))
    unit = match.group(2)
    seconds = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 60 * 60 * 24,
        "w": 60 * 60 * 24 * 7,
    }[unit] * magnitude
    return timedelta(seconds=seconds)


def bars_from_span(
    start_ts: datetime,
    interval: str,
    *,
    end_ts: datetime | None = None,
    duration: timedelta | None = None,
    n_bars: int | None = None,
    trading_hours: str = "24x7",
) -> int:
    """Resolve number of bars from duration or end timestamp."""

    horizon_sources = sum(value is not None for value in (n_bars, duration, end_ts))
    if horizon_sources != 1:
        raise SyntheticValidationError("Provide exactly one of n_bars, duration, or end_ts")

    if n_bars is not None:
        if n_bars <= 0:
            raise SyntheticValidationError("n_bars must be greater than 0")
        return n_bars

    if duration is None and end_ts is None:
        raise SyntheticValidationError("Provide n_bars or one of duration/end_ts")

    step = parse_interval(interval)
    if duration is not None:
        if duration <= timedelta(0):
            raise SyntheticValidationError("duration must be greater than 0")
        if trading_hours == "24x7":
            return math.ceil(duration.total_seconds() / step.total_seconds())
        return _estimate_session_bars(start_ts, duration, step)

    if end_ts <= start_ts:
        raise SyntheticValidationError("end_ts must be after start_ts")

    if trading_hours == "24x7":
        return math.ceil((end_ts - start_ts).total_seconds() / step.total_seconds())

    return _business_bar_count(start_ts, end_ts, step)


def generate_timestamps(
    start_ts: datetime,
    interval: str,
    n_bars: int,
    *,
    trading_hours: str = "24x7",
) -> tuple[datetime, ...]:
    """Generate contiguous timestamps according to trading-hours policy."""

    if n_bars <= 0:
        return tuple()

    step = parse_interval(interval)
    cursor = _normalize_utc(start_ts)

    if trading_hours == "24x7":
        return tuple(cursor + step * i for i in range(n_bars))

    if trading_hours == "equity_sessions":
        stamps: list[datetime] = []
        while len(stamps) < n_bars:
            if cursor.weekday() < 5:
                stamps.append(cursor)
            cursor += step
        return tuple(stamps)

    raise SyntheticValidationError(f"Unsupported trading_hours policy: {trading_hours}")


def _business_bar_count(start_ts: datetime, end_ts: datetime, step: timedelta) -> int:
    if end_ts <= start_ts:
        return 0

    count = 0
    cursor = _normalize_utc(start_ts)
    end = _normalize_utc(end_ts)
    while cursor < end:
        if cursor.weekday() < 5:
            count += 1
        cursor += step
    return count


def _estimate_session_bars(start_ts: datetime, duration: timedelta, step: timedelta) -> int:
    if duration <= timedelta(0):
        return 0
    end_ts = start_ts + duration
    return _business_bar_count(start_ts, end_ts, step)


def _normalize_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
