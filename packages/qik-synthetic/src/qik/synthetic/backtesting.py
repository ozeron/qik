"""Backtesting interoperability helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import SyntheticConversionError
from .types import SyntheticBatch


REQUIRED_FIELDS = ("symbol", "venue", "interval", "dataset")
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def to_backtesting_frame(
    batch: SyntheticBatch,
    *,
    identity_override: Mapping[str, str] | None = None,
):
    """Return a DataFrame-like object with OHLCV rows."""
    identity = _resolve_identity(batch, identity_override=identity_override)
    rows = [
        {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "symbol": identity["symbol"],
            "venue": identity["venue"],
            "dataset": identity["dataset"],
            "interval": identity["interval"],
        }
        for row in batch.bars
    ]

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise SyntheticConversionError("pandas is required for to_backtesting_frame") from exc

    # Keep payload columns stable and deterministic for downstream contracts.
    rows = sorted(
        rows,
        key=lambda row: row["timestamp"],
    )
    return pd.DataFrame(rows)


def to_backtesting_payload(
    batch: SyntheticBatch,
    *,
    identity_override: Mapping[str, str] | None = None,
    run_context: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Create a research adapter payload for candle-research style pipelines."""
    identity = _resolve_identity(batch, identity_override=identity_override)
    missing = [field for field in REQUIRED_FIELDS if not identity.get(field)]
    if missing:
        raise SyntheticConversionError("Missing required identity field(s): " + ", ".join(missing))

    frame = to_backtesting_frame(batch, identity_override=identity)
    first_timestamp = frame["timestamp"].iloc[0] if len(frame) else None
    last_timestamp = frame["timestamp"].iloc[-1] if len(frame) else None
    first_close = float(frame["close"].iloc[0]) if len(frame) else None

    payload: dict[str, object] = {
        "candle_table": frame,
        "required_columns": REQUIRED_COLUMNS,
        "request_id": batch.metadata.request_id,
        "run_id": batch.metadata.run_id,
        "seed": batch.metadata.seed,
        "model_name": batch.metadata.model_name,
        "context_id": batch.metadata.context_id,
        "metadata": {
            "n_bars": batch.n_rows,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "first_close": first_close,
            "run_context": dict(run_context or {}),
            "identity": identity,
            "symbol": identity["symbol"],
            "venue": identity["venue"],
            "dataset": identity["dataset"],
            "interval": identity["interval"],
        },
    }

    validate_backtesting_payload(payload)
    return payload


def validate_backtesting_payload(payload: Mapping[str, object]) -> None:
    """Validate payload boundary before downstream consumption."""
    for key in {"candle_table", "required_columns", "request_id", "run_id", "seed", "model_name", "context_id"}:
        if key not in payload:
            raise SyntheticConversionError(f"Backtesting payload missing key: {key}")

    required_columns = payload.get("required_columns")
    if not isinstance(required_columns, tuple) and not isinstance(required_columns, list):
        raise SyntheticConversionError("required_columns must be list/tuple")

    table = payload["candle_table"]
    columns = getattr(table, "columns", ())
    for column in REQUIRED_COLUMNS:
        if column not in columns:
            raise SyntheticConversionError(f"Missing required OHLCV column: {column}")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SyntheticConversionError("metadata must be a mapping")

    identity = metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise SyntheticConversionError("metadata.identity must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if not identity.get(field)]
    if missing:
        raise SyntheticConversionError("Payload metadata missing identity fields: " + ", ".join(missing))


def _resolve_identity(
    batch: SyntheticBatch,
    *,
    identity_override: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    identity: dict[str, str | None] = {
        "symbol": batch.context.symbol,
        "venue": batch.context.venue,
        "interval": batch.request.interval,
        "dataset": batch.context.dataset,
    }
    if identity_override:
        identity.update(identity_override)
    return identity
