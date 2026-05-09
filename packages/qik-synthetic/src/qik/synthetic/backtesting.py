"""Backtesting interoperability helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import SyntheticConversionError
from .types import SyntheticBatch, SyntheticMetadata


REQUIRED_FIELDS = ("symbol", "venue", "interval", "dataset")
REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def to_backtesting_frame(
    batch: SyntheticBatch,
    *,
    identity_override: Mapping[str, str] | None = None,
    compact_identity: bool = False,
):
    """Return a DataFrame-like object with OHLCV rows."""
    identity = _resolve_identity(batch, identity_override=identity_override)

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise SyntheticConversionError("pandas is required for to_backtesting_frame") from exc

    frame = pd.DataFrame.from_records(batch.bars, columns=REQUIRED_COLUMNS)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if not frame["timestamp"].is_monotonic_increasing:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return _add_identity_columns(frame, identity, compact_identity=compact_identity)


def to_backtesting_payload(
    batch: SyntheticBatch,
    *,
    identity_override: Mapping[str, str] | None = None,
    run_context: Mapping[str, str] | None = None,
    compact_identity: bool = False,
) -> Mapping[str, object]:
    """Create a research adapter payload for candle-research style pipelines."""
    identity = _resolve_identity(batch, identity_override=identity_override)
    missing = [field for field in REQUIRED_FIELDS if not identity.get(field)]
    if missing:
        raise SyntheticConversionError("Missing required identity field(s): " + ", ".join(missing))

    frame = to_backtesting_frame(batch, identity_override=identity, compact_identity=compact_identity)
    return to_backtesting_payload_from_frame(
        frame,
        metadata=batch.metadata,
        identity=identity,
        n_bars=batch.n_rows,
        run_context=run_context,
    )


def to_backtesting_payload_from_frame(
    frame,
    *,
    metadata: SyntheticMetadata,
    identity: Mapping[str, str | None],
    n_bars: int,
    run_context: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Create a backtesting payload from an already materialized candle frame."""
    missing = [field for field in REQUIRED_FIELDS if not identity.get(field)]
    if missing:
        raise SyntheticConversionError("Missing required identity field(s): " + ", ".join(missing))

    first_timestamp = frame["timestamp"].iloc[0] if len(frame) else None
    last_timestamp = frame["timestamp"].iloc[-1] if len(frame) else None
    first_close = float(frame["close"].iloc[0]) if len(frame) else None

    payload: dict[str, object] = {
        "candle_table": frame,
        "required_columns": REQUIRED_COLUMNS,
        "request_id": metadata.request_id,
        "run_id": metadata.run_id,
        "seed": metadata.seed,
        "model_name": metadata.model_name,
        "context_id": metadata.context_id,
        "metadata": {
            "n_bars": n_bars,
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


def _add_identity_columns(frame, identity: Mapping[str, str | None], *, compact_identity: bool):
    frame = frame.copy()
    for column in ("symbol", "venue", "dataset", "interval"):
        frame[column] = identity[column]
    if compact_identity:
        for column in ("symbol", "venue", "dataset", "interval"):
            frame[column] = frame[column].astype("category")
    return frame


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
