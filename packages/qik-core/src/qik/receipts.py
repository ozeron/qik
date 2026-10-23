"""Pinned, append-only old/new field comparison receipts (no execution imports)."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qik.trial_storage import atomic_json, canonical, file_ref, sync_directory


@dataclass(frozen=True)
class Pins:
    source_git_sha: str
    tape_sha256: str
    session_set_hash: str
    window: tuple[str, str]
    params_hash: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_git_sha):
            raise ValueError("source_git_sha must be a full git SHA")
        for name in ("tape_sha256", "session_set_hash", "params_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise ValueError(f"{name} must be SHA256")
        if len(self.window) != 2 or date.fromisoformat(self.window[0]) > date.fromisoformat(
            self.window[1]
        ):
            raise ValueError("window must be an ordered ISO date pair")


@dataclass(frozen=True)
class FieldDiff:
    """Positional comparison; callers explicitly align keyed trades beforehand.

    Unpaired trailing rows remain differences. Both-null cells compare equal,
    one-null cells differ, infinities are refused. Numeric tolerance is absolute
    (rtol=0), either one value or a per-field mapping; omitted fields are exact.
    """

    old_rows: int
    new_rows: int
    equal_rows: int
    changed_rows: int
    changes: pd.DataFrame

    @property
    def equal(self) -> bool:
        return self.changed_rows == 0


def diff_fields(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    fields: Sequence[str],
    tol: float | Mapping[str, float],
) -> FieldDiff:
    names = tuple(fields)
    if not names or len(set(names)) != len(names):
        raise ValueError("fields must be nonempty and unique")
    tolerances = dict.fromkeys(names, tol) if not isinstance(tol, Mapping) else dict(tol)
    if set(tolerances) - set(names):
        raise ValueError("tolerance names must be compared fields")
    if any(not np.isfinite(v) or v < 0 for v in tolerances.values()):
        raise ValueError("tolerances must be finite and nonnegative")
    for frame in (old, new):
        if not frame.columns.is_unique or set(names) - set(frame):
            raise ValueError("missing or duplicate comparison fields")
    a, b = old.reset_index(drop=True), new.reset_index(drop=True)
    changes: list[dict[str, Any]] = []
    changed: set[int] = set()
    for i in range(max(len(a), len(b))):
        if i >= min(len(a), len(b)):
            changed.add(i)
            changes.append(
                {
                    "row": i,
                    "field": "__row__",
                    "old": "present" if i < len(a) else None,
                    "new": "present" if i < len(b) else None,
                    "delta": None,
                }
            )
            continue
        for name in names:
            x, y = a.at[i, name], b.at[i, name]
            delta = None
            if pd.isna(x) or pd.isna(y):
                equal = pd.isna(x) and pd.isna(y)
            elif isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)):
                if not np.isfinite([x, y]).all():
                    raise ValueError(f"nonfinite comparison value at row {i}, field {name}")
                delta = (y.item() if isinstance(y, np.generic) else y) - (
                    x.item() if isinstance(x, np.generic) else x
                )
                equal = abs(delta) <= tolerances.get(name, 0.0)
            else:
                equal = x == y
            if not equal:
                changed.add(i)
                changes.append({"row": i, "field": name, "old": x, "new": y, "delta": delta})
    return FieldDiff(
        len(a),
        len(b),
        max(len(a), len(b)) - len(changed),
        len(changed),
        pd.DataFrame(
            changes, columns=pd.Index(["row", "field", "old", "new", "delta"]), dtype=object
        ),
    )


@dataclass(frozen=True)
class Receipt:
    path: Path
    ref: Mapping[str, Any]
    diff: FieldDiff
    pins: Pins


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON scalars without rounding floats or truncating timestamp precision."""

    def scalar(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, date)):
            return value.isoformat()
        return value.item() if isinstance(value, np.generic) else value

    return [
        {str(key): scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _read(value: Path | str | pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        frame = value.copy(deep=True)
        # pandas.to_json rounds floats even at its maximum precision. Canonical
        # Python JSON preserves their round-trip representation instead.
        payload = {
            "columns": frame.columns.tolist(),
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "rows": _records(frame),
        }
        return frame, {
            "kind": "dataframe",
            "sha256": hashlib.sha256(canonical(payload)).hexdigest(),
        }
    path = Path(value)
    ref = file_ref(path)
    frame = pd.read_csv(path, float_precision="round_trip")  # ty: ignore[no-matching-overload] -- pandas supports round_trip; its bundled annotation omits it
    if file_ref(path) != ref:
        raise ValueError("receipt source changed while reading")
    return frame, ref


def write_receipt(
    path: str | Path,
    *,
    old: Path | str | pd.DataFrame,
    new: Path | str | pd.DataFrame,
    pins: Pins,
    fields: Sequence[str],
) -> Receipt:
    """Write exact comparison, references and pins; atomically refuse any overwrite.

    ``diff_fields`` supports tolerances for diagnostics; a receipt is always
    exact. It does not classify a DIFF as acceptable or infer missing pins.
    """
    path = Path(path)
    before, old_ref = _read(old)
    after, new_ref = _read(new)
    diff = diff_fields(before, after, fields=fields, tol=0.0)
    changes = _records(diff.changes)
    payload = {
        "version": 1,
        "pins": asdict(pins),
        "old": old_ref,
        "new": new_ref,
        "fields": list(fields),
        "tolerance": 0.0,
        "verdict": "EQUAL" if diff.equal else "DIFF",
        "old_rows": diff.old_rows,
        "new_rows": diff.new_rows,
        "equal_rows": diff.equal_rows,
        "changed_rows": diff.changed_rows,
        "changes": changes,
    }
    # atomic_json replaces its target, so stage privately and publish with link(2),
    # which fails atomically if a file or symlink already occupies the destination.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qik-receipt-", dir=path.parent) as directory:
        staged = Path(directory) / "receipt.json"
        atomic_json(staged, payload)
        os.link(staged, path)
        sync_directory(path.parent)
    return Receipt(path, file_ref(path), diff, pins)
