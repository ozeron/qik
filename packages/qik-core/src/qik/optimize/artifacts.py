"""Date-first loading of frozen, eligible-session panels (zeros must be present)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

SEALED_FROM = date(2024, 1, 1)


def validate_dates(values: list[str], *, unsealed: bool = False) -> tuple[date, ...]:
    """Validate every date before the caller interprets any economic value.

    Explicit unsealing is an API capability, never automatic authorization.
    """
    dates = tuple(date.fromisoformat(value) for value in values)
    if any(
        d.isoformat() != value or d < date(1900, 1, 1)
        for d, value in zip(dates, values, strict=True)
    ):
        raise ValueError("malformed date: use ISO YYYY-MM-DD")
    if not unsealed and any(d >= SEALED_FROM for d in dates):
        raise ValueError("sealed row on or after 2024-01-01")
    return dates


@dataclass(frozen=True)
class SessionPanel:
    dates: tuple[date, ...]
    names: tuple[str, ...]
    values: np.ndarray

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]


def load_sessions(path: Path, *, unsealed: bool = False) -> SessionPanel:
    """Read a wide session_date CSV; refuse duplicates, gaps in values or sealed rows.

    Calendar completeness is a caller's G1 provenance obligation. Never inner-join
    arms or infer the eligible calendar from nonzero trades.
    """
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = tuple(reader.fieldnames or ())
        rows = list(reader)
    if len(set(names)) != len(names) or not names or names[0] != "session_date":
        raise ValueError("expected unique session_date then arm columns")
    dates = validate_dates([row["session_date"] for row in rows], unsealed=unsealed)
    if len(dates) < 2 or tuple(sorted(set(dates))) != dates or len(names) < 3:
        raise ValueError("need >=2 sorted unique sessions and >=2 arms")
    values = np.array([[float(row[name]) for name in names[1:]] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("nonfinite session value")
    return SessionPanel(dates, names[1:], values)
