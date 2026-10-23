"""Immutable, ordered snapshots of caller-defined candidate columns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray


class _SnapshotColumns(Mapping[str, NDArray]):
    """Expose fresh array views so callers cannot mutate stored metadata."""

    def __init__(self, columns: Mapping[str, NDArray]) -> None:
        self._columns = MappingProxyType(dict(columns))

    def __getitem__(self, name: str) -> NDArray:
        column = self._columns[name]
        raw = column.base
        assert isinstance(raw, bytes)
        return np.frombuffer(raw, dtype=column.dtype)

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


@dataclass(frozen=True, eq=False)
class CandidatePool:
    """Freeze aligned 1-D columns without filtering, sorting or deduplication.

    Include every column used by the scoring callback in the snapshot. The hash
    binds column names, dtypes, row order and bytes, not external callback state.
    Bytes-backed arrays cannot be made writeable again. Object arrays are refused.
    """

    columns: Mapping[str, NDArray]
    expected_count: int
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.expected_count < 0 or not self.columns:
            raise ValueError("nonempty columns and nonnegative expected_count required")
        frozen: dict[str, NDArray] = {}
        digest = hashlib.sha256(b"qik.battery.pool.v1\0")
        for name, values in sorted(self.columns.items()):
            a = np.asarray(values)
            if a.ndim != 1 or a.dtype.hasobject or a.dtype.fields is not None:
                raise ValueError(f"{name}: plain non-object 1-D column required")
            if len(a) != self.expected_count:
                raise ValueError(f"{name}: {len(a)} rows != expected {self.expected_count}")
            raw = a.tobytes(order="C")
            header = json.dumps([name, a.dtype.str, len(a)], separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(raw)
            frozen[name] = np.frombuffer(raw, dtype=a.dtype)
        object.__setattr__(self, "columns", _SnapshotColumns(frozen))
        object.__setattr__(self, "sha256", digest.hexdigest())

    def assert_counts(self, columns: tuple[str, ...], expected: Mapping[tuple, int]) -> None:
        """Assert the complete stratum counts, including absence of extra cells."""
        if not columns:
            raise ValueError("at least one stratum column required")
        counts: dict[tuple, int] = {}
        for row in zip(*(self.columns[c] for c in columns), strict=True):
            counts[row] = counts.get(row, 0) + 1
        if counts != dict(expected):
            raise ValueError(f"candidate counts differ: {counts} != {dict(expected)}")
