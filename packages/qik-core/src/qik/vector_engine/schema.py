"""Typed author schemas backed by numeric records and named tuples of arrays."""

from __future__ import annotations

import hashlib
from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import Any, ClassVar, TypeVar, dataclass_transform, get_origin, get_type_hints

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass_transform()
class _RecordMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], ns: dict[str, Any]) -> Any:
        cls: Any = super().__new__(mcls, name, bases, ns)
        if not bases:
            return cls
        hints = get_type_hints(cls)
        fields = {}
        defaults = {}
        for field, kind in hints.items():
            if get_origin(kind) is ClassVar:
                continue
            scalar = {float: np.float64, int: np.int64, bool: np.bool_}.get(kind)
            if isinstance(kind, type) and issubclass(kind, IntEnum):
                if any(not -(2**63) <= int(v) < 2**63 for v in kind):
                    raise TypeError(f"{name}.{field}: enum must fit int64")
                scalar = np.int64
            if scalar is None:
                raise TypeError(f"{name}.{field}: unsupported field type {kind!r}")
            fields[field] = scalar
            if hasattr(cls, field):
                defaults[field] = getattr(cls, field)
            elif cls._required_defaults:
                raise TypeError(f"{name}.{field}: State requires a default")
        cls.dtype = np.dtype(list(fields.items()))
        cls._defaults = defaults
        return cls


class _Record(metaclass=_RecordMeta):
    dtype: ClassVar[np.dtype]
    _defaults: ClassVar[dict[str, Any]] = {}
    _required_defaults: ClassVar[bool] = False

    @classmethod
    def _allocate(cls, n: int) -> np.recarray:
        rows = np.zeros(n, dtype=cls.dtype).view(np.recarray)
        for field, value in cls._defaults.items():
            rows[field] = value
        return rows


class State(_Record):
    """Mutable scalar schema; every field requires a default, including NaN."""

    _required_defaults = True

    @classmethod
    def new(cls, n: int) -> np.recarray:
        """Allocate n independent rows with declared defaults and attribute access."""
        return cls._allocate(n)

    @classmethod
    def _header(cls) -> bytes:
        return b"QIKS1" + hashlib.sha256(repr(cls.dtype.descr).encode()).digest()

    @classmethod
    def to_bytes(cls, row: np.record) -> bytes:
        """Serialize one matching row exactly, with a versioned dtype fingerprint."""
        if row.dtype != cls.dtype:
            raise ValueError(f"{cls.__name__}: checkpoint dtype mismatch")
        return cls._header() + row.tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> np.record:
        """Restore a writable record, refusing a different schema or payload size."""
        header = cls._header()
        if not payload.startswith(header) or len(payload) != len(header) + cls.dtype.itemsize:
            raise ValueError(f"{cls.__name__}: checkpoint fingerprint or size mismatch")
        if cls.dtype.itemsize == 0:
            return cls.new(1)[0]
        return np.frombuffer(payload[len(header) :], dtype=cls.dtype).copy().view(np.recarray)[0]


class Params(_Record):
    """Scalar grid schema; omitted fields must have defaults."""

    @classmethod
    def from_grid(cls, grid: Mapping[str, Sequence[Any]]) -> np.recarray:
        """Cartesian product in declaration order, rightmost field varying fastest.

        Mapping insertion order is ignored. Omitted defaults contribute one value;
        an empty axis yields zero rows. An empty schema yields one empty row.
        """
        names = cls.dtype.names or ()
        if unknown := set(grid).difference(names):
            raise ValueError(f"{cls.__name__}: unknown grid keys {sorted(unknown)}")
        axes = []
        for field in names:
            if field in grid:
                axes.append(grid[field])
            elif field in cls._defaults:
                axes.append([cls._defaults[field]])
            else:
                raise ValueError(f"{cls.__name__}.{field}: missing grid value")
        values = list(product(*axes))
        rows = cls._allocate(len(values))
        for i, value in enumerate(values):
            rows[i] = value
        return rows


@dataclass_transform()
class _FeaturesMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], ns: dict[str, Any]) -> Any:
        cls: Any = super().__new__(mcls, name, bases, ns)
        if not bases:
            return cls
        hints = get_type_hints(cls)
        for field, kind in hints.items():
            if kind not in (FloatArray, IntArray, BoolArray):
                raise TypeError(f"{name}.{field}: unsupported feature type {kind!r}")
        record = namedtuple(f"_{name}Arrays", hints)
        return super().__new__(mcls, name, (record, *bases), ns)


class Features(metaclass=_FeaturesMeta):
    """Named tuple schema of 1-D arrays, usable directly by numba."""

    __slots__ = ()

    def validate(self, n_bars: int) -> None:
        """Require all declared arrays with exact float64/int64/bool dtype and length."""
        type(self).validate_values(self, n_bars)

    @classmethod
    def validate_values(cls, values: Features | Mapping[str, Any], n_bars: int) -> None:
        """Validate a typed feature tuple or a plain dictionary against this schema."""
        for field, kind in get_type_hints(cls).items():
            if isinstance(values, Mapping):
                if field not in values:
                    raise ValueError(f"{cls.__name__}.{field}: missing feature")
                value = values[field]
            else:
                value = getattr(values, field)
            dtype = {FloatArray: np.float64, IntArray: np.int64, BoolArray: np.bool_}[kind]
            if not isinstance(value, np.ndarray) or value.dtype != dtype:
                raise TypeError(f"{cls.__name__}.{field}: expected 1-D {np.dtype(dtype)} array")
            if value.shape != (n_bars,):
                raise ValueError(
                    f"{cls.__name__}.{field}: expected length {n_bars}, got {value.shape}"
                )


@dataclass(frozen=True)
class StrategySpec:
    """Schema and Python function binding consumed by the forthcoming run adapter."""

    state: type[State]
    params: type[Params]
    features: type[Features]
    fn: Callable[..., Any]


_Fn = TypeVar("_Fn", bound=Callable[..., Any])


def strategy(
    *, state: type[State], params: type[Params], features: type[Features]
) -> Callable[[_Fn], _Fn]:
    """Validate and attach a StrategySpec; execution and compilation belong to run."""
    for schema, base in ((state, State), (params, Params), (features, Features)):
        if not isinstance(schema, type) or not issubclass(schema, base):
            raise TypeError(f"strategy: expected {base.__name__} schema, got {schema!r}")

    def decorate(fn: _Fn) -> _Fn:
        setattr(fn, "__strategy__", StrategySpec(state, params, features, fn))
        return fn

    return decorate
