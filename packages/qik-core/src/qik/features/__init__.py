"""Opt-in shared 1-D feature preparation, independent of the engine runner.

Builders receive declared, immutable Series at the base decision clock. The opt-in
run wrapper streams parameter groups through the unchanged engine runner.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import FunctionType, MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from . import _identity
from ._identity import array_identity, code_identity, digest, function_identity, normalize, scalar
from ._runner import run

__all__ = ["FeaturePlan", "feature", "param", "with_features", "run", "quantity_from_risk"]

Array = np.ndarray
Verification = Literal["sampled", "exhaustive"] | None
MAX_SAMPLED_PREFIX_BARS = 100_000_000


@dataclass(frozen=True)
class Param:
    name: str


def param(name: str) -> Param:
    """Declare a scalar dependency, resolved from a binding or managed grid row."""
    if not name.isidentifier():
        raise ValueError(f"invalid param name {name!r}")
    return Param(name)


@dataclass(frozen=True, eq=False)
class Feature:
    fn: Callable[..., Array]
    inputs: Mapping[str, str | Feature]
    dtype: np.dtype
    warmup: int
    allow_nan: bool
    bindings: Mapping[str, object]

    def __call__(self, **bindings: object) -> Feature:
        """Bind builder keyword arguments to scalar literals or param(name)."""
        return replace(self, bindings=MappingProxyType({**self.bindings, **bindings}))


def feature(
    *,
    inputs: Sequence[str] | Mapping[str, str | Feature],
    dtype: str | type,
    warmup: int,
    allow_nan: bool = False,
) -> Callable[[Callable[..., Array]], Feature]:
    """Declare a causal builder. Warmup counts unavailable leading bars (not period)."""
    kind = np.dtype(dtype)
    if kind not in (np.dtype("float64"), np.dtype("int64"), np.dtype("bool")):
        raise TypeError(f"unsupported feature dtype {kind}")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if warmup and kind != np.dtype("float64"):
        raise ValueError("bool/int warmup requires readiness metadata (not supported in F1)")
    selected = dict(inputs) if isinstance(inputs, Mapping) else {n: n for n in inputs}
    if any(not n.isidentifier() or n.startswith("_") for n in selected):
        raise ValueError("input aliases must be public identifiers")
    if any(not isinstance(v, (str, Feature)) for v in selected.values()):
        raise TypeError("feature inputs must name source lanes or feature nodes")
    return lambda fn: Feature(
        fn, MappingProxyType(selected), kind, warmup, allow_nan, MappingProxyType({})
    )


def _immutable(value: Array) -> Array:
    # A write-protected owning ndarray can be made writable again. Bytes cannot.
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


class FeaturePlan[T]:
    """Map schema lanes to a DAG of shared builders; produce ordinary engine arrays."""

    def __init__(self, schema: type[T], **nodes: Feature) -> None:
        if set(nodes) != set(getattr(schema, "_fields")):
            raise ValueError(f"{schema.__name__}: plan/schema fields differ")
        if any(not isinstance(n, Feature) for n in nodes.values()):
            raise TypeError("plan outputs must be declared features")
        getattr(schema, "validate_values")(
            {k: np.empty(0, dtype=n.dtype) for k, n in nodes.items()}, 0
        )
        self.schema = schema
        self.nodes = MappingProxyType(nodes)
        self._ordered()  # detect cycles before any builder runs

    def _ordered(self) -> list[Feature]:
        done: set[Feature] = set()
        active: set[Feature] = set()
        ordered: list[Feature] = []

        def visit(node: Feature) -> None:
            if node in active:
                raise ValueError(f"{getattr(node.fn, '__name__', 'builder')}: dependency cycle")
            if node in done:
                return
            active.add(node)
            for source in node.inputs.values():
                if isinstance(source, Feature):
                    visit(source)
            active.remove(node)
            done.add(node)
            ordered.append(node)

        for node in self.nodes.values():
            visit(node)
        return ordered

    def materialize(
        self,
        inputs: pd.DataFrame | Mapping[str, Array],
        *,
        index: pd.DatetimeIndex | None = None,
        params: Mapping[str, object] | None = None,
        source_manifest: Mapping[str, object] | None = None,
        verify_features: Verification = None,
        cache: Path | Literal[False] | None = None,
        _shared: dict[Feature, tuple[Array, str]] | None = None,
    ) -> T:
        """Prepare one scalar binding. Verification always bypasses cached prefix outputs.

        A DatetimeIndex is the caller's decision/availability clock: all inputs must
        already be aligned and known by that bar. External tables belong to F4.
        """
        if verify_features not in (None, "sampled", "exhaustive"):
            raise ValueError(f"unknown verification mode {verify_features!r}")
        if index is None and isinstance(inputs, pd.DataFrame):
            index = inputs.index if isinstance(inputs.index, pd.DatetimeIndex) else None
        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.hasnans
            or not index.is_monotonic_increasing
            or not index.is_unique
        ):
            raise ValueError("features require a unique increasing decision DatetimeIndex")
        index = pd.DatetimeIndex(_immutable(index.asi8), dtype=index.dtype)
        ordered = self._ordered()
        required = {s for node in ordered for s in node.inputs.values() if isinstance(s, str)}
        sources: dict[str, Array] = {}
        for name in sorted(required):
            if name not in inputs:
                raise ValueError(f"missing feature input {name}")
            value = np.asarray(inputs[name])
            if value.shape != (len(index),):
                raise ValueError(f"{name}: expected 1-D input of length {len(index)}")
            array_identity(value)  # refuse object inputs, even without caching
            sources[name] = _immutable(value)
        bound: dict[Feature, dict[str, object]] = {}
        params = params or {}
        for node in ordered:
            bound[node] = {}
            for name, value in node.bindings.items():
                if isinstance(value, Param):
                    if value.name not in params:
                        raise ValueError(
                            f"{getattr(node.fn, '__name__', 'builder')}: unknown param {value.name}"
                        )
                    value = params[value.name]
                bound[node][name] = normalize(value)
        selected_cache = (
            cache
            if cache is not None
            else os.environ.get("QIK_FEATURE_CACHE", "~/data/desk/cache/qik/features/v1")
        )
        root = (
            Path(selected_cache).expanduser()
            if selected_cache is not False and selected_cache != ""
            else None
        )
        identities: dict[Feature, str] = {}
        provenance: dict[Feature, dict[str, object]] = {}
        values: dict[Feature, Array] = {}
        missed: set[Feature] = set()
        clock = {"values": array_identity(index.asi8), "dtype": str(index.dtype)}
        cuts = _cuts(index, [n.warmup for n in ordered], verify_features)
        if verify_features == "sampled" and len(ordered) * sum(cuts) > MAX_SAMPLED_PREFIX_BARS:
            raise ValueError(
                "sampled verification budget exceeded (100000000 graph-prefix bars); "
                "use bounded input or explicitly request exhaustive"
            )
        labels = {
            node: next(
                (k for k, v in self.nodes.items() if v is node),
                getattr(node.fn, "__name__", "builder"),
            )
            for node in ordered
        }
        source_keys = {k: array_identity(v) for k, v in sources.items()}
        planner = [
            code_identity(fn.__code__)
            for fn in (
                _compute,
                _check_nan,
                _cuts,
                _read,
                _write,
                _publish,
                FeaturePlan.materialize,
                FeaturePlan._ordered,
                Feature.__call__,
                *(
                    value
                    for value in vars(_identity).values()
                    if isinstance(value, FunctionType) and value.__module__ == _identity.__name__
                ),
            )
        ]
        independent: dict[Feature, bool] = {}
        for node in ordered:
            independent[node] = not any(
                isinstance(v, Param) for v in node.bindings.values()
            ) and all(independent[v] for v in node.inputs.values() if isinstance(v, Feature))
            if _shared is not None and node in _shared:
                values[node], identities[node] = _shared[node]
                continue
            label = labels[node]
            identity = {
                "version": 2,
                "builder": function_identity(node.fn),
                "params": scalar(bound[node]),
                "dtype": node.dtype.str,
                "warmup": node.warmup,
                "allow_nan": node.allow_nan,
                "clock": clock,
                "manifest": scalar(dict(source_manifest or {})),
                "inputs": {
                    k: identities[v] if isinstance(v, Feature) else source_keys[v]
                    for k, v in node.inputs.items()
                },
                "runtime": [
                    sys.version,
                    np.__version__,
                    pd.__version__,
                    _identity.backend_identity(),
                ],
                "planner": planner,
            }
            provenance[node] = identity
            key = digest(identity)
            identities[node] = key
            path = root / f"{key}.npz" if root is not None else None
            value = _read(path, key, node, len(index)) if path is not None else None
            if value is None:
                missed.add(node)
                value = _compute(node, sources, values, index, bound[node], label)
            values[node] = value
            if _shared is not None and independent[node]:
                _shared[node] = (value, key)
        # Recompute the ENTIRE graph for each prefix, not a slice of cached dependencies.
        for cut in cuts:
            prefix: dict[Feature, Array] = {}
            for node in ordered:
                label = labels[node]
                prefix[node] = _compute(
                    node,
                    {k: v[:cut] for k, v in sources.items()},
                    prefix,
                    index[:cut],
                    bound[node],
                    label,
                )
                same = np.all(
                    prefix[node].view(np.uint8).reshape(cut, -1)
                    == values[node][:cut].view(np.uint8).reshape(cut, -1),
                    axis=1,
                )
                if not np.all(same):
                    bar = int(np.flatnonzero(~same)[0])
                    raise ValueError(f"{label}: prefix mismatch at bar {bar} (cut {cut})")
        result = {name: values[node] for name, node in self.nodes.items()}
        getattr(self.schema, "validate_values")(result, len(index))
        # Publish only after all validation/verification succeeds.
        if root is not None:
            for node in ordered:
                if node not in provenance or (node not in missed and verify_features is None):
                    continue
                _write(
                    root / f"{identities[node]}.npz",
                    identities[node],
                    values[node],
                    cuts,
                    verify_features,
                    provenance[node],
                )
        return cast(Callable[..., T], self.schema)(**result)


def _compute(
    node: Feature,
    sources: Mapping[str, Array],
    values: Mapping[Feature, Array],
    index: pd.DatetimeIndex,
    bindings: Mapping[str, object],
    label: str,
) -> Array:
    lanes = {
        name: pd.Series(
            values[s] if isinstance(s, Feature) else sources[s], index=index, copy=False
        )
        for name, s in node.inputs.items()
    }
    value = node.fn(SimpleNamespace(**lanes), **bindings)
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != node.dtype
        or value.shape != (len(index),)
    ):
        raise TypeError(
            f"{label}: expected {node.dtype}[{len(index)}], got "
            f"{getattr(value, 'dtype', type(value).__name__)}{getattr(value, 'shape', '')}"
        )
    value = value.copy(order="C")
    if node.warmup:
        value[: node.warmup] = np.nan
    _check_nan(node, value, label)
    return _immutable(value)


def _check_nan(node: Feature, value: Array, label: str) -> None:
    if node.dtype == np.dtype("float64") and not node.allow_nan:
        bad = np.flatnonzero(np.isnan(value[node.warmup :]))
        if bad.size:
            raise ValueError(
                f"{label}: NaN after warmup at bar {int(bad[0]) + node.warmup}; "
                "declare allow_nan=True for intentional missing values"
            )


def _cuts(index: pd.DatetimeIndex, warmups: Sequence[int], mode: Verification) -> list[int]:
    n = len(index)
    if mode is None or n < 2:
        return []
    if mode == "exhaustive":
        return list(range(1, n))
    # Every day boundary, plus a sqrt(n) seeded sample (capped at 512).
    count = min(n - 1, 512, max(16, int(np.ceil(np.sqrt(n)))))
    candidates = {
        1,
        n - 1,
        *(int(v) for v in np.linspace(1, n - 1, 17)),
        *(w + delta for w in warmups for delta in (-1, 0, 1, 2)),
        *(int(v) + 1 for v in np.random.default_rng(0).choice(n - 1, size=count, replace=False)),
    }
    boundaries = (
        np.flatnonzero(getattr(index, "normalize")()[1:] != getattr(index, "normalize")()[:-1]) + 1
    )
    candidates.update(int(c) + delta for c in boundaries for delta in (-1, 0, 1))
    return sorted(c for c in candidates if 0 < c < n)


def _read(path: Path, key: str, node: Feature, n: int) -> Array | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as entry:
            value = entry["value"]
            meta = json.loads(entry["metadata"].tobytes())
        if (
            meta["key"] != key
            or meta["array"] != array_identity(value)
            or value.dtype != node.dtype
            or value.shape != (n,)
        ):
            raise ValueError("checksum/schema mismatch")
    except Exception as exc:
        raise ValueError(f"corrupt feature cache {path}: {exc}") from exc
    _check_nan(node, value, getattr(node.fn, "__name__", "builder"))
    return _immutable(value)


def _write(
    path: Path,
    key: str,
    value: Array,
    cuts: list[int],
    mode: Verification,
    identity: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _publish(path, key, value, cuts, mode, identity)


def _publish(
    path: Path,
    key: str,
    value: Array,
    cuts: list[int],
    mode: Verification,
    identity: dict[str, object],
) -> None:
    # Remove abandoned cache publications, while preserving recent writes.
    now = time.time_ns() / 1e9
    for stale in path.parent.glob("*.tmp"):
        try:
            if now - stale.stat().st_mtime > 3600:
                stale.unlink()
        except FileNotFoundError:
            pass  # another writer cleaned it first
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            previous = json.loads(existing["metadata"].tobytes())
        rank = {None: 0, "sampled": 1, "exhaustive": 2}
        if rank[previous["verification"]] > rank[mode]:
            return  # a weaker hit must not downgrade a stronger receipt
    metadata = {
        "key": key,
        "array": array_identity(value),
        "verification": mode,
        "cuts": cuts,
        "seed": 0,
        "verification_version": 2,
        "identity": identity,
    }
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="qik-feature-", suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            np.savez(
                stream,
                value=value,
                metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def with_features(plan: FeaturePlan[Any]) -> Callable[[Any], Any]:
    """Attach a plan for qik.features.run; the raw engine runner stays unchanged."""

    def attach(step: Any) -> Any:
        if hasattr(step, "feature_plan"):
            raise ValueError("strategy already has a feature plan")
        spec = getattr(step, "__strategy__", None)
        if spec is not None and getattr(spec, "features", plan.schema) is not plan.schema:
            raise ValueError("strategy/plan feature schema differs")
        step.feature_plan = plan
        return step

    return attach


def quantity_from_risk(
    distance: Feature,
    *,
    risk_usd: object,
    multiple: object,
    point_value: object,
    minimum: object = 0.0,
) -> Feature:
    """Decision-known desired quantity; Q1 owns lot rounding, min/cap admission."""
    return feature(inputs={"distance": distance}, dtype="float64", warmup=0, allow_nan=True)(
        _quantity
    )(risk_usd=risk_usd, multiple=multiple, point_value=point_value, minimum=minimum)


def _quantity(
    x: Any, *, risk_usd: float, multiple: float, point_value: float, minimum: float
) -> Array:
    if not all(np.isfinite(v) and v > 0 for v in (risk_usd, multiple, point_value)):
        raise ValueError("quantity risk_usd, multiple and point_value must be positive finite")
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("quantity minimum must be nonnegative finite")
    distance = x.distance.to_numpy()
    out = np.full(len(distance), np.nan)
    good = np.isfinite(distance) & (distance > 0)
    out[good] = np.maximum(minimum, risk_usd / (distance[good] * multiple * point_value))
    return out
