"""Bounded, opt-in preparation around the unchanged typed runner."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._identity import digest, normalize, scalar


def run(
    step: Any,
    bars: Any,
    *,
    inputs: Any = None,
    feature_plan: Any = None,
    grid: Any = None,
    features: Any = None,
    index: Any = None,
    source_manifest: Any = None,
    verify_features: Any = None,
    cache: Any = None,
    **kwargs: Any,
) -> Any:
    """Run raw arrays unchanged, or stream plan groups in schema Cartesian order.

    Explicit inputs carry the decision clock. An attached plan is used only when
    no raw features are supplied; explicitly combining raw arrays and a plan refuses.
    """
    from qik.vector_engine import run as engine_run

    attached = getattr(step, "feature_plan", None)
    if feature_plan is not None and attached is not None and feature_plan is not attached:
        raise ValueError("conflicting feature plans")
    if features is not None:
        if (
            feature_plan is not None
            or attached is not None
            or inputs is not None
            or verify_features is not None
        ):
            raise ValueError("raw features conflict with managed preparation")
        return engine_run(step, bars, grid=grid, features=features, **kwargs)
    plan = feature_plan if feature_plan is not None else attached
    if plan is None:
        return engine_run(step, bars, grid=grid, **kwargs)
    from qik.vector_engine.compiled import GridResult
    from qik.vector_engine.fixed import _feature_write_risk

    from . import Param

    spec = getattr(step, "__strategy__", None)
    if spec is None:
        raise TypeError("managed features require a @strategy-decorated function")
    if spec.features is not plan.schema:
        raise ValueError("strategy/plan feature schema differs")
    if risk := _feature_write_risk(step):
        raise ValueError(f"managed strategy may write features: {risk}")
    nodes = plan._ordered()
    dependencies = {v.name for n in nodes for v in n.bindings.values() if isinstance(v, Param)}
    names = spec.params.dtype.names or ()
    if unknown := dependencies.difference(names):
        raise ValueError(f"feature plan: unknown strategy params {sorted(unknown)}")
    rows = spec.params.from_grid({} if grid is None else grid)
    if not len(rows):
        raise ValueError("grid axes must be nonempty")
    qty = getattr(kwargs.get("sizing"), "qty_feature", None)
    if qty is not None:
        if qty not in plan.nodes:
            raise ValueError(f"qty_feature={qty!r} is not a declared strategy feature")
        if plan.nodes[qty].dtype.kind not in "fiu":
            raise ValueError("qty_feature must be a numeric quantity lane")
    if inputs is None:
        raise ValueError("managed features require explicit inputs and decision clock")
    clock = inputs.index if index is None and isinstance(inputs, pd.DataFrame) else index
    if not isinstance(clock, pd.DatetimeIndex) or len(clock) != len(bars):
        raise ValueError("feature inputs/decision clock must match bars")
    if bars.index is not None and not clock.equals(bars.index):
        raise ValueError("feature decision clock differs from bars.index")
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key = digest({n: scalar(row[n].item()) for n in names if n in dependencies})
        groups.setdefault(key, []).append(i)
    results: list[Any] = [None] * len(rows)
    states = spec.state.new(len(rows))
    # Only parameter-independent nodes survive between launches. Never retain
    # an array per grid row or all parameterized group lanes.
    shared: dict[Any, Any] = {}
    for indices in groups.values():
        first = rows[indices[0]]
        group_grid = dict(grid or {})
        for name in dependencies:
            original = group_grid.get(name, [first[name].item()])
            group_grid[name] = [
                v for v in original if scalar(normalize(v)) == scalar(first[name].item())
            ]
        lanes = plan.materialize(
            inputs,
            index=clock,
            params={n: first[n].item() for n in names},
            source_manifest=source_manifest,
            verify_features=verify_features,
            cache=cache,
            _shared=shared,
        )
        batch = engine_run(step, bars, grid=group_grid, features=lanes, **kwargs)
        if len(batch) != len(indices):
            raise RuntimeError("feature group lost Cartesian rows")
        for local, original in enumerate(indices):
            results[original] = batch[local]
            states[original] = batch.states[local]
            results[original].states = states[original : original + 1]
        del lanes, batch
    if grid is None:
        return results[0]
    table = pd.DataFrame.from_records(rows) if names else pd.DataFrame(index=range(len(rows)))
    return GridResult(tuple(results), table, states)
