"""Typed scripted-decision fixtures shared by the qik engine tests.

A script is the open-loop decision columns a test used to hand the engine as array
signals: side and exit per bar, plus the optional fields it declares. Absent and
declared all-NaN fields stay distinct. ``replay`` executes each column through
``replay_decisions`` (typed ``run``, empty state, original decision clock) and
assembles one Result numbered by configuration, in column order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from qik.vector_engine import (
    Bar,
    Bars,
    Costs,
    Decision,
    Features,
    FloatArray,
    Instrument,
    IntArray,
    Params,
    Position,
    Result,
    Rules,
    Sizing,
    State,
    strategy,
)
from qik.vector_engine.testing import SIGNAL_FIELDS, replay_decisions

_OPTIONAL = SIGNAL_FIELDS[2:]
_FLAGS = frozenset(("flatten", "exit_long", "exit_short", "entry_from_open"))


def _columns(values: ArrayLike, dtype: type) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    return array[:, None] if array.ndim == 1 else array


@dataclass(frozen=True)
class Script:
    """Decision columns, one per configuration, and the optional fields they declare."""

    entry_side: np.ndarray  # (n_bars, n_configs)
    exit: np.ndarray
    fields: Mapping[str, np.ndarray]

    @property
    def n_configs(self) -> int:
        return self.entry_side.shape[1]

    def get(self, name: str) -> np.ndarray | None:
        """A declared field's (n_bars, n_configs) values, or None when absent."""
        return self.fields.get(name)

    def replace(self, **fields: ArrayLike | None) -> Script:
        """The same columns with fields declared, overwritten or (None) removed."""
        merged: dict[str, ArrayLike | None] = dict(self.fields)
        merged.update(fields)
        return script(self.entry_side, self.exit, **merged)

    def decisions(self, j: int) -> tuple[Decision, ...]:
        """Column j as numeric Decisions; absent fields hold the Decision defaults."""
        n = len(self.entry_side)
        values = [self.entry_side[:, j].tolist(), self.exit[:, j].tolist()]
        for name, default in zip(_OPTIONAL, Decision()[2:], strict=True):
            field = self.fields.get(name)
            values.append([default] * n if field is None else field[:, j].tolist())
        return tuple(Decision._make(row) for row in zip(*values, strict=True))


def script(entry_side: ArrayLike, exit: ArrayLike, **fields: ArrayLike | None) -> Script:
    """Columns from 1-D (one config) or (n_bars, n_configs) arrays; None leaves a field absent."""
    if unknown := set(fields).difference(_OPTIONAL):
        raise TypeError(f"unknown script fields: {sorted(unknown)}")
    side = _columns(entry_side, np.int64)
    declared = {
        name: _columns(value, bool if name in _FLAGS else np.float64)
        for name, value in fields.items()
        if value is not None
    }
    exits = _columns(exit, bool)
    if any(a.shape[1] != side.shape[1] for a in (exits, *declared.values())):
        raise ValueError("all script fields must have the same number of configs")
    return Script(side, exits, declared)


def replay(
    bars: Bars,
    scripts: Script | Sequence[Script],
    *,
    rules: Rules = Rules(),
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
) -> Result:
    """Replay every column in order; the scripts' declared fields are their union."""
    scripts = [scripts] if isinstance(scripts, Script) else list(scripts)
    declared = [name for name in _OPTIONAL if any(s.get(name) is not None for s in scripts)]
    results = [
        replay_decisions(
            s.decisions(j),
            bars,
            signal_fields=("entry_side", "exit", *declared),
            rules=rules,
            instrument=instrument,
            costs=costs,
            sizing=sizing,
            initial_cash=initial_cash,
            jit=jit,
        )
        for s in scripts
        for j in range(s.n_configs)
    ]
    return results[0] if len(results) == 1 else combine(results)


def combine(results: Sequence[Result]) -> Result:
    """One Result over independent single-config results, numbered in order."""

    def frames(name: str) -> pd.DataFrame:
        parts = [getattr(r, name).assign(config=np.int64(j)) for j, r in enumerate(results)]
        nonempty = [p for p in parts if len(p)] or parts[:1]
        return pd.concat(nonempty, ignore_index=True)

    return replace(
        results[0],
        trades=frames("trades"),
        open_positions=frames("open_positions"),
        n_configs=len(results),
        research_only=any(r.research_only for r in results),
    )


class Empty(State):
    """Fixture strategies below keep no state."""


class NoParams(Params):
    """Fixture strategies below take no parameters."""


class StopUpdates(Features):
    stop_update: FloatArray


@strategy(state=Empty, params=NoParams, features=StopUpdates)
def stop_update_step(
    state: Empty, params: NoParams, features: StopUpdates, bar: Bar, pos: Position
) -> Decision:
    """Never enters; amends only, from a precomputed column (look-ahead fixtures)."""
    return Decision(stop_update=features.stop_update[bar.i])


@strategy(state=Empty, params=NoParams, features=StopUpdates)
def infinite_stop_update_step(
    state: Empty, params: NoParams, features: StopUpdates, bar: Bar, pos: Position
) -> Decision:
    """Emits a non-finite amendment, which execution must refuse rather than drop."""
    return Decision(stop_update=np.inf)


class BracketTicks(Params):
    stop_ticks: int
    target_ticks: int
    tick: float = 0.25


class SidesAndCloses(Features):
    side: IntArray
    close: FloatArray


@strategy(state=Empty, params=BracketTicks, features=SidesAndCloses)
def bracket_step(
    state: Empty, params: BracketTicks, features: SidesAndCloses, bar: Bar, pos: Position
) -> Decision:
    """A fixed side column with brackets a grid of tick distances from each close."""
    side = features.side[bar.i]
    close = features.close[bar.i]
    return Decision(
        side,
        False,
        close - side * params.stop_ticks * params.tick,
        close + side * params.target_ticks * params.tick,
    )
