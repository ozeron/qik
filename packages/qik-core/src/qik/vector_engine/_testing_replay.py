"""Immutable numeric decision tapes for typed fixture replay."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .api import Bars, Costs, Instrument, Result, Rules, Sizing
from .compiled import NAUTILUS, run
from .schema import BoolArray, Features, FloatArray, IntArray, Params, State, strategy
from .types import Bar, Decision, Position

SIGNAL_FIELDS = ("entry_side", *Decision._fields[1:])


class ReplayState(State):
    """Replay never consults or changes execution state."""


class ReplayParams(Params):
    """One recorded column; caller owns configuration ordering."""


class ReplayFeatures(Features):
    """One immutable array per numeric Decision field, in ABI order."""

    side: IntArray
    exit: BoolArray
    stop: FloatArray
    target: FloatArray
    entry_limit: FloatArray
    stop_points: FloatArray
    target_points: FloatArray
    entry_stop: FloatArray
    stop_update: FloatArray
    target_update: FloatArray
    breakeven_after_points: FloatArray
    breakeven_offset_points: FloatArray
    flatten: BoolArray
    exit_long: BoolArray
    exit_short: BoolArray
    entry_limit_ioc: FloatArray
    entry_stop_long: FloatArray
    entry_limit_ioc_long: FloatArray
    stop_long: FloatArray
    target_long: FloatArray
    stop_points_long: FloatArray
    target_points_long: FloatArray
    entry_stop_short: FloatArray
    entry_limit_ioc_short: FloatArray
    stop_short: FloatArray
    target_short: FloatArray
    stop_points_short: FloatArray
    target_points_short: FloatArray


@strategy(state=ReplayState, params=ReplayParams, features=ReplayFeatures)
def replay_step(
    state: ReplayState, params: ReplayParams, features: ReplayFeatures, bar: Bar, pos: Position
) -> Decision:
    """Emit the recorded decision at the original bar index, independent of fills."""
    return Decision(
        features.side[bar.i],
        features.exit[bar.i],
        features.stop[bar.i],
        features.target[bar.i],
        features.entry_limit[bar.i],
        features.stop_points[bar.i],
        features.target_points[bar.i],
        features.entry_stop[bar.i],
        features.stop_update[bar.i],
        features.target_update[bar.i],
        features.breakeven_after_points[bar.i],
        features.breakeven_offset_points[bar.i],
        features.flatten[bar.i],
        features.exit_long[bar.i],
        features.exit_short[bar.i],
        features.entry_limit_ioc[bar.i],
        features.entry_stop_long[bar.i],
        features.entry_limit_ioc_long[bar.i],
        features.stop_long[bar.i],
        features.target_long[bar.i],
        features.stop_points_long[bar.i],
        features.target_points_long[bar.i],
        features.entry_stop_short[bar.i],
        features.entry_limit_ioc_short[bar.i],
        features.stop_short[bar.i],
        features.target_short[bar.i],
        features.stop_points_short[bar.i],
        features.target_points_short[bar.i],
    )


def replay_features(decisions: Sequence[Decision]) -> ReplayFeatures:
    """Copy a column into arrays backed by immutable bytes (including NaN payloads).

    No field discovery from finite values: declaration masks belong to the caller
    and are passed explicitly as ``signal_fields`` to ``replay_decisions``.
    """
    if not all(isinstance(row, Decision) for row in decisions):
        raise TypeError("replay requires 28-field numeric Decisions")
    if any(row.side not in (-1, 0, 1) for row in decisions):
        raise ValueError("replay side must be -1, 0 or +1")

    def floats(name: str) -> FloatArray:
        values = np.asarray([getattr(row, name) for row in decisions], dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("levels must be finite or NaN")
        return np.frombuffer(values.tobytes(), dtype=np.float64)

    def flags(name: str) -> BoolArray:
        values = np.asarray([getattr(row, name) for row in decisions], dtype=np.bool_)
        return np.frombuffer(values.tobytes(), dtype=np.bool_)

    side = np.asarray([row.side for row in decisions], dtype=np.int64)
    return ReplayFeatures(
        side=np.frombuffer(side.tobytes(), dtype=np.int64),
        exit=flags("exit"),
        stop=floats("stop"),
        target=floats("target"),
        entry_limit=floats("entry_limit"),
        stop_points=floats("stop_points"),
        target_points=floats("target_points"),
        entry_stop=floats("entry_stop"),
        stop_update=floats("stop_update"),
        target_update=floats("target_update"),
        breakeven_after_points=floats("breakeven_after_points"),
        breakeven_offset_points=floats("breakeven_offset_points"),
        flatten=flags("flatten"),
        exit_long=flags("exit_long"),
        exit_short=flags("exit_short"),
        entry_limit_ioc=floats("entry_limit_ioc"),
        entry_stop_long=floats("entry_stop_long"),
        entry_limit_ioc_long=floats("entry_limit_ioc_long"),
        stop_long=floats("stop_long"),
        target_long=floats("target_long"),
        stop_points_long=floats("stop_points_long"),
        target_points_long=floats("target_points_long"),
        entry_stop_short=floats("entry_stop_short"),
        entry_limit_ioc_short=floats("entry_limit_ioc_short"),
        stop_short=floats("stop_short"),
        target_short=floats("target_short"),
        stop_points_short=floats("stop_points_short"),
        target_points_short=floats("target_points_short"),
    )


def replay_decisions(
    decisions: Sequence[Decision],
    bars: Bars,
    *,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    cash_accumulation: str = "batch",
) -> Result[ReplayState]:
    """Run one fixed decision column through the ordinary typed strategy path.

    Preserve the original declaration mask, including explicitly all-NaN fields.
    Replay does not infer masks, transform terminal decisions, or consult Position.
    Empty final state is intentional; source-core state is not reconstructed.
    """
    return run(
        replay_step,
        bars,
        features=replay_features(decisions),
        signal_fields=signal_fields,
        rules=rules,
        instrument=instrument,
        costs=costs,
        sizing=sizing,
        initial_cash=initial_cash,
        jit=jit,
        cash_accumulation=cash_accumulation,
    )


def prefix_bars(bars: Bars, n: int) -> Bars:
    """Slice a tape without changing its clock, volume or session labels."""
    return Bars(
        bars.open[:n],
        bars.high[:n],
        bars.low[:n],
        bars.close[:n],
        index=None if bars.index is None else bars.index[:n],
        volume=None if bars.volume is None else bars.volume[:n],
        session_labels=None if bars.session_labels is None else bars.session_labels[:n],
    )


def signal_rows(capture: object, n: int) -> list[FloatArray]:
    """Normalize captured field arrays for prefix checks; absent fields stay NaN.

    Accept the capture boundary structurally, without depending on legacy Core.
    """
    shape = np.asarray(getattr(capture, "entry_side")).shape
    rows = []
    for name in SIGNAL_FIELDS:
        value = getattr(capture, name)
        array = np.full(shape, np.nan) if value is None else np.asarray(value, dtype=np.float64)
        rows.append(array[:n])
    return rows


def declared_signal_fields(capture: object) -> tuple[str, ...]:
    """Retain absent versus declared all-NaN fields from a captured column."""
    return tuple(name for name in SIGNAL_FIELDS if getattr(capture, name) is not None)


def captured_decisions(capture: object) -> tuple[Decision, ...]:
    """Convert one recorded column to immutable numeric decisions, without execution."""
    rows = signal_rows(capture, len(getattr(capture, "entry_side")))
    if any(row.ndim != 1 for row in rows):
        raise ValueError("capture must contain exactly one decision column")
    decisions = []
    for values in zip(*rows, strict=True):
        values = list(values)
        if values[0] not in (-1, 0, 1):
            raise ValueError("capture side must be -1, 0 or +1")
        values[0] = int(values[0])
        for i in (1, 12, 13, 14):
            values[i] = False if np.isnan(values[i]) else bool(values[i])
        decisions.append(Decision(*values))
    return tuple(decisions)
