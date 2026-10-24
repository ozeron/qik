"""Strategy cores: one per-bar decision function shared by research and the live sleeve.

A core sees bars one at a time, keeps its own state, and returns a ``Decision`` at each bar's close:

    class Orb:
        def __init__(self, range_bars: int, reward: float) -> None: ...
        def step(self, bar: BarView) -> Decision: ...

    sig = signals_from_core(lambda: Orb(6, 2.0), bars)          # one config, (n,) Signals
    sig = stack(params, lambda **p: signals_from_core(lambda: Orb(**p), bars))

Because ``step`` only ever receives bars up to now, a core cannot look ahead, and the same object can
drive a live strategy's ``on_bar``. A core decides from market data only: the engine's position is
not an input (it is an output of execution, and feeding it back would make configs path-dependent
on their own fills). An entry while a position is open is ignored by the engine, as in ``Signals``.

Vectorized signal functions are faster but can peek; ``check_causal`` tests one against its own
prefixes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd

from .api import SIGNAL_FIELDS, Bars, Signals
from .types import Decision as NumericDecision

__all__ = ["BarView", "Core", "Decision", "check_causal", "signals_from_core"]


class BarView(NamedTuple):
    """One completed bar as a core sees it."""

    i: int
    ts: pd.Timestamp | None
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Decision:
    """What to do at this bar's close; the same fields as one row of ``Signals``."""

    side: int = 0  # +1 / -1: entry decided at this close when flat (market, or the limit below)
    exit: bool = False  # exit decided at this close when in a position; cancels a working entry
    stop: float = math.nan  # absolute level placed with this bar's entry
    target: float = math.nan
    entry_limit: float = math.nan  # LimitThenMarket price for this bar's entry (A15)
    stop_points: float = math.nan  # stop as points from the entry anchor instead of a level
    target_points: float = math.nan
    entry_stop: float = math.nan
    stop_update: float = math.nan
    target_update: float = math.nan
    breakeven_after_points: float = math.nan
    breakeven_offset_points: float = math.nan

    flatten: bool = False
    exit_long: bool = False
    exit_short: bool = False
    entry_limit_ioc: float = math.nan
    entry_stop_long: float = math.nan
    entry_limit_ioc_long: float = math.nan
    stop_long: float = math.nan
    target_long: float = math.nan
    stop_points_long: float = math.nan
    target_points_long: float = math.nan
    entry_stop_short: float = math.nan
    entry_limit_ioc_short: float = math.nan
    stop_short: float = math.nan
    target_short: float = math.nan
    stop_points_short: float = math.nan
    target_points_short: float = math.nan
    entry_from_open: bool = False
    stop_fraction: float = math.nan
    target_fraction: float = math.nan

    def __post_init__(self) -> None:
        if self.side not in (-1, 0, 1):
            raise ValueError(f"side must be -1, 0 or +1, got {self.side!r}")


class Core(Protocol):
    def step(self, bar: BarView) -> Decision | NumericDecision: ...


_LEVELS = tuple(
    name
    for name in SIGNAL_FIELDS
    if name not in ("entry_side", "exit", "exit_long", "exit_short", "flatten", "entry_from_open")
)


def signals_from_core(make: Callable[[], Core], bars: Bars) -> Signals:
    """Run a fresh core over every bar and collect its decisions as one-config ``Signals``.

    A level field no decision ever sets is left ``None``.
    """

    core = make()
    n = len(bars)
    side = np.zeros(n, np.int8)
    exit_ = np.zeros(n, np.bool_)
    exits = {name: np.zeros(n, np.bool_) for name in ("exit_long", "exit_short", "flatten")}
    levels = {name: np.full(n, np.nan) for name in _LEVELS}
    from_open = np.zeros(n, np.bool_)
    vol = bars.volume if bars.volume is not None else np.full(n, np.nan)
    for i in range(n):
        ts = bars.index[i] if bars.index is not None else None
        d = core.step(
            BarView(i, ts, bars.open[i], bars.high[i], bars.low[i], bars.close[i], vol[i])
        )
        if isinstance(d, NumericDecision):
            d = Decision(*d)
        if not isinstance(d, Decision):
            raise TypeError(f"bar {i}: step must return a Decision, got {type(d).__name__}")
        side[i] = d.side
        exit_[i] = d.exit
        for name, arr in exits.items():
            arr[i] = getattr(d, name)
        for name, arr in levels.items():
            arr[i] = getattr(d, name)
        from_open[i] = d.entry_from_open
    used = {name: (arr if not np.isnan(arr).all() else None) for name, arr in levels.items()}
    if from_open.any():
        used["entry_from_open"] = from_open
    return Signals(side, exit_, **exits, **used)


def _prefix(bars: Bars, m: int) -> Bars:
    return Bars(
        bars.open[:m],
        bars.high[:m],
        bars.low[:m],
        bars.close[:m],
        index=None if bars.index is None else bars.index[:m],
        volume=None if bars.volume is None else bars.volume[:m],
        session_labels=None if bars.session_labels is None else bars.session_labels[:m],
    )


def _rows(sig: Signals, m: int) -> list[np.ndarray]:
    """The first ``m`` rows of every ``Signals`` field as float; a missing level is all NaN."""

    shape = np.asarray(sig.entry_side).shape
    out = []
    for name in SIGNAL_FIELDS:
        a = getattr(sig, name)
        arr = np.full(shape, np.nan) if a is None else np.asarray(a, dtype=np.float64)
        out.append(arr[:m])
    return out


def _decision_rows(sig: Signals) -> np.ndarray:
    side = np.asarray(sig.entry_side)
    exit_ = np.asarray(sig.exit)
    any_ = (side != 0) | exit_
    for name in ("exit_long", "exit_short", "flatten"):
        a = getattr(sig, name)
        if a is not None:
            any_ = any_ | np.asarray(a, dtype=bool)
    for name in (
        "stop_update",
        "target_update",
        "entry_stop_long",
        "entry_stop_short",
        "entry_limit_ioc_long",
        "entry_limit_ioc_short",
    ):
        a = getattr(sig, name)
        if a is not None:
            any_ = any_ | np.isfinite(a)
    return np.flatnonzero(any_.reshape(any_.shape[0], -1).any(axis=1))


def check_causal(
    fn: Callable[[Bars], Signals],
    bars: Bars,
    cuts: Sequence[int] | None = None,
    *,
    spaced: int = 100,
) -> int:
    """Raise if ``fn``'s decisions before a cut change when the bars after the cut are removed.

    ``fn`` is a vectorized signal function of the bars alone (bind its parameters first). By default
    the tape is cut right after every bar where the full run decides something (that is where a
    peek that moves or keeps a decision shows) plus ``spaced`` evenly spaced bars (where a peek that
    suppresses one may show). A peek on a bar that is never cut is not seen. Returns the number of
    cuts checked.
    """

    full = fn(bars)
    n = len(bars)
    if cuts is None:
        grid_cuts = np.linspace(1, n - 1, num=min(spaced, n - 1), dtype=np.int64)
        cut_set = set((_decision_rows(full) + 1).tolist()) | set(grid_cuts.tolist())
        cuts = sorted(m for m in cut_set if 0 < m < n)
    for m in cuts:
        if not 0 < m < n:
            raise ValueError(f"cut {m} must be inside (0, {n})")
        part = fn(_prefix(bars, m))
        for name, a, b in zip(SIGNAL_FIELDS, _rows(full, m), _rows(part, m), strict=True):
            if a.shape != b.shape:
                raise ValueError(
                    f"{name}: shape {a.shape} with the full tape but {b.shape} with {m} bars"
                )
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            if bad.any():
                where = np.argwhere(bad.reshape(bad.shape[0], -1))[0]
                i, j = int(where[0]), int(where[1])
                a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
                raise ValueError(
                    f"look-ahead: {name} at bar {i}, config {j} is {float(a2[i, j])!r} with the "
                    f"full tape but {float(b2[i, j])!r} with only the first {m} bars"
                )
    return len(cuts)
