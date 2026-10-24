"""Parameter grids: one row per config, and the stacking of one-config signals into ``(n, k)``.

    params = grid(sl=[4, 8, 12], tp=[8, 16])              # 6 configs, index = config number
    sig = stack(params, lambda sl, tp: orb(bars, sl, tp))  # each call returns one-config Signals
    res = simulate(bars, sig, ...)
    res.summary().join(params)                             # results next to their parameters

For speed, build the ``(n, k)`` arrays directly by broadcasting the columns instead:
``stop = close[:, None] - side * params.sl.to_numpy()``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .api import SIGNAL_FIELDS, Signals

__all__ = ["grid", "stack"]


def grid(**axes: Sequence[Any]) -> pd.DataFrame:
    """Cartesian product of the axes, first axis varying slowest. Index ``config`` = 0..k-1."""

    if not axes:
        raise ValueError("grid needs at least one axis")
    for name, values in axes.items():
        if isinstance(values, str) or len(values) == 0:
            raise ValueError(f"axis {name!r} must be a non-empty sequence of values")
    rows = list(itertools.product(*axes.values()))
    frame = pd.DataFrame(rows, columns=pd.Index(list(axes)))
    frame.index.name = "config"
    return frame


def _column(a: Any, n: int, name: str, j: int, dtype: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape != (n,):
        raise ValueError(f"config {j}: {name} must have shape ({n},) or ({n}, 1), got {arr.shape}")
    return arr


def stack(params: pd.DataFrame, build: Callable[..., Signals]) -> Signals:
    """Call ``build(**row)`` for every row of ``params`` and stack the one-config results.

    Column ``j`` of the result is row ``j`` of ``params`` (by position). Every ``Signals`` field is
    stacked; a config without one (say a stop) gets NaN in a grid where other configs have it.
    """

    if len(params) == 0:
        raise ValueError("params has no rows")
    built = [build(**row) for row in params.to_dict("records")]
    n = int(np.asarray(built[0].entry_side).shape[0])
    nan = np.full(n, np.nan)

    def columns(name: str, dtype: Any) -> list[np.ndarray]:
        out = []
        for j, sig in enumerate(built):
            a = getattr(sig, name)
            if a is None and name in ("entry_side", "exit"):
                raise ValueError(f"config {j}: {name} is required")
            out.append(nan if a is None else _column(a, n, name, j, dtype))
        return out

    def levels(name: str) -> np.ndarray | None:
        if all(getattr(sig, name) is None for sig in built):
            return None
        return np.column_stack(columns(name, np.float64))

    lv = {
        name: levels(name)
        for name in SIGNAL_FIELDS[2:]
        if name not in ("exit_long", "exit_short", "flatten", "entry_from_open")
    }
    for name in ("exit_long", "exit_short", "flatten", "entry_from_open"):
        if any(getattr(sig, name) is not None for sig in built):
            lv[name] = np.column_stack(
                [
                    np.zeros(n, bool)
                    if getattr(sig, name) is None
                    else _column(getattr(sig, name), n, name, j, bool)
                    for j, sig in enumerate(built)
                ]
            )
    return Signals(
        np.column_stack(columns("entry_side", np.int8)),
        np.column_stack(columns("exit", np.bool_)),
        **lv,
    )
