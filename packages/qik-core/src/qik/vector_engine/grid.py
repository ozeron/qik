"""Cartesian parameter tables; typed execution accepts grids directly via ``run``."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import pandas as pd

__all__ = ["grid"]


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
