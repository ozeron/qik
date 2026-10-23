"""Causal comparisons; indicator/range construction and entry gates belong to the core."""

from __future__ import annotations

import numpy as np
from numba.extending import register_jitable
from numpy.typing import ArrayLike, NDArray


@register_jitable
def side_of(value: float, reference: float) -> int:
    """Strict side of an MA/indicator; equality and an unready NaN are neutral.

    Admitted: P060 OvernightCore and C001 Seykota EMA bias.
    """
    return 1 if value > reference else -1 if value < reference else 0


def side_of_array(value: ArrayLike, reference: ArrayLike) -> NDArray[np.int8]:
    """Elementwise ``side_of``; numpy broadcasting is explicit in the inputs."""
    v, r = np.asarray(value), np.asarray(reference)
    return np.where(v > r, 1, np.where(v < r, -1, 0)).astype(np.int8)


@register_jitable
def range_breakout(close: float, high: float, low: float) -> int:
    """Strict close breakout, long first; pass completed range bounds only.

    Admitted: P059 BreakoutCore features and P063 StreakCore confirmation.
    NaN bounds disable their own side. Infinite bounds can explicitly disable a side.
    """
    return 1 if close > high else -1 if close < low else 0


def range_breakout_array(close: ArrayLike, high: ArrayLike, low: ArrayLike) -> NDArray[np.int8]:
    """Elementwise ``range_breakout``; never computes a future range."""
    c, h, lo = np.asarray(close), np.asarray(high), np.asarray(low)
    return np.where(c > h, 1, np.where(c < lo, -1, 0)).astype(np.int8)
