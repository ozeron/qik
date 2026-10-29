"""Independent Q1 order/quantity interpreter. Standard library only; no engine imports."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import isfinite


def quantity(value: float, lot: float = 1, minimum: float = 1, cap: float | None = None) -> float:
    if not isfinite(value) or value <= 0:
        raise ValueError("invalid quantity")
    v, step = Fraction(str(value)), Fraction(str(lot))
    result = (v // step) * step
    if cap is not None:
        result = min(result, Fraction(str(cap)))
    if lot == 1 and minimum == 1 and result >= 2**63:
        raise ValueError("unrepresentable quantity")
    if result < Fraction(str(minimum)):
        raise ValueError("below minimum")
    return float(result)


def entry(
    bars: Sequence[tuple[float, float, float, float]],
    lane: Sequence[float],
    decision: int,
    kind: str,
    level: float = 0,
    chase: int = 10,
    delay: int = 0,
    lot: float = 1,
    minimum: float = 1,
    cap: float | None = None,
) -> tuple[int, float, float]:
    """One long order; the submission retains its quantity through every matching event."""
    if kind == "close":
        fill, price = decision, bars[decision][3]
    elif kind == "next_open":
        fill = decision + 1
        price = bars[fill][0]
    else:
        fill, price = -1, 0.0
        for i in range(decision + delay, len(bars)):
            o, h, lo, c = bars[i]
            if kind == "limit" and (c <= level if i == decision + delay else lo <= level):
                fill, price = i, (c if i == decision + delay else level)
                break
            if kind == "stop" and i > decision and h >= level:
                fill, price = i, max(o, level)
                break
            if kind == "limit" and i == decision + chase:
                fill, price = i, c
                break
        assert fill >= 0, "fixture must fill"
    return fill, price, quantity(lane[decision], lot, minimum, cap)
