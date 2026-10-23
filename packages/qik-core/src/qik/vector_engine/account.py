"""Shared sizing and fill economics; no scheduling or pandas result assembly.

Costs are subtracted in two steps. Combining fixed and proportional costs changes
rounding. Sequential cash addition and pandas grouped cumsum remain at their
original call sites because they have distinct accumulation contracts.
"""

from __future__ import annotations

import numba as nb
import numpy as np

MODE_CONTRACTS = 0
MODE_RISK_USD = 1
MODE_RISK_FRACTION = 2
ERR_NONE = 0
ERR_ZERO = 1
ERR_NO_STOP = 2


@nb.njit(cache=True)
def quantity(
    mode: int,
    amount: float,
    cap: float,
    equity: float,
    risk_per_contract: float,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
) -> float:
    """Contracts for one entry, or a negative error code."""

    if mode == MODE_CONTRACTS:
        return amount
    if not (risk_per_contract > 0.0):  # NaN stop
        return float(-ERR_NO_STOP)
    budget = amount if mode == MODE_RISK_USD else amount * equity
    if not (budget > 0.0):
        return float(-ERR_ZERO)
    n = np.floor(budget / risk_per_contract / lot_step + 1e-9) * lot_step
    if cap > 0 and n > cap:
        n = cap
    if n < min_qty - 1e-10:
        return float(-ERR_ZERO)
    return n


@nb.njit(cache=True)
def fixed_cost(
    commission: float,
    slippage: float,
    tick: float,
    point_value: float,
    round_trip: float = 2.0,
) -> float:
    """Fixed round-trip friction per unit; retain multiplication order."""
    return round_trip * (commission + slippage * tick * point_value)


@nb.njit(cache=True)
def fill_cost(
    entry: float | np.ndarray, exit: float | np.ndarray, point_value: float, bps: float
) -> float | np.ndarray:
    """Proportional round-trip cost per unit, for scalar or array fills."""
    return (np.abs(entry) + np.abs(exit)) * point_value * bps / 10000


@nb.njit(cache=True)
def settle(
    gross_points: float | np.ndarray,
    point_value: float,
    friction: float,
    proportional: float | np.ndarray,
) -> float | np.ndarray:
    """Net per unit, preserving the original two subtractions."""
    return gross_points * point_value - friction - proportional
