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
MODE_NOTIONAL = 3  # stop-free fixed notional (A24); 4 is reserved for cash_fraction
ERR_NONE = 0
ERR_ZERO = 1
ERR_NO_STOP = 2
ERR_UNAFFORDABLE = 3


@nb.njit(cache=True)
def quantity(
    mode: int,
    amount: float,
    cap: float,
    equity: float,
    risk_per_contract: float,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
    clamp_min_qty: bool = False,
    notional_per_unit: float = 0.0,
    entry_cost_per_unit: float = 0.0,
    check_cash: bool = False,
) -> float:
    """Contracts for one entry, or a negative error code.

    ``MODE_NOTIONAL`` sizes from the entry fill's notional per unit (``notional_per_unit``, no
    stop) and, with ``check_cash``, refuses an entry whose notional plus the entry fill's costs
    (``entry_cost_per_unit``) exceeds ``equity``. The other modes ignore the last three arguments.
    """

    if mode == MODE_CONTRACTS:
        return amount
    if mode == MODE_NOTIONAL:
        risk_per_contract = notional_per_unit
    if not (risk_per_contract > 0.0):  # NaN stop
        return float(-ERR_NO_STOP)
    budget = amount * equity if mode == MODE_RISK_FRACTION else amount
    if not (budget > 0.0):
        return float(-ERR_ZERO)
    n = np.floor(budget / risk_per_contract / lot_step + 1e-9) * lot_step
    if cap > 0 and n > cap:
        n = cap
    if clamp_min_qty:
        n = max(min_qty, n)
    if n < min_qty - 1e-10:
        return float(-ERR_ZERO)
    # Relative slack absorbs float rounding of an exact decimal cost (3 x 100.01 is
    # 300.03000000000003); a real shortfall of a cent on any balance is still refused.
    if check_cash and n * (notional_per_unit + entry_cost_per_unit) > equity + 1e-12 * max(
        abs(equity), 1.0
    ):
        return float(-ERR_UNAFFORDABLE)
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
