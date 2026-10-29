"""Shared sizing and fill economics; no scheduling or pandas result assembly.

Costs are subtracted in two steps. Combining fixed and proportional costs changes
rounding. Sequential cash addition and pandas grouped cumsum remain at their
original call sites because they have distinct accumulation contracts.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from .economic import (
    E_ADMISSION,
    E_AMOUNT,
    E_BPS,
    E_CAP,
    E_CHECK_CASH,
    E_CLAMP_MIN,
    E_ENTRY_NOTIONAL,
    E_FRICTION,
    E_LOT_STEP,
    E_MIN_QTY,
    E_MODE,
    E_POINT_VALUE,
)

MODE_CONTRACTS = 0
MODE_RISK_USD = 1
MODE_RISK_FRACTION = 2
MODE_NOTIONAL = 3  # stop-free fixed notional (A24); 4 is reserved for cash_fraction
MODE_QTY_FEATURE = 5  # 4 remains reserved for cash_fraction
ERR_INVALID_QTY = 4
ERR_QTY_RANGE = 5
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
    if mode == MODE_QTY_FEATURE:
        # This mode's submitted sizing input is a quantity, not a stop distance.
        if not np.isfinite(risk_per_contract) or risk_per_contract <= 0:
            return float(-ERR_INVALID_QTY)
        n = np.floor(risk_per_contract / lot_step + 1e-9) * lot_step
        if cap > 0 and n > cap:
            n = cap
        # Whole-contract Result columns use int64; refuse before a cast can wrap
        # a large positive quantity into a negative position. Apply the cap first.
        if not np.isfinite(n) or (lot_step == 1.0 and min_qty == 1.0 and n >= 2.0**63):
            return float(-ERR_QTY_RANGE)
        return float(-ERR_ZERO) if n < min_qty - 1e-10 else n
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


# F0a shadow account, packed after the unchanged six callback bookkeeping cells.
# settled, free, reserved, unmarked cash snapshot, signed quantity, entry
# quantity, entries, skips, submits, releases, closes, entry price, live order
# count, current/last entry identities, last retirement bar, spare, independently
# observed open orders. Actual marked equity and rejection records are F0b scope.
@nb.njit(cache=True)
def _f0a_entry(
    a: np.ndarray,
    economic: np.ndarray,
    side: int,
    price: float,
    stop: float,
    risk_points: float = np.nan,
    allow_skip: bool = True,
) -> bool:
    pv, friction, bps = economic[E_POINT_VALUE], economic[E_FRICTION], economic[E_BPS]
    unit = abs(price) * pv
    qty = quantity(
        int(economic[E_MODE]),
        economic[E_AMOUNT],
        economic[E_CAP],
        a[6],
        # Q1's trade-ordinal snapshot is already a wanted quantity, not a
        # price distance. Resolve it with the very same lot/cap/min helper.
        risk_points
        if economic[E_MODE] == MODE_QTY_FEATURE
        else (abs(price - stop) if np.isnan(risk_points) else risk_points) * pv,
        economic[E_LOT_STEP],
        economic[E_MIN_QTY],
        bool(economic[E_CLAMP_MIN]),
        unit,
        friction * 0.5 + unit * bps / 10000,
        bool(economic[E_CHECK_CASH]),
    )
    if allow_skip and economic[E_ADMISSION] != 0 and (qty == -ERR_ZERO or qty == -ERR_UNAFFORDABLE):
        a[8] = 0.0
        a[7], a[9] = a[6], a[6]
        a[13] += 1
        return False
    # The legacy sizing pass remains the refusal authority in this spike.
    # Invalid shadow quantities cannot alter default matching or errors.
    qty = max(qty, 0.0)
    a[8] = 0.0
    a[7] = a[6] - qty * (friction * 0.5 + unit * bps / 10000)
    a[9] = a[7]
    a[10], a[11], a[17] = side * qty, qty, price
    a[12] += 1
    return True


@nb.njit(cache=True)
def _f0a_exit(a: np.ndarray, economic: np.ndarray, side: int, entry: float, price: float) -> None:
    proportional = fill_cost(
        entry,
        entry if economic[E_ENTRY_NOTIONAL] else price,
        economic[E_POINT_VALUE],
        economic[E_BPS],
    )
    a[6] += a[11] * settle(
        side * (price - entry), economic[E_POINT_VALUE], economic[E_FRICTION], proportional
    )
    a[7], a[9] = a[6], a[6]
    a[8], a[10], a[11] = 0.0, 0.0, 0.0
    a[16] += 1


@nb.njit(cache=True)
def _f0a_mask(long_stop: float, long_ioc: float, short_stop: float, short_ioc: float) -> int:
    return int(not np.isnan(long_stop) or not np.isnan(long_ioc)) | (
        int(not np.isnan(short_stop) or not np.isnan(short_ioc)) << 1
    )


@nb.njit(cache=True)
def _f0a_submit(a: np.ndarray, orders: np.ndarray, key: int, mask: int) -> None:
    """Register entry legs once, including queued replacements. No principal reserve yet."""
    free = -1
    for j in range(orders.shape[1]):
        if orders[0, j] == key:
            extra = mask & ~orders[1, j]
            a[14] += (extra & 1) + ((extra >> 1) & 1)
            a[18] += (extra & 1) + ((extra >> 1) & 1)
            orders[1, j] |= mask
            return
        if orders[0, j] < 0 and free < 0:
            free = j
    if free < 0:
        a[2] = 15  # explicit overflow status, never an exception swallowed by prange
        return
    orders[0, free], orders[1, free] = key, mask
    legs = (mask & 1) + ((mask >> 1) & 1)
    a[14] += legs
    a[18] += legs


@nb.njit(cache=True)
def _f0a_release(a: np.ndarray, orders: np.ndarray, key: int, mask: int, bar: int) -> None:
    for j in range(orders.shape[1]):
        if orders[0, j] == key:
            retired = orders[1, j] & mask
            legs = (retired & 1) + ((retired >> 1) & 1)
            a[15] += legs
            a[18] -= legs
            orders[1, j] &= ~mask
            if orders[1, j] == 0:
                orders[0, j] = -1
                a[20], a[21] = key, bar
                if a[19] == key:
                    a[19] = -1
            return


@nb.njit(cache=True)
def _f0a_pending(a: np.ndarray, orders: np.ndarray, key: int, mask: int, bar: int) -> None:
    if key < 0:
        if a[19] >= 0:
            _f0a_release(a, orders, int(a[19]), 3, bar)
        a[19] = -1
    else:
        _f0a_submit(a, orders, key, mask)
        a[19] = key


@nb.njit(cache=True)
def _f0a_complete(a: np.ndarray, orders: np.ndarray, key: int, bar: int) -> None:
    # Some matchers clear pend just before committing a fill. Its order was
    # already released at that event; do not fabricate another submission.
    present = False
    for j in range(orders.shape[1]):
        present = present or orders[0, j] == key
    if not present and (a[20] != key or a[21] != bar):
        _f0a_submit(a, orders, key, 1)
    _f0a_release(a, orders, key, 3, bar)  # fill plus a straddle/OCO sibling cancel


@nb.njit(cache=True)
def _f0a_clear(a: np.ndarray, orders: np.ndarray, bar: int) -> None:
    for j in range(orders.shape[1]):
        key = orders[0, j]
        if key >= 0:
            _f0a_release(a, orders, key, 3, bar)
    a[19], a[8] = -1, 0.0
