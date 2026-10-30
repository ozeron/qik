"""Shared sizing and fill economics; no scheduling or pandas result assembly.

Costs are subtracted in two steps. Combining fixed and proportional costs changes
rounding. Sequential cash addition and pandas grouped cumsum remain at their
original call sites because they have distinct accumulation contracts.
"""

from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np

from .economic import (
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
    E_ON_REJECT,
    E_POINT_VALUE,
    E_QTY_START,
)
from .protection import levels

type AccountInt = int | np.int64

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

ACCOUNT_PREFIX = 45
ORDER_MULTIPLIER = 8
REGISTRY_ROWS = 8


@nb.njit(cache=True)
def quantity(
    mode: AccountInt,
    amount: float,
    cap: float,
    equity: float,
    risk_per_contract: float,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
    clamp_min_qty: bool | np.bool_ = False,
    notional_per_unit: float = 0.0,
    entry_cost_per_unit: float = 0.0,
    check_cash: bool | np.bool_ = False,
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


# Account prefix after the unchanged six callback bookkeeping cells:
# settled, free, reserved, event mark, signed quantity, entry quantity,
# entries, skips, submits, releases, closes, entry price, live order count,
# current/last entry identities, last retirement bar, order row count, and
# independently observed open orders. Cells 24..39 hold event/transaction metadata;
# fill quantities and packed order rows follow that fixed prefix.
@nb.njit(cache=True)
def _f0a_entry(
    a: np.ndarray,
    economic: np.ndarray,
    side: AccountInt,
    price: float,
    stop: float,
    risk_points: float = np.nan,
    bar: int = -1,
    decision_bar: int = -1,
    capacity: int = 0,
) -> bool:
    a[34] = a[6]
    a[35] = a[10]
    a[37] = a[11]
    a[38] = a[17]
    a[39] = a[31]
    _account_resolve_terminal(a, decision_bar, side, bar)
    a[33] = max(a[33], decision_bar)
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
    record = len(economic) > E_ON_REJECT and (int(economic[E_ON_REJECT]) & 1) != 0
    exhausted = record and a[6] <= 0
    if exhausted:
        qty = -ERR_UNAFFORDABLE
    # Q1 may have classified its invalid lane immediately before this hook.
    if record and a[2] == 1:
        a[2], a[3], a[4] = 0, -1, 0
    if record and qty < 0:
        wanted = quantity(
            int(economic[E_MODE]),
            economic[E_AMOUNT],
            economic[E_CAP],
            a[6],
            risk_points
            if economic[E_MODE] == MODE_QTY_FEATURE
            else (abs(price - stop) if np.isnan(risk_points) else risk_points) * pv,
            economic[E_LOT_STEP],
            0.0,
            np.bool_(False),
            unit,
            friction * 0.5 + unit * bps / 10000,
            np.bool_(False),
        )
        _account_order(
            a,
            capacity,
            bar,
            decision_bar,
            side,
            wanted if wanted >= 0 else np.nan,
            0.0,
            2 if exhausted else 3 if qty in (-ERR_ZERO, -ERR_UNAFFORDABLE) else 2,
            13 if exhausted else int(-qty),
        )
        a[7], a[9] = a[6] - a[8], a[6]
        a[13] += 1
        return False
    # Default refusals retain their legacy host-side precedence.
    qty = max(qty, 0.0)
    a[7] = a[6] - a[8]
    a[9] = a[6] - qty * (friction * 0.5 + unit * bps / 10000)
    a[31] = qty * (friction * 0.5 + unit * bps / 10000)
    a[10], a[11], a[17] = side * qty, qty, price
    a[12] += 1
    _account_order(a, capacity, bar, decision_bar, side, qty, qty, np.int64(1), np.int64(0))
    return True


@nb.njit(cache=True)
def _f0a_exit(
    a: np.ndarray, economic: np.ndarray, side: AccountInt, entry: float, price: float
) -> None:
    proportional = fill_cost(
        entry,
        entry if economic[E_ENTRY_NOTIONAL] else price,
        economic[E_POINT_VALUE],
        economic[E_BPS],
    )
    a[6] += a[11] * settle(
        float(side) * (price - entry), economic[E_POINT_VALUE], economic[E_FRICTION], proportional
    )
    a[7], a[9] = a[6] - a[8], a[6]
    a[10], a[11], a[31] = 0.0, 0.0, 0.0
    a[16] += 1


@nb.njit(cache=True)
def _f0a_mask(long_stop: float, long_ioc: float, short_stop: float, short_ioc: float) -> int:
    return int(not np.isnan(long_stop) or not np.isnan(long_ioc)) | (
        int(not np.isnan(short_stop) or not np.isnan(short_ioc)) << 1
    )


@nb.njit(cache=True)
def _f0a_submit(
    a: np.ndarray,
    orders: np.ndarray,
    key: AccountInt,
    mask: AccountInt,
    reserves: tuple[float, float],
) -> None:
    """Register entry legs and their cash once, including queued replacements."""
    cash = orders.view(np.float64)
    free = -1
    for j in range(orders.shape[1]):
        if orders[0, j] == key:
            extra = mask & ~orders[1, j]
            a[14] += (extra & 1) + ((extra >> 1) & 1)
            a[18] += (extra & 1) + ((extra >> 1) & 1)
            orders[1, j] |= mask
            if extra & 1:
                cash[2, j] = reserves[0]
            if extra & 2:
                cash[3, j] = reserves[1]
            _account_reserved(a, orders)
            _account_submitted(a, orders, j, extra)
            return
        if orders[0, j] < 0 and free < 0:
            free = j
    if free < 0:
        a[2] = 15  # explicit overflow status, never an exception swallowed by prange
        return
    orders[0, free], orders[1, free] = key, mask
    cash[2, free] = reserves[0] if mask & 1 else 0.0
    cash[3, free] = reserves[1] if mask & 2 else 0.0
    legs = (mask & 1) + ((mask >> 1) & 1)
    a[14] += legs
    a[18] += legs
    _account_reserved(a, orders)
    _account_submitted(a, orders, free, mask)


@nb.njit(cache=True)
def _f0a_release(
    a: np.ndarray, orders: np.ndarray, key: AccountInt, mask: AccountInt, bar: AccountInt
) -> None:
    if key < 0:
        return
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
            _account_reserved(a, orders)
            cash = orders.view(np.float64)
            for leg in range(2):
                if retired & (1 << leg):
                    _account_order(
                        a,
                        np.int64(a[29]),
                        bar,
                        key,
                        np.int64(cash[4 + 2 * leg, j]),
                        cash[5 + 2 * leg, j],
                        0.0,
                        np.int64(a[26]),
                        np.int64(a[27]),
                    )
            return


@nb.njit(cache=True)
def _f0a_pending(
    a: np.ndarray,
    orders: np.ndarray,
    key: AccountInt,
    mask: AccountInt,
    bar: AccountInt,
    reserves: tuple[float, float],
) -> None:
    if key < 0:
        if a[19] >= 0:
            _f0a_release(a, orders, np.int64(a[19]), np.int64(3), bar)
        a[19] = -1
    else:
        _f0a_submit(a, orders, key, mask, reserves)
        a[19] = key


@nb.njit(cache=True)
def _f0a_complete(
    a: np.ndarray,
    orders: np.ndarray,
    key: AccountInt,
    bar: AccountInt,
    reserves: tuple[float, float],
    side: AccountInt,
) -> None:
    a[28] = side
    # Some matchers clear pend just before committing a fill. Its order was
    # already released at that event; do not fabricate another submission.
    present = False
    for j in range(orders.shape[1]):
        present = present or orders[0, j] == key
    if not present and (a[20] != key or a[21] != bar):
        _f0a_submit(a, orders, key, np.int64(1), reserves)
    _f0a_release(a, orders, key, np.int64(3), bar)  # fill plus a straddle/OCO sibling cancel


@nb.njit(cache=True)
def _f0a_clear(a: np.ndarray, orders: np.ndarray, bar: AccountInt) -> None:
    for j in range(orders.shape[1]):
        key = orders[0, j]
        if key >= 0:
            _f0a_release(a, orders, key, np.int64(3), bar)
    a[19], a[8] = -1, 0.0


# One packed row layout for event writers and the host schema. No bar allocations.
ORDER_WIDTH = 12


@nb.njit(cache=True, inline="never")
def _account_finish(a: np.ndarray, orders: np.ndarray) -> None:
    """Audit the registry at return and retain its owner through all borrowed calls."""
    legs = 0
    for j in range(orders.shape[1]):
        if orders[0, j] >= 0:
            mask = orders[1, j]
            legs += (mask & 1) + ((mask >> 1) & 1)
    if legs != a[18] or legs != a[23] or a[14] != a[15] + legs:
        a[2] = 15


@nb.njit(cache=True, inline="never")
def _account_order(
    a: np.ndarray,
    capacity: AccountInt,
    bar: AccountInt,
    decision_bar: AccountInt,
    side: AccountInt,
    requested: float,
    filled: float,
    status: AccountInt,
    reason: AccountInt,
) -> None:
    if capacity == 0 or (a[44] != 0 and status == 3 and reason == 10):
        return
    row = int(a[22])
    if row >= ORDER_MULTIPLIER * capacity:
        a[2] = 9  # replay with larger event/trade storage, outside prange
        return
    at = ACCOUNT_PREFIX + capacity + row * ORDER_WIDTH
    a[at], a[at + 1], a[at + 2] = bar, decision_bar, side
    a[at + 3], a[at + 4], a[at + 5], a[at + 6] = requested, filled, status, reason
    for q in range(5):
        a[at + 7 + q] = a[6 + q]
    a[22] += 1


@nb.njit(cache=True, inline="never")
def _account_reserved(a: np.ndarray, orders: np.ndarray) -> None:
    cash = orders.view(np.float64)
    reserved = 0.0
    for j in range(orders.shape[1]):
        if orders[0, j] >= 0:
            if orders[1, j] & 1:
                reserved += cash[2, j]
            if orders[1, j] & 2:
                reserved += cash[3, j]
    a[8] = reserved
    a[7] = a[6] - reserved


@nb.njit(cache=True, inline="never")
def _account_fill_reserve(
    a: np.ndarray, economic: np.ndarray, price: float, stop: float, points: float, key: AccountInt
) -> tuple[float, float]:
    pv = economic[E_POINT_VALUE]
    unit = abs(price) * pv
    risk = (
        economic[E_QTY_START + key]
        if economic[E_MODE] == MODE_QTY_FEATURE
        else (abs(price - stop) if np.isnan(points) else points) * pv
    )
    cost = economic[E_FRICTION] * 0.5 + unit * economic[E_BPS] / 10000
    qty = quantity(
        int(economic[E_MODE]),
        economic[E_AMOUNT],
        economic[E_CAP],
        a[6],
        risk,
        economic[E_LOT_STEP],
        economic[E_MIN_QTY],
        bool(economic[E_CLAMP_MIN]),
        unit,
        cost,
        np.bool_(False),
    )
    a[30] = qty if qty >= 0 else np.nan
    a[32] = a[30]
    cash = qty * (unit + cost) if qty > 0 else 0.0
    return cash, cash


@nb.njit(cache=True, inline="never")
def _account_reserve_leg(
    a: np.ndarray,
    economic: np.ndarray,
    view: Any,
    key: AccountInt,
    ident: AccountInt,
    leg: AccountInt,
    pair: bool,
    quote: float,
    stop_lane: np.ndarray,
    points_lane: np.ndarray,
    limit_lane: np.ndarray,
    trigger_lane: np.ndarray,
    pair_lanes: np.ndarray,
    fraction_lane: np.ndarray,
    market: bool,
) -> float:
    if key < 0:
        return 0.0
    fraction = np.nan
    if view.event_mode == 2:
        offset = 11 + leg * 6 if pair else 0
        stop_row = view.level_rows[offset + 2 if pair else 0]
        points_row = view.level_rows[offset + 4 if pair else 3]
        first_row = view.level_rows[offset if pair else 2]
        second_row = view.level_rows[offset + 1 if pair else 5]
        stop = view.ev_val[ident, stop_row] if ident >= 0 and stop_row >= 0 else np.nan
        points = view.ev_val[ident, points_row] if ident >= 0 and points_row >= 0 else np.nan
        first = view.ev_val[ident, first_row] if ident >= 0 and first_row >= 0 else np.nan
        second = view.ev_val[ident, second_row] if ident >= 0 and second_row >= 0 else np.nan
        fraction_row = view.level_rows[24]
        if not pair and ident >= 0 and fraction_row >= 0:
            fraction = view.ev_val[ident, fraction_row]
    elif pair:
        offset = leg * 6
        stop, points = pair_lanes[offset + 2, key], pair_lanes[offset + 4, key]
        first, second = pair_lanes[offset, key], pair_lanes[offset + 1, key]
    else:
        stop, points = stop_lane[key], points_lane[key]
        first, second = limit_lane[key], trigger_lane[key]
        if key < len(fraction_lane):
            fraction = fraction_lane[key]
    price = (
        quote
        if market
        else first
        if not np.isnan(first)
        else second
        if not np.isnan(second)
        else quote
    )
    side = int(1 - 2 * leg) if pair else int(a[28])
    if economic[E_MODE] == MODE_RISK_USD or economic[E_MODE] == MODE_RISK_FRACTION:
        stop = levels(
            side,
            quote if a[42] == 1 else price,
            stop,
            np.nan,
            points,
            np.nan,
            a[40],
            int(a[41]) if a[40] > 0 else 0,
            int(a[43]),
            price,
            fraction,
            np.nan,
        )[0]
    return _account_fill_reserve(a, economic, price, stop, np.nan, key)[0]


@nb.njit(cache=True, inline="never")
def _account_reserves(
    a: np.ndarray,
    economic: np.ndarray,
    view: Any,
    key: AccountInt,
    ident: AccountInt,
    pair: bool,
    quote: float,
    stop_lane: np.ndarray,
    points_lane: np.ndarray,
    limit_lane: np.ndarray,
    trigger_lane: np.ndarray,
    pair_lanes: np.ndarray,
    side: AccountInt,
    fraction_lane: np.ndarray,
    market: bool,
) -> tuple[float, float]:
    a[28] = side
    left = _account_reserve_leg(
        a,
        economic,
        view,
        key,
        ident,
        np.int64(0),
        pair,
        quote,
        stop_lane,
        points_lane,
        limit_lane,
        trigger_lane,
        pair_lanes,
        fraction_lane,
        market,
    )
    left_quantity = a[30]
    right = (
        _account_reserve_leg(
            a,
            economic,
            view,
            key,
            ident,
            np.int64(1),
            pair,
            quote,
            stop_lane,
            points_lane,
            limit_lane,
            trigger_lane,
            pair_lanes,
            fraction_lane,
            market,
        )
        if pair
        else 0.0
    )
    a[32] = a[30]
    a[30] = left_quantity
    return left, right


@nb.njit(cache=True, inline="never")
def _account_clock(
    a: np.ndarray,
    economic: np.ndarray,
    bar: AccountInt,
    quote: float,
    status: AccountInt,
    reason: AccountInt,
) -> None:
    """Only event sites call this; hold bars never read or write account cells."""
    a[24] = bar
    a[25] = quote
    a[26] = status
    a[27] = reason
    if a[10] != 0:
        a[9] = a[6] + a[10] * (quote - a[17]) * economic[E_POINT_VALUE] - a[31]
    else:
        a[9] = a[6]


@nb.njit(cache=True, inline="never")
def _account_submitted(
    a: np.ndarray, orders: np.ndarray, slot: AccountInt, mask: AccountInt
) -> None:
    cash = orders.view(np.float64)
    key = orders[0, slot]
    a[33] = max(a[33], key)
    for leg in range(2):
        if mask & (1 << leg):
            side = np.int64(a[28]) if mask == 1 and a[28] != 0 else 1 - 2 * leg
            wanted = a[30] if leg == 0 else a[32]
            cash[4 + 2 * leg, slot] = side
            cash[5 + 2 * leg, slot] = wanted
            _account_order(
                a,
                np.int64(a[29]),
                np.int64(a[24]),
                key,
                np.int64(side),
                wanted,
                0.0,
                np.int64(0),
                np.int64(0),
            )


@nb.njit(cache=True, inline="never")
def _account_resolve_terminal(
    a: np.ndarray, key: AccountInt, side: AccountInt, bar: AccountInt
) -> None:
    """Coalesce scheduler retire-before-fill into the atomic fill/sibling outcome."""
    capacity = np.int64(a[29])
    for row in range(np.int64(a[22]) - 1, -1, -1):
        at = ACCOUNT_PREFIX + capacity + row * ORDER_WIDTH
        if a[at] != bar:
            break
        if a[at + 1] == key and a[at + 2] == side and a[at + 5] == 0:
            break
        if a[at + 1] == key and (a[at + 5] == 4 or a[at + 5] == 5):
            if a[at + 2] == side:
                a[at + 5] = 6  # internal retired row, replaced by the committed fill/refusal
            else:
                a[at + 5] = 4
                a[at + 6] = 9  # OCO/straddle sibling
                for q in range(5):
                    a[at + 7 + q] = a[6 + q]


@nb.njit(cache=True, inline="never")
def _account_attempt(a: np.ndarray, key: AccountInt, side: AccountInt, paired: bool) -> None:
    if a[33] == key:
        return
    _account_order(
        a,
        np.int64(a[29]),
        np.int64(a[24]),
        key,
        np.int64(1 if paired else side),
        a[30],
        0.0,
        np.int64(3),
        np.int64(10),
    )
    if paired:
        _account_order(
            a,
            np.int64(a[29]),
            np.int64(a[24]),
            key,
            np.int64(-1),
            a[32],
            0.0,
            np.int64(3),
            np.int64(10),
        )
    a[33] = max(a[33], key)


@nb.njit(cache=True, inline="never")
def _account_kernel(
    a: np.ndarray,
    orders: np.ndarray,
    error: AccountInt,
    key: AccountInt,
    side: AccountInt,
    final_side: AccountInt,
) -> None:
    """A kernel refusal terminates this config; retain its processed event prefix."""
    requested = a[30] if side != 0 else np.nan
    if final_side == 0 and a[10] != 0:
        # The scheduler rejected the entry before committing its position tail.
        # Preserve earlier settled trades while undoing that tentative exposure.
        requested = a[11]
        side = np.int64(1 if a[10] > 0 else -1)
        a[6] = a[34]
        a[10] = a[35]
        a[11] = a[37]
        a[17] = a[38]
        a[31] = a[39]
        a[12] -= 1
        a[7] = a[6] - a[8]
        a[9] = a[6]
        key = np.int64(a[33])
        if side == 0:
            side = np.int64(a[28])
        for row in range(np.int64(a[22]) - 1, -1, -1):
            at = ACCOUNT_PREFIX + np.int64(a[29]) + row * ORDER_WIDTH
            if a[at] != a[24]:
                break
            if a[at + 1] == key and a[at + 5] == 1:
                a[at + 5] = 6
    else:
        # A committed entry is already FILLED. Later protection/command errors
        # are configuration events, not a second terminal outcome for that entry.
        side = np.int64(0)
        requested = np.nan
    before = a[22]
    a[26] = 2
    a[27] = 100 + error
    _f0a_clear(a, orders, np.int64(a[24]))
    if a[22] == before:
        _account_order(
            a,
            np.int64(a[29]),
            np.int64(a[24]),
            key,
            side,
            requested,
            0.0,
            np.int64(2),
            np.int64(100 + error),
        )
