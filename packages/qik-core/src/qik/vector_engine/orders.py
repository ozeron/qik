"""Public order table projection of the engine's single packed event layout."""

from __future__ import annotations

import numpy as np
import pandas as pd

ORDER_STATUSES = ("OPEN", "FILLED", "REFUSED", "SKIPPED", "CANCELLED", "EXPIRED")
ORDER_REASONS = (
    "",
    "below_min",
    "stop_required",
    "insufficient_cash",
    "invalid_quantity",
    "quantity_range",
    "cancelled",
    "day_expiry",
    "chase_cancel",
    "oco_sibling",
    "busy",
    "stop_entry_skip",
    "ioc_expiry",
    "cash_exhausted",
)
ORDER_SCHEMA_VERSION = 1
KERNEL_REASONS = (
    "",
    "stop_in_market",
    "target_in_market",
    "stale_book",
    "protection_rejected",
    "pending_new_day",
    "next_open_session_end",
    "stop_entry_in_market",
    "mit_in_market",
    "amend_rejected",
    "amend_no_order",
    "stop_replace_latency",
    "straddle_side_cancel",
    "busy_entry_dropped",
    "rearm_before_close",
    "ring_overflow",
)
ORDER_COLUMNS = (
    "config",
    "order_id",
    "bar",
    "time",
    "decision_bar",
    "side",
    "requested_quantity",
    "filled_quantity",
    "status",
    "reason_code",
    "reason",
    "settled",
    "free",
    "reserved",
    "marked",
    "quantity",
)


def order_frame(
    rows: np.ndarray, clock: pd.DatetimeIndex | None, *, terminal: bool = False
) -> pd.DataFrame:
    rows = rows[rows[:, 5] != 6]
    count = len(rows)
    bar = rows[:, 0].astype(np.int64)
    # First submissions retain their v1 identity. Replacement submissions and
    # configuration-only refusals use unique negative row identities.
    identities = np.empty(count, np.int64)
    active: dict[tuple[int, int], int] = {}
    next_id = -1
    for i, row in enumerate(rows):
        key = (int(row[1]), int(row[2]))
        if key[1] == 0:
            ident = next_id
            next_id -= 1
        elif row[5] == 0:
            ident = next_id if key in active else 2 * key[0] + (key[1] < 0)
            if key in active:
                next_id -= 1
            active[key] = ident
        else:
            ident = active.get(key, 2 * key[0] + (key[1] < 0))
        identities[i] = ident
    frame = pd.DataFrame(
        {
            "config": np.zeros(count, np.int64),
            "order_id": identities,
            "bar": bar,
            "time": pd.to_datetime(np.full(count, np.datetime64("NaT", "ns")), utc=True)
            if clock is None
            else clock[bar],
            "decision_bar": rows[:, 1].astype(np.int64),
            "side": rows[:, 2].astype(np.int64),
            "requested_quantity": rows[:, 3],
            "filled_quantity": rows[:, 4],
            "status": pd.Categorical(
                [ORDER_STATUSES[int(x)] for x in rows[:, 5]], categories=ORDER_STATUSES
            ),
            "reason_code": rows[:, 6].astype(np.int64),
            "reason": pd.Series(
                [
                    ORDER_REASONS[int(x)] if x < 100 else f"kernel:{KERNEL_REASONS[int(x) - 100]}"
                    for x in rows[:, 6]
                ],
                dtype="str",
            ),
            **{
                name: rows[:, 7 + i]
                for i, name in enumerate(("settled", "free", "reserved", "marked", "quantity"))
            },
        }
    )
    if terminal:
        frame = frame[frame.status != "OPEN"].reset_index(drop=True)
    return frame.loc[:, ORDER_COLUMNS]
