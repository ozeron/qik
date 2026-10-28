"""Paired-entry schema and admission, shared by array and close-time callers."""

import numpy as np
from numba.extending import register_jitable

PAIR_FIELDS = tuple(
    f"{field}_{side}"
    for side in ("long", "short")
    for field in ("entry_stop", "entry_limit_ioc", "stop", "target", "stop_points", "target_points")
)


PAIR_MESSAGES = (
    "straddle levels must be finite and positive or NaN",
    "straddle cannot share a decision with single entry, protection or commands",
    "straddle needs exactly one stop or IOC level per leg",
    "straddle stop level and points need stop_combine=tighter",
    "straddle target level and points need target_combine=tighter",
    "straddle stop triggers must be ordered short < long",
    "straddle arm_stop=next_bar composition is unproved",
    "straddle day_order_expiry=cancel composition is unproved",
    "straddle OTO composition is unproved; use callback protection",
    "straddle ONE_TICK execution is unproved",
    "straddle ONE_TICK IOC execution is unproved",
    "straddle ONE_TICK targets execution is unproved",
)


@register_jitable
def pair_refusal(
    pair: np.ndarray,
    legacy: np.ndarray,
    side: float,
    commands: bool,
    combine: int,
    oto: bool,
    one_tick: bool,
    arm_stop: bool = False,
    day_cancel: bool = False,
    next_open: bool = False,
) -> int:
    """What admission says of a pair decision, as a code, never raising: 0 for no pair at all,
    1 for a valid pair, ``2 + k`` for the refusal ``PAIR_MESSAGES[k]``. The compiled column
    code carries it out of the parallel loop in the account row, where a raise would be lost."""
    all_nan = True
    for value in pair:
        if not np.isnan(value):
            all_nan = False
            break
    if all_nan:
        return 0
    for value in pair:
        if np.isinf(value) or value <= 0:
            return 2
    legacy_finite = False
    for value in legacy:
        if np.isfinite(value):
            legacy_finite = True
            break
    if side or commands or legacy_finite:
        return 3
    for offset in (0, 6):
        if np.isnan(pair[offset]) == np.isnan(pair[offset + 1]):
            return 4
        if not combine & 1 and not np.isnan(pair[offset + 2]) and not np.isnan(pair[offset + 4]):
            return 5
        if not combine & 2 and not np.isnan(pair[offset + 3]) and not np.isnan(pair[offset + 5]):
            return 6
    if not np.isnan(pair[0]) and not np.isnan(pair[6]) and pair[6] >= pair[0]:
        return 7
    if arm_stop:
        return 8
    if day_cancel:
        return 9
    if oto:
        return 10
    if one_tick:
        if not next_open:
            return 11
        if not np.isnan(pair[1]) or not np.isnan(pair[7]):
            return 12
        for offset in (0, 6):
            if not np.isnan(pair[offset + 3]) or not np.isnan(pair[offset + 5]):
                return 13
    return 1


@register_jitable
def validate_pair(
    pair: np.ndarray,
    legacy: np.ndarray,
    side: float,
    commands: bool,
    combine: int,
    oto: bool,
    one_tick: bool,
    arm_stop: bool = False,
    day_cancel: bool = False,
    next_open: bool = False,
) -> bool:
    """Validate before an order can be sent: ``pair_refusal`` as an exception, for the array API
    and the Python reference (``jit=False``). Returns whether there is a pair to admit.

    The messages are literals, not ``PAIR_MESSAGES`` lookups: compiled, a raise takes constants
    (a tuple lookup raised the first message for every code); the table is for Python callers,
    and ``test_scalar_validation`` pins the two together."""
    code = pair_refusal(
        pair, legacy, side, commands, combine, oto, one_tick, arm_stop, day_cancel, next_open
    )
    if code == 2:
        raise ValueError("straddle levels must be finite and positive or NaN")
    if code == 3:
        raise ValueError(
            "straddle cannot share a decision with single entry, protection or commands"
        )
    if code == 4:
        raise ValueError("straddle needs exactly one stop or IOC level per leg")
    if code == 5:
        raise ValueError("straddle stop level and points need stop_combine=tighter")
    if code == 6:
        raise ValueError("straddle target level and points need target_combine=tighter")
    if code == 7:
        raise ValueError("straddle stop triggers must be ordered short < long")
    if code == 8:
        raise ValueError("straddle arm_stop=next_bar composition is unproved")
    if code == 9:
        raise ValueError("straddle day_order_expiry=cancel composition is unproved")
    if code == 10:
        raise ValueError("straddle OTO composition is unproved; use callback protection")
    if code == 11:
        raise ValueError("straddle ONE_TICK execution is unproved")
    if code == 12:
        raise ValueError("straddle ONE_TICK IOC execution is unproved")
    if code == 13:
        raise ValueError("straddle ONE_TICK targets execution is unproved")
    return code == 1
