"""Paired-entry schema and admission, shared by array and close-time callers."""

import numpy as np
from numba.extending import register_jitable

PAIR_FIELDS = tuple(
    f"{field}_{side}"
    for side in ("long", "short")
    for field in ("entry_stop", "entry_limit_ioc", "stop", "target", "stop_points", "target_points")
)


@register_jitable
def validate_pair(
    pair: np.ndarray,
    legacy: np.ndarray,
    side: float,
    commands: bool,
    combine: int,
    oto: bool,
    one_tick: bool,
) -> bool:
    """Validate before an order can be sent, including from a compiled strategy."""
    if np.isnan(pair).all():
        return False
    if np.isinf(pair).any() or np.any(pair <= 0):
        raise ValueError("straddle levels must be finite and positive or NaN")
    if side or commands or np.isfinite(legacy).any():
        raise ValueError(
            "straddle cannot share a decision with single entry, protection or commands"
        )
    for offset in (0, 6):
        if np.isnan(pair[offset]) == np.isnan(pair[offset + 1]):
            raise ValueError("straddle needs exactly one stop or IOC level per leg")
        if not combine & 1 and not np.isnan(pair[offset + 2]) and not np.isnan(pair[offset + 4]):
            raise ValueError("straddle stop level and points need stop_combine=tighter")
        if not combine & 2 and not np.isnan(pair[offset + 3]) and not np.isnan(pair[offset + 5]):
            raise ValueError("straddle target level and points need target_combine=tighter")
    if not np.isnan(pair[0]) and not np.isnan(pair[6]) and pair[6] >= pair[0]:
        raise ValueError("straddle stop triggers must be ordered short < long")
    if oto:
        raise ValueError("straddle OTO composition is unproved; use callback protection")
    if one_tick:
        raise ValueError("straddle ONE_TICK execution is unproved")
    return True
