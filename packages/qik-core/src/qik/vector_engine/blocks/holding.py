"""Position-age predicate; the caller owns entry index and bar-count convention."""

from numba.extending import register_jitable


@register_jitable
def held_bars_exit(held_bars: int, limit: int) -> bool:
    """Exit when caller-counted completed bars reach an inclusive limit.

    Some callers use i-entry_i, others i-entry_i+1 (entry bar included).
    Pass that count explicitly; missing clock minutes do not become held bars.
    Flat positions must be gated by the strategy; this does not inspect fills.
    """
    if held_bars < 0 or limit < 1:
        raise ValueError("held_bars must be nonnegative and limit positive")
    return held_bars >= limit
