"""Exit comparisons and raw targets; clocks, rounding and fill anchors stay explicit."""

from __future__ import annotations

import math
from datetime import time

from .stops import _side


def time_exit[Clock: (int, time)](
    clock: Clock, cutoff: Clock, *, at_or_after: bool = False
) -> bool:
    """Compare a caller-supplied completed-bar clock (time or integer offset).

    Admitted: P060 OvernightCore (exact) and P063 StreakCore (at-or-after).
    No timezone conversion, session discovery or terminal-bar inference occurs.
    """
    return clock >= cutoff if at_or_after else clock == cutoff


def r_target(anchor: float, side: int, risk: float, reward: float) -> float:
    """Raw R target, with positive risk units and caller-owned tick conversion.

    Admitted: P063 native brackets and P073 plan. ATR is also a valid risk unit.
    This is a target specification, never a signal to exit on an observed touch.
    """
    _side(side)
    if not math.isfinite(risk) or risk < 0 or not math.isfinite(reward) or reward <= 0:
        raise ValueError("risk must be finite/nonnegative and reward finite/positive")
    return anchor + side * risk * reward
