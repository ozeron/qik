"""Small causal entry/stop/exit blocks composed inside ``step(BarView)``.

No block imports a project, computes fills or guesses position state. Each is
admitted by equivalence to at least two named cores; see the Blocks documentation.
"""

from .entries import range_breakout, range_breakout_array, side_of, side_of_array
from .exits import r_target, time_exit
from .holding import held_bars_exit
from .stops import atr_stop, trailing_stop

__all__ = [
    "atr_stop",
    "held_bars_exit",
    "r_target",
    "range_breakout",
    "range_breakout_array",
    "side_of",
    "side_of_array",
    "time_exit",
    "trailing_stop",
]
