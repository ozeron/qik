"""Settings/error/level codes and the float64 settings record shared by every matcher."""

from __future__ import annotations

from collections import namedtuple

import numpy as np

# Stable shared settings ABI. Float64 exactly represents the bounded integer/boolean
# rule values; only tick is fractional. Array shape and dtype never depend on rules.
RULES_WIDTH = 29
R_CHASE = 0
R_ADAPTIVE = 1
R_REENTER = 2
R_STALE_DROP = 3
R_PROTECT_FLATTEN = 4
R_NEXT_OPEN = 5
R_EXIT_NEXT_OPEN = 6
R_LATENCY_BARS = 7
R_ONE_TICK = 8
R_TICK = 9
R_ADVERSE_FIRST = 10
R_BRACKET_ANCHOR = 11
R_TARGET_GAP = 12
R_TICK_ROUND = 13
R_CROSS_SESSION = 14
R_MIT = 15
R_OTO = 16
R_AMEND_KEEP = 17
R_RATCHET = 18
R_STOP_SKIP = 19
R_FLATTEN = 20
R_BOUNDARY = 21
R_COMBINE = 22
R_AMEND_CLOSE = 23
R_COMPLETED_CLOSE = 24
R_ARM_STOP = 25
R_DAY_CANCEL = 26
R_NEXT_BAR = 27
R_EXIT_AND_REARM = 28


def rule_settings(
    *,
    chase: int = 1,
    adaptive: bool = False,
    reenter: bool = False,
    stale_drop: bool = False,
    protect_flatten: bool = False,
    next_open: bool = False,
    exit_next_open: bool | None = None,
    latency_bars: int = 0,
    one_tick: bool = False,
    tick: float = 0.0,
    adverse_first: bool = False,
    bracket_anchor: int = 0,
    target_gap: bool = False,
    tick_round: int = 0,
    cross_session: bool = False,
    mit: bool = False,
    oto: bool = False,
    amend_keep: bool = False,
    ratchet: bool = False,
    stop_skip: bool = False,
    flatten: int = 0,
    boundary: int = 0,
    combine: int = 0,
    amend_close: bool = False,
    completed_close: bool = False,
    arm_stop: bool = False,
    day_cancel: bool = False,
    next_bar: bool = False,
    exit_and_rearm: bool = False,
) -> np.ndarray:
    """Build one settings record at the Python boundary, shared by every matcher.

    chase=1 matches Rules.limit_chase_bars: an unfilled limit chases after one
    bar under the default rules. Zero would mean same-bar chasing and is not a
    valid Rules value. Production boundaries pass their chosen chase explicitly.
    """
    if exit_next_open is None:
        exit_next_open = next_open
    settings = np.empty(RULES_WIDTH, dtype=np.float64)
    settings[R_CHASE] = chase
    settings[R_ADAPTIVE] = adaptive
    settings[R_REENTER] = reenter
    settings[R_STALE_DROP] = stale_drop
    settings[R_PROTECT_FLATTEN] = protect_flatten
    settings[R_NEXT_OPEN] = next_open
    settings[R_EXIT_NEXT_OPEN] = exit_next_open
    settings[R_LATENCY_BARS] = latency_bars
    settings[R_ONE_TICK] = one_tick
    settings[R_TICK] = tick
    settings[R_ADVERSE_FIRST] = adverse_first
    settings[R_BRACKET_ANCHOR] = bracket_anchor
    settings[R_TARGET_GAP] = target_gap
    settings[R_TICK_ROUND] = tick_round
    settings[R_CROSS_SESSION] = cross_session
    settings[R_MIT] = mit
    settings[R_OTO] = oto
    settings[R_AMEND_KEEP] = amend_keep
    settings[R_RATCHET] = ratchet
    settings[R_STOP_SKIP] = stop_skip
    settings[R_FLATTEN] = flatten
    settings[R_BOUNDARY] = boundary
    settings[R_COMBINE] = combine
    settings[R_AMEND_CLOSE] = amend_close
    settings[R_COMPLETED_CLOSE] = completed_close
    settings[R_ARM_STOP] = arm_stop
    settings[R_DAY_CANCEL] = day_cancel
    settings[R_NEXT_BAR] = next_bar
    settings[R_EXIT_AND_REARM] = exit_and_rearm
    return settings


REASON_SIGNAL = 1  # market exit at the decision close (R1: the next open)
REASON_STOP = 2  # resting stop-market hit during a bar, filled at the stop level
REASON_STOP_GAP = 4  # resting stop already through at the bar's open, filled at the open

ERR_STOP_IN_MARKET = 1  # stop level not strictly beyond the entry close (Nautilus rejects it)
ERR_TARGET_IN_MARKET = 2  # target level not strictly beyond the entry close (would fill at once)
ERR_STALE_BOOK = 3  # new stop falls inside the stale quote a same-bar limit fill left (A13)
ERR_PENDING_NEW_DAY = 5  # a limit entry still working when the New York date changes (A15)
ERR_NEXT_OPEN_SESSION_END = 6  # R1 decision has no following bar on the same New York date
ERR_STOP_ENTRY_IN_MARKET = 7
ERR_MIT_IN_MARKET = 8
ERR_AMEND_REJECTED = 9
ERR_AMEND_NO_ORDER = 10
ERR_STOP_REPLACE_LATENCY = 11
ERR_STRADDLE_SIDE_CANCEL = 12
ERR_BUSY_ENTRY_DROPPED = 13  # R15b: a whole-bar busy guard would silently discard an entry
ERR_REARM_BEFORE_CLOSE = 14  # A23: replacement fills at this close, the old close at the next open

# Account refusals 9/10/11 are buffer overflow, dropped fields and callback failure.
ACCOUNT_INVALID_SIDE = 12
# Refusals the column code used to raise, which a parallel loop loses (the column's remaining work
# is skipped and the run returns normally): carried in the account row like the others, raised
# in Python with the message the reference run raises. 16.. are ``straddle.PAIR_MESSAGES``.
ACCOUNT_BOOLEAN_DECISION = 15
ACCOUNT_PAIR_REFUSAL = 16
# ``counts`` of a column the parallel loop never completed; any other value was written by the
# column itself, so this one is never a result.
COLUMN_UNRUN = -1000

_SLOT_SUBMIT = 1
_SLOT_PROTECT = 2
_SLOT_CANCEL = 3
_SLOT_CHASE = 4
_SLOT_REPLACE = 5
_SLOT_CLOSE = 6
_SLOT_PROTECT_FAIL = 7
_SLOT_AMEND_STOP = 8
_SLOT_AMEND_TARGET = 9
_SLOT_ARM_STOP = 10
_SLOT_ARM_TARGET = 11
_SLOT_PAIR_IOC = 12  # paired IOC attempts at scheduled open


# Stable numeric ABI shared with compiled strategy cores and live hosts.
V_SIDE, V_QTY, V_ENTRY_I, V_ENTRY_PX, V_STOP, V_TARGET, V_WORK_SIDE, V_WORK_LEVEL = range(8)
V_LAST_EXIT_I, V_LAST_EXIT_PX, V_LAST_EXIT_REASON, V_EQUITY = range(8, 12)

CoreContext = namedtuple(
    "CoreContext",
    "step state params feats volume position_view decision sides exits levels long_exits short_exits "
    "force_close force_open days ends economic account flatten_mode boundary tick rounding layout exit_next_open pair_rules sizing_risks ev_count ev_bar ev_mask ev_val event_mode event_state field_rows level_rows pair_scratch leg_scratch masks",
)
# CoreContext.layout is six integers: an array field access refcounts and a tuple indexed
# at run time spills, and the core runs once per bar. CoreContext.levels rows, in order:
# stored fields outside the straddle legs (Signals order); the raw decision of each leg
# whose level _core_decision transforms; the twelve-row leg block the matcher reads (a
# transformed leg's own row, otherwise that leg's decision); own rows for levels 2 and 5.


EVENT_FLAG_BITS = (1 << 0) | (1 << 1) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 28)


PAIR_BITS = ((1 << 12) - 1) << 16


__all__ = [
    "ACCOUNT_BOOLEAN_DECISION",
    "ACCOUNT_INVALID_SIDE",
    "ACCOUNT_PAIR_REFUSAL",
    "COLUMN_UNRUN",
    "CoreContext",
    "ERR_AMEND_NO_ORDER",
    "ERR_AMEND_REJECTED",
    "ERR_BUSY_ENTRY_DROPPED",
    "ERR_MIT_IN_MARKET",
    "ERR_NEXT_OPEN_SESSION_END",
    "ERR_PENDING_NEW_DAY",
    "ERR_REARM_BEFORE_CLOSE",
    "ERR_STALE_BOOK",
    "ERR_STOP_ENTRY_IN_MARKET",
    "ERR_STOP_IN_MARKET",
    "ERR_STOP_REPLACE_LATENCY",
    "ERR_STRADDLE_SIDE_CANCEL",
    "ERR_TARGET_IN_MARKET",
    "REASON_SIGNAL",
    "REASON_STOP",
    "REASON_STOP_GAP",
    "RULES_WIDTH",
    "R_ADAPTIVE",
    "R_ADVERSE_FIRST",
    "R_AMEND_CLOSE",
    "R_AMEND_KEEP",
    "R_ARM_STOP",
    "R_BOUNDARY",
    "R_BRACKET_ANCHOR",
    "R_CHASE",
    "R_COMBINE",
    "R_COMPLETED_CLOSE",
    "R_CROSS_SESSION",
    "R_DAY_CANCEL",
    "R_EXIT_AND_REARM",
    "R_EXIT_NEXT_OPEN",
    "R_FLATTEN",
    "R_LATENCY_BARS",
    "R_MIT",
    "R_NEXT_BAR",
    "R_NEXT_OPEN",
    "R_ONE_TICK",
    "R_OTO",
    "R_PROTECT_FLATTEN",
    "R_RATCHET",
    "R_REENTER",
    "R_STALE_DROP",
    "R_STOP_SKIP",
    "R_TARGET_GAP",
    "R_TICK",
    "R_TICK_ROUND",
    "V_ENTRY_I",
    "V_ENTRY_PX",
    "V_EQUITY",
    "V_LAST_EXIT_I",
    "V_LAST_EXIT_PX",
    "V_LAST_EXIT_REASON",
    "V_QTY",
    "V_SIDE",
    "V_STOP",
    "V_TARGET",
    "V_WORK_LEVEL",
    "V_WORK_SIDE",
    "_SLOT_AMEND_STOP",
    "_SLOT_AMEND_TARGET",
    "_SLOT_ARM_STOP",
    "_SLOT_ARM_TARGET",
    "_SLOT_CANCEL",
    "_SLOT_CHASE",
    "_SLOT_CLOSE",
    "_SLOT_PAIR_IOC",
    "_SLOT_PROTECT",
    "_SLOT_PROTECT_FAIL",
    "_SLOT_REPLACE",
    "_SLOT_SUBMIT",
    "rule_settings",
    "EVENT_FLAG_BITS",
    "PAIR_BITS",
]
