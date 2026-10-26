"""Typed core decision: strategy callback transport into the level store and account."""

from __future__ import annotations

import numba as nb
import numpy as np

from qik.vector_engine.account import fill_cost, settle
from qik.vector_engine.account import quantity as _contracts_for
from qik.vector_engine.protection import _round_price
from qik.vector_engine.straddle import validate_pair

from .codes import (
    ACCOUNT_INVALID_SIDE,
    V_ENTRY_I,
    V_ENTRY_PX,
    V_EQUITY,
    V_LAST_EXIT_I,
    V_LAST_EXIT_PX,
    V_LAST_EXIT_REASON,
    V_QTY,
    V_SIDE,
    V_STOP,
    V_TARGET,
    V_WORK_LEVEL,
    V_WORK_SIDE,
)
from .levels import _put_level


@nb.njit(cache=False, inline="always")
def _core_decision(
    core,
    i,
    o,
    h,
    lo,
    c,
    pos,
    ei,
    ep,
    stop,
    target,
    work_side,
    work_level,
    risk_stop,
    cnt,
    out_ei,
    out_xi,
    out_sd,
    out_ep,
    out_xp,
    out_why,
    out_st,
    work_long=np.nan,
    work_short=np.nan,
    next_px=np.nan,
):
    """Observe settled fills, call a core, and publish its close decision (no matching here).

    ``next_px`` (the following bar's open) resolves open-relative triggers after the core
    returned; the core never sees it, and the position view publishes it only later.
    """
    account = core.account
    seen = int(account[0])
    if account[2]:
        return
    if cnt > len(out_ei):
        # The trade buffer is smaller than the worst case; the caller reruns this config.
        account[2] = 9
        return
    pv, friction, bps, mode, amount, cap, lot, minimum = core.economic[:8]
    clamp = len(core.economic) > 8 and bool(core.economic[8])
    for k in range(seen, cnt):
        points = core.sizing_risks[out_ei[k]]
        risk = (points if not np.isnan(points) else abs(out_ep[k] - out_st[k])) * pv
        qty = _contracts_for(mode, amount, cap, account[1], risk, lot, minimum, clamp)
        if qty < 0:
            # Entry bar and sizing error code: the array path's refusal text needs both.
            account[2], account[3], account[4] = 1, out_ei[k], -qty
            return
        net = settle(
            out_sd[k] * (out_xp[k] - out_ep[k]),
            pv,
            friction,
            fill_cost(out_ep[k], out_xp[k], pv, bps),
        )
        account[1] += qty * net
        core.position_view[V_LAST_EXIT_I] = out_xi[k]
        core.position_view[V_LAST_EXIT_PX] = out_xp[k]
        core.position_view[V_LAST_EXIT_REASON] = out_why[k]
    account[0] = cnt
    view = core.position_view
    view[V_SIDE], view[V_QTY] = pos, 0
    view[V_ENTRY_I], view[V_ENTRY_PX] = (ei if pos else -1), (ep if pos else np.nan)
    view[V_STOP], view[V_TARGET] = (stop if pos else np.nan), (target if pos else np.nan)
    if pos:
        points = core.sizing_risks[ei]
        risk = (points if not np.isnan(points) else abs(ep - risk_stop)) * pv
        qty = _contracts_for(mode, amount, cap, account[1], risk, lot, minimum, clamp)
        if qty < 0:
            account[2], account[3], account[4] = 2, ei, -qty
            return
        view[V_QTY] = qty
    view[V_WORK_SIDE], view[V_WORK_LEVEL] = work_side, work_level
    view[V_EQUITY] = account[1]
    view[12], view[13] = work_long, work_short
    d = core.decision
    d[:] = np.nan
    d[0], d[1], d[12], d[13], d[14], d[28] = 0, 0, 0, 0, 0, 0
    core.step(core.state, core.params, i, core.feats, o, h, lo, c, core.volume[i], view, d)
    levels, layout = core.levels, core.layout
    stored, declared, legs, block, row = layout[0], layout[1], layout[2], layout[4], 0
    dropped = False
    for q in range(len(d)):
        if (q < 16 or q > 27) and stored >> q & 1:
            levels[row, i] = d[q]
            row += 1
        flag = q == 12 or q == 13 or q == 14 or q == 28
        if not declared >> q & 1 and (d[q] != 0 if flag else np.isfinite(d[q])):
            account[5] = float(int(account[5]) | (1 << q))
        # Any non-default value (including +-inf) in a field this layout drops.
        if not stored >> q & 1 and (d[q] != 0 if flag else not np.isnan(d[q])):
            dropped = True
    if dropped:
        # The matcher would read a different decision from here on: stop, and let the
        # caller rerun with every field stored.
        account[2] = 10
        return
    if stored >> 16 & 1:
        for leg in range(12):
            if legs >> leg & 1:
                levels[row, i] = d[16 + leg]
                row += 1
            else:
                levels[block + leg, i] = d[16 + leg]
    if d[0] != -1 and d[0] != 0 and d[0] != 1:
        account[2], account[3] = ACCOUNT_INVALID_SIDE, i
        return
    for col in (1, 12, 13, 14, 28):
        if d[col] != 0 and d[col] != 1:
            raise ValueError("compiled core: boolean decision must be 0 or 1")
    core.sides[i], core.exits[i] = int(d[0]), bool(d[1])
    for col in range(10):
        _put_level(levels, layout, col, i, d[col + 2])
    _put_level(levels, layout, 10, i, d[15])
    # Carry these refusals out of prange like sizing refusals; raising inside
    # a compiled callback can otherwise leave uninitialized result buffers.
    pair = d[16:28].copy()
    if d[28]:
        pair[0] += next_px
        pair[6] += next_px
    if np.isfinite(d[16:28]).any():
        if core.pair_rules & 16:
            account[2] = 3
            return
        if core.pair_rules & 32:
            account[2] = 4
            return
        if core.pair_rules & 8:
            if not core.pair_rules & 64:
                account[2] = 5
                return
            if not np.isnan(pair[1]) or not np.isnan(pair[7]):
                account[2] = 7
                return
            if (
                not np.isnan(pair[3])
                or not np.isnan(pair[5])
                or not np.isnan(pair[9])
                or not np.isnan(pair[11])
            ):
                account[2] = 6
                return
    if d[12] and core.pair_rules & 128 and core.flatten_mode == 0:
        # A23 settles a flatten at this close only; a future-open flatten is refused.
        account[2] = 8
        return
    validate_pair(
        pair,
        d[2:12],
        d[0],
        bool(
            d[13]
            or d[14]
            or not np.isnan(d[15])
            or ((d[1] or d[12]) and not (core.pair_rules & 128))
        ),
        core.pair_rules & 3,
        bool(core.pair_rules & 4),
        bool(core.pair_rules & 8),
        bool(core.pair_rules & 16),
        bool(core.pair_rules & 32),
        bool(core.pair_rules & 64),
    )
    for col in range(12):
        _put_level(levels, layout, 11 + col, i, pair[col])
    for offset, direction in ((0, 1), (6, -1)):
        _put_level(
            levels,
            layout,
            11 + offset,
            i,
            _round_price(pair[offset], core.tick, core.rounding, direction),
        )
        _put_level(
            levels,
            layout,
            12 + offset,
            i,
            _round_price(pair[1 + offset], core.tick, core.rounding, -direction),
        )
    _put_level(levels, layout, 2, i, _round_price(d[4], core.tick, core.rounding, -d[0]))
    trigger = d[7] + next_px if d[28] else d[7]
    _put_level(levels, layout, 5, i, _round_price(trigger, core.tick, core.rounding, d[0]))
    _put_level(levels, layout, 23, i, d[28])
    _put_level(levels, layout, 24, i, d[29])
    _put_level(levels, layout, 25, i, d[30])
    core.long_exits[i], core.short_exits[i] = bool(d[13]), bool(d[14])
    # Match shared validation boundary transformation on the current decision, without future signals.
    if core.boundary == 1 and core.ends[i] and core.exit_next_open:
        core.force_close[i] |= int(core.exits[i]) | (2 * int(d[13])) | (4 * int(d[14]))
        core.exits[i], core.long_exits[i], core.short_exits[i] = False, False, False
    if core.flatten_mode == 2:
        if core.ends[i]:
            core.force_close[i] |= 1
    elif d[12]:
        if core.flatten_mode == 1:
            core.force_close[i] |= 1
        elif i + 1 == len(core.sides) or (core.days[i + 1] and core.boundary != 2):
            core.force_close[i] |= 1 if core.boundary == 1 else 8
        else:
            core.force_open[i + 1] = True


__all__ = [
    "_core_decision",
]
