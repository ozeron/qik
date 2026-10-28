"""Legacy venue matcher, moved verbatim: the differential oracle of the venue route.

Production does not import this module; the matcher differential tests and ``legacy.single`` do.
Deleted with the rest of the legacy matchers in the last stage of the single-matcher plan."""

from __future__ import annotations

from collections import namedtuple

import numba as nb
import numpy as np
from numba.extending import overload

from qik.vector_engine.engine.codes import (
    ERR_NEXT_OPEN_SESSION_END,
    ERR_PENDING_NEW_DAY,
    R_ADAPTIVE,
    R_ADVERSE_FIRST,
    R_BRACKET_ANCHOR,
    R_CHASE,
    R_COMBINE,
    R_CROSS_SESSION,
    R_DAY_CANCEL,
    R_LATENCY_BARS,
    R_ONE_TICK,
    R_PROTECT_FLATTEN,
    R_REENTER,
    R_TARGET_GAP,
    R_TICK,
    R_TICK_ROUND,
    REASON_SIGNAL,
    REASON_STOP,
    REASON_STOP_GAP,
)
from qik.vector_engine.engine.decision import _core_decision
from qik.vector_engine.engine.events import (
    _close_at,
    _current_flag,
    _flag_at,
    _level_at,
    _open_at,
    _side_at,
)
from qik.vector_engine.engine.fills import (
    _cancel_entry_commands,
    _day_entry_expires,
    _exit_for,
    _force_for,
    _high_first,
    _target_fill_price,
    _venue_record,
)
from qik.vector_engine.engine.levels import _any_in, _optional_at
from qik.vector_engine.protection import (
    ERR_NONE,
    ERR_PROTECT_REJECTED,
    REASON_PROTECT_FAIL,
    REASON_TARGET,
    _anchor,
    _in_market,
    levels,
)


@nb.njit(cache=True, nogil=True)
def _venue_peek(
    count_only,
    k,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
):
    """Row ``k`` as ``_venue_record`` would overwrite it (zeros where it would skip the write)."""
    if not count_only and k < len(out_entry_i):
        return (
            np.int64(out_entry_i[k]),
            np.int64(out_exit_i[k]),
            np.int64(out_side[k]),
            np.float64(out_entry_px[k]),
            np.float64(out_exit_px[k]),
            np.int64(out_reason[k]),
            np.float64(out_stop[k]),
        )
    return (np.int64(0), np.int64(0), np.int64(0), 0.0, 0.0, np.int64(0), 0.0)


@nb.njit(cache=True, nogil=True)
def _venue_unrecord(
    count_only,
    k,
    prior,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
):
    """Put back the row ``_venue_peek`` read, so a refused bar leaves nothing past the count.

    Both helpers are plain calls: with ``inline="always"`` into the venue's inlined bar step,
    numba compiles this restore to a no-op.
    """
    if not count_only and k < len(out_entry_i):
        out_entry_i[k], out_exit_i[k], out_side[k] = prior[0], prior[1], prior[2]
        out_entry_px[k], out_exit_px[k] = prior[3], prior[4]
        out_reason[k], out_stop[k] = prior[5], prior[6]


@nb.njit(cache=True, nogil=True)
def _venue_column(
    o,
    h,
    lo,
    c,
    side_sig,
    exit_sig,
    entry_stop,
    entry_target,
    entry_limit,
    stop_pts,
    target_pts,
    new_day,
    rules,
    long_exits,
    short_exits,
    force_close,
    force_open,
    count_only,
    out_entry_i,
    out_exit_i,
    out_side,
    out_entry_px,
    out_exit_px,
    out_reason,
    out_stop,
    base,
    core=None,
    ioc_limits=None,
    expiry_days=None,
):
    """A16/A17 venue settlement, with a bounded-by-flight-time FIFO command list.

    Commands: 1 market, 2 limit, 3 protection, 4 cancel entry, 5 cancel protection, 6 close.
    A command's generation is its entry decision row; cancels cannot erase a later trade's legs.
    """
    commands = [(0, 0, 0)]
    commands.pop()
    state = VenueState(
        commands, 0, -1, -1, np.nan, np.nan, np.nan, -1,
        False, False, 0, False, REASON_SIGNAL, False, 0,
    )  # fmt: skip
    inputs = (
        o,
        h,
        lo,
        c,
        side_sig,
        exit_sig,
        entry_stop,
        entry_target,
        entry_limit,
        stop_pts,
        target_pts,
        new_day,
        rules,
        long_exits,
        short_exits,
        force_close,
        force_open,
        count_only,
        out_entry_i,
        out_exit_i,
        out_side,
        out_entry_px,
        out_exit_px,
        out_reason,
        out_stop,
        base,
        ioc_limits,
        expiry_days,
    )
    for i in range(len(c)):
        state, err, eb = _venue_column_advance(state, i, inputs, core)
        if err:
            return state.cnt, err, eb, state.pos, state.ei, state.ep, state.stop
    return state.cnt, ERR_NONE, -1, state.pos, state.ei, state.ep, state.stop


VenueState = namedtuple(
    "VenueState",
    "commands pos ei gen ep stop target pending resting chasing protection closing close_reason rejected_stop cnt",
)


@nb.njit(cache=True, nogil=True, inline="always")
def _venue_column_advance(state, i, inputs, core=None):
    (
        o,
        h,
        lo,
        c,
        side_sig,
        exit_sig,
        entry_stop,
        entry_target,
        entry_limit,
        stop_pts,
        target_pts,
        new_day,
        rules,
        long_exits,
        short_exits,
        force_close,
        force_open,
        count_only,
        out_entry_i,
        out_exit_i,
        out_side,
        out_entry_px,
        out_exit_px,
        out_reason,
        out_stop,
        base,
        ioc_limits,
        expiry_days,
    ) = inputs
    chase = int(rules[R_CHASE])
    adaptive = bool(rules[R_ADAPTIVE])
    reenter = bool(rules[R_REENTER])
    protect_flatten = bool(rules[R_PROTECT_FLATTEN])
    latency = int(rules[R_LATENCY_BARS])
    slip = rules[R_TICK] if rules[R_ONE_TICK] else 0.0
    adverse_first = bool(rules[R_ADVERSE_FIRST])
    bracket_anchor = int(rules[R_BRACKET_ANCHOR])
    target_gap = bool(rules[R_TARGET_GAP])
    tick_round = int(rules[R_TICK_ROUND])
    round_tick = rules[R_TICK]
    cross_session = bool(rules[R_CROSS_SESSION])
    combine = int(rules[R_COMBINE])
    day_cancel = bool(rules[R_DAY_CANCEL])
    """Advance one bar; the batch and compiled-core drivers share this matcher."""
    (
        commands,
        pos,
        ei,
        gen,
        ep,
        stop,
        target,
        pending,
        resting,
        chasing,
        protection,
        closing,
        close_reason,
        rejected_stop,
        cnt,
    ) = state
    start_cnt = cnt  # the refusal below returns ``state``, whose count excludes this bar's exit
    ci = c[i]
    long_exit_i = _flag_at(core, long_exits, i, 13)
    short_exit_i = _flag_at(core, short_exits, i, 14)
    close_mask = _close_at(core, force_close, i)
    open_flatten = _open_at(core, force_open, i)
    exit_now = _exit_for(
        _flag_at(core, exit_sig, i, 1),
        long_exit_i,
        short_exit_i,
        pos if pos else (_side_at(core, side_sig, pending) if pending >= 0 else 0),
    )
    exited = False
    # The target/stop exit below writes its row before the bar can still refuse (ERR 6 at the
    # re-entry IOC); keep what it overwrites so the refusal can put it back.
    prior = (np.int64(0), np.int64(0), np.int64(0), 0.0, 0.0, np.int64(0), 0.0)
    busy = pending >= 0
    if (
        pending >= 0
        and not np.isnan(_level_at(core, entry_limit, pending, 2))
        and _day_entry_expires(day_cancel, _optional_at(expiry_days, i, new_day[i]))
    ):
        _cancel_entry_commands(commands, pending)
        pending, resting, chasing, closing, busy = -1, False, False, False, False
    if (
        pending >= 0
        and not np.isnan(_level_at(core, entry_limit, pending, 2))
        and new_day[i]
        and not day_cancel
    ):
        return (
            VenueState(
                commands,
                pos,
                ei,
                gen,
                ep,
                stop,
                target,
                pending,
                resting,
                chasing,
                protection,
                closing,
                close_reason,
                rejected_stop,
                cnt,
            ),
            ERR_PENDING_NEW_DAY,
            i,
        )
    if open_flatten:
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                o[i] - pos * slip,
                REASON_SIGNAL,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
        commands.clear()
        pos, pending, protection, closing, resting, chasing = 0, -1, 0, False, False, False
    for q in range(len(commands) - 1, -1, -1):
        due, kind, d = commands[q]
        if kind == 7 and due <= i:
            commands.pop(q)
            if pending != d:
                continue
            s = _side_at(core, side_sig, d)
            cap = _level_at(core, ioc_limits, d, 10)
            if s * (o[i] - cap) > 0:
                pending, resting, chasing, closing = -1, False, False, False
                continue
            pos, ei, ep, gen = s, i, o[i], d
            stop, target = levels(
                s,
                _anchor(bracket_anchor, c[d], ep, ep),
                _level_at(core, entry_stop, d, 0),
                _level_at(core, entry_target, d, 1),
                _level_at(core, stop_pts, d, 3),
                _level_at(core, target_pts, d, 4),
                round_tick,
                tick_round,
                combine=combine,
                fill=ep,
            )
            pending, resting, chasing, rejected_stop = -1, False, False, False
            protection = 0 if closing else 1
            commands.append((i + latency, 6 if closing else 3, d))
    if closing and pending < 0 and pos == 0 and (protection == 0 or protection == 3):
        closing = False
    high_first = _high_first(adaptive, o[i], h[i], lo[i])
    # Locate the first matching point. A newly filled entry has no resting protection yet.
    if pending >= 0 and resting:
        d = pending
        s = _side_at(core, side_sig, d)
        extreme = lo[i] if s > 0 else h[i]
        if (s > 0 and extreme + slip <= _level_at(core, entry_limit, d, 2)) or (
            s < 0 and extreme - slip >= _level_at(core, entry_limit, d, 2)
        ):
            px = _level_at(core, entry_limit, d, 2)
            if slip:
                for m in range(4):
                    pt = (
                        o[i]
                        if m == 0
                        else (c[i] if m == 3 else (h[i] if (m == 1) == high_first else lo[i]))
                    )
                    candidate = pt + s * slip
                    if (s > 0 and candidate <= _level_at(core, entry_limit, d, 2)) or (
                        s < 0 and candidate >= _level_at(core, entry_limit, d, 2)
                    ):
                        px = candidate
                        break
            pos, ei, ep, gen = s, i, px, d
            stop, target = levels(
                s,
                _anchor(bracket_anchor, c[d], px, _level_at(core, entry_limit, d, 2)),
                _level_at(core, entry_stop, d, 0),
                _level_at(core, entry_target, d, 1),
                _level_at(core, stop_pts, d, 3),
                _level_at(core, target_pts, d, 4),
                round_tick,
                tick_round,
                combine=combine,
                fill=px,
            )
            pending, resting, chasing = -1, False, False
            rejected_stop = False
            protection = 0 if closing else 1
            commands.append((i + latency, 6 if closing else 3, d))
    elif pos == 0 and protection == 2 and rejected_stop and not np.isnan(target):
        extreme = h[i] if _side_at(core, side_sig, gen) > 0 else lo[i]
        if _in_market(_side_at(core, side_sig, gen), target, extreme, False):
            protection = 0  # flat reduce-only target is cancelled before liquidity is checked
    elif pos and protection >= 2:
        adverse_hit = (
            adverse_first
            and not rejected_stop
            and not np.isnan(stop)
            and (lo[i] <= stop if pos > 0 else h[i] >= stop)
        )
        reason = 0
        px = 0.0
        for m in range(4):
            pt = (
                o[i]
                if m == 0
                else (c[i] if m == 3 else (h[i] if (m == 1) == high_first else lo[i]))
            )
            if not rejected_stop and _in_market(pos, stop, pt, True):
                reason = REASON_STOP_GAP if m == 0 else REASON_STOP
                px = pt - pos * slip if slip else (pt if m == 0 else stop)
                break
            candidate = pt - pos * slip
            if not adverse_hit and _in_market(pos, target, candidate, False):
                reason = REASON_TARGET
                px = _target_fill_price(pt, target, m == 0, target_gap, pos, slip)
                break
        if reason:
            prior = _venue_peek(
                count_only,
                base + cnt,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                px,
                reason,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
            pos, exited = 0, True
            commands.append((i + latency, 5, gen))
            protection = 3
    if core is not None:
        _core_decision(
            core,
            i,
            o[i],
            h[i],
            lo[i],
            c[i],
            pos,
            ei,
            ep,
            (stop if protection >= 2 and not rejected_stop else np.nan),
            (target if protection >= 2 else np.nan),
            _side_at(core, side_sig, pending) if pending >= 0 else 0,
            _level_at(core, entry_limit, pending, 2) if pending >= 0 else np.nan,
            stop,
            cnt,
            out_entry_i,
            out_exit_i,
            out_side,
            out_entry_px,
            out_exit_px,
            out_reason,
            out_stop,
        )
        long_exit_i, short_exit_i = (
            _current_flag(core, core.long_exits, i),
            _current_flag(core, core.short_exits, i),
        )
        close_mask, open_flatten = (
            _close_at(core, core.force_close, i),
            _open_at(core, core.force_open, i),
        )
    if close_mask & 8:
        if pos or pending >= 0:
            return (
                VenueState(
                    commands,
                    pos,
                    ei,
                    gen,
                    ep,
                    stop,
                    target,
                    pending,
                    resting,
                    chasing,
                    protection,
                    closing,
                    close_reason,
                    rejected_stop,
                    cnt,
                ),
                ERR_NEXT_OPEN_SESSION_END,
                i,
            )
        return (
            VenueState(
                commands,
                pos,
                ei,
                gen,
                ep,
                stop,
                target,
                pending,
                resting,
                chasing,
                protection,
                closing,
                close_reason,
                rejected_stop,
                cnt,
            ),
            ERR_NONE,
            -1,
        )
    if (
        _force_for(
            close_mask, pos if pos else (_side_at(core, side_sig, pending) if pending >= 0 else 0)
        )
        or open_flatten
    ):
        if pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                ci - pos * slip,
                REASON_SIGNAL,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
        commands.clear()
        pos, pending, protection, closing, resting, chasing = 0, -1, 0, False, False, False
        return (
            VenueState(
                commands,
                pos,
                ei,
                gen,
                ep,
                stop,
                target,
                pending,
                resting,
                chasing,
                protection,
                closing,
                close_reason,
                rejected_stop,
                cnt,
            ),
            ERR_NONE,
            -1,
        )
    if closing and pending < 0 and pos == 0 and (protection == 0 or protection == 3):
        closing = False
    if closing and pos == 0 and protection == 2:
        commands.append((i + latency, 5, gen))
        protection = 3
    exit_now = _exit_for(
        _flag_at(core, exit_sig, i, 1),
        long_exit_i,
        short_exit_i,
        pos if pos else (_side_at(core, side_sig, pending) if pending >= 0 else 0),
    )
    if (
        pending >= 0
        and not closing
        and not np.isnan(_level_at(core, entry_limit, pending, 2))
        and i == pending + chase
    ):
        chasing = True
        commands.append((i + latency, 4, pending))
    if exit_now:
        chasing = False
        if pending >= 0:
            if not closing and not np.isnan(_level_at(core, entry_limit, pending, 2)):
                commands.append((i + latency, 4, pending))
            closing, close_reason = True, REASON_SIGNAL
        if pos:
            if protection == 2:
                commands.append((i + latency, 5, gen))
                protection = 3
            if not closing:
                commands.append((i + latency, 6, gen))
            closing, close_reason = True, REASON_SIGNAL
    if (
        pos == 0
        and pending < 0
        and not busy
        and not closing
        and _side_at(core, side_sig, i) != 0
        and (reenter or not exited)
    ):
        pending = i
        if not np.isnan(_level_at(core, ioc_limits, i, 10)):
            due = i + latency + 1
            if due >= len(c) or (not cross_session and _any_in(new_day, i + 1, due + 1)):
                if exited:
                    _venue_unrecord(
                        count_only,
                        base + start_cnt,
                        prior,
                        out_entry_i,
                        out_exit_i,
                        out_side,
                        out_entry_px,
                        out_exit_px,
                        out_reason,
                        out_stop,
                    )
                return state, ERR_NEXT_OPEN_SESSION_END, i
            commands.append((due, 7, i))
        else:
            commands.append(
                (i + latency, 1 if np.isnan(_level_at(core, entry_limit, i, 2)) else 2, i)
            )
    while len(commands):
        ready = -1
        for q in range(len(commands)):
            if commands[q][0] <= i and commands[q][1] != 7:
                ready = q
                break
        if ready < 0:
            break
        _, kind, d = commands.pop(ready)
        if kind == 1 or kind == 2:
            if pending != d:
                continue
            s = _side_at(core, side_sig, d)
            px = ci + s * slip
            if (
                kind == 1
                or (s > 0 and px <= _level_at(core, entry_limit, d, 2))
                or (s < 0 and px >= _level_at(core, entry_limit, d, 2))
            ):
                anchor = _anchor(
                    bracket_anchor,
                    c[d],
                    px,
                    px if kind == 1 else _level_at(core, entry_limit, d, 2),
                )
                pos, ei, ep, gen = s, i, px, d
                stop, target = levels(
                    s,
                    anchor,
                    _level_at(core, entry_stop, d, 0),
                    _level_at(core, entry_target, d, 1),
                    _level_at(core, stop_pts, d, 3),
                    _level_at(core, target_pts, d, 4),
                    round_tick,
                    tick_round,
                    combine=combine,
                    fill=px,
                )
                pending, resting, chasing = -1, False, False
                rejected_stop = False
                protection = 0 if closing else 1
                commands.append((i + latency, 6 if closing else 3, d))
            else:
                resting = True
        elif kind == 4:
            if pending == d:
                pending, resting = -1, False
                if chasing:
                    chasing = False
                    pending = d
                    commands.append((i + latency, 1, d))
        elif kind == 3 and gen == d:
            protection = 2
            if pos:
                if _in_market(pos, stop, ci, True):
                    if not protect_flatten:
                        return (
                            VenueState(
                                commands,
                                pos,
                                ei,
                                gen,
                                ep,
                                stop,
                                target,
                                pending,
                                resting,
                                chasing,
                                protection,
                                closing,
                                close_reason,
                                rejected_stop,
                                cnt,
                            ),
                            ERR_PROTECT_REJECTED,
                            i,
                        )
                    rejected_stop = True
                    if np.isnan(target):
                        protection = 0
                    if not closing:
                        close_reason = REASON_PROTECT_FAIL
                        commands.append((i + latency, 6, d))
                    closing = True
                px = ci - pos * slip
                if _in_market(pos, target, px, False):
                    _venue_record(
                        count_only,
                        base + cnt,
                        ei,
                        i,
                        pos,
                        ep,
                        px,
                        REASON_TARGET,
                        stop,
                        out_entry_i,
                        out_exit_i,
                        out_side,
                        out_entry_px,
                        out_exit_px,
                        out_reason,
                        out_stop,
                    )
                    cnt += 1
                    pos, exited = 0, True
                    commands.append((i + latency, 5, d))
                    protection = 3
        elif kind == 5 and gen == d:
            protection = 0
        elif kind == 6 and pos:
            _venue_record(
                count_only,
                base + cnt,
                ei,
                i,
                pos,
                ep,
                ci - pos * slip,
                close_reason,
                stop,
                out_entry_i,
                out_exit_i,
                out_side,
                out_entry_px,
                out_exit_px,
                out_reason,
                out_stop,
            )
            cnt += 1
            pos, exited = 0, True
    return (
        VenueState(
            commands,
            pos,
            ei,
            gen,
            ep,
            stop,
            target,
            pending,
            resting,
            chasing,
            protection,
            closing,
            close_reason,
            rejected_stop,
            cnt,
        ),
        ERR_NONE,
        -1,
    )


# Keep external callback/jitclass signatures out of numeric batch cache indices.
# Both dispatchers compile the same source; the matcher is never copied.
_compiled_venue_column = nb.njit(cache=False, nogil=True)(
    getattr(_venue_column, "py_func", _venue_column)
)


def _venue_driver(core):
    return _venue_column if core is None else _compiled_venue_column


@overload(_venue_driver)
def _venue_driver_type(core):
    # Select a dispatcher at typing time: a first-class-function union loses
    # Numba's omitted-default argument specialization for the numeric driver.
    if isinstance(core, nb.types.NoneType):

        def numeric(core):
            return _venue_column

        return numeric

    def generic(core):
        return _compiled_venue_column

    return generic


__all__ = [
    "VenueState",
    "_compiled_venue_column",
    "_venue_column",
    "_venue_column_advance",
    "_venue_driver",
    "_venue_driver_type",
]
