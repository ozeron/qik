"""The slots arm of the unified matcher (single-matcher plan, stage 4).

The slots matcher as a flat, event-gated loop
over a fixed command ring, in the shape of ``book._book_loop``: scalar state in locals, no state
tuple and no per-branch return, the rule bits decoded once, and a bar with nothing resting, nothing
in flight and nothing to enter skipped before it costs anything. The route's differences from the
single and venue routes are the compat fields of ``book`` (``COMPAT_SLOTS``).

A bar runs in the old matcher's order:

1. the pre-phase: DAY expiry of a working entry, the new-date refusal (ERR 5), a forced flatten
   at the open;
2. phase 0, the open: commands that settle at an open (next-open fills, IOC attempts, the
   protection of an entry that just filled);
3. phase 1, the close: the path O, e1, e2, C (resting entries and the live stop and target), the
   GTD expiry of a stop entry, the strategy callback, the close decision (forced flatten, exits,
   amendments, breakeven, entry staging), then every command that is due settles in insertion
   order.

Differences from the old matcher that change nothing it reports:

* the command list is ``slot_ring``: a fixed ring with an insertion number per entry;
* the decision a command or a working entry belongs to is read through its *held identity*
  (``book._held_ident``), and what a working entry needs again (side, trigger, limit, straddle,
  from-open) is read once when it is created;
* the path loop runs only when something can fill: a resting entry, or a live stop or target that
  the bar's range reaches. A refusal sets ``error`` and ends the bar where the old matcher returned.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from qik.vector_engine.account import MODE_QTY_FEATURE
from qik.vector_engine.economic import E_MODE
from qik.vector_engine.protection import (
    ERR_NONE,
    ERR_PROTECT_REJECTED,
    REASON_PROTECT_FAIL,
    REASON_TARGET,
    _anchor,
    _in_market,
    _round_price,
    levels,
)

from .book import (
    C_NEW_DAY_MASK,
    C_NEW_DAY_SHIFT,
    COMPAT_SLOTS,
    F_ADAPTIVE,
    F_ADVERSE_FIRST,
    F_CROSS_SESSION,
    F_DAY_CANCEL,
    F_EXIT_NEXT_OPEN,
    F_NEXT_OPEN,
    F_ONE_TICK,
    F_PROTECT_FLATTEN,
    F_REENTER,
    F_STALE_DROP,
    F_TARGET_GAP,
    NEW_DAY_REFUSE_RESTING,
    _exit_flags,
    _has_lane,
    _held_ident,
    _held_level,
    _held_side,
    _lane_at,
    _rule_flags,
    _scratch_at,
)
from .codes import (
    _SLOT_AMEND_STOP,
    _SLOT_AMEND_TARGET,
    _SLOT_ARM_STOP,
    _SLOT_ARM_TARGET,
    _SLOT_CANCEL,
    _SLOT_CHASE,
    _SLOT_CLOSE,
    _SLOT_PAIR_IOC,
    _SLOT_PROTECT,
    _SLOT_PROTECT_FAIL,
    _SLOT_REPLACE,
    _SLOT_SUBMIT,
    ERR_AMEND_NO_ORDER,
    ERR_AMEND_REJECTED,
    ERR_BUSY_ENTRY_DROPPED,
    ERR_MIT_IN_MARKET,
    ERR_NEXT_OPEN_SESSION_END,
    ERR_PENDING_NEW_DAY,
    ERR_REARM_BEFORE_CLOSE,
    ERR_STALE_BOOK,
    ERR_STOP_ENTRY_IN_MARKET,
    ERR_STOP_REPLACE_LATENCY,
    ERR_STRADDLE_SIDE_CANCEL,
    R_AMEND_CLOSE,
    R_AMEND_KEEP,
    R_ARM_STOP,
    R_BRACKET_ANCHOR,
    R_CHASE,
    R_COMBINE,
    R_COMPLETED_CLOSE,
    R_EXIT_AND_REARM,
    R_LATENCY_BARS,
    R_MIT,
    R_NEXT_BAR,
    R_OTO,
    R_RATCHET,
    R_STOP_SKIP,
    R_TICK,
    R_TICK_ROUND,
    REASON_SIGNAL,
    REASON_STOP,
    REASON_STOP_GAP,
)
from .decision import _core_decision, _entry_quantity
from .events import _close_at, _open_at
from .fills import _high_first, _target_fill_price, _venue_record
from .levels import _any_in, _store_risk
from .ring import ERR_RING_OVERFLOW
from .slot_ring import (
    RING_ROWS,
    RW_D,
    RW_DUE,
    RW_IDENT,
    RW_KIND,
    RW_PAIR,
    RW_VALUE,
    SCAP,
    _event_masks,
    _event_stored,
    _held_lifetime_crosses,
    _held_open_in_market,
    _held_pair,
    _held_pair_any,
    _held_pair_level,
    _held_pair_rests,
    _held_stop_pair,
    _held_view,
    _pair_cols,
    _pair_possible,
    _sbound,
    _sdrop,
    _spop,
    _spush,
)

# Slots-only rule bits, decoded once from the settings record (the shared bits are ``book.F_*``).
(
    S_MIT,
    S_OTO,
    S_AMEND_KEEP,
    S_RATCHET,
    S_STOP_SKIP,
    S_AMEND_CLOSE,
    S_COMPLETED_CLOSE,
    S_ARM_STOP,
    S_NEXT_BAR,
    S_EXIT_AND_REARM,
) = (1 << k for k in range(10))


@nb.njit(cache=True, nogil=True)
def _slot_flags(rules):
    """The slots-only rule bits as one word, read once per column."""
    flags = 0
    for bit, index in (
        (S_MIT, R_MIT),
        (S_OTO, R_OTO),
        (S_AMEND_KEEP, R_AMEND_KEEP),
        (S_RATCHET, R_RATCHET),
        (S_STOP_SKIP, R_STOP_SKIP),
        (S_AMEND_CLOSE, R_AMEND_CLOSE),
        (S_COMPLETED_CLOSE, R_COMPLETED_CLOSE),
        (S_ARM_STOP, R_ARM_STOP),
        (S_NEXT_BAR, R_NEXT_BAR),
        (S_EXIT_AND_REARM, R_EXIT_AND_REARM),
    ):
        if rules[index] != 0.0:
            flags |= bit
    return flags


@nb.njit(cache=True, nogil=True)
def _pend_load(
    hv, ident, side_sig, triggers, entry_limit, from_open, straddle,
    has_trig, has_limit, has_from, has_pairs,
):  # fmt: skip
    """What a working entry needs again, read once from its decision: ``(side, trigger, limit,
    straddle, from_open)``. A plain call: it runs once per entry order."""
    return (
        _held_side(hv, side_sig, ident),
        _held_level(hv, triggers, ident, 5, np.nan) if has_trig else np.nan,
        _held_level(hv, entry_limit, ident, 2, np.nan) if has_limit else np.nan,
        has_pairs and _held_pair_any(hv, straddle, ident),
        bool(_held_level(hv, from_open, ident, 23, 0.0)) if has_from else False,
    )


@nb.njit(cache=True, nogil=True)
def _entry_resolve(
    hv, side, anchor, fill, ident, paired, entry_stop, entry_target, stop_pts, target_pts,
    stop_fracs, target_fracs, straddle, has_es, has_et, has_sp, has_tp, has_sf, has_tf,
    tick, tick_round, combine, fracs,
):  # fmt: skip
    """The protection levels of the held decision ``ident`` for an entry of ``side`` that filled
    at ``fill``: its own legs, or the leg of the straddle side that fired. ``fracs`` False leaves
    the fractional levels out (the R15b sizing risk is taken from the trigger geometry)."""
    leg_stop = _held_level(hv, entry_stop, ident, 0, np.nan) if has_es else np.nan
    leg_target = _held_level(hv, entry_target, ident, 1, np.nan) if has_et else np.nan
    leg_sp = _held_level(hv, stop_pts, ident, 3, np.nan) if has_sp else np.nan
    leg_tp = _held_level(hv, target_pts, ident, 4, np.nan) if has_tp else np.nan
    if paired:
        offset = 0 if side > 0 else 6
        leg_stop = _held_pair(hv, straddle, offset + 2, ident)
        leg_target = _held_pair(hv, straddle, offset + 3, ident)
        leg_sp = _held_pair(hv, straddle, offset + 4, ident)
        leg_tp = _held_pair(hv, straddle, offset + 5, ident)
    return levels(
        side,
        anchor,
        leg_stop,
        leg_target,
        leg_sp,
        leg_tp,
        tick,
        tick_round,
        combine=combine,
        fill=fill,
        stop_frac=_held_level(hv, stop_fracs, ident, 24, np.nan) if (fracs and has_sf) else np.nan,
        target_frac=_held_level(hv, target_fracs, ident, 25, np.nan)
        if (fracs and has_tf)
        else np.nan,
    )


@nb.njit(cache=True, nogil=True)
def _session_end(ring, qmask, i, n, new_day, cross_session, next_bar, hv, triggers, has_trig, pair):
    """A next-open command still ahead at a close, with no following bar on the same New York
    date (and no session crossing allowed): the refusal ERR 6. Scans every ring entry."""
    m = qmask
    slot = 0
    while m != 0:
        if m & 1 and ring[RW_DUE, slot] > i:
            if i + 1 == n or (
                new_day[i + 1]
                and not cross_session
                and not _held_lifetime_crosses(
                    hv, next_bar, int(ring[RW_KIND, slot]), int(ring[RW_IDENT, slot]),
                    triggers, has_trig, pair,
                )
            ):  # fmt: skip
                return True
        m >>= 1
        slot += 1
    return False


@nb.njit(cache=True, nogil=True)
def _slots_loop(
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
    triggers,
    su,
    tu,
    ba,
    bo,
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
    straddle=None,
    from_open=None,
    stop_fracs=None,
    target_fracs=None,
    sizing_risks=None,
    compat=COMPAT_SLOTS,
):
    """The slots matcher loop: numeric ABI, optional strategy callback, the route in ``compat``.

    Returns ``(n_trades, err, err_bar, final_side, final_entry_i, final_entry_px,
    final_risk_stop)``. A callback or account refusal returns ``err == ERR_NONE`` with
    ``core.account[2]`` set. A straddle with ``arm_stop`` or DAY cancel is refused (ValueError).
    """
    n = o.shape[0]
    flags = _rule_flags(rules, compat)
    sflags = _slot_flags(rules)
    adaptive = flags & F_ADAPTIVE != 0
    adverse_first = flags & F_ADVERSE_FIRST != 0
    reenter = flags & F_REENTER != 0
    stale_drop = flags & F_STALE_DROP != 0
    target_gap = flags & F_TARGET_GAP != 0
    protect_flatten = flags & F_PROTECT_FLATTEN != 0
    next_open = flags & F_NEXT_OPEN != 0
    exit_next_open = flags & F_EXIT_NEXT_OPEN != 0
    cross_session = flags & F_CROSS_SESSION != 0
    day_cancel = flags & F_DAY_CANCEL != 0
    refuse_new_day = (flags & C_NEW_DAY_MASK) >> C_NEW_DAY_SHIFT == NEW_DAY_REFUSE_RESTING
    mit = sflags & S_MIT != 0
    oto = sflags & S_OTO != 0
    amend_keep = sflags & S_AMEND_KEEP != 0
    ratchet = sflags & S_RATCHET != 0
    stop_skip = sflags & S_STOP_SKIP != 0
    amend_close = sflags & S_AMEND_CLOSE != 0
    completed_close = sflags & S_COMPLETED_CLOSE != 0
    arm_stop = sflags & S_ARM_STOP != 0
    next_bar = sflags & S_NEXT_BAR != 0
    exit_and_rearm = sflags & S_EXIT_AND_REARM != 0
    tick = rules[R_TICK]
    slip = tick if flags & F_ONE_TICK != 0 else 0.0
    tslip = 0.0 if mit else slip  # a MIT target fills on the touch, the others a tick through
    tick_round = int(rules[R_TICK_ROUND])
    combine = int(rules[R_COMBINE])
    bracket_anchor = int(rules[R_BRACKET_ANCHOR])
    chase = int(rules[R_CHASE])
    latency = int(rules[R_LATENCY_BARS])
    # Absent columns (None or zero length) read as their default; decided once, not per bar.
    has_exit = _has_lane(exit_sig)
    has_long = _has_lane(long_exits)
    has_short = _has_lane(short_exits)
    has_es = _has_lane(entry_stop)
    has_et = _has_lane(entry_target)
    has_sp = _has_lane(stop_pts)
    has_tp = _has_lane(target_pts)
    has_limit = _has_lane(entry_limit)
    has_trig = _has_lane(triggers)
    has_su = _has_lane(su)
    has_tu = _has_lane(tu)
    has_ba = _has_lane(ba)
    has_bo = _has_lane(bo)
    has_from = _has_lane(from_open)
    has_sf = _has_lane(stop_fracs)
    has_tf = _has_lane(target_fracs)
    has_ioc = _has_lane(ioc_limits)
    has_close = _has_lane(force_close)
    has_open = _has_lane(force_open)
    has_expiry = _has_lane(expiry_days)
    has_pairs = _pair_cols(straddle) > 0
    hv = _held_view(core)
    pair_masks = _event_masks(core)
    event_stored = _event_stored(core)
    # Phase 0 (the open) settles only commands that can be due at an open.
    phase0 = next_open or exit_next_open or has_ioc or has_pairs
    has_straddle = False
    if straddle is not None and has_pairs:
        for value in straddle.flat:
            if np.isfinite(value):
                has_straddle = True
                break
    if has_straddle:
        if arm_stop:
            raise ValueError("straddle arm_stop=next_bar composition is unproved")
        if day_cancel:
            raise ValueError("straddle day_order_expiry=cancel composition is unproved")

    # Loop-carried scalars start as int64, not as literals (see ``book._book_loop``).
    zero = n - n
    unset = zero - 1
    pos, cnt, ei, gen, gident = zero, zero, unset, unset, unset
    ep, stop, target, risk_stop = np.nan, np.nan, np.nan, np.nan
    live, resting, chasing, closing, be_done = False, False, False, False, False
    # The working entry: its decision bar and held identity, and what it needs again.
    pend, pident = unset, unset
    pside, ptrig, plim, pend_pair, pfrom = zero, np.nan, np.nan, False, False
    ring = np.empty((RING_ROWS, SCAP))
    qmask, qseq = zero, zero
    ovf = False
    err, eb = ERR_NONE, -1
    for i in range(n):
        at = _scratch_at(core, i)
        oi = o[i]
        hi = h[i]
        li = lo[i]
        ci = c[i]
        if core is None:
            s = int(side_sig[i])
            xg, xl, xs = _exit_flags(
                exit_sig, long_exits, short_exits, i, has_exit, has_long, has_short
            )
            cmask = _lane_at(force_close, i, 0) if has_close else 0
            oflat = _lane_at(force_open, i, False) if has_open else False
            cur = i
            cur_pair = has_pairs and _held_pair_any(hv, straddle, i)
            if (
                pos == 0
                and pend < 0
                and qmask == 0
                and s == 0
                and not xg
                and cmask == 0
                and not oflat
                and not cur_pair
            ):
                closing = False
                continue  # nothing rests, nothing is in flight, nothing to enter
        else:
            # Nothing is published yet at this bar's open: the strategy decides at the close.
            s = 0
            xg, xl, xs = False, False, False
            cmask = 0
            cur = i
            cur_pair = False
            oflat = _open_at(core, force_open, i)
        idle = pos == 0 and pend < 0 and qmask == 0 and not oflat
        exited = False
        busy = pend >= 0  # a working entry: the sleeve cannot place another
        pinned, pin = zero, np.nan  # side and price of a quote a target fill left (A13)
        error = zero
        halt = False
        if not idle:
            if (
                pend >= 0
                and day_cancel
                and (_lane_at(expiry_days, i, False) if has_expiry else new_day[i])
                and (not np.isnan(ptrig) or not np.isnan(plim))
            ):
                qmask = _sdrop(ring, qmask, pend)
                pend, resting, chasing, closing, busy = unset, False, False, False, False
            if (
                refuse_new_day
                and pend >= 0
                and new_day[i]
                and not day_cancel
                and not (next_bar and i == pend + 1)
                and ((pend_pair and _held_pair_rests(hv, straddle, pident))
                     or not np.isnan(ptrig) or not np.isnan(plim))
            ):  # fmt: skip
                err, eb = ERR_PENDING_NEW_DAY, i
                break
            if oflat:
                if pos != 0:
                    _venue_record(
                        count_only, base + cnt, ei, i, pos, ep, oi - pos * slip, REASON_SIGNAL,
                        risk_stop, out_entry_i, out_exit_i, out_side, out_entry_px,
                        out_exit_px, out_reason, out_stop,
                    )  # fmt: skip
                    cnt += 1
                qmask = zero
                pos, pend, live, resting, chasing, closing = zero, unset, False, False, False, False
        prev = oi
        for phase in range(2):
            if phase == 0 and (idle or qmask == 0 or not phase0):
                continue
            cx = oi if phase == 0 else ci  # the price this phase settles at
            if phase == 1:
                if not idle:
                    # --- the path O, e1, e2, C: only when something can fill ---
                    run_path = pend >= 0 and resting
                    if not run_path and pos != 0 and live:
                        mn = min(min(oi, ci), min(hi, li))
                        mx = max(max(oi, ci), max(hi, li))
                        if pos > 0:
                            run_path = stop >= mn or target <= mx - tslip
                        else:
                            run_path = stop <= mx or target >= mn + tslip
                    if run_path:
                        adverse_hit = (
                            adverse_first
                            and pos != 0
                            and live
                            and not np.isnan(stop)
                            and (li <= stop if pos > 0 else hi >= stop)
                        )
                        high_first = _high_first(adaptive, oi, hi, li)
                        if pend >= 0 and resting and i == pend + 1 and pfrom and (
                            _held_open_in_market(
                                hv, triggers, has_trig, straddle, pside, pident, oi
                            )
                        ):  # fmt: skip
                            # The trigger is submitted at this open: at or through it is in the market.
                            if stop_skip and not pend_pair:
                                pend, resting = unset, False
                            else:
                                err, eb = ERR_STOP_ENTRY_IN_MARKET, i
                                halt = True
                                break
                        for m in range(4):
                            pt = (
                                oi
                                if m == 0
                                else (ci if m == 3 else (hi if (m == 1) == high_first else li))
                            )
                            if (pinned > 0 and pt < prev) or (pinned < 0 and pt > prev):
                                pinned = zero
                            prev = pt
                            entered = False
                            if pend >= 0 and resting:
                                d, s_hit = pend, pside
                                stop_order = not np.isnan(ptrig)
                                level = ptrig if stop_order else plim
                                if pend_pair:
                                    s_hit = zero
                                    for k in range(2):
                                        offset = 6 * k
                                        direction = 1 - 2 * k
                                        trigger = _held_pair(hv, straddle, offset, pident)
                                        if (
                                            not np.isnan(trigger)
                                            and direction * (pt - trigger) >= 0
                                        ):
                                            s_hit, stop_order, level = direction, True, trigger
                                            break
                                match = s_hit != 0 and _in_market(
                                    -s_hit,
                                    level,
                                    pt if stop_order else pt + s_hit * slip,
                                    stop_order,
                                )
                                if match:
                                    if slip and next_open and stop_order:
                                        # R15b: the A4 price (trigger on a touch, open on a gap) plus one tick.
                                        px = _round_price(
                                            (pt if m == 0 else level) + s_hit * slip,
                                            tick,
                                            tick_round,
                                            s_hit,
                                        )
                                    else:
                                        px = (
                                            pt + s_hit * slip
                                            if slip
                                            else (pt if stop_order and m == 0 else level)
                                        )
                                    anchor = _anchor(
                                        bracket_anchor,
                                        c[d],
                                        px,
                                        plim if not np.isnan(plim) else px,
                                    )
                                    stop, target = _entry_resolve(
                                        hv, s_hit, anchor, px, pident, pend_pair, entry_stop,
                                        entry_target, stop_pts, target_pts, stop_fracs,
                                        target_fracs, straddle, has_es, has_et, has_sp, has_tp,
                                        has_sf, has_tf, tick, tick_round, combine, True,
                                    )  # fmt: skip
                                    if slip and next_open and stop_order:
                                        # Quantity is submitted against the trigger geometry, before slippage.
                                        submitted_stop, _ = _entry_resolve(
                                            hv, s_hit,
                                            _anchor(bracket_anchor, c[d], level, level),
                                            level, pident, pend_pair, entry_stop, entry_target,
                                            stop_pts, target_pts, stop_fracs, target_fracs,
                                            straddle, has_es, has_et, has_sp, has_tp, has_sf,
                                            has_tf, tick, tick_round, combine, False,
                                        )  # fmt: skip
                                        submitted_risk = abs(level - submitted_stop)
                                        _store_risk(sizing_risks, i, submitted_risk)
                                        if (
                                            core is not None
                                            and core.economic[E_MODE] != MODE_QTY_FEATURE
                                        ):
                                            core.sizing_risks[i] = submitted_risk
                                    if (
                                        core is not None
                                        and core.economic[E_MODE] == MODE_QTY_FEATURE
                                    ):
                                        _entry_quantity(core, d, cnt)
                                    pos, ei, ep, gen, risk_stop = s_hit, i, px, d, stop
                                    gident = pident
                                    be_done = False
                                    pend, resting, chasing = unset, False, False
                                    live, entered = False, True
                                    if oto:
                                        if mit and _in_market(pos, target, pt, False):
                                            error = ERR_MIT_IN_MARKET
                                        elif _in_market(pos, stop, pt, True):
                                            error = ERR_PROTECT_REJECTED
                                        else:
                                            live = True
                                            if _in_market(pos, target, pt, False):
                                                _venue_record(
                                                    count_only, base + cnt, ei, i, pos, ep,
                                                    pt - pos * slip, REASON_TARGET, risk_stop,
                                                    out_entry_i, out_exit_i, out_side,
                                                    out_entry_px, out_exit_px, out_reason,
                                                    out_stop,
                                                )  # fmt: skip
                                                cnt += 1
                                                pos, exited = zero, True
                                    else:
                                        qmask, qseq, ok = _spush(
                                            ring, qmask, qseq,
                                            i + latency + (int(exit_next_open) if closing else 0),
                                            _SLOT_CLOSE if closing else _SLOT_PROTECT, d, gident,
                                            0.0, 0, pend_pair,
                                        )  # fmt: skip
                                        ovf = ovf or not ok
                            if pos != 0 and live and not entered:
                                reason, px = 0, 0.0
                                if _in_market(pos, stop, pt, True):
                                    reason = REASON_STOP_GAP if m == 0 else REASON_STOP
                                    if slip and next_open:
                                        px = (pt if m == 0 else stop) - pos * slip
                                    else:
                                        px = pt - pos * slip if slip else (pt if m == 0 else stop)
                                elif not adverse_hit and _in_market(
                                    pos, target, pt if mit else pt - pos * slip, False
                                ):
                                    reason = REASON_TARGET
                                    px = _target_fill_price(
                                        pt, target, m == 0, mit or target_gap, pos, slip
                                    )
                                    if not mit and not slip and (m == 1 or m == 2):
                                        pinned, pin = pos, px
                                if reason:
                                    _venue_record(
                                        count_only, base + cnt, ei, i, pos, ep, px, reason,
                                        risk_stop, out_entry_i, out_exit_i, out_side,
                                        out_entry_px, out_exit_px, out_reason, out_stop,
                                    )  # fmt: skip
                                    cnt += 1
                                    pos, exited, live = zero, True, False
                    if (
                        next_bar
                        and pend >= 0
                        and resting
                        and i > pend
                        and (not np.isnan(ptrig) or (pend_pair and _held_pair_rests(hv, straddle, pident)))
                    ):  # fmt: skip
                        # GTD next bar: an unfilled stop expires before this close's decision.
                        qmask = _sdrop(ring, qmask, pend)
                        pend, resting, busy = unset, False, False
                    if error:
                        err, eb = error, i
                        halt = True
                        break
                if core is not None:
                    _core_decision(
                        core, i, oi, hi, li, ci, pos, ei, ep,
                        stop if live else np.nan, target if live else np.nan,
                        pside if pend >= 0 else 0,
                        (ptrig if not np.isnan(ptrig) else plim) if pend >= 0 else np.nan,
                        risk_stop, cnt, out_entry_i, out_exit_i, out_side, out_entry_px,
                        out_exit_px, out_reason, out_stop,
                        _held_pair_level(hv, straddle, pident, 0, pend, i, latency)
                        if (pend >= 0 and has_pairs)
                        else np.nan,
                        _held_pair_level(hv, straddle, pident, 6, pend, i, latency)
                        if (pend >= 0 and has_pairs)
                        else np.nan,
                        o[i + 1] if i + 1 < n else np.nan,
                    )  # fmt: skip
                    if core.account[2] != 0:
                        halt = True
                        break
                    # The decision just published is read back from the scratch.
                    cur = _held_ident(hv, i)
                    s = int(side_sig[at])
                    xg, xl, xs = _exit_flags(
                        exit_sig, long_exits, short_exits, at, has_exit, has_long, has_short
                    )
                    cmask = _close_at(core, force_close, i)
                    oflat = _open_at(core, force_open, i)
                    cur_pair = (
                        has_pairs
                        and _pair_possible(pair_masks, event_stored, cur)
                        and _held_pair_any(hv, straddle, cur)
                    )
                if (
                    pos == 0
                    and pend < 0
                    and qmask == 0
                    and s == 0
                    and not xg
                    and cmask == 0
                    and not oflat
                    and not cur_pair
                ):
                    closing = False
                    continue  # the strategy has observed the bar; there is nothing to act on
                if cmask & 8:
                    if pos != 0 or pend >= 0:
                        err, eb = ERR_NEXT_OPEN_SESSION_END, i
                        halt = True
                        break
                    continue
                signal = s != 0 or cur_pair
                fside = pos if pos != 0 else (pside if pend >= 0 else 0)
                if (
                    (cmask & 1) != 0
                    or (fside > 0 and (cmask & 2) != 0)
                    or (fside < 0 and (cmask & 4) != 0)
                    or oflat
                ):
                    if pos != 0:
                        _venue_record(
                            count_only, base + cnt, ei, i, pos, ep, ci - pos * slip, REASON_SIGNAL,
                            risk_stop, out_entry_i, out_exit_i, out_side, out_entry_px,
                            out_exit_px, out_reason, out_stop,
                        )  # fmt: skip
                        cnt += 1
                    qmask = zero
                    pos, pend, live, resting, chasing, closing = (
                        zero, unset, False, False, False, False,
                    )  # fmt: skip
                    if not (exit_and_rearm and reenter and signal):
                        continue
                    exited, busy = True, False
                xside = pos if pos != 0 else (pside if pend >= 0 else 0)
                exit_now = xg or (xside > 0 and xl) or (xside < 0 and xs)
                if pend >= 0 and pend_pair and (xl or xs or (cmask & 6) != 0):
                    err, eb = ERR_STRADDLE_SIDE_CANCEL, i
                    halt = True
                    break
                if pos == 0 and pend < 0:
                    closing = False
                if pend >= 0 and not np.isnan(plim) and not closing and i == pend + chase:
                    chasing = True
                    qmask, qseq, ok = _spush(
                        ring, qmask, qseq,
                        i + latency + (int(exit_next_open) if exit_now else int(next_open)),
                        _SLOT_CANCEL, pend, pident, 0.0, 0, pend_pair,
                    )  # fmt: skip
                    ovf = ovf or not ok
                if exit_now:
                    chasing = False
                    if pend >= 0 and exit_and_rearm and reenter and signal:
                        # A23: this close cancels the working entry, then submits the new one.
                        qmask = _sdrop(ring, qmask, pend)
                        pend, resting, closing = unset, False, False
                    elif pend >= 0:
                        qmask, qseq, ok = _spush(
                            ring, qmask, qseq, i + latency + int(exit_next_open), _SLOT_CANCEL,
                            pend, pident, 0.0, 0, pend_pair,
                        )  # fmt: skip
                        ovf = ovf or not ok
                        closing = not oto
                    if pos != 0 and not closing:
                        qmask, qseq, ok = _spush(
                            ring, qmask, qseq, i + latency + int(exit_next_open), _SLOT_CLOSE,
                            gen, gident, 0.0, 0, False,
                        )  # fmt: skip
                        ovf = ovf or not ok
                        closing = True
                elif pos != 0:
                    update = _lane_at(su, at, np.nan) if has_su else np.nan
                    if (
                        not be_done
                        and has_ba
                        and not np.isnan(_held_level(hv, ba, gident, 8, np.nan))
                    ):
                        ba_gen = _held_level(hv, ba, gident, 8, np.nan)
                        reached = hi >= ep + ba_gen if pos > 0 else li <= ep - ba_gen
                        if reached:
                            if not np.isnan(update):
                                error = ERR_AMEND_REJECTED
                            bo_gen = _held_level(hv, bo, gident, 9, np.nan) if has_bo else np.nan
                            update = ep + pos * (0.0 if np.isnan(bo_gen) else bo_gen)
                            be_done = True
                    if not np.isnan(update):
                        update = _round_price(update, tick, tick_round, -pos)
                        has_stop, bound = _sbound(ring, qmask, gen, stop, pos)
                        if not has_stop and arm_stop:
                            qmask, qseq, ok = _spush(
                                ring, qmask, qseq, i + latency, _SLOT_ARM_STOP, gen, gident,
                                update, 0, False,
                            )  # fmt: skip
                            ovf = ovf or not ok
                        elif not has_stop:
                            error = ERR_AMEND_NO_ORDER
                        elif not ratchet or pos * (update - bound) > 0:
                            qmask, qseq, ok = _spush(
                                ring, qmask, qseq,
                                i + latency + int(next_open and not amend_close),
                                _SLOT_AMEND_STOP, gen, gident, update, 0, False,
                            )  # fmt: skip
                            ovf = ovf or not ok
                    tgt_update = _lane_at(tu, at, np.nan) if has_tu else np.nan
                    if not np.isnan(tgt_update):
                        if (
                            np.isnan(target)
                            and arm_stop
                            and np.isnan(stop)
                            and not np.isnan(update)
                        ):
                            qmask, qseq, ok = _spush(
                                ring, qmask, qseq, i + latency, _SLOT_ARM_TARGET, gen, gident,
                                _round_price(tgt_update, tick, tick_round, pos), 0, False,
                            )  # fmt: skip
                            ovf = ovf or not ok
                        elif np.isnan(target):
                            error = ERR_AMEND_NO_ORDER
                        else:
                            qmask, qseq, ok = _spush(
                                ring, qmask, qseq,
                                i + latency + int(next_open and not amend_close),
                                _SLOT_AMEND_TARGET, gen, gident,
                                _round_price(tgt_update, tick, tick_round, pos), 0, False,
                            )  # fmt: skip
                            ovf = ovf or not ok
                rearm = exit_and_rearm and exit_now and reenter
                if (
                    pos == 0
                    and pend >= 0
                    and (pend_pair or not np.isnan(ptrig))
                    and signal
                    and not exit_now
                ):
                    if latency:
                        err, eb = ERR_STOP_REPLACE_LATENCY, i
                        halt = True
                        break
                    qmask, qseq, ok = _spush(
                        ring, qmask, qseq, i + latency + int(next_open), _SLOT_CANCEL, pend,
                        pident, 0.0, 0, pend_pair,
                    )  # fmt: skip
                    ovf = ovf or not ok
                    qmask, qseq, ok = _spush(
                        ring, qmask, qseq, i + latency + int(next_open), _SLOT_REPLACE, i, cur,
                        0.0, 0, cur_pair,
                    )  # fmt: skip
                    ovf = ovf or not ok
                elif (
                    (pos == 0 or rearm)
                    and pend < 0
                    and (not busy or rearm)
                    and (not closing or rearm)
                    and signal
                    and (reenter or not exited)
                ):
                    cur_ioc = (
                        not np.isnan(_held_level(hv, ioc_limits, cur, 10, np.nan))
                        if has_ioc
                        else False
                    )
                    cur_trig = _lane_at(triggers, at, np.nan) if has_trig else np.nan
                    if (
                        exit_and_rearm
                        and exit_now
                        and pos != 0
                        and exit_next_open
                        and not next_open
                        and not cur_pair
                        and np.isnan(cur_trig)
                        and not cur_ioc
                    ):
                        error = ERR_REARM_BEFORE_CLOSE
                    pend, pident = i, cur
                    pside, ptrig, plim, pend_pair, pfrom = _pend_load(
                        hv, cur, side_sig, triggers, entry_limit, from_open, straddle,
                        has_trig, has_limit, has_from, has_pairs,
                    )  # fmt: skip
                    if cur_pair or cur_ioc:
                        due = i + latency + 1
                        if due >= n or (
                            not cross_session
                            and not (next_bar and cur_pair and _held_stop_pair(hv, straddle, cur))
                            and _any_in(new_day, i + 1, due + 1)
                        ):
                            error = ERR_NEXT_OPEN_SESSION_END
                    qmask, qseq, ok = _spush(
                        ring, qmask, qseq, i + latency + (1 if cur_ioc else int(next_open)),
                        _SLOT_SUBMIT, i, cur, 0.0, 1 if cur_ioc else 0, cur_pair,
                    )  # fmt: skip
                    ovf = ovf or not ok
                elif (
                    slip != 0.0
                    and next_open
                    and pos == 0
                    and pend < 0
                    and busy
                    and not closing
                    and signal
                    and (reenter or not exited)
                    and not error
                ):
                    # R15b: refuse instead of letting the whole-bar busy guard drop this entry.
                    error = ERR_BUSY_ENTRY_DROPPED
                if next_open and exit_now and pos != 0 and reenter and s != 0 and pend < 0:
                    pend, pident = i, cur
                    pside, ptrig, plim, pend_pair, pfrom = _pend_load(
                        hv, cur, side_sig, triggers, entry_limit, from_open, straddle,
                        has_trig, has_limit, has_from, has_pairs,
                    )  # fmt: skip
                    cur_ioc = (
                        not np.isnan(_held_level(hv, ioc_limits, cur, 10, np.nan))
                        if has_ioc
                        else False
                    )
                    qmask, qseq, ok = _spush(
                        ring, qmask, qseq, i + 1, _SLOT_SUBMIT, i, cur, 0.0, 1 if cur_ioc else 0,
                        cur_pair,
                    )  # fmt: skip
                    ovf = ovf or not ok
            # --- the commands that are due settle, in insertion order ---
            # A next-open command still ahead at this close is checked here, before anything
            # settles. Nothing pushed by the settle loop below can fail the same test: with no
            # latency, only a straddle submission's IOC attempt is due after this close
            # (``d + 1``), and the settle-time pair check just passed ``d + 1 < n`` with no new
            # date there (or a session-crossing mode, or no IOC leg to push), so the scan would
            # find nothing.
            if qmask != 0 and not error and not latency and (next_open or exit_next_open):
                if _session_end(
                    ring,
                    qmask,
                    i,
                    n,
                    new_day,
                    cross_session,
                    next_bar,
                    hv,
                    triggers,
                    has_trig,
                    straddle,
                ):
                    error = ERR_NEXT_OPEN_SESSION_END
            while qmask != 0 and error == 0:
                slot = _spop(ring, qmask, i, next_open or exit_next_open, phase == 1, gen, ei)
                if slot < 0:
                    break
                kind = int(ring[RW_KIND, slot])
                d = int(ring[RW_D, slot])
                dident = int(ring[RW_IDENT, slot])
                value = ring[RW_VALUE, slot]
                dpair = ring[RW_PAIR, slot] != 0.0
                qmask &= ~(1 << slot)
                if kind == _SLOT_REPLACE:
                    if pos != 0 or pend >= 0:
                        continue
                    pend, pident, kind = d, dident, _SLOT_SUBMIT
                    pside, ptrig, plim, pend_pair, pfrom = _pend_load(
                        hv, dident, side_sig, triggers, entry_limit, from_open, straddle,
                        has_trig, has_limit, has_from, has_pairs,
                    )  # fmt: skip
                if kind == _SLOT_SUBMIT and pend == d:
                    if dpair:
                        due = d + latency + 1
                        if due >= n or (
                            not cross_session
                            and not (next_bar and _held_stop_pair(hv, straddle, dident))
                            and _any_in(new_day, d + 1, due + 1)
                        ):
                            error = ERR_NEXT_OPEN_SESSION_END
                            continue
                        long_trigger = _held_pair(hv, straddle, 0, dident)
                        short_trigger = _held_pair(hv, straddle, 6, dident)
                        if (
                            not (phase == 0 and next_open)
                            and not pfrom
                            and (
                                (
                                    not np.isnan(long_trigger)
                                    and (pin if pinned == -1 else cx) >= long_trigger
                                )
                                or (
                                    not np.isnan(short_trigger)
                                    and (pin if pinned == 1 else cx) <= short_trigger
                                )
                            )
                        ):
                            error = ERR_STOP_ENTRY_IN_MARKET
                        else:
                            resting = True
                            if not (
                                np.isnan(_held_pair(hv, straddle, 1, dident))
                                and np.isnan(_held_pair(hv, straddle, 7, dident))
                            ):
                                qmask, qseq, ok = _spush(
                                    ring, qmask, qseq, d + latency + 1, _SLOT_PAIR_IOC, d, dident,
                                    0.0, 1, True,
                                )  # fmt: skip
                                ovf = ovf or not ok
                        continue
                    fs = pside
                    cap = _held_level(hv, ioc_limits, dident, 10, np.nan) if has_ioc else np.nan
                    if not np.isnan(cap) and fs * (cx - cap) > 0:
                        pend, resting, chasing, closing = unset, False, False, False
                        continue
                    q = pin if pinned == -fs else cx
                    if not np.isnan(ptrig):
                        if (
                            not (phase == 0 and next_open)
                            and not pfrom
                            and _in_market(-fs, ptrig, q, True)
                        ):
                            if stop_skip:
                                pend, resting = unset, False
                            else:
                                error = ERR_STOP_ENTRY_IN_MARKET
                        else:
                            resting = True
                        continue
                    if not np.isnan(plim) and fs * (cx + fs * slip - plim) > 0:
                        resting = True
                        continue
                    px = cx + fs * slip
                    anchor = _anchor(bracket_anchor, c[d], px, plim if not np.isnan(plim) else px)
                elif kind == _SLOT_PAIR_IOC:
                    # Stable long-then-short priority at one opening price. A stop leg gapped
                    # at this open wins in the path at m == 0; an eligible IOC fills here.
                    fs, px, anchor = zero, 0.0, 0.0
                    if pend == d and resting and pos == 0 and phase == 0:
                        for k in range(2):
                            offset = 6 * k
                            direction = 1 - 2 * k
                            trigger = _held_pair(hv, straddle, offset, dident)
                            cap = _held_pair(hv, straddle, offset + 1, dident)
                            if not np.isnan(trigger) and direction * (cx - trigger) >= 0:
                                break
                            if not np.isnan(cap) and direction * (cx - cap) <= 0:
                                fs = direction
                                break
                        px = cx + fs * slip
                        anchor = _anchor(bracket_anchor, c[d], px, px)
                        if (
                            fs == 0
                            and np.isnan(_held_pair(hv, straddle, 0, dident))
                            and np.isnan(_held_pair(hv, straddle, 6, dident))
                        ):
                            pend, resting = unset, False
                elif kind == _SLOT_CHASE and pend == d:
                    fs = pside
                    px = cx + fs * slip
                    anchor = _anchor(bracket_anchor, c[d], px, px)
                else:
                    fs, px, anchor = zero, 0.0, 0.0
                if fs != 0:
                    # A market entry at this open is settled before the completed-close
                    # decision. Resting limit/stop callbacks keep their A15 busy contract.
                    if completed_close and phase == 0 and np.isnan(plim) and np.isnan(ptrig):
                        busy = False
                    if core is not None and core.economic[E_MODE] == MODE_QTY_FEATURE:
                        _entry_quantity(core, d, cnt)
                    pos, ei, ep, gen, gident = fs, i, px, d, dident
                    stop, target = _entry_resolve(
                        hv, fs, anchor, px, dident, dpair, entry_stop, entry_target, stop_pts,
                        target_pts, stop_fracs, target_fracs, straddle, has_es, has_et, has_sp,
                        has_tp, has_sf, has_tf, tick, tick_round, combine, True,
                    )  # fmt: skip
                    risk_stop, be_done = stop, False
                    pend, resting, chasing, live = unset, False, False, False
                    if oto:
                        # Native child processing is synchronous within the entry fill.
                        q = pin if pinned == pos else cx
                        if mit and _in_market(pos, target, q, False):
                            error = ERR_MIT_IN_MARKET
                        elif _in_market(pos, stop, q, True):
                            if pinned == pos and not _in_market(pos, stop, cx, True):
                                if not stale_drop:
                                    error = ERR_STALE_BOOK
                                else:
                                    stop, target = np.nan, np.nan
                            else:
                                error = ERR_PROTECT_REJECTED
                        else:
                            live = True
                            if _in_market(pos, target, q, False):
                                _venue_record(
                                    count_only, base + cnt, ei, i, pos, ep, q - pos * slip,
                                    REASON_TARGET, risk_stop, out_entry_i, out_exit_i, out_side,
                                    out_entry_px, out_exit_px, out_reason, out_stop,
                                )  # fmt: skip
                                cnt += 1
                                pos, exited, live = zero, True, False
                    else:
                        qmask, qseq, ok = _spush(
                            ring, qmask, qseq, i + latency,
                            _SLOT_CLOSE if closing else _SLOT_PROTECT, d, dident, 0.0, 0, dpair,
                        )  # fmt: skip
                        ovf = ovf or not ok
                elif kind == _SLOT_CANCEL and pend == d:
                    pend, resting = unset, False
                    if chasing:
                        pend = d
                        qmask, qseq, ok = _spush(
                            ring, qmask, qseq, i + latency, _SLOT_CHASE, d, dident, 0.0, 0, dpair
                        )  # fmt: skip
                        ovf = ovf or not ok
                elif pos != 0 and gen == d:
                    q = pin if pinned == pos else cx
                    reason = 0
                    if kind == _SLOT_PROTECT:
                        if mit and _in_market(pos, target, q, False):
                            error = ERR_MIT_IN_MARKET
                        elif _in_market(pos, stop, q, True):
                            if not protect_flatten:
                                error = ERR_PROTECT_REJECTED
                            else:
                                qmask, qseq, ok = _spush(
                                    ring, qmask, qseq, i + latency, _SLOT_PROTECT_FAIL, gen,
                                    gident, 0.0, 0, False,
                                )  # fmt: skip
                                ovf = ovf or not ok
                        else:
                            live = True
                            if _in_market(pos, target, q, False):
                                reason = REASON_TARGET
                    elif kind == _SLOT_CLOSE or kind == _SLOT_PROTECT_FAIL:
                        reason = REASON_SIGNAL if kind == _SLOT_CLOSE else REASON_PROTECT_FAIL
                    elif kind == _SLOT_ARM_STOP:
                        if _in_market(pos, value, q, True):
                            if not protect_flatten:
                                error = ERR_PROTECT_REJECTED
                            else:
                                qmask, qseq, ok = _spush(
                                    ring, qmask, qseq, i + latency, _SLOT_PROTECT_FAIL, gen,
                                    gident, 0.0, 0, False,
                                )  # fmt: skip
                                ovf = ovf or not ok
                        else:
                            stop, live = value, True
                    elif kind == _SLOT_ARM_TARGET and not np.isnan(stop):
                        if mit and _in_market(pos, value, q, False):
                            error = ERR_MIT_IN_MARKET
                        else:
                            target = value
                            if _in_market(pos, value, q, False):
                                reason = REASON_TARGET
                    elif kind == _SLOT_AMEND_STOP or kind == _SLOT_AMEND_TARGET:
                        is_stop = kind == _SLOT_AMEND_STOP
                        invalid = _in_market(pos, value, q, is_stop)
                        if invalid and (is_stop or mit):
                            if not amend_keep:
                                error = ERR_AMEND_REJECTED
                        elif is_stop:
                            stop = value
                        else:
                            target = value
                            if invalid:
                                reason = REASON_TARGET
                    if reason:
                        _venue_record(
                            count_only, base + cnt, ei, i, pos, ep, q - pos * slip, reason,
                            risk_stop, out_entry_i, out_exit_i, out_side, out_entry_px,
                            out_exit_px, out_reason, out_stop,
                        )  # fmt: skip
                        cnt += 1
                        if (next_open or exit_and_rearm) and pend >= 0 and pend != gen:
                            closing = False
                        pos, exited, live = zero, True, False
        if halt:
            break
        if ovf:
            err, eb = ERR_RING_OVERFLOW, i
            break
        if error:
            err, eb = error, i
            break
    return cnt, err, eb, pos, ei, ep, risk_stop


__all__ = [
    "S_AMEND_CLOSE",
    "S_AMEND_KEEP",
    "S_ARM_STOP",
    "S_COMPLETED_CLOSE",
    "S_EXIT_AND_REARM",
    "S_MIT",
    "S_NEXT_BAR",
    "S_OTO",
    "S_RATCHET",
    "S_STOP_SKIP",
    "_entry_resolve",
    "_pend_load",
    "_session_end",
    "_slot_flags",
    "_slots_loop",
]
