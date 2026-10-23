"""Reference implementation: slow, literal, structured differently from ``kernel.py``.

Where the kernel evaluates closed-form conditions per bar, the reference replays each bar as four
price updates (open, then high and low in the configured order, then close) and matches the resting
stop and limit against every update, the way a venue does. Both implement the spec in
``docs/vector-engine.md``; tests require exact agreement on thousands of generated inputs.

It exists to be read and trusted, not to be fast. Do not optimise it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .kernel import (
    ERR_NEXT_OPEN_SESSION_END,
    ERR_NONE,
    ERR_PENDING_NEW_DAY,
    ERR_PROTECT_REJECTED,
    ERR_STALE_BOOK,
    ERR_STOP_IN_MARKET,
    ERR_TARGET_IN_MARKET,
    REASON_PROTECT_FAIL,
    REASON_SIGNAL,
    REASON_STOP,
    REASON_STOP_GAP,
    REASON_TARGET,
)


@dataclass
class RefTrade:
    entry_i: int
    exit_i: int
    side: int
    entry_px: float
    exit_px: float
    reason: int
    stop: float  # the position's stop level (NaN = none)


@dataclass
class RefColumn:
    trades: list[RefTrade]
    err: int
    err_bar: int
    final_side: int
    final_entry_i: int
    final_entry_px: float = math.nan
    final_stop: float = math.nan


def _price_points(
    o: float, h: float, lo: float, c: float, adaptive: bool
) -> list[tuple[str, float]]:
    high_first = True
    if adaptive:
        high_first = abs(o - h) < abs(o - lo)  # tie -> low first (A6)
    extremes = [("H", h), ("L", lo)] if high_first else [("L", lo), ("H", h)]
    return [("O", o), *extremes, ("C", c)]


def _stop_fill(side: int, stop: float, label: str, price: float) -> float | None:
    """Price at which a resting stop-market closing ``side`` fills on this update, or None."""

    through = price <= stop if side > 0 else price >= stop
    if not through:
        return None
    return price if label == "O" else stop  # gap at the open fills at the open (A4)


def _limit_fill(side: int, limit: float, price: float) -> float | None:
    """Fill price of a resting limit closing ``side``: always the limit level once reached (A5)."""

    reached = price >= limit if side > 0 else price <= limit
    return limit if reached else None


class _Book:
    """The two L1 quote sides as Nautilus leaves them after a bar (A13).

    Only a resting limit fill during the high/low legs is modelled: it pins the side it trades
    against (a long's SELL limit the bid, a short's BUY limit the ask) at the limit price. Any later update that moves price the
    other way refreshes that side to the market again; one that does not leaves it stale.
    """

    def __init__(self) -> None:
        self.pinned_side = 0  # +1 bid pinned (long's limit), -1 ask pinned, 0 none
        self.pinned_px = math.nan
        self._last = math.nan

    def update(self, price: float) -> None:
        if self.pinned_side > 0 and price < self._last:
            self.pinned_side = 0
        elif self.pinned_side < 0 and price > self._last:
            self.pinned_side = 0
        self._last = price

    def pin(self, side: int, price: float) -> None:
        self.pinned_side, self.pinned_px = side, price


def _levels(
    side: int, anchor: float, stop: float, target: float, stop_pts: float, target_pts: float
) -> tuple[float, float]:
    """Absolute stop and target: given levels as they are, point offsets from ``anchor``."""

    if not math.isnan(stop_pts):
        stop = anchor - side * stop_pts
    if not math.isnan(target_pts):
        target = anchor + side * target_pts
    return stop, target


OPEN = 0
REFUSE = -1


def _protect_at_close(side: int, stop: float, target: float, close: float, flatten: bool) -> int:
    """V2 sends a limit-then-market entry's stop and target when its fill event arrives, which a
    backtest delivers at the close (A15). Nautilus rejects a stop at or through the close (V2 then
    flattens there) and fills a target at or through it at once, at the close. Returns ``OPEN``,
    ``REFUSE`` or the reason of the trade that closes at once."""

    if not math.isnan(stop) and (stop >= close if side > 0 else stop <= close):
        return REASON_PROTECT_FAIL if flatten else REFUSE
    if not math.isnan(target) and (target <= close if side > 0 else target >= close):
        return REASON_TARGET
    return OPEN


@dataclass
class _Pending:
    """A working limit-then-market entry (``LimitThenMarket``): rests, then is chased."""

    side: int
    limit: float
    chase_bar: int
    stop: float
    target: float
    stop_pts: float
    target_pts: float


@dataclass
class _MarketEntry:
    """R1 entry decided at a close, held until the following open."""

    side: int
    stop: float
    target: float
    stop_pts: float
    target_pts: float


def simulate_column_reference(
    o: np.ndarray,
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    side_sig: np.ndarray,
    exit_sig: np.ndarray,
    entry_stop: np.ndarray,
    entry_target: np.ndarray,
    *,
    adaptive: bool = False,
    reenter: bool = True,
    stale_drop: bool = False,
    entry_limit: np.ndarray | None = None,
    stop_pts: np.ndarray | None = None,
    target_pts: np.ndarray | None = None,
    chase_after: int = 1,
    protect_flatten: bool = False,
    new_day: np.ndarray | None = None,
    next_open: bool = False,
    latency_bars: int = 0,
    one_tick: bool = False,
    tick: float = 0.0,
) -> RefColumn:
    if latency_bars or one_tick:
        return _simulate_venue_reference(
            o,
            h,
            lo,
            c,
            side_sig,
            exit_sig,
            entry_stop,
            entry_target,
            adaptive=adaptive,
            reenter=reenter,
            entry_limit=entry_limit,
            stop_pts=stop_pts,
            target_pts=target_pts,
            chase_after=chase_after,
            protect_flatten=protect_flatten,
            new_day=new_day,
            latency_bars=latency_bars,
            tick=tick if one_tick else 0.0,
        )
    n = len(o)
    starts_day = np.zeros(n, bool) if new_day is None else new_day
    nan = np.full(n, np.nan)
    lim_sig = nan if entry_limit is None else entry_limit
    stop_pts_sig = nan if stop_pts is None else stop_pts
    tgt_pts_sig = nan if target_pts is None else target_pts
    trades: list[RefTrade] = []
    pos = 0
    entry_i = -1
    entry_px = 0.0
    stop = math.nan
    target = math.nan
    pending: _Pending | None = None
    market_entry: _MarketEntry | None = None
    market_exit = False
    for i in range(n):
        exited = False
        busy = False
        book = _Book()
        close = float(c[i])
        if market_exit:
            trades.append(RefTrade(entry_i, i, pos, entry_px, float(o[i]), REASON_SIGNAL, stop))
            pos = 0
            exited = True
            market_exit = False
        if market_entry is not None:
            m = market_entry
            px = float(o[i])
            st, tg = _levels(m.side, px, m.stop, m.target, m.stop_pts, m.target_pts)
            # R1's newly placed protection cannot retroactively gap-fill across the entry open.
            if not math.isnan(st) and (st >= px if m.side > 0 else st <= px):
                return RefColumn(trades, ERR_STOP_IN_MARKET, i, pos, entry_i)
            if not math.isnan(tg) and (tg <= px if m.side > 0 else tg >= px):
                return RefColumn(trades, ERR_TARGET_IN_MARKET, i, pos, entry_i)
            pos, entry_i, entry_px, stop, target = m.side, i, px, st, tg
            market_entry = None
        if pending is not None and bool(starts_day[i]):
            # V2 drops a staged chase on a new calendar day and the DAY limit keeps resting; the
            # engine does not model that, so the input is refused (A15).
            return RefColumn(trades, ERR_PENDING_NEW_DAY, i, pos, entry_i)
        if pending is not None:
            # A resting limit entry fills at its own level once reached, also on a gap (A5). V2
            # places the protection when the fill event arrives, which in a backtest is after the
            # whole bar: the stop and target meet the close, not the rest of this bar (A15).
            p = pending
            filled = False
            for _label, price in _price_points(
                float(o[i]), float(h[i]), float(lo[i]), close, adaptive
            ):
                if (price <= p.limit) if p.side > 0 else (price >= p.limit):
                    filled = True
                    break
            busy = filled or i == p.chase_bar  # the sleeve still sees a working entry this bar
            if filled or (i == p.chase_bar and not bool(exit_sig[i])):
                # Not filled by the chase bar: cancel, market at this close (V2's coordinator runs
                # before the sleeve's own decisions on the bar); levels from that fill.
                pending = None
                px = p.limit if filled else close
                st, tg = _levels(p.side, px, p.stop, p.target, p.stop_pts, p.target_pts)
                verdict = _protect_at_close(p.side, st, tg, close, protect_flatten)
                if verdict == REFUSE:
                    return RefColumn(trades, ERR_PROTECT_REJECTED, i, pos, entry_i)
                # A target filled at this close leaves nothing for this close's exit signal: V2's
                # flatten is a reduce-only close_all_positions, rejected on a flat position (A15).
                if verdict == OPEN:
                    pos, entry_i, entry_px, stop, target = p.side, i, px, st, tg
                else:
                    trades.append(RefTrade(i, i, p.side, px, close, verdict, st))
                    exited = True
        elif pos != 0:
            fill: tuple[int, float] | None = None
            for label, price in _price_points(
                float(o[i]), float(h[i]), float(lo[i]), float(c[i]), adaptive
            ):
                book.update(price)
                if fill is not None:
                    continue  # position already closed: only keep the quote state current
                if not math.isnan(stop):
                    px = _stop_fill(pos, stop, label, price)
                    if px is not None:
                        fill = (REASON_STOP_GAP if label == "O" else REASON_STOP, px)
                        continue
                if not math.isnan(target):
                    px = _limit_fill(pos, target, price)
                    if px is not None:
                        fill = (REASON_TARGET, px)
                        if label != "O":  # a limit already through at the open pins nothing (A13)
                            book.pin(pos, px)
            if fill is not None:
                trades.append(RefTrade(entry_i, i, pos, entry_px, fill[1], fill[0], stop))
                pos = 0
                exited = True
        if next_open:
            # Gate at decision time. An exit can stage a replacement entry at the same open;
            # unrelated entry signals while holding are still ignored (one NETTING position).
            market_exit = bool(exit_sig[i]) and pos != 0
            enter = (
                (pos == 0 or (market_exit and reenter))
                and int(side_sig[i]) != 0
                and (reenter or not exited)
            )
            if market_exit or enter:
                if i + 1 == n or bool(starts_day[i + 1]):
                    return RefColumn(trades, ERR_NEXT_OPEN_SESSION_END, i, pos, entry_i)
                if enter:
                    market_entry = _MarketEntry(
                        int(side_sig[i]),
                        float(entry_stop[i]),
                        float(entry_target[i]),
                        float(stop_pts_sig[i]),
                        float(tgt_pts_sig[i]),
                    )
            continue
        if bool(exit_sig[i]):
            if pos != 0:
                trades.append(RefTrade(entry_i, i, pos, entry_px, close, REASON_SIGNAL, stop))
                pos = 0
                exited = True
            pending = None  # an exit also cancels a working entry
        if (
            pos == 0
            and pending is None
            and not busy
            and int(side_sig[i]) != 0
            and (reenter or not exited)
        ):
            s = int(side_sig[i])
            lim = float(lim_sig[i])
            spec = (float(entry_stop[i]), float(entry_target[i]),
                    float(stop_pts_sig[i]), float(tgt_pts_sig[i]))  # fmt: skip
            if not math.isnan(lim):
                # A quote left stale by this bar's target fill does not affect a limit entry (A13,
                # pinned by the ltm_bracket golden strategy).
                if (lim >= close) if s > 0 else (lim <= close):
                    # Marketable: fills at the close (A11). V2 still anchors point levels to the
                    # limit price, not the fill.
                    st, tg = _levels(s, lim, *spec)
                    verdict = _protect_at_close(s, st, tg, close, protect_flatten)
                    if verdict == REFUSE:
                        return RefColumn(trades, ERR_PROTECT_REJECTED, i, pos, entry_i)
                    if verdict == OPEN:
                        pos, entry_i, entry_px, stop, target = s, i, close, st, tg
                    else:
                        trades.append(RefTrade(i, i, s, close, close, verdict, st))
                        exited = True
                else:
                    pending = _Pending(s, lim, i + chase_after, *spec)
                continue
            st, tg = _levels(s, close, *spec)
            if not math.isnan(st) and (st >= close if s > 0 else st <= close):
                return RefColumn(trades, ERR_STOP_IN_MARKET, i, pos, entry_i)
            if not math.isnan(tg) and (tg <= close if s > 0 else tg >= close):
                return RefColumn(trades, ERR_TARGET_IN_MARKET, i, pos, entry_i)
            # A13: the stop is rejected against the stale quote. A long's SELL stop needs
            # stop < pinned bid; a short's BUY stop needs stop > pinned ask.
            pinned = book.pinned_side != 0 and book.pinned_side == s
            if (
                pinned
                and not math.isnan(st)
                and (st >= book.pinned_px if s > 0 else st <= book.pinned_px)
            ):
                if not stale_drop:
                    return RefColumn(trades, ERR_STALE_BOOK, i, pos, entry_i)
                st, tg = math.nan, math.nan
            pos, entry_i, entry_px, stop, target = s, i, close, st, tg
    return RefColumn(trades, ERR_NONE, -1, pos, entry_i, entry_px, stop)


@dataclass(frozen=True)
class _VenueCommand:
    due: int
    kind: str
    entry: int


def _simulate_venue_reference(
    o: np.ndarray,
    h: np.ndarray,
    lo: np.ndarray,
    c: np.ndarray,
    side_sig: np.ndarray,
    exit_sig: np.ndarray,
    entry_stop: np.ndarray,
    entry_target: np.ndarray,
    *,
    adaptive: bool,
    reenter: bool,
    entry_limit: np.ndarray | None,
    stop_pts: np.ndarray | None,
    target_pts: np.ndarray | None,
    chase_after: int,
    protect_flatten: bool,
    new_day: np.ndarray | None,
    latency_bars: int,
    tick: float,
) -> RefColumn:
    """A16/A17: literal venue replay. Commands keep send order and incur a hop each time.

    Protection is V2's fill-callback pattern, not an exchange-native OTO order list.
    """
    n = len(c)
    nan = np.full(n, np.nan)
    limits = nan if entry_limit is None else entry_limit
    spts = nan if stop_pts is None else stop_pts
    tpts = nan if target_pts is None else target_pts
    days = np.zeros(n, bool) if new_day is None else new_day
    commands: list[_VenueCommand] = []
    trades: list[RefTrade] = []
    pos, entry_i, generation = 0, -1, -1
    entry_px, stop, target = math.nan, math.nan, math.nan
    pending = -1
    resting = False
    chasing = False
    protection = 0  # 0 absent, 1 in flight, 2 resting, 3 cancel in flight (still matches)
    closing = False
    close_reason = REASON_SIGNAL
    rejected_stop = False

    def send(kind: str, d: int) -> None:
        commands.append(_VenueCommand(i + latency_bars, kind, d))

    def close(price: float, reason: int) -> None:
        nonlocal pos, exited
        if pos:
            trades.append(RefTrade(entry_i, i, pos, entry_px, price, reason, stop))
            pos = 0
            exited = True

    def fill_entry(d: int, price: float, is_limit: bool) -> None:
        nonlocal pos, entry_i, entry_px, stop, target, generation
        nonlocal pending, resting, chasing, protection, rejected_stop
        pos, entry_i, entry_px, generation = int(side_sig[d]), i, price, d
        anchor = float(limits[d]) if is_limit else price
        stop, target = _levels(
            pos,
            anchor,
            float(entry_stop[d]),
            float(entry_target[d]),
            float(spts[d]),
            float(tpts[d]),
        )
        pending, resting, chasing = -1, False, False
        rejected_stop = False
        if closing:
            # Flatten cleared the logical attempt before its fill arrived. V2's
            # late-entry guard unwinds it, without submitting protective legs.
            protection = 0
            send("close", d)
        else:
            protection = 1
            send("protect", d)

    for i in range(n):
        exited = False
        busy = pending >= 0
        ci = float(c[i])
        if pending >= 0 and not math.isnan(limits[pending]) and days[i]:
            return RefColumn(trades, ERR_PENDING_NEW_DAY, i, pos, entry_i)
        if closing and pending < 0 and pos == 0 and protection in (0, 3):
            closing = False
        # Exchange's complete price path precedes the sleeve decision and command settlement.
        for label, price in _price_points(float(o[i]), float(h[i]), float(lo[i]), ci, adaptive):
            if pending >= 0 and resting:
                s = int(side_sig[pending])
                px = price + s * tick
                if px <= limits[pending] if s > 0 else px >= limits[pending]:
                    fill_entry(pending, px if tick else float(limits[pending]), True)
            if not pos and protection == 2 and rejected_stop and not math.isnan(target):
                # A rejected-stop flatten can leave its accepted target behind.
                # NT cancels it on a touch against a flat position, before asking
                # the fill model for liquidity (so even an exact touch clears it).
                s = int(side_sig[generation])
                if price >= target if s > 0 else price <= target:
                    protection = 0
            if pos and protection >= 2:
                if (
                    not rejected_stop
                    and not math.isnan(stop)
                    and (price <= stop if pos > 0 else price >= stop)
                ):
                    px = price - pos * tick if tick else (price if label == "O" else stop)
                    close(px, REASON_STOP_GAP if label == "O" else REASON_STOP)
                    send("cancel_protect", generation)
                    protection = 3
                elif not math.isnan(target):
                    px = price - pos * tick
                    if px >= target if pos > 0 else px <= target:
                        close(px if tick else target, REASON_TARGET)
                        send("cancel_protect", generation)
                        protection = 3
        # V2 sync sees protective fills/cancellations from this sweep before deciding.
        if closing and pending < 0 and pos == 0 and protection in (0, 3):
            closing = False
        # A live protective leg which lost a flatten race is cancelled on V2's next sync.
        if closing and pos == 0 and protection == 2:
            send("cancel_protect", generation)
            protection = 3
        if (
            pending >= 0
            and not closing
            and not math.isnan(limits[pending])
            and i == pending + chase_after
        ):
            chasing = True
            send("cancel_entry", pending)
        if bool(exit_sig[i]):
            chasing = False
            if pending >= 0:
                if not closing and not math.isnan(limits[pending]):
                    send("cancel_entry", pending)
                closing, close_reason = True, REASON_SIGNAL
            if pos:
                if protection == 2:
                    send("cancel_protect", generation)
                    protection = 3
                if not closing:
                    send("close", generation)
                closing, close_reason = True, REASON_SIGNAL
        if (
            not pos
            and pending < 0
            and not busy
            and not closing
            and side_sig[i]
            and (reenter or not exited)
        ):
            pending = i
            send("market" if math.isnan(limits[i]) else "limit", i)
        # FIFO settlement; a fill callback may append another command due now (zero latency).
        while commands and commands[0].due <= i:
            cmd = commands.pop(0)
            d = cmd.entry
            if cmd.kind in ("market", "limit"):
                if pending != d:
                    continue
                s = int(side_sig[d])
                px = ci + s * tick
                if cmd.kind == "market":
                    fill_entry(d, px, False)
                elif px <= limits[d] if s > 0 else px >= limits[d]:
                    fill_entry(d, px, True)
                else:
                    resting = True
            elif cmd.kind == "cancel_entry":
                if pending == d:
                    pending, resting = -1, False
                    if chasing:
                        chasing = False
                        pending = d
                        send("market", d)
            elif cmd.kind == "protect" and generation == d:
                protection = 2
                if pos:
                    verdict = _protect_at_close(pos, stop, target, ci, protect_flatten)
                    if verdict == REFUSE:
                        return RefColumn(trades, ERR_PROTECT_REJECTED, i, pos, entry_i)
                    if verdict == REASON_PROTECT_FAIL:
                        rejected_stop = True
                        if math.isnan(target):
                            protection = 0
                        if not closing:
                            close_reason = REASON_PROTECT_FAIL
                            send("close", d)
                        closing = True
                    if not math.isnan(target):
                        px = ci - pos * tick
                        if px >= target if pos > 0 else px <= target:
                            close(px, REASON_TARGET)
                            send("cancel_protect", d)
                            protection = 3
            elif cmd.kind == "cancel_protect" and generation == d:
                protection = 0
            elif cmd.kind == "close":
                close(ci - pos * tick, close_reason)
    return RefColumn(trades, ERR_NONE, -1, pos, entry_i, entry_px, stop)
