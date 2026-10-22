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
) -> RefColumn:
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
    for i in range(n):
        exited = False
        busy = False
        book = _Book()
        close = float(c[i])
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
