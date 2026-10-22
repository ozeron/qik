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
    ERR_STALE_BOOK,
    ERR_STOP_IN_MARKET,
    ERR_TARGET_IN_MARKET,
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


@dataclass
class RefColumn:
    trades: list[RefTrade]
    err: int
    err_bar: int
    final_side: int
    final_entry_i: int


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
) -> RefColumn:
    trades: list[RefTrade] = []
    pos = 0
    entry_i = -1
    entry_px = 0.0
    stop = math.nan
    target = math.nan
    for i in range(len(o)):
        exited = False
        book = _Book()
        if pos != 0:
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
                trades.append(RefTrade(entry_i, i, pos, entry_px, fill[1], fill[0]))
                pos = 0
                exited = True
        if pos != 0 and bool(exit_sig[i]):
            trades.append(RefTrade(entry_i, i, pos, entry_px, float(c[i]), REASON_SIGNAL))
            pos = 0
            exited = True
        if pos == 0 and int(side_sig[i]) != 0 and (reenter or not exited):
            s = int(side_sig[i])
            st, tg = float(entry_stop[i]), float(entry_target[i])
            close = float(c[i])
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
    return RefColumn(trades, ERR_NONE, -1, pos, entry_i)
