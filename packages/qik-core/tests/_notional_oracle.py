"""Independent oracle for stop-free notional sizing and the entry-notional fee basis (A24).

A bar-by-bar cash/position ledger in exact rational arithmetic. It shares nothing with the engine:
no import from ``qik.vector_engine`` (a test asserts that on the AST), prices and money are the
decimals the author wrote (``Fraction(Decimal(repr(x)))``, never their binary neighbours), floors are
exact ``//`` with no epsilon, and fills, fees, cash and the terminal mark are produced here from raw
OHLC rows and a declarative scenario. The engine instead sizes a finished trade list with one
vectorized float64 rule, so the two agree only if the rule is the same.

Scenario language (everything the fixtures need, nothing the engine's matcher decides for us):

* entry on ``entry_bar`` filling at that bar's close (``"close"``) or the next bar's open
  (``"next_open"``);
* leaving by a signal exit at ``exit_bar``'s close, by a protective ``stop`` level (the first bar
  after the entry fill whose range touches it fills at the level, or at the open when the bar opens
  through it), or not at all (position open at the last bar, marked at the last close).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Literal


def q(x: float | int | str | Fraction | Decimal) -> Fraction:
    """The decimal the author wrote, exactly."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Decimal):
        return Fraction(x)
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(Decimal(repr(float(x)) if not isinstance(x, str) else x))


@dataclass(frozen=True)
class Rule:
    """The economics under test, as plain numbers."""

    notional: float
    point_value: float = 1.0
    tick: float = 0.01
    lot_step: float = 1.0
    min_qty: float = 1.0
    cap: float | None = None
    commission_per_side: float = 0.0
    slippage_ticks: float = 0.0
    bps: float = 0.0
    fee_basis: Literal["fill_notional", "entry_notional"] = "fill_notional"
    cash: float | None = None
    affordability: Literal["refuse", "unchecked"] = "unchecked"


@dataclass(frozen=True)
class Attempt:
    """One position attempt on a tape."""

    entry_bar: int
    side: int = 1
    fill: Literal["close", "next_open"] = "close"
    exit_bar: int | None = None  # signal exit at that bar's close
    stop: float | None = None  # protective level; fills at the level, or the open when gapped


@dataclass
class Fill:
    bar: int
    price: Fraction
    qty: Fraction
    fee: Fraction


@dataclass
class Closed:
    attempt: Attempt
    qty: Fraction
    entry: Fill
    exit: Fill
    gross: Fraction
    fees: Fraction
    net: Fraction
    equity_after: Fraction | None


@dataclass
class Open:
    attempt: Attempt
    qty: Fraction
    entry: Fill
    mark_price: Fraction
    terminal_mark: Fraction  # side*qty*(last close - entry)*pv - entry fill's fees


@dataclass
class Refused:
    attempt: Attempt
    why: Literal["zero", "unaffordable"]
    detail: str = ""


@dataclass
class Ledger:
    closed: list[Closed] = field(default_factory=list)
    open: Open | None = None
    refused: Refused | None = None
    cash_events: list[tuple[int, str, Fraction]] = field(
        default_factory=list
    )  # (bar, what, balance)


def _floor_units(n: Fraction, step: Fraction) -> Fraction:
    return (n // step) * step


def quantity(rule: Rule, price: Fraction, equity: Fraction | None) -> Fraction | str:
    """Exact quantity for one entry fill, or ``"zero"`` / ``"unaffordable"``."""
    pv = q(rule.point_value)
    unit = abs(price) * pv
    step = q(rule.lot_step)
    n = _floor_units(q(rule.notional) / unit, step)
    if rule.cap is not None and n > q(rule.cap):
        n = q(rule.cap)
    if n < q(rule.min_qty):
        return "zero"
    if rule.affordability == "refuse":
        assert equity is not None, "affordability needs a balance"
        if n * (unit + _entry_costs(rule, price)) > equity:
            return "unaffordable"
    return n


def _entry_costs(rule: Rule, price: Fraction) -> Fraction:
    """One unit's costs at the entry fill (fixed side cost and its proportional fee)."""
    pv = q(rule.point_value)
    fixed = q(rule.commission_per_side) + q(rule.slippage_ticks) * q(rule.tick) * pv
    return fixed + abs(price) * pv * q(rule.bps) / 10000


def _exit_costs(rule: Rule, entry: Fraction, exit_: Fraction) -> Fraction:
    pv = q(rule.point_value)
    fixed = q(rule.commission_per_side) + q(rule.slippage_ticks) * q(rule.tick) * pv
    basis = entry if rule.fee_basis == "entry_notional" else exit_
    return fixed + abs(basis) * pv * q(rule.bps) / 10000


def _fills(
    rows: list[tuple[float, float, float, float]], a: Attempt
) -> tuple[int, Fraction, tuple[int, Fraction] | None]:
    """Entry bar index of the fill and price, and the exit (bar, price) if the attempt leaves."""
    opens = [q(r[0]) for r in rows]
    highs = [q(r[1]) for r in rows]
    lows = [q(r[2]) for r in rows]
    closes = [q(r[3]) for r in rows]
    if a.fill == "close":
        fb, price = a.entry_bar, closes[a.entry_bar]
    else:
        fb = a.entry_bar + 1
        price = opens[fb]
    leave: tuple[int, Fraction] | None = None
    if a.stop is not None:
        level = q(a.stop)
        for b in range(fb + 1, len(rows)):
            if a.side > 0 and lows[b] <= level:
                leave = (b, opens[b] if opens[b] <= level else level)
                break
            if a.side < 0 and highs[b] >= level:
                leave = (b, opens[b] if opens[b] >= level else level)
                break
    if a.exit_bar is not None and (leave is None or a.exit_bar < leave[0]):
        leave = (a.exit_bar, closes[a.exit_bar])
    return fb, price, leave


def run(
    rows: list[tuple[float, float, float, float]],
    attempts: list[Attempt],
    rule: Rule,
) -> Ledger:
    """Walk the attempts in order against one book (one symbol, one config)."""
    ledger = Ledger()
    equity = q(rule.cash) if rule.cash is not None else None
    pv = q(rule.point_value)
    free_from = 0  # bar from which a new entry may be decided
    for a in attempts:
        if a.entry_bar < free_from:
            continue  # still in a position: the signal is not an entry
        fb, price, leave = _fills(rows, a)
        n = quantity(rule, price, equity)
        if isinstance(n, str):
            ledger.refused = Refused(a, n)  # type: ignore[arg-type]
            return ledger
        entry_fee = n * _entry_costs(rule, price)
        entry = Fill(fb, price, n, entry_fee)
        if leave is None:
            mark = q(rows[-1][3])
            ledger.open = Open(a, n, entry, mark, a.side * n * (mark - price) * pv - entry_fee)
            return ledger
        xb, xp = leave
        exit_fee = n * _exit_costs(rule, price, xp)
        gross = a.side * n * (xp - price) * pv
        net = gross - entry_fee - exit_fee
        if equity is not None:
            equity += net
        ledger.closed.append(
            Closed(a, n, entry, Fill(xb, xp, n, exit_fee), gross, entry_fee + exit_fee, net, equity)
        )
        free_from = xb + 1  # fixtures never re-enter on the exit bar's own close
    return ledger
