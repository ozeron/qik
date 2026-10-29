"""Importable strategy fixtures for the independent quantity oracle."""

from __future__ import annotations

from qik.vector_engine import (
    HOLD,
    Bar,
    Decision,
    Features,
    FloatArray,
    Params,
    Position,
    State,
    strategy,
)


class Seen(State):
    qty: float = 0.0
    equity: float = 0.0
    observed: float = 0.0


class Order(Params):
    kind: int = 0
    level: float = 90.0
    decision: int = 0
    leave: int = 5


class Lanes(Features):
    wanted: FloatArray
    other: FloatArray


@strategy(state=Seen, params=Order, features=Lanes)
def probe(s: Seen, p: Order, f: Lanes, bar: Bar, pos: Position) -> Decision:
    if pos.side:
        s.qty = pos.qty
    s.equity = pos.equity
    s.observed = f.wanted[bar.i]
    if bar.i == p.decision:
        if p.kind == 1:
            return Decision(side=1, entry_limit=p.level, stop_points=50.0)
        if p.kind == 4:
            return Decision(side=1, entry_limit=80.0, stop_points=50.0)
        if p.kind == 2:
            return Decision(side=1, entry_stop=p.level)
        if p.kind == 5:
            return Decision(side=1, entry_stop=108.0, stop_points=50.0, target_points=0.5)
        if p.kind == 3:
            return Decision(entry_stop_long=p.level, entry_stop_short=p.level - 30)
        return Decision(side=1)
    if p.kind == 5 and bar.i == 1:
        return Decision(side=1, stop_points=50.0)
    if p.kind == 4 and bar.i == 2:
        return Decision(side=1, exit=True, entry_limit=90.0, stop_points=50.0)
    if bar.i == p.leave:
        return Decision(exit=True)
    return HOLD


@strategy(state=Seen, params=Order, features=Lanes)
def mutate(s: Seen, p: Order, f: Lanes, bar: Bar, pos: Position) -> Decision:
    if pos.side:
        s.qty = pos.qty
    if bar.i == 0:
        f.wanted[0] = 100.0
        f.wanted[1] = 200.0
        return Decision(side=1)
    if bar.i == 2:
        return Decision(exit=True)
    return HOLD
