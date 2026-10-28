"""Typed probe for the notional-sizing tests: a scripted side/exit column that records what the
strategy is shown (``pos.qty``, ``pos.equity``), so the in-column mirror of the sizing pass can be
compared with the Result. Module-level because the compiled build imports the strategy by name."""

from __future__ import annotations

from qik.vector_engine import (
    HOLD,
    Bar,
    Decision,
    Features,
    IntArray,
    Params,
    Position,
    State,
    strategy,
)


class Seen(State):
    qty_sum: float = 0.0  # Σ pos.qty over distinct entries the strategy observed
    entries: int = 0
    last_entry_i: int = -1
    last_equity: float = 0.0


class Lane(Params):
    lane: int = 0  # which of the two entry columns this config follows


class Column(Features):
    side: IntArray
    leave: IntArray
    side2: IntArray


@strategy(state=Seen, params=Lane, features=Column)
def probe(s: Seen, p: Lane, f: Column, bar: Bar, pos: Position) -> Decision:
    s.last_equity = pos.equity
    if pos.side != 0 and pos.entry_i != s.last_entry_i:
        s.last_entry_i = pos.entry_i
        s.qty_sum += pos.qty
        s.entries += 1
    if f.leave[bar.i] != 0:
        return Decision(0, True)
    side = f.side[bar.i] if p.lane == 0 else f.side2[bar.i]
    if side != 0:
        return Decision(side)
    return HOLD
