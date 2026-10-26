"""Shared typed record strategies use the shared simple, venue and slot matchers."""

from __future__ import annotations

import numba as nb
import numpy as np
import pytest

from qik.vector_engine import Bar, Features, Params, Position, State, TypedDecision, strategy
from qik.vector_engine import kernel as K
from qik.vector_engine.api import SIGNAL_FIELDS
from qik.vector_engine.compiled import (
    RULES_WIDTH,
    _Buffers,
    _layout,
    _python_module,
    _python_runner,
)
from qik.vector_engine.fixed import runner as fixed_runner


class Calls(State):
    calls: int = 0


class Distance(Params):
    distance: float = 2.0


class NoFeatures(Features):
    pass


@strategy(state=Calls, params=Distance, features=NoFeatures)
def record_step(
    state: Calls, params: Distance, features: NoFeatures, bar: Bar, pos: Position
) -> TypedDecision:
    state.calls += 1
    return TypedDecision(
        side=1 if state.calls == 1 else 0,
        stop=bar.close - params.distance,
        target=bar.close + params.distance,
    )


@pytest.mark.parametrize("mode", ["simple", "venue", "slots"])
@pytest.mark.parametrize("rules_width", [22, 25, 26, RULES_WIDTH - 1, RULES_WIDTH, RULES_WIDTH + 1])
def test_shared_record_matches_batch(mode: str, rules_width: int):
    runner = (
        _python_runner(_python_module(record_step))
        if nb.config.DISABLE_JIT
        else fixed_runner(record_step).run
    )
    x = np.array([100.0, 101.0, 102.0, 100.0])
    states = Calls.new(2)
    params = Distance.from_grid({"distance": [2.0, 3.0]})
    rules = np.zeros(rules_width)
    rules[0] = 1
    rules[7] = int(mode == "venue")
    days = np.zeros(len(x), bool)
    everything = frozenset(range(len(SIGNAL_FIELDS)))
    layout = _layout(everything, everything, False, False)
    store, trades = _Buffers().take(2, layout.nrows, len(x), 2 * len(x))
    blank = np.full(len(x), np.nan)
    args = (
        states,
        params,
        (np.full(len(x), -1, np.int64),),
        x,
        x + 1,
        x - 1,
        x,
        np.zeros(len(x)),
        days,
        days,
        rules,
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        1000.0,
        mode == "slots",
        layout.context,
        layout.level_rows,
        store,
        trades,
        blank,
        np.full((12, 0), np.nan),
        layout.risk_row,
        None,
        np.empty((0, 2, 0), np.int8),
        np.empty(0, np.bool_),
        np.empty(0, np.bool_),
        False,
    )
    if rules_width != RULES_WIDTH:
        with pytest.raises(
            ValueError, match=f"compiled Rules ABI requires exactly {RULES_WIDTH} values"
        ):
            runner(*args)
        return
    raw = runner(*args)
    # Every field stored: the decision rows are the store rows in Signals order.
    decisions = store[:, layout.rows]
    assert decisions.shape == (2, 31, len(x))
    levels = [np.ascontiguousarray(decisions[:, q]) for q in range(2, 12)]
    batch = K.run_columns(
        x,
        x + 1,
        x - 1,
        x,
        decisions[:, 0].astype(np.int8),
        decisions[:, 1].astype(bool),
        *levels[:5],
        days,
        1,
        False,
        False,
        False,
        False,
        latency_bars=int(mode == "venue"),
        entry_trigger=levels[5] if mode == "slots" else None,
        stop_updates=levels[6] if mode == "slots" else None,
        target_updates=levels[7] if mode == "slots" else None,
        be_after=levels[8] if mode == "slots" else None,
        be_offset=levels[9] if mode == "slots" else None,
    )
    np.testing.assert_array_equal(states.calls, 0)
    np.testing.assert_array_equal(raw[0], batch[0])
    for j, count in enumerate(raw[0]):
        start, end = batch[1][j : j + 2]
        np.testing.assert_array_equal(raw[1][j, :2], [batch[2][j], batch[3][j]])
        for q in range(7):
            np.testing.assert_array_equal(trades[q][j, :count], batch[q + 8][start:end])
    if not nb.config.DISABLE_JIT:
        assert not any(
            item.library.has_dynamic_globals
            for item in fixed_runner(record_step).engine.invoke.overloads.values()
        )
