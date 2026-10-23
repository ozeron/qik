"""Fully bound record strategies use the shared simple, venue and slot matchers."""

from __future__ import annotations

import importlib
import sys

import numba as nb
import numpy as np
import pytest

from qik.vector_engine import kernel as K
from qik.vector_engine.bound import write_bound_module


@nb.njit(cache=True)
def record_step(state, params, features, bar, pos):
    state.calls += 1
    return (
        1 if state.calls == 1 else 0,
        0,
        bar.close - params[0],
        bar.close + params[0],
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0,
        0,
        0,
    )


@pytest.fixture(scope="module")
def bound_runner(tmp_path_factory):
    directory = tmp_path_factory.mktemp("bound_record")
    sys.path.insert(0, str(directory))
    name = "bound_record_test"
    write_bound_module(directory / f"{name}.py", __name__, "record_step")
    try:
        yield importlib.import_module(name)
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop(name, None)


@pytest.mark.parametrize("mode", ["simple", "venue", "slots"])
def test_bound_record_matches_batch(bound_runner, mode: str):
    module = bound_runner
    x = np.array([100.0, 101.0, 102.0, 100.0])
    states = np.zeros(2, dtype=[("calls", "i8")]).view(np.recarray)
    params = np.array([[2.0], [3.0]])
    rules = np.zeros(22)
    rules[0] = 1
    rules[7] = int(mode == "venue")
    days = np.zeros(len(x), bool)
    raw = module.run(
        states,
        params,
        np.zeros((len(x), 1)),
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
    )
    decisions = raw[-1]
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
            np.testing.assert_array_equal(raw[q + 2][j, :count], batch[q + 8][start:end])
    assert not any(item.library.has_dynamic_globals for item in module.run.overloads.values())
