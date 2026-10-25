"""Historical tuple ABIs and defaulted constructors retain Python/JIT parity."""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
import pytest

from qik.vector_engine import (
    HOLD,
    Bars,
    Decision,
    Features,
    Params,
    Rules,
    State,
    run,
    strategy,
)
from qik.vector_engine.testing import replay_features

_15 = (
    1,
    False,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    False,
    False,
    False,
)
_16 = (*_15, np.nan)
_28 = (*_16, *(np.nan for _ in range(12)))
_31 = (*_28, False, np.nan, np.nan)


@nb.njit(cache=True)
def constructor_probe(i):
    if i == 0:
        return HOLD
    if i == 15:
        return Decision(*_15)
    if i == 16:
        return Decision(*_16)
    if i == 28:
        return Decision(*_28)
    if i == 31:
        return Decision(*_31)
    return Decision(side=1)


@pytest.mark.parametrize("width", [0, 1, 15, 16, 28, 31])
def test_defaulted_constructor_mixed_returns(width):
    expected = HOLD if width == 0 else Decision(side=1)
    actual = constructor_probe(width)
    assert type(actual) is Decision
    assert len(actual) == len(Decision._fields)
    assert np.asarray(actual, float).tobytes() == np.asarray(expected, float).tobytes()


@pytest.mark.parametrize("values", [_15, _16, _28, _31], ids=["15", "16", "28", "31"])
def test_recorded_tuple_width_takes_decision_defaults(values):
    features = replay_features([values])
    out = np.array([np.asarray(column, float)[0] for column in features])
    assert out.tobytes() == np.asarray(Decision(*values), float).tobytes()


@pytest.mark.parametrize("width", [14, 17, 27, 29, 30, 32])
def test_recorded_tuple_refuses_unknown_width(width):
    with pytest.raises(TypeError, match="15, 16, 28 or 31 fields"):
        replay_features([(0.0,) * width])


class EmptyState(State):
    pass


class EmptyParams(Params):
    pass


class EmptyFeatures(Features):
    pass


@strategy(state=EmptyState, params=EmptyParams, features=EmptyFeatures)
def mixed_record(s, p, f, bar, pos):
    if bar.i == 0:
        return HOLD
    if bar.i == 1:
        return Decision(*_15)
    if bar.i == 2:
        return Decision(*_16)
    return Decision(exit=True)


def _bars():
    x = np.arange(100.0, 105.0)
    return Bars(x, x + 1, x - 1, x)


def test_mixed_record_python_equals_jit():
    python = run(mixed_record, _bars(), rules=Rules(), jit=False)
    compiled = run(mixed_record, _bars(), rules=Rules(), jit=True)
    pd.testing.assert_frame_equal(python.trades, compiled.trades, check_exact=True)
    assert len(compiled.trades) == 1
