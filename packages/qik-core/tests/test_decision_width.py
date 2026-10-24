"""Historical tuple ABIs and defaulted constructors retain Python/JIT parity."""

from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
import pytest
from numba.experimental import jitclass

from qik.vector_engine import (
    HOLD,
    Bars,
    Decision,
    Features,
    LegacyDecision,
    Params,
    Rules,
    State,
    run,
    strategy,
)
from qik.vector_engine.compiled import _jitclass_step

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
_EXIT15 = (0, True, *_15[2:])
_EXIT16 = (*_EXIT15, np.nan)
_EXIT28 = (*_EXIT16, *(np.nan for _ in range(12)))


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
def test_jitclass_adapter_historical_tuple_width(values):
    @jitclass([])
    class OldCore:
        def __init__(self):
            pass

        def step(self, bar, pos):
            return values

    out = np.full(len(Decision._fields), -999.0)
    view = np.full(14, np.nan)
    view[[0, 2, 6, 8]] = 0
    _jitclass_step(
        OldCore(), np.empty(0), 0, np.zeros((1, 1)), 100.0, 101.0, 99.0, 100.0, 1.0, view, out
    )
    assert out.tobytes() == np.asarray(Decision(*values), float).tobytes()


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


@strategy(state=EmptyState, params=EmptyParams, features=EmptyFeatures)
def record15(s, p, f, bar, pos):
    return _15 if bar.i == 1 else _EXIT15


@strategy(state=EmptyState, params=EmptyParams, features=EmptyFeatures)
def record16(s, p, f, bar, pos):
    return _16 if bar.i == 1 else _EXIT16


@strategy(state=EmptyState, params=EmptyParams, features=EmptyFeatures)
def record28(s, p, f, bar, pos):
    return _28 if bar.i == 1 else _EXIT28


@pytest.mark.parametrize("step", [record15, record16, record28])
def test_record_tuple_width_python_equals_jit(step):
    python = run(step, _bars(), jit=False)
    compiled = run(step, _bars(), jit=True)
    pd.testing.assert_frame_equal(python.trades, compiled.trades, check_exact=True)
    assert len(compiled.trades) == 1


def _bars():
    x = np.arange(100.0, 105.0)
    return Bars(x, x + 1, x - 1, x)


def test_mixed_record_python_equals_jit():
    python = run(mixed_record, _bars(), rules=Rules(), jit=False)
    compiled = run(mixed_record, _bars(), rules=Rules(), jit=True)
    pd.testing.assert_frame_equal(python.trades, compiled.trades, check_exact=True)
    assert len(compiled.trades) == 1


@pytest.mark.parametrize("values", [_15, _16, _28, _31], ids=["15", "16", "28", "31"])
def test_legacy_core_tuple_shims(values):
    class OldCore:
        def step(self, bar):
            return values if bar.i == 1 else LegacyDecision(exit=bar.i == 3)

    class ExpectedCore:
        def step(self, bar):
            return LegacyDecision(*values) if bar.i == 1 else LegacyDecision(exit=bar.i == 3)

    actual = run(OldCore, _bars())
    expected = run(ExpectedCore, _bars())
    pd.testing.assert_frame_equal(actual.trades, expected.trades, check_exact=True)
    assert len(actual.trades) == 1


@pytest.mark.parametrize("width", [14, 17, 27, 29, 30, 32])
def test_legacy_core_refuses_unknown_tuple_width(width):
    class BadCore:
        def step(self, bar):
            return (0.0,) * width

    with pytest.raises(ValueError, match="Decision ABI requires"):
        run(BadCore, _bars())


def test_legacy_core_mixed_tuple_and_dataclass_returns():
    class MixedCore:
        def step(self, bar):
            if bar.i == 0:
                return HOLD
            if bar.i == 1:
                return _15
            if bar.i == 2:
                return _16
            return LegacyDecision(exit=True)

    actual = run(MixedCore, _bars())
    expected = run(mixed_record, _bars())
    pd.testing.assert_frame_equal(actual.trades, expected.trades, check_exact=True)
    assert len(actual.trades) == 1
