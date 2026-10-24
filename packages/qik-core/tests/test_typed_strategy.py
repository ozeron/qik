"""Author API contracts, compiled records and real ty diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
from enum import IntEnum
from pathlib import Path

import numba as nb
import numpy as np
import pytest

from qik.vector_engine import kernel
from qik.vector_engine.schema import (
    BoolArray,
    Features,
    FloatArray,
    IntArray,
    Params,
    State,
    strategy,
)
from qik.vector_engine.testing import SIGNAL_FIELDS
from qik.vector_engine.types import (
    HOLD,
    Bar,
    Decision,
    ExitReason,
    Position,
    Side,
    enter,
    exit,
    flatten,
    move_target,
    trail,
)


class Mode(IntEnum):
    OFF = 0
    ON = 1


class S(State):
    high: float = np.nan
    count: int = -1
    active: bool = False
    mode: Mode = Mode.OFF


class P(Params):
    k: float
    length: int = 2


class F(Features):
    atr: FloatArray
    up: BoolArray


@strategy(state=S, params=P, features=F)
def step(s: S, p: P, f: F, bar: Bar, pos: Position) -> Decision:
    s.high = f.atr[bar.i] * p.k
    return enter(1, stop=s.high)


@nb.njit(cache=True, parallel=True)
def mutate(rows):
    for j in nb.prange(len(rows)):
        i = np.int64(j)
        row = rows[i]
        row.high = i + 0.5
        row.count += 1


@nb.njit(cache=True)
def helper_probe():
    return (
        HOLD,
        enter(
            1, stop=1.0, target=4.0, stop_points=2.0, target_points=3.0, limit=2.0, entry_stop=3.0
        ),
        exit(),
        trail(3.0),
        move_target(4.0),
        flatten(),
    )


def test_schema_grid_checkpoint():
    rows = S.new(3)
    assert isinstance(rows, np.recarray)
    assert np.isnan(rows[0].high) and rows[1].count == -1
    assert rows.dtype.names == ("high", "count", "active", "mode")
    assert rows.dtype["count"] == np.int64 and rows.dtype["active"] == np.bool_
    rows[0].high = np.array([0x7FF8000000000042], dtype=np.uint64).view(np.float64)[0]
    payload = S.to_bytes(rows[0])
    assert S.to_bytes(S.from_bytes(payload)) == payload

    class Other(State):
        different: float = 0.0

    with pytest.raises(ValueError, match="fingerprint"):
        Other.from_bytes(payload)
    with pytest.raises(ValueError):
        S.from_bytes(payload[:-1])
    with pytest.raises(ValueError, match="dtype"):
        Other.to_bytes(rows[0])
    grid = P.from_grid({"length": [2, 3], "k": [1.0, 2.0]})
    assert grid.tolist() == [(1.0, 2), (1.0, 3), (2.0, 2), (2.0, 3)]
    assert P.from_grid({"k": [5.0]}).tolist() == [(5.0, 2)]
    assert len(P.from_grid({"k": []})) == 0
    with pytest.raises(ValueError, match="unknown"):
        P.from_grid({"oops": [1]})
    with pytest.raises(ValueError, match="missing"):
        P.from_grid({})


def test_invalid_schemas_and_features():
    with pytest.raises(TypeError, match="Bad.field"):

        class Bad(State):
            field: str = "bad"

    with pytest.raises(TypeError, match="Private._field"):

        class Private(State):
            _field: str = "bad"

    with pytest.raises(TypeError, match="Missing.field"):

        class Missing(State):
            field: float

    with pytest.raises(TypeError, match="BadFeatures.field"):

        class BadFeatures(Features):
            field: float

    f = F(atr=np.ones(3), up=np.ones(3, dtype=bool))
    f.validate(3)
    F.validate_values({"atr": f.atr, "up": f.up}, 3)
    with pytest.raises(ValueError, match="missing"):
        F.validate_values({"atr": f.atr}, 3)
    with pytest.raises(ValueError, match="length"):
        f.validate(2)
    with pytest.raises(TypeError, match="float64"):
        F.validate_values({"atr": np.ones(3, dtype=np.float32), "up": f.up}, 3)
    with pytest.raises(TypeError, match="State"):
        strategy(state=P, params=P, features=F)


def test_decision_order_and_reason_codes():
    assert ("entry_side", *Decision._fields[1:]) == SIGNAL_FIELDS
    assert ExitReason.NONE == -1  # compiled position adapter uses -1 before an exit
    for name in ("SIGNAL", "STOP", "TARGET", "STOP_GAP", "PROTECT_FAIL"):
        assert ExitReason[name] == getattr(kernel, "REASON_" + name)


def test_record_mutation_and_cached_step():
    rows = S.new(4)
    mutate(rows)
    np.testing.assert_array_equal(rows.high, np.arange(4) + 0.5)
    np.testing.assert_array_equal(rows.count, 0)
    spec = step.__strategy__
    assert spec.fn is step and spec.state is S and spec.features is F
    compiled = nb.njit(cache=True)(step)
    result = compiled(
        rows[0],
        P.from_grid({"k": [2.0]})[0],
        F(np.ones(1), np.ones(1, dtype=bool)),
        Bar(0, 1, 1.0, 1.0, 1.0, 1.0, 1.0),
        Position(0, 0.0, -1, np.nan, np.nan, np.nan, 0, np.nan, -1, np.nan, 0, 100.0),
    )
    assert result.stop == 2.0 and rows[0].high == 2.0


def test_helpers_python_jit_and_disabled():
    np.testing.assert_array_equal(np.asarray(helper_probe()), np.asarray(helper_probe.py_func()))
    code = """
import numpy as np
from test_typed_strategy import S, mutate, helper_probe, mixed_numeric_decision
rows = S.new(4)
mutate(rows)
np.testing.assert_array_equal(rows.high, np.arange(4) + .5)
np.testing.assert_array_equal(rows.count, 0)
for which in range(4):
    mixed_numeric_decision(which)
np.save(__import__('sys').argv[1], np.asarray(helper_probe()))
"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "disabled.npy")
        env = dict(os.environ, NUMBA_DISABLE_JIT="1")
        env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", code, target], env=env, capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        np.testing.assert_array_equal(np.load(target), np.asarray(helper_probe()))


def test_author_typing_gate(tmp_path):
    good = """from qik.vector_engine import State, Params, Features, FloatArray, strategy
from qik.vector_engine.types import Bar, Position, Decision, enter
class S(State):
    high: float = 0.0
class P(Params):
    k: float = 2.0
class F(Features):
    atr: FloatArray
@strategy(state=S, params=P, features=F)
def step(s: S, p: P, f: F, bar: Bar, pos: Position) -> Decision:
    s.high = p.k * float(f.atr[bar.i])
    return enter(1, stop=s.high)
f = F(atr=__import__('numpy').ones(3))
"""
    clean = tmp_path / "correct.py"
    clean.write_text(good)
    bad = tmp_path / "incorrect.py"
    bad.write_text(
        good
        + """
def broken(s: S, p: P, f: F) -> Decision:
    s.hihg = 3.0
    s.high = 'bad'
    p.kk
    f.atrr
    return 42
@strategy(state=S, params=P, features=F)
def wrong_return(s: S, p: P, f: F, bar: Bar, pos: Position) -> int:
    return 42
@strategy(state=S, params=P, features=F)
def wrong_args(s: str, p: P, f: F, bar: Bar, pos: Position) -> Decision:
    return enter(1)
"""
    )

    config = tmp_path / "ty.toml"
    source = Path(__file__).resolve().parents[1] / "src"
    config.write_text(f'[environment]\nextra-paths = ["{source}"]\n')

    def check(path):
        return subprocess.run(
            [
                "ty",
                "check",
                "--config-file",
                str(config),
                "--python",
                sys.prefix,
                str(path),
                "--output-format",
                "concise",
                "--color",
                "never",
            ],
            capture_output=True,
            text=True,
        )

    proc = check(clean)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = check(bad)
    assert proc.returncode == 1
    for text in ("hihg", "high", "kk", "atrr", "return", "wrong_return", "wrong_args"):
        assert text in proc.stdout, proc.stdout
    print(proc.stdout)


def test_empty_schemas_and_integer_features():
    class Empty(State):
        pass

    class EmptyParams(Params):
        pass

    class Integers(Features):
        count: IntArray

    row = Empty.new(1)[0]
    assert Empty.to_bytes(Empty.from_bytes(Empty.to_bytes(row))) == Empty.to_bytes(row)
    assert len(EmptyParams.from_grid({})) == 1
    Integers(np.arange(3, dtype=np.int64)).validate(3)
    with pytest.raises(TypeError, match="int64"):
        Integers(np.arange(3, dtype=np.int32)).validate(3)
    with pytest.raises(ValueError, match="length"):
        F(np.ones((3, 1)), np.ones(3, dtype=bool)).validate(3)
    with pytest.raises(TypeError):
        enter(1, 3.0)


@nb.njit(cache=True)
def mixed_numeric_decision(which):
    if which == 1:
        return enter(
            Side.LONG, stop=10, target=20, stop_points=2, target_points=4, limit=12, entry_stop=13
        )
    if which == 2:
        return trail(10)
    if which == 3:
        return move_target(20)
    return HOLD


def test_helpers_normalize_enum_and_integer_literals():
    for which in range(4):
        np.testing.assert_array_equal(
            np.asarray(mixed_numeric_decision(which)),
            np.asarray(mixed_numeric_decision.py_func(which)),
        )


@pytest.mark.parametrize("field", ["shape", "dtype", "new", "to_bytes", "_allocate"])
def test_reject_reserved_record_fields(field):
    with pytest.raises(TypeError, match=f"Reserved.{field}"):
        type("Reserved", (State,), {"__annotations__": {field: float}, field: 1.0})


def test_reject_unannotated_fields():
    for base in (State, Params, Features):
        with pytest.raises(TypeError, match="Missing.value.*annotation"):
            type("Missing", (base,), {"value": 1})


@pytest.mark.parametrize(
    "kind,value",
    [(int, 1.9), (int, 2**63), (bool, 2), (Mode, 3), (float, "1.5")],
)
def test_reject_lossy_defaults_and_grid_values(kind, value):
    with pytest.raises(TypeError, match="Bad.value"):
        type("Bad", (State,), {"__annotations__": {"value": kind}, "value": value})
    schema = type("Grid", (Params,), {"__annotations__": {"value": kind}})
    with pytest.raises(TypeError, match="Grid.value"):
        schema.from_grid({"value": [value]})


def test_feature_field_cannot_shadow_validation():
    with pytest.raises(TypeError, match="Bad.validate"):
        type("Bad", (Features,), {"__annotations__": {"validate": FloatArray}})
