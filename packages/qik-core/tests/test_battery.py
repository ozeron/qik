"""Failure contracts that matter independently of project callback dispatch."""

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from qik.battery import CandidatePool, NullJob, ablate, run_nulls, sweep_windows, walk_forward
from qik.falsify import random_entry_null


def test_pool_cannot_be_changed_through_inputs_or_public_arrays() -> None:
    source = np.array([1, 2, 3])
    p = CandidatePool({"entry": source}, 3)
    sha = p.sha256
    source[0] = 99
    np.testing.assert_array_equal(p.columns["entry"], [1, 2, 3])
    with pytest.raises(ValueError):
        p.columns["entry"].flags.writeable = True
    with pytest.raises(TypeError):
        p.columns["other"] = source
    assert p.sha256 == sha
    assert p.sha256 == CandidatePool({"entry": np.array([1, 2, 3])}, 3).sha256
    assert p.sha256 != CandidatePool({"entry": np.array([3, 2, 1])}, 3).sha256
    assert p.sha256 != CandidatePool({"entry": np.array([1.0, 2.0, 3.0])}, 3).sha256
    p.assert_counts(("entry",), {(1,): 1, (2,): 1, (3,): 1})
    with pytest.raises(ValueError, match="counts differ"):
        p.assert_counts(("entry",), {(1,): 1})


@pytest.mark.parametrize(
    "columns,count",
    [
        ({}, 0),
        ({"x": np.array([1])}, 2),
        ({"x": np.array([object()])}, 1),
        ({"x": np.ones((2, 2))}, 2),
    ],
)
def test_malformed_pool_refused(columns, count) -> None:
    with pytest.raises(ValueError):
        CandidatePool(columns, count)


def test_public_array_metadata_cannot_change_pool_snapshot() -> None:
    pool = CandidatePool({"entry": np.array([1, 2, 3], dtype=np.int64)}, 3)
    sha = pool.sha256
    exposed = pool.columns["entry"]
    assert isinstance(exposed.base, bytes)
    exposed.dtype = np.uint8
    exposed.shape = (3, 8)
    original = pool.columns["entry"]
    assert original.dtype == np.dtype("int64")
    assert original.shape == (3,)
    np.testing.assert_array_equal(original, [1, 2, 3])
    pool.assert_counts(("entry",), {(1,): 1, (2,): 1, (3,): 1})
    assert pool.sha256 == sha


def test_hash_and_null_seeds_replay_across_fresh_processes() -> None:
    script = """
import hashlib
import json
import sys
import numpy as np
from qik.battery import CandidatePool, NullJob, run_nulls
from qik.falsify import random_entry_null
columns = {name: np.arange(10, dtype=np.int64) for name in {'entry', 'exit'}}
pool = CandidatePool(columns, 10)
def run(seed, draws):
    return random_entry_null(
        year=np.full(10, 2020), side=np.ones(10, int),
        entry=pool.columns['entry'], exit=pool.columns['exit'],
        target_year=np.array([2020, 2020]), target_side=np.array([1, 1]),
        score=lambda ix: float(ix.sum()), observed=3.0, seed=seed, draws=draws,
    )
result = run_nulls([NullJob('test', 123, 50, run)])['test']
assert 'qik.vector_engine' not in sys.modules
print(json.dumps([pool.sha256, result.seed, result.draws, result.counts_sha256,
                  hashlib.sha256(result.stats.tobytes()).hexdigest()]))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        )
        for seed in ("1", "987")
    ]
    assert outputs[0] == outputs[1]


def test_callbacks_keep_order_and_fail_without_partial_stitch() -> None:
    called = []

    def run(i):
        called.append(i)
        if i == 2:
            raise RuntimeError("failed real run")
        return i

    with pytest.raises(RuntimeError, match="failed real run"):
        walk_forward(run, [1, 2, 3], lambda _: pytest.fail("must not stitch partial results"))
    assert called == [1, 2]
    with pytest.raises(ValueError):
        ablate(lambda *_: pytest.fail("validate before runs"), {}, ["x", "x"])
    assert sweep_windows(run, []) == ()


def null(seed: int, draws: int):
    return random_entry_null(
        year=np.full(10, 2020),
        side=np.ones(10, int),
        entry=np.arange(10),
        exit=np.arange(10),
        target_year=np.array([2020]),
        target_side=np.array([1]),
        score=lambda ix: float(ix.sum()),
        observed=3.0,
        seed=seed,
        draws=draws,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 3},
        {"draws": 3},
        {"count_assertions": 0},
        {"counts_sha256": ""},
        {"stats": np.ones(1)},
    ],
)
def test_incomplete_nulls_fail_closed(change) -> None:
    with pytest.raises(ValueError, match="receipt"):
        run_nulls([NullJob("bad", 1, 10, lambda s, n: replace(null(s, n), **change))])


def test_seed_replays_and_suite_does_not_retry_failure() -> None:
    a = run_nulls([NullJob("first", 1, 10, null)])["first"]
    b = run_nulls([NullJob("first", 1, 10, null)])["first"]
    np.testing.assert_array_equal(a.stats, b.stats)
    calls = []

    def fail(seed, draws):
        calls.append(seed)
        raise RuntimeError("EXHAUSTED")

    with pytest.raises(RuntimeError, match="EXHAUSTED"):
        run_nulls([NullJob("first", 1, 10, fail), NullJob("second", 2, 10, null)])
    assert calls == [1]
