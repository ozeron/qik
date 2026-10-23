"""Failure contracts that matter independently of project callback dispatch."""

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
