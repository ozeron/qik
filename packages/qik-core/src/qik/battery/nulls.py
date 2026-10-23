"""Count-receipted suites over existing qik.falsify nulls."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from qik.falsify import NullResult


@dataclass(frozen=True)
class NullJob:
    """Bind a qik.falsify call; callback arguments are (seed, draws).

    Use random_entry_null or matched_removal_permutation. Bind pool, target,
    overlap, score and tie policy explicitly in the callback. Legacy samplers
    with different RNG consumption remain project callbacks outside this suite.
    """

    name: str
    seed: int
    draws: int
    run: Callable[[int, int], NullResult]


def run_nulls(jobs: Iterable[NullJob]) -> dict[str, NullResult]:
    """Run each fixed-seed job once; refuse incomplete or mismatched receipts."""
    plan = tuple(jobs)
    if len({j.name for j in plan}) != len(plan):
        raise ValueError("null job names must be unique")
    if any(not j.name or j.draws < 1 or j.seed < 0 for j in plan):
        raise ValueError("nonempty name, positive draws and nonnegative seed required")
    out: dict[str, NullResult] = {}
    for job in plan:
        result = job.run(job.seed, job.draws)
        if (
            result.seed != job.seed
            or result.draws != job.draws
            or result.count_assertions != job.draws
            or result.stats.shape != (job.draws,)
            or not result.counts_sha256
        ):
            raise ValueError(f"{job.name}: incomplete or mismatched null receipt")
        out[job.name] = result
    return out
