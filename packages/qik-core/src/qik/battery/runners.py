"""Ordered callback orchestration; all experiment policy belongs to callers."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def sweep_windows[W, R](run_fn: Callable[[W], R], windows: Iterable[W]) -> tuple[R, ...]:
    """Run each explicit window once, in order; propagate every failure."""
    return tuple(run_fn(window) for window in windows)


def walk_forward[W, R, S](
    run_fn: Callable[[W], R], folds: Iterable[W], stitch: Callable[[tuple[R, ...]], S]
) -> S:
    """Run caller-owned train/test folds and apply the explicit stitching policy.

    The callback owns training, warmup, test isolation, trial accounting and
    abstention. No selector or state is reused or inferred by this dispatcher.
    """
    return stitch(sweep_windows(run_fn, folds))


def ablate[P, R](
    run_fn: Callable[[P, str | None], R], params: P, drop: Iterable[str]
) -> dict[str | None, R]:
    """Run baseline (None), then one declared rule removal per callback.

    Params are passed unchanged. The callback maps rule names to actual config
    changes or source-plan regeneration. Duplicate rule names fail before runs.
    """
    names = tuple(drop)
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("drop must contain unique nonempty rule names")
    return {name: run_fn(params, name) for name in (None, *names)}
