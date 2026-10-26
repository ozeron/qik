"""Seeded bulk equivalents of _case's support, without Hypothesis draw overhead."""

from __future__ import annotations

from typing import Any

import numpy as np

SEED = 98241
COUNT = 1500
AXES = {
    "slots": 2,
    "next_open": 2,
    "exit_next_open": 2,
    "latency": 2,
    "one_tick": 2,
    "adverse": 2,
    "rounding": 3,
    "anchor": 3,
    "gap": 2,
}


def bulk_cases(seed: int = SEED, count: int = COUNT) -> list[dict[str, Any]]:
    """Vectorize independent draws over cases; only the 30-bar price recurrence loops.

    Preserve _case's integer endpoints, price floors, missing protections, side
    and exit probabilities. Stratify all 1152 scheduler/rule combinations once,
    then shuffle deterministically. Case indices refer to this shuffled order.
    """
    rng = np.random.default_rng(seed)
    width = 30
    lengths = rng.integers(2, width + 1, count)
    previous = rng.integers(380, 421, count)
    step = rng.integers(-8, 9, (count, width))
    gaps = rng.integers(-6, 7, (count, width)) * rng.integers(0, 2, (count, width))
    up = rng.integers(0, 9, (count, width))
    down = rng.integers(0, 9, (count, width))
    prices = {name: np.empty((count, width)) for name in ("o", "h", "lo", "c")}
    for i in range(width):
        opening = previous + gaps[:, i]
        close = np.maximum(opening + step[:, i], 4)
        prices["o"][:, i] = opening * 0.25
        prices["c"][:, i] = close * 0.25
        prices["h"][:, i] = (np.maximum(opening, close) + up[:, i]) * 0.25
        prices["lo"][:, i] = np.maximum(np.minimum(opening, close) - down[:, i], 1) * 0.25
        previous = close
    side = np.array([0, 0, 0, 1, -1], np.int8)[rng.integers(0, 5, (count, width))]
    exits = rng.integers(0, 4, (count, width)) == 3
    stop_offsets = rng.integers(-1, 9, (count, width))
    target_offsets = rng.integers(-1, 9, (count, width))
    stop = prices["c"] - side * stop_offsets * 0.25
    target = prices["c"] + side * target_offsets * 0.25
    stop[(side == 0) | (rng.integers(0, 2, (count, width)) == 0)] = np.nan
    target[(side == 0) | (rng.integers(0, 2, (count, width)) == 0)] = np.nan
    modes = rng.integers(0, 2, (count, 3)).astype(bool)
    ids = np.arange(count) % int(np.prod(list(AXES.values())))
    rng.shuffle(ids)
    flags = {}
    for name, size in AXES.items():
        flags[name] = ids % size
        ids = ids // size
    cases = []
    for j, n in enumerate(lengths):
        row = {
            name: int(values[j]) if name in ("latency", "rounding", "anchor") else bool(values[j])
            for name, values in flags.items()
        }
        row["case"] = (
            {name: values[j, :n].copy() for name, values in prices.items()},
            side[j, :n].copy(),
            exits[j, :n].copy(),
            stop[j, :n].copy(),
            target[j, :n].copy(),
            *(bool(v) for v in modes[j]),
        )
        cases.append(row)
    return cases


def coverage(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Case-class contract: every rule cell and each original tape class occurs."""
    seen = {tuple(row[name] for name in AXES) for row in cases}
    assert len(seen) == 1152, len(seen)
    lengths = {len(row["case"][1]) for row in cases}
    assert lengths == set(range(2, 31)), lengths
    sides = set(np.concatenate([row["case"][1] for row in cases]).tolist())
    assert sides == {-1, 0, 1}, sides
    for column in (3, 4):
        values = np.concatenate([row["case"][column] for row in cases])
        assert np.isnan(values).any() and np.isfinite(values).any()
    for column in (5, 6, 7):
        assert {row["case"][column] for row in cases} == {False, True}
    assert len({row["case"][5:] for row in cases}) == 8
    for column in (3, 4):
        offsets = set()
        for row in cases:
            arrays, side, _, stop, target, *_ = row["case"]
            levels = stop if column == 3 else target
            active = (side != 0) & np.isfinite(levels)
            values = (arrays["c"][active] - levels[active]) * side[active] / 0.25
            offsets.update((values if column == 3 else -values).tolist())
        assert offsets == set(range(-1, 9)), offsets
    return dict(
        cases=len(cases), rule_cells=len(seen), lengths=sorted(lengths), sides=sorted(sides)
    )
