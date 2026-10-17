"""Engine-agnostic falsification nulls shared by strategy repro and optimize runs.

The library never simulates bars. The caller supplies *candidate trades* (arrays) and a *scoring
callback* that turns a chosen subset (or a mirrored side vector) into one number, exactly the way
H0 is scored (same costs, same exits). The library only decides *which* candidates a null history
contains, asserts the count contract on every draw and reports where the observed statistic sits.

Contents
    random_entry_null            count-matched per year x side random entries (overlap or packed)
    matched_removal_permutation  random removal matched to a filter's kept count (per stratum)
    side_flip                    mirror every entry
    percentile_of                percentile of an observed value in a null, documented tie policy
    circular_shift_entries /
    random_direction /
    stationary_bootstrap         null tapes for the optimize noise canary (L13c)

Conventions
    * A *candidate pool* is a set of arrays aligned by row: ``year``, ``side`` (+1/-1), ``entry`` and
      ``exit`` (monotone ordinals such as bar or session indices; ``exit >= entry``).
    * ``score(idx)`` receives the selected pool row indices (sorted by entry) and returns a float.
    * Results hold the stat vector only (never per-draw selections).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from qik.runs import params_hash

Tie = Literal["mid", "weak", "strict"]
Score = Callable[[np.ndarray], float]

DEFAULT_DRAWS = 1000
MAX_ATTEMPTS = 200_000


# --------------------------------------------------------------------------- percentile


def percentile_of(observed: float, null: Sequence[float] | np.ndarray, tie: Tie = "mid") -> float:
    """Percentile (0-100) of ``observed`` in ``null``: the share of null draws it beats.

    Tie policy (matters for discrete statistics such as win % or trade counts):
      ``mid``    (default) ties count half:  (#below + 0.5 * #equal) / n
      ``weak``   ties count fully:           #(null <= observed) / n   (optimistic)
      ``strict`` ties count zero:            #(null <  observed) / n   (conservative)
    NaN null draws are dropped; an empty or all-NaN null raises ``ValueError``.
    """
    v = np.asarray(null, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise ValueError("empty null")
    below, equal = float(np.sum(v < observed)), float(np.sum(v == observed))
    if tie == "mid":
        return 100.0 * (below + 0.5 * equal) / v.size
    if tie == "weak":
        return 100.0 * (below + equal) / v.size
    if tie == "strict":
        return 100.0 * below / v.size
    raise ValueError(f"unknown tie policy {tie!r}")


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class NullResult:
    """Compact outcome of a null: the stat vector plus the contract evidence. No per-draw selections."""

    observed: float
    stats: np.ndarray  # one statistic per draw
    percentile: float  # percentile_of(observed, stats, tie)
    tie: str
    draws: int
    seed: int
    counts_sha256: str  # sha256 of the (stratum, count) contract asserted on every draw
    count_assertions: int  # how many draws the count contract was asserted on
    mode: str  # "overlap" | "nonoverlap" | "removal"
    attempts: int = 0  # rejection-sampling attempts (nonoverlap), 0 otherwise

    def summary(self) -> dict[str, Any]:
        """JSON-ready dict for ``summary.json`` (no arrays): percentile, quantiles, hash, seed."""
        v = self.stats[~np.isnan(self.stats)]
        d = asdict(self)
        d.pop("stats")
        d["percentile"] = round(self.percentile, 1)
        d["null_median"] = float(np.median(v))
        d["null_p5_p95"] = [float(x) for x in np.percentile(v, [5, 95])]
        return d


def _counts_hash(strata: Sequence[tuple[Any, int]]) -> str:
    return params_hash({"counts": [[str(k), int(c)] for k, c in strata]})


def _factorize(year: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    keys = sorted({(int(y), int(s)) for y, s in zip(year, side, strict=True)})
    lookup = {k: i for i, k in enumerate(keys)}
    sid = np.fromiter(
        (lookup[(int(y), int(s))] for y, s in zip(year, side, strict=True)),
        dtype=np.int64,
        count=len(year),
    )
    return sid, keys


# --------------------------------------------------------------------------- random entry


def _successors(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Per candidate (entry-sorted): first row whose entry is strictly after this one's exit."""
    return np.searchsorted(entry, exit_, side="right")


def _expected_count(nxt: np.ndarray, p: float) -> float:
    """E[#trades] of the flat-waiting Bernoulli(p) process: E[k] = p(1+E[nxt k]) + (1-p)E[k+1]."""
    n = len(nxt)
    e = np.zeros(n + 1)
    for k in range(n - 1, -1, -1):
        e[k] = p * (1.0 + e[nxt[k]]) + (1.0 - p) * e[k + 1]
    return float(e[0])


def _calibrate_p(nxt: np.ndarray, target: int) -> float:
    """Fixed trigger probability with E[N(p)] = target. Uses counts only, never PnL."""
    if _expected_count(nxt, 1.0) < target:
        raise ValueError(
            f"target {target} exceeds the {_expected_count(nxt, 1.0):.1f} trades packable"
        )
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _expected_count(nxt, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def _packed_draw(
    nxt: np.ndarray, p: float, target: int, rng: np.random.Generator, max_attempts: int
) -> tuple[np.ndarray, int]:
    """One chronological first-come history with exactly ``target`` trades (rejection on the count).

    While flat, the next trigger is a geometric wait over candidate rows, then the position is held
    through that candidate's own exit. The trigger depends on entry time only, never on duration, so
    hold lengths are sampled like the pool (random-priority packing prefers short trades and is biased).
    """
    n = len(nxt)
    for attempt in range(1, max_attempts + 1):
        picked: list[int] = []
        k = int(rng.geometric(p)) - 1
        while k < n and len(picked) <= target:
            picked.append(k)
            k = int(nxt[k]) + int(rng.geometric(p)) - 1
        if len(picked) == target:
            return np.asarray(picked, dtype=np.int64), attempt
    raise RuntimeError(f"no history with exactly {target} trades in {max_attempts} attempts")


def random_entry_null(
    *,
    year: np.ndarray,
    side: np.ndarray,
    entry: np.ndarray,
    exit: np.ndarray,
    target_year: np.ndarray,
    target_side: np.ndarray,
    score: Score,
    observed: float,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    overlap: bool = True,
    tie: Tie = "mid",
    max_attempts: int = MAX_ATTEMPTS,
) -> NullResult:
    """Random entries with EXACTLY H0's count per year x side on every draw.

    ``year/side/entry/exit`` describe the candidate pool (each row carries its own exit, so stops and
    targets drawn from same-year signals are baked in by the caller). ``target_year/target_side`` list
    H0's trades (one element per trade). ``score(idx)`` scores the selected rows, ``observed`` is H0's
    value of the same statistic.

    ``overlap=True``   uniform without-replacement subset per stratum; trades may overlap (a per-trade
                       control, not an executable history).
    ``overlap=False``  chronological first-come packing within each stratum: one position at a time,
                       trigger probability calibrated on counts only, conditioned on the exact count.
                       No duration bias (see ``_packed_draw``). Different sides pack independently.
    """
    year, side = np.asarray(year), np.asarray(side)
    entry, exit_ = np.asarray(entry), np.asarray(exit)
    if draws < 1:
        raise ValueError("draws must be >= 1")
    if not len(year) == len(side) == len(entry) == len(exit_):
        raise ValueError("pool arrays must have equal length")
    if (exit_ < entry).any():
        raise ValueError("exit must be >= entry")
    order = np.argsort(entry, kind="stable")  # per-stratum rows stay entry-sorted
    sid, keys = _factorize(year, side)
    want: dict[tuple[int, int], int] = {}
    for y, s in zip(np.asarray(target_year), np.asarray(target_side), strict=True):
        want[(int(y), int(s))] = want.get((int(y), int(s)), 0) + 1
    unknown = set(want) - set(keys)
    if unknown:
        raise ValueError(f"no candidates for strata {sorted(unknown)}")
    strata = sorted(want.items())
    target_vec = np.zeros(len(keys), dtype=np.int64)
    rows: list[np.ndarray] = []
    plans: list[tuple[int, int, np.ndarray, np.ndarray | None, float]] = []
    for key, c in strata:
        i = keys.index(key)
        target_vec[i] = c
        r = order[sid[order] == i]
        rows.append(r)
        if overlap:
            if len(r) < c:
                raise ValueError(f"stratum {key}: {len(r)} candidates < {c} H0 trades")
            plans.append((i, c, r, None, 0.0))
        else:
            nxt = _successors(entry[r], exit_[r])
            plans.append((i, c, r, nxt, _calibrate_p(nxt, c)))
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    attempts = 0
    for d in range(draws):
        parts: list[np.ndarray] = []
        for _, c, r, nxt, p in plans:
            if nxt is None:
                parts.append(rng.choice(r, size=c, replace=False))
            else:
                pick, a = _packed_draw(nxt, p, c, rng, max_attempts)
                attempts += a
                parts.append(r[pick])
        idx = np.concatenate(parts)
        idx = idx[np.argsort(entry[idx], kind="stable")]
        got = np.bincount(sid[idx], minlength=len(keys))
        assert (got == target_vec).all(), f"draw {d}: year x side counts differ from H0"
        out[d] = score(idx)
    return NullResult(
        observed=float(observed),
        stats=out,
        percentile=percentile_of(observed, out, tie),
        tie=tie,
        draws=draws,
        seed=seed,
        counts_sha256=_counts_hash(strata),
        count_assertions=draws,
        mode="overlap" if overlap else "nonoverlap",
        attempts=attempts,
    )


# --------------------------------------------------------------------------- matched removal


def matched_removal_permutation(
    *,
    kept: np.ndarray,
    score: Score,
    strata: np.ndarray | None = None,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    tie: Tie = "mid",
) -> NullResult:
    """Does a filter / selection rule beat removing the same number of trades at random?

    ``kept`` is a boolean mask over the PARENT trades (True = the filter keeps it). Each draw keeps a
    random subset with exactly the same kept count per ``strata`` label (default: one stratum, e.g.
    pass year x side labels to match those too). ``score(idx)`` scores kept rows (sorted indices);
    the observed value is ``score(flatnonzero(kept))``.
    """
    kept = np.asarray(kept, dtype=bool)
    lab = np.zeros(len(kept), dtype=np.int64) if strata is None else _labels(strata, len(kept))
    groups = [(g, np.flatnonzero(lab == g)) for g in np.unique(lab)]
    need = {int(g): int(kept[r].sum()) for g, r in groups}
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        idx = np.sort(
            np.concatenate([rng.choice(r, size=need[int(g)], replace=False) for g, r in groups])
        )
        assert all(int((lab[idx] == g).sum()) == need[int(g)] for g, _ in groups), (
            f"draw {d}: kept counts differ"
        )
        out[d] = score(idx)
    obs = float(score(np.flatnonzero(kept)))
    return NullResult(
        observed=obs,
        stats=out,
        percentile=percentile_of(obs, out, tie),
        tie=tie,
        draws=draws,
        seed=seed,
        counts_sha256=_counts_hash(sorted(need.items())),
        count_assertions=draws,
        mode="removal",
    )


def _labels(strata: np.ndarray, n: int) -> np.ndarray:
    s = np.asarray(strata)
    if len(s) != n:
        raise ValueError("strata must align with kept")
    return np.unique(s, return_inverse=True)[1].astype(np.int64)


# --------------------------------------------------------------------------- side flip


@dataclass(frozen=True)
class SideFlipResult:
    observed: float
    flipped: float

    @property
    def delta(self) -> float:
        return self.observed - self.flipped


def side_flip(score_sides: Callable[[np.ndarray], float], side: np.ndarray) -> SideFlipResult:
    """Score H0's entries as given and mirrored (every side multiplied by -1; costs unchanged).

    ``score_sides(side)`` must score the same entries and exits with the supplied side vector.
    """
    s = np.asarray(side)
    return SideFlipResult(observed=float(score_sides(s)), flipped=float(score_sides(-s)))


# --------------------------------------------------------------------------- null tapes (L13c)


def circular_shift_entries(
    entry_session: np.ndarray, session_year: np.ndarray, *, n_tapes: int, seed: int = 0
) -> Iterator[np.ndarray]:
    """Yield ``n_tapes`` arrays of entry sessions, each year's entries shifted circularly.

    ``session_year[k]`` is the year of session ``k`` (sessions sorted). Within each year, all entries
    move by one random offset (>= 1 when the year has more than one session) modulo the year's session
    count, which keeps per-year counts, the spacing of entries and the market intact while breaking
    the link between signal and outcome.
    """
    e = np.asarray(entry_session, dtype=np.int64)
    yrs = np.asarray(session_year)
    ey = yrs[e]
    spans = {int(y): np.flatnonzero(yrs == y) for y in np.unique(ey)}
    rng = np.random.default_rng(seed)
    for _ in range(n_tapes):
        out = e.copy()
        for y, sess in spans.items():
            m = ey == y
            lo, n = int(sess[0]), len(sess)
            shift = int(rng.integers(1, n)) if n > 1 else 0
            out[m] = lo + (e[m] - lo + shift) % n
        yield out


def random_direction(side: np.ndarray, *, n_tapes: int, seed: int = 0) -> Iterator[np.ndarray]:
    """Yield ``n_tapes`` +1/-1 vectors: same timestamps, each trade's direction an independent coin."""
    n = len(np.asarray(side))
    rng = np.random.default_rng(seed)
    for _ in range(n_tapes):
        yield rng.choice(np.array([-1, 1]), size=n)


def stationary_bootstrap(
    returns: np.ndarray,
    *,
    n_tapes: int,
    seed: int = 0,
    mean_block: float | None = None,
) -> Iterator[np.ndarray]:
    """Politis-Romano stationary block bootstrap of a (daily) return series.

    Each tape has the same length as ``returns``. Starting from a uniform random index, every next
    position continues the block (index + 1, circular) with probability 1 - 1/L, otherwise jumps to a
    new uniform index; L = ``mean_block`` (default ``ceil(n ** (1/3))``). Keeps short-range dependence,
    destroys the signal's alignment with the market.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        raise ValueError("empty returns")
    block = float(math.ceil(n ** (1 / 3))) if mean_block is None else float(mean_block)
    if block < 1:
        raise ValueError("mean_block must be >= 1")
    rng = np.random.default_rng(seed)
    pos = np.arange(n)
    for _ in range(n_tapes):
        restart = rng.random(n) < 1.0 / block
        restart[0] = True
        base = rng.integers(0, n, size=n) - pos  # index = base + pos at a restart
        last = np.maximum.accumulate(np.where(restart, pos, 0))
        yield r[(base[last] + pos) % n]
