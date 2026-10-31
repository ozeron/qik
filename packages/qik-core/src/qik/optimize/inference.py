"""Common circular blocks, skeptical Normal diagnostic and incremental max test.

The smooth Sharpe test uses paired influence differences, preserving inherited
H0 edge. Nonsmooth endpoints use resampled_max_test on full endpoint/selector
replays. Neither adjusts unrecorded human adaptation. No absolute no-edge null
is substituted for the incremental null.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from qik.metrics import ann_sharpe

NORMAL = NormalDist()


def circular_indices(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    if n < 2 or not 1 <= block <= n or draws < 2:
        raise ValueError("need n>=2, 1<=block<=n, draws>=2")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def normal_posterior(effect: float, se: float, prior_sd: float) -> dict[str, float]:
    """Normal likelihood diagnostic, conditional on a checked sampling model."""
    if not all(math.isfinite(x) for x in (effect, se, prior_sd)) or min(se, prior_sd) <= 0:
        raise ValueError("finite effect and positive SE/prior SD required")
    variance = 1 / (1 / prior_sd**2 + 1 / se**2)
    mean = variance * effect / se**2
    sd = math.sqrt(variance)
    return {
        "prior_sd": prior_sd,
        "mean": mean,
        "sd": sd,
        "p_positive": NORMAL.cdf(mean / sd),
        "p_below_minus100": NORMAL.cdf((-100 - mean) / sd),
    }


@dataclass(frozen=True)
class MaxStatistic:
    effects: np.ndarray
    se: np.ndarray
    adjusted_p: np.ndarray
    draws: int
    block: int
    seed: int
    family_id: str = ""
    pair_ids: tuple[str, ...] = ()
    endpoint: str = ""


def resampled_max_test(
    effects: np.ndarray,
    resampled_effects: np.ndarray,
    *,
    block: int,
    seed: int,
) -> MaxStatistic:
    """Centered, studentized single-step max test over EVERY frozen alternative.

    Columns are all candidate/endpoint/market alternatives; rows are common-block
    full-procedure replicates. Effects must be oriented so positive means benefit.
    SE is the common-bootstrap SD, fixed across null replicates (single-level
    studentization). Undefined replicates refuse, never disappear from a tail.
    Exactly zero-effect, zero-variance columns are conservative p=1.
    """
    effects = np.asarray(effects, dtype=float)
    samples = np.asarray(resampled_effects, dtype=float)
    if (
        effects.ndim != 1
        or samples.ndim != 2
        or samples.shape[1] != len(effects)
        or len(effects) == 0
        or len(samples) < 2
        or not np.isfinite(samples).all()
        or not np.isfinite(effects).all()
    ):
        raise ValueError("finite full-family effects and >=2 common replicates required")
    se = samples.std(axis=0, ddof=1)
    active = se > 1e-12
    if np.any(~active & (np.abs(effects) > 1e-10)):
        raise ValueError("unidentified nonzero effect with zero bootstrap variance")
    p = np.ones(len(effects))
    if active.any():
        # Center each score at its sample estimate, imposing zero incremental
        # benefit while retaining the observed cross-candidate covariance.
        null_max = np.max((samples[:, active] - effects[active]) / se[active], axis=1)
        observed = effects[active] / se[active]
        p[active] = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (len(samples) + 1)
    return MaxStatistic(effects, se, p, len(samples), block, seed)


def _sharpe_influence(x: np.ndarray) -> np.ndarray:
    n = len(x)
    mu, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("undefined Sharpe in family")
    centered = x - mu
    # Sample-SD Sharpe functional; influence is explicitly centered below.
    return math.sqrt(252) * (
        centered / sd - mu * (centered**2 - centered.var(axis=0)) * n / (n - 1) / (2 * sd**3)
    )


def incremental_max_test(
    baseline: np.ndarray,
    candidates: np.ndarray,
    *,
    block: int = 10,
    draws: int = 4_999,
    seed: int = 20261031,
) -> MaxStatistic:
    """Paired Sharpe-increment max test; candidates shape (eligible dates, family).

    Common circular block sums avoid materializing B×dates×family arrays. A
    truncated final block preserves exactly n observations, also when n%block!=0.
    """
    baseline = np.asarray(baseline, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if (
        baseline.ndim not in (1, 2)
        or candidates.ndim != 2
        or len(baseline) != len(candidates)
        or candidates.shape[1] == 0
        or not np.isfinite(baseline).all()
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("finite aligned baseline and complete candidate matrix required")
    n = len(baseline)
    if n < 3 or not 1 <= block <= n or draws < 2:
        raise ValueError("invalid circular block design")
    if baseline.ndim == 1:
        baseline = np.repeat(baseline[:, None], candidates.shape[1], axis=1)
    if baseline.shape != candidates.shape:
        raise ValueError("baseline matrix must match candidate/date columns")
    effects = np.array(
        [ann_sharpe(x) - ann_sharpe(b) for x, b in zip(candidates.T, baseline.T, strict=True)]
    )
    influence = _sharpe_influence(candidates) - _sharpe_influence(baseline)
    influence -= influence.mean(axis=0)
    rng = np.random.default_rng(seed)
    full, remainder = divmod(n, block)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    offsets = np.arange(n)[:, None] + np.arange(block)
    sums = influence[offsets % n].sum(axis=1)
    errors = sums[starts[:, :full]].sum(axis=1)
    if remainder:
        partial = influence[(np.arange(n)[:, None] + np.arange(remainder)) % n].sum(axis=1)
        errors += partial[starts[:, -1]]
    return resampled_max_test(effects, effects + errors / n, block=block, seed=seed)
