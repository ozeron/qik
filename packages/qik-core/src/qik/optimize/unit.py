"""Paired unit uncertainty and predeclared A/R/E research screens.

Headlines call qik.metrics. Blocks retain all eligible sessions, including zeros.
These screens cannot supply missing G1, G4 or G5 evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qik.metrics import ann_sharpe, max_drawdown_usd
from qik.optimize.inference import (
    NORMAL,
    MaxStatistic,
    circular_indices,
    normal_posterior,
    resampled_max_test,
)

RiskEndpoint = Literal["es5", "drawdown", "parking"]
UnitRoute = Literal["A", "R", "E"]


def endpoint_max_test(
    baseline: np.ndarray,
    candidates: np.ndarray,
    *,
    endpoint: RiskEndpoint,
    block: int = 10,
    draws: int = 4_999,
    seed: int = 20261031,
) -> MaxStatistic:
    """Recompute full-family nonsmooth ES5/DD/parking endpoints per common draw.

    Unlike the smooth Sharpe IF, these statistics are recomputed on resampled
    paths. A nonsmooth bootstrap still needs population-specific calibration.
    """
    baseline, candidates = np.asarray(baseline, float), np.asarray(candidates, float)
    if baseline.ndim != 1 or candidates.ndim != 2 or len(baseline) != len(candidates):
        raise ValueError("aligned baseline and full candidate matrix required")
    if not np.isfinite(baseline).all() or not np.isfinite(candidates).all():
        raise ValueError("nonfinite endpoint input")
    if endpoint == "parking" and not (
        np.isin(baseline, [0, 1]).all() and np.isin(candidates, [0, 1]).all()
    ):
        raise ValueError("parking endpoint requires binary path indicators")

    def effect(b: np.ndarray, a: np.ndarray) -> np.ndarray:
        if endpoint == "es5":
            k = math.ceil(0.05 * len(b))
            return (
                np.partition(a, k - 1, axis=0)[:k].mean(axis=0) - np.partition(b, k - 1)[:k].mean()
            )
        if endpoint == "drawdown":
            return np.array([max_drawdown_usd(x) - max_drawdown_usd(b) for x in a.T])
        if endpoint == "parking":
            return b.mean() - a.mean(axis=0)
        raise ValueError("unknown endpoint")

    indices = circular_indices(len(baseline), block, draws, seed)
    effects = effect(baseline, candidates)
    samples = np.array([effect(baseline[i], candidates[i]) for i in indices])
    return resampled_max_test(effects, samples, block=block, seed=seed)


def loss_es5(x: np.ndarray) -> float:
    k = math.ceil(0.05 * len(x))
    return -float(np.partition(x, k - 1)[:k].mean())


def unit_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "net": float(x.sum()),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "sharpe": ann_sharpe(x),
        "worst_day": float(x.min()),
        "es5": loss_es5(x),
        "drawdown": float(max_drawdown_usd(x)),
    }


@dataclass(frozen=True)
class UnitScore:
    baseline: dict[str, float]
    candidate: dict[str, float]
    block: int
    draws: int
    seed: int
    changed_days: int
    ni_margin: float
    delta_sharpe: float
    se_sharpe: float
    ci90_sharpe: tuple[float, float]
    p_sharpe_positive: float
    p_sharpe_ni: float
    p_mean_ni: float
    p_mean_positive: float
    es5_reduction: float
    p_es5_lower: float
    dd_reduction: float
    p_dd_lower: float
    parking_reduction: float | None
    p_parking_lower: float | None
    skeptical: dict[str, float]
    mde80: float  # 90pct superiority-only planning diagnostic, excludes route point/prior/NI/G4
    pair_id: str = ""
    calendar_id: str = ""
    stress_reference_id: str = ""


def score_unit(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    block: int = 10,
    draws: int = 10_000,
    seed: int = 20261041,
    parking: tuple[np.ndarray, np.ndarray] | None = None,
) -> UnitScore:
    """Compute one paired block sensitivity. Run blocks10/20/40 before nomination.

    parking is a pair of aligned per-session (or explicit path-grid) binary
    parking indicators; it must have exactly the same resampling units. Account
    parking on a different grid belongs in book.prop.optimize, never a fake join.
    """
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if (
        a.ndim != 1
        or a.shape != b.shape
        or len(a) < 3
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("finite, aligned eligible-session pair required")
    base, pick = unit_stats(b), unit_stats(a)
    if not all(math.isfinite(x) for x in (*base.values(), *pick.values())):
        raise ValueError("undefined unit statistic")
    parked_base, parked_pick = (np.zeros(len(a)), np.zeros(len(a))) if parking is None else parking
    if parking is not None and any(
        np.shape(x) != a.shape or not np.isin(x, [0, 1]).all() for x in parking
    ):
        raise ValueError("parking must be paired aligned binary indicators")
    indices = circular_indices(len(a), block, draws, seed)
    samples = np.empty((draws, 6))
    for j, idx in enumerate(indices):
        aa, bb = a[idx], b[idx]
        samples[j, :5] = (
            ann_sharpe(aa) - ann_sharpe(bb),
            float((aa - bb).mean()),
            float(aa.mean()),
            loss_es5(aa) - loss_es5(bb),
            max_drawdown_usd(bb) - max_drawdown_usd(aa),
        )
        samples[j, 5] = (
            0 if parking is None else float((parked_pick[idx] - parked_base[idx]).mean())
        )
    if not np.isfinite(samples).all():
        raise ValueError("undefined bootstrap replicate; no dropping allowed")
    se = float(samples[:, 0].std(ddof=1))
    if se <= 1e-12:
        raise ValueError("unidentified Sharpe increment: zero paired uncertainty")
    margin = min(0.1 * base["mean"], 0.15 * base["sd"] / math.sqrt(252))
    delta = pick["sharpe"] - base["sharpe"]
    lo, hi = np.quantile(samples[:, 0], [0.05, 0.95])
    return UnitScore(
        base,
        pick,
        block,
        draws,
        seed,
        int(np.count_nonzero(np.abs(a - b) > 1e-7)),
        margin,
        delta,
        se,
        (float(lo), float(hi)),
        float((samples[:, 0] > 0).mean()),
        float((samples[:, 0] > -0.15).mean()),
        float((samples[:, 1] > -margin).mean()),
        float((samples[:, 2] > 0).mean()),
        1 - pick["es5"] / base["es5"] if base["es5"] > 0 else 0.0,
        float((samples[:, 3] < 0).mean()),
        1 - pick["drawdown"] / base["drawdown"] if base["drawdown"] < 0 else 0.0,
        float((samples[:, 4] < 0).mean()),
        None if parking is None else float((parked_base - parked_pick).mean()),
        None if parking is None else float((samples[:, 5] < 0).mean()),
        normal_posterior(delta, se, 0.20),
        (NORMAL.inv_cdf(0.9) + NORMAL.inv_cdf(0.8)) * se,
    )


def unit_screen(
    score: UnitScore,
    route: UnitRoute,
    *,
    endpoint: RiskEndpoint = "es5",
    formal: bool = False,
) -> bool:
    """G2′/G3′ numerical screen only. G1 assertions live in the decision API."""
    if route not in ("A", "R", "E") or endpoint not in ("es5", "drawdown", "parking"):
        raise ValueError("unknown route/endpoint")
    feasible_margin = score.ni_margin > 0
    if route == "A":
        return (
            feasible_margin
            and score.delta_sharpe >= 0.10
            and score.p_sharpe_positive >= 0.80
            and score.p_mean_ni >= 0.80
            and score.skeptical["p_positive"] >= 0.70
        )
    if route == "E":
        if endpoint != "es5":
            raise ValueError("E uses the predeclared ES5 repair endpoint")
        return (
            score.es5_reduction >= 0.20
            and score.p_es5_lower >= 0.90
            and score.p_mean_positive >= 0.95
        )
    point, support = {
        "es5": (score.es5_reduction, score.p_es5_lower),
        "drawdown": (score.dd_reduction, score.p_dd_lower),
        "parking": (score.parking_reduction, score.p_parking_lower),
    }[endpoint]
    if point is None or support is None:
        raise ValueError("declared parking endpoint has no paired observations")
    dd_guard = (
        endpoint == "drawdown" or score.candidate["drawdown"] >= 1.25 * score.baseline["drawdown"]
    )
    return (
        feasible_margin
        and point >= (0.05 if endpoint == "parking" else 0.20)
        and support >= 0.90
        and score.p_mean_ni >= (0.90 if formal else 0.80)
        and score.p_sharpe_ni >= 0.90
        and dd_guard
    )
