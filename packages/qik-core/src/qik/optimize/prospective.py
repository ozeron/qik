"""Prospective v2-recal-1 cluster inference; original S2 kernels stay frozen.

Automatic blocks are chosen on the full family before selecting a winner.
Too few dependence clusters is evidence insufficiency, never iid fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, t

from qik.optimize.artifacts import SessionPanel, validate_dates
from qik.optimize.identity import array_digest
from qik.optimize.inference import _sharpe_influence
from qik.optimize.unit import loss_es5, unit_stats

POLICY = "v2-recal-1"
ALPHA = 0.10
DRAWS = 4999


class InsufficientInformation(ValueError):
    """A predeclared information boundary, not a discarded history."""

    def __init__(self, message: str, **diagnostics: object) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def dependence_time(values: np.ndarray) -> float:
    """Largest initial-positive-sequence integrated autocorrelation time.

    Pair adjacent lags (1,2), (3,4), stopping at the first nonpositive pair.
    FFT uses zero padding, so this is linear rather than circular covariance.
    """
    x = np.asarray(values, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("finite aligned family with >=8 observations required")
    x = x - x.mean(axis=0)
    n = len(x)
    spectrum = np.fft.rfft(x, n=2 ** math.ceil(math.log2(2 * n)), axis=0)
    acov = np.fft.irfft(spectrum * spectrum.conj(), axis=0)[: min(n // 4, 500) + 1]
    active = acov[0] > 1e-14
    if not active.any():
        return 1.0
    rho = acov[:, active] / acov[0, active]
    pairs = rho[1 : 1 + 2 * ((len(rho) - 1) // 2)].reshape(-1, 2, active.sum()).sum(axis=1)
    included = np.cumprod(pairs > 0, axis=0)
    return float(max(1.0, np.max(1 + 2 * (pairs * included).sum(axis=0))))


def family_blocks(baselines: np.ndarray, candidates: np.ndarray) -> tuple[int, int, int]:
    influence = _sharpe_influence(candidates) - _sharpe_influence(baselines)
    features = np.column_stack(
        (
            baselines,
            candidates,
            influence,
            candidates - baselines,
            baselines <= np.quantile(baselines, 0.05, axis=0),
            candidates <= np.quantile(candidates, 0.05, axis=0),
        )
    )
    block = max(10, math.ceil(4 * dependence_time(features)))
    return block, 2 * block, 4 * block


def cluster_slices(n: int, block: int) -> tuple[slice, ...]:
    count = n // block
    if count < 8:
        raise InsufficientInformation(
            f"only {count} dependence clusters at block {block}; need 8",
            clusters=count,
            block=block,
            sessions=n,
        )
    return tuple(slice(i * block, (i + 1) * block if i < count - 1 else n) for i in range(count))


def support(effect: float, se: float, df: int) -> float:
    if se <= 1e-12:
        return float(effect > 0)
    return float(t.cdf(effect / se, df))


def skeptical_t(effect: float, se: float, df: int, prior_sd: float) -> dict[str, float]:
    """Integrate Student-t likelihood times the declared Normal skeptical prior."""
    if se <= 1e-12 or not math.isfinite(effect + se) or df < 1:
        raise InsufficientInformation("unidentified Student-t likelihood")
    # Scale by the density at zero to avoid underflow far from the prior center.
    anchor = float(t.logpdf(-effect / se, df))

    def density(z: float) -> float:
        return float(np.exp(t.logpdf((prior_sd * z - effect) / se, df) - anchor) * norm.pdf(z))

    peak = min(11.999, max(-11.999, effect / prior_sd))
    mass = quad(density, -12, 12, points=[peak], epsabs=1e-9, limit=150)[0]
    positive = quad(density, 0, 12, points=[peak] if peak > 0 else None, epsabs=1e-9, limit=150)[0]
    loss = (
        quad(density, -12, -100 / prior_sd, epsabs=1e-9, limit=150)[0]
        if -100 / prior_sd > -12
        else 0.0
    )
    mean = quad(lambda z: prior_sd * z * density(z), -12, 12, epsabs=1e-8, limit=150)[0]
    if not math.isfinite(mass + positive + loss + mean) or mass <= 0:
        raise InsufficientInformation("undefined skeptical Student-t integration")
    return {
        "mean": mean / mass,
        "p_positive": positive / mass,
        "p_below_minus100": loss / mass,
        "prior_sd": prior_sd,
    }


@dataclass(frozen=True)
class ClusterTest:
    effects: np.ndarray
    se: np.ndarray
    adjusted_p: np.ndarray
    clusters: int
    block: int
    draws: int = DRAWS


def cluster_max_test(
    effects: np.ndarray, pseudo: np.ndarray, *, block: int, seed: int
) -> ClusterTest:
    """Common centered cluster multipliers plus a finite-cluster t union bound."""
    effects, pseudo = np.asarray(effects, float), np.asarray(pseudo, float)
    k, family = pseudo.shape
    if k < 8 or effects.shape != (family,) or not np.isfinite(pseudo).all():
        raise ValueError("finite complete family with >=8 clusters required")
    centered = pseudo - pseudo.mean(axis=0)
    se = pseudo.std(axis=0, ddof=1) / math.sqrt(k)
    active = se > 1e-12
    p = np.ones(family)
    if active.any():
        signs = np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=(DRAWS, k))
        null = (signs @ centered[:, active]) / k / se[active]
        maximum = null.max(axis=1)
        observed = effects[active] / se[active]
        empirical = (1 + (maximum[:, None] >= observed).sum(axis=0)) / (DRAWS + 1)
        p[active] = np.maximum(empirical, np.minimum(1, family * t.sf(observed, k - 1)))
    # A nonzero constant empirical effect has no estimated sampling variation.
    # It is unidentified here, not permission to assign an infinite t statistic.
    return ClusterTest(effects, se, p, k, block)


def _endpoints(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    base, pick = unit_stats(b), unit_stats(a)
    return np.array(
        [pick["sharpe"] - base["sharpe"], (a - b).mean(), a.mean(), loss_es5(b) - loss_es5(a)]
    )


def unit_family(
    baseline: np.ndarray,
    candidates: np.ndarray,
    route: Literal["A", "R", "E"],
    *,
    seed: int,
    alpha: float = ALPHA,
) -> dict[str, object]:
    """Full-family statistical envelope; raw arrays are for synthetic inference.

    Dated campaign entry points must validate their whole calendar before this
    numerical function; this cannot certify execution or native account gates.
    """
    if alpha not in (0.10, 0.05):
        raise ValueError("only predeclared structural/tightened levels are supported")
    a = np.asarray(candidates, float)
    b = np.asarray(baseline, float)
    if a.ndim != 2 or b.ndim not in (1, 2) or len(a) != len(b) or not np.isfinite(a).all():
        raise ValueError("finite aligned complete family required")
    if b.ndim == 1:
        b = np.repeat(b[:, None], a.shape[1], axis=1)
    if a.shape != b.shape or not np.isfinite(b).all() or route not in ("A", "R", "E"):
        raise ValueError("invalid baseline family or route")
    blocks = family_blocks(b, a)
    points = np.column_stack([_endpoints(bb, aa) for bb, aa in zip(b.T, a.T, strict=True)])
    primary = 0 if route == "A" else 3
    selected = int(np.argmax(points[primary]))
    bs, ps = unit_stats(b[:, selected]), unit_stats(a[:, selected])
    eligible = ps["worst_day"] > -500 and (
        route == "A" or (bs["worst_day"] > -500 if route == "R" else bs["worst_day"] <= -500)
    )
    tests, provisional, formal = [], [], []
    for block in blocks:
        slices = cluster_slices(len(a), block)
        k = len(slices)
        # Unequal final cluster uses delete-group weights, rather than pretending
        # the merged remainder contains the same number of days as a full block.
        pseudo = []
        for cluster in slices:
            keep = np.ones(len(a), dtype=bool)
            keep[cluster] = False
            deleted = np.column_stack(
                [_endpoints(bb[keep], aa[keep]) for bb, aa in zip(b.T, a.T, strict=True)]
            )
            fraction = (cluster.stop - cluster.start) / len(a)
            pseudo.append((points - (1 - fraction) * deleted) / fraction)
        # A merged remainder may be twice the regular cluster size. Inflate
        # uncertainty by the largest relative cluster mass, conservatively
        # avoiding an equal-cluster variance claim for unequal delete groups.
        pseudo_array = np.array(pseudo)
        inflation = math.sqrt(max(s.stop - s.start for s in slices) / (len(a) / k))
        pseudo_array = points + (pseudo_array - points) * inflation
        test = cluster_max_test(
            points[primary], pseudo_array[:, primary], block=block, seed=seed + block
        )
        tests.append(test)
        if not eligible:
            provisional.append(False)
            formal.append(False)
            continue
        se = pseudo_array[:, :, selected].std(axis=0, ddof=1) / math.sqrt(k)
        delta = points[:, selected]
        margin = min(0.1 * bs["mean"], 0.15 * bs["sd"] / math.sqrt(252))
        ni = support(delta[1] + margin, se[1], k - 1)
        if route == "A":
            post = skeptical_t(delta[0], se[0], k - 1, 0.20)
            passed = (
                margin > 0
                and delta[0] >= 0.10
                and support(delta[0], se[0], k - 1) >= 0.80
                and ni >= 0.80
                and post["p_positive"] >= 0.70
            )
            provisional.append(passed)
            formal.append(passed)
        else:
            risk = bs["es5"] > 0 and 1 - ps["es5"] / bs["es5"] >= 0.20
            risk = risk and support(delta[3], se[3], k - 1) >= 0.90
            if route == "E":
                passed = risk and support(delta[2], se[2], k - 1) >= 0.95
                provisional.append(passed)
                formal.append(passed)
            else:
                guard = (
                    margin > 0
                    and risk
                    and support(delta[0] + 0.15, se[0], k - 1) >= 0.90
                    and ps["drawdown"] >= 1.25 * bs["drawdown"]
                )
                provisional.append(guard and ni >= 0.80)
                formal.append(guard and ni >= 0.90)
    return {
        "policy": POLICY,
        "route": route,
        "supported": bool(all(formal) and all(t.adjusted_p[selected] <= alpha for t in tests)),
        "provisional": bool(all(provisional)),
        "g4": bool(all(t.adjusted_p[selected] <= alpha for t in tests)),
        "g4_family": bool(all(np.min(t.adjusted_p) <= alpha for t in tests)),
        "alpha": alpha,
        "inconclusive": False,
        "selected": selected,
        "blocks": blocks,
        "clusters": [t.clusters for t in tests],
        "family_size": a.shape[1],
        "p": [float(t.adjusted_p[selected]) for t in tests],
        "eligible": bool(eligible),
        "calendar_validated": False,
        "family_id": array_digest(b, a),
    }


def unit_panel_family(
    panel: SessionPanel,
    baseline: str,
    candidates: tuple[str, ...],
    route: Literal["A", "R", "E"],
    *,
    seed: int,
    alpha: float = ALPHA,
) -> dict[str, object]:
    """Campaign entry: validate the entire eligible calendar before economics."""
    dates = validate_dates([d.isoformat() for d in panel.dates])
    if len(dates) != len(panel.values) or tuple(sorted(set(dates))) != dates:
        raise ValueError("sorted unique complete eligible calendar required")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("unique complete candidate family required")
    result = unit_family(
        panel.column(baseline),
        np.column_stack([panel.column(c) for c in candidates]),
        route,
        seed=seed,
        alpha=alpha,
    )
    return {
        **result,
        "calendar_validated": True,
        "calendar_id": "|".join(d.isoformat() for d in dates),
    }
