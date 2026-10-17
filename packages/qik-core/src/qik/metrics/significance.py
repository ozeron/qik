"""Sharpe selection diagnostics under independent, stationary observations/trials."""

from __future__ import annotations

import math
from numbers import Integral
from statistics import NormalDist


def deflated_sharpe(
    sharpe: float,
    *,
    n_observations: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey/Lopez de Prado PSR against the expected maximum null Sharpe.

    ``sharpe`` is mean / sample standard deviation per observation (NOT
    annualized). ``sharpe_variance`` is the explicitly supplied across-trial
    variance of Sharpes in those same nonannual units, not return variance or
    the sampling variance of the selected Sharpe. ``skew`` and ``kurtosis``
    describe the selected return/PnL distribution; kurtosis is NON-excess.
    Defaults explicitly assume Gaussian moments. At least four observations
    are required for this four-moment approximation.

    The benchmark is sqrt(sharpe_variance) times the expected maximum of N
    independent standard Normals, approximated using Euler's constant. N=1
    has benchmark zero. Zero across-trial variance is allowed: the null
    Sharpes are deterministically identical at zero, so the benchmark is
    zero for any N. A constant return series has undefined Sharpe and must
    not be encoded as finite zero; the caller must reject it upstream.

    PSR = Phi((SR - benchmark) * sqrt(T - 1) /
              sqrt(1 - skew*SR + (kurtosis - 1)*SR**2/4)).

    Stationary independent observations, finite fourth moments and independent
    trials underpin this approximation. Correlated trials and serially
    dependent returns violate those assumptions. This is not a serially
    corrected significance estimate or proof of an out-of-sample edge.
    """
    for name, value, minimum in (
        ("n_observations", n_observations, 4),
        ("n_trials", n_trials, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (
        ("sharpe", sharpe),
        ("sharpe_variance", sharpe_variance),
        ("skew", skew),
        ("kurtosis", kurtosis),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if sharpe_variance < 0:
        raise ValueError("sharpe_variance must be nonnegative")
    if kurtosis < 1:
        raise ValueError("non-excess kurtosis must be >= 1")
    denominator = 1 - skew * sharpe + (kurtosis - 1) * sharpe**2 / 4
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("PSR moment denominator must be finite and positive")
    normal = NormalDist()
    benchmark = 0.0
    if n_trials > 1 and sharpe_variance > 0:
        gamma = 0.5772156649015329
        # Symmetry avoids cancellation in 1 - 1/N for large trial counts.
        expected_max = -(1 - gamma) * normal.inv_cdf(1 / n_trials) - gamma * normal.inv_cdf(
            1 / n_trials / math.e
        )
        benchmark = math.sqrt(sharpe_variance) * expected_max
    z = (sharpe - benchmark) * math.sqrt(n_observations - 1) / math.sqrt(denominator)
    return normal.cdf(z)
