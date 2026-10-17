"""Independent direct-formula checks for the nonannual Sharpe correction."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from qik.metrics import deflated_sharpe


@pytest.mark.parametrize("skew,kurtosis", [(0.0, 3.0), (-0.7, 5.0)])
def test_direct_psr_formula(skew: float, kurtosis: float) -> None:
    sr, variance, n, trials = 0.12, 0.0025, 600, 25
    normal = NormalDist()
    gamma = 0.5772156649015329
    benchmark = math.sqrt(variance) * (
        (1 - gamma) * normal.inv_cdf(1 - 1 / trials)
        + gamma * normal.inv_cdf(1 - 1 / (trials * math.e))
    )
    denominator = math.sqrt(1 - skew * sr + (kurtosis - 1) * sr**2 / 4)
    expected = 0.5 * math.erfc(-(sr - benchmark) * math.sqrt(n - 1) / denominator / math.sqrt(2))
    assert deflated_sharpe(
        sr,
        n_observations=n,
        n_trials=trials,
        sharpe_variance=variance,
        skew=skew,
        kurtosis=kurtosis,
    ) == pytest.approx(expected, abs=1e-14)


def test_one_trial_is_psr_against_zero() -> None:
    sr, n = 0.1, 252
    expected = NormalDist().cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr**2 / 2))
    assert deflated_sharpe(sr, n_observations=n, n_trials=1, sharpe_variance=99) == expected
    assert deflated_sharpe(0, n_observations=n, n_trials=1, sharpe_variance=0) == 0.5


def test_increasing_trials_or_variance_lowers_probability() -> None:
    scores = [
        deflated_sharpe(0.1, n_observations=252, n_trials=n, sharpe_variance=0.002)
        for n in [1, 2, 10, 100, 1000]
    ]
    assert all(a > b for a, b in zip(scores, scores[1:], strict=False))
    assert deflated_sharpe(0.1, n_observations=252, n_trials=20, sharpe_variance=0.004) < scores[2]


def test_positive_return_scaling_preserves_sharpe_moments_and_dsr() -> None:
    pnl = np.array([-2, 1, 3, -1, 4, -3, 2, 0, 1], dtype=float)
    values = []
    for scale in [0.01, 1, 100]:
        x = pnl * scale
        centered = x - x.mean()
        m2 = float(np.mean(centered**2))
        values.append(
            deflated_sharpe(
                float(x.mean() / x.std(ddof=1)),
                n_observations=len(x),
                n_trials=20,
                sharpe_variance=0.01,
                skew=float(np.mean(centered**3) / m2**1.5),
                kurtosis=float(np.mean(centered**4) / m2**2),
            )
        )
    assert values == pytest.approx([values[0]] * 3, abs=1e-14)


def test_zero_across_trial_variance_is_documented_deterministic_benchmark() -> None:
    one = deflated_sharpe(0.1, n_observations=100, n_trials=1, sharpe_variance=0)
    assert deflated_sharpe(0.1, n_observations=100, n_trials=1000, sharpe_variance=0) == one


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_observations", 3),
        ("n_observations", 1),
        ("n_observations", 10.5),
        ("n_observations", True),
        ("n_trials", 0),
        ("n_trials", -1),
        ("n_trials", 2.5),
        ("n_trials", True),
        ("sharpe", float("nan")),
        ("sharpe", float("inf")),
        ("sharpe_variance", -0.1),
        ("sharpe_variance", float("nan")),
        ("skew", float("nan")),
        ("kurtosis", float("inf")),
        ("kurtosis", 0),
    ],
)
def test_invalid_inputs_fail_explicitly(field: str, value: float) -> None:
    args = dict(sharpe=0.1, n_observations=252, n_trials=10, sharpe_variance=0.01)
    args[field] = value
    with pytest.raises(ValueError):
        deflated_sharpe(**args)


def test_nonpositive_moment_denominator_is_rejected() -> None:
    with pytest.raises(ValueError, match="moment denominator"):
        deflated_sharpe(1, n_observations=100, n_trials=1, sharpe_variance=0, skew=2, kurtosis=1)
