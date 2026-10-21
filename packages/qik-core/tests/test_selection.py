"""Unit checks for qik.selection (no project imports; project equivalence is in repo-root tests/)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qik.metrics import DEFAULT_PERIODS_PER_YEAR, deflated_sharpe
from qik.selection import (
    NEffParticipation,
    deflated_sharpe_from_returns,
    dsr_inputs,
    n_eff_mean_corr,
    n_eff_participation,
    pbo_summary,
    trial_sharpe_variance,
)


def _returns(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, n))


def test_dsr_inputs_moments() -> None:
    r = _returns()
    got = dsr_inputs(r, constant="raise")
    x = r.to_numpy()
    z = (x - x.mean()) / x.std(ddof=0)
    assert got.sr == float(x.mean() / x.std(ddof=1))
    assert got.n_obs == len(x)
    assert got.skew == float((z**3).mean())
    assert got.kurt == float((z**4).mean())


def test_dsr_inputs_drops_nan() -> None:
    r = _returns()
    r.iloc[5] = np.nan
    assert dsr_inputs(r, constant="raise").n_obs == len(r) - 1


@pytest.mark.parametrize("bad", [np.zeros(20), np.array([0.01]), np.array([])])
def test_dsr_inputs_constant_policy(bad: np.ndarray) -> None:
    assert dsr_inputs(bad, constant="none") is None
    with pytest.raises(ValueError, match="undefined"):
        dsr_inputs(bad, constant="raise")
    with pytest.raises(ValueError, match="constant must be"):
        dsr_inputs(bad, constant="skip")  # type: ignore[call-overload]


def test_trial_sharpe_variance_units_and_policy() -> None:
    ts = [1.0, 2.0, 4.0]
    assert trial_sharpe_variance(ts, periods_per_year=1.0, nonfinite="raise") == float(
        np.var(ts, ddof=1)
    )
    assert (
        trial_sharpe_variance(ts, periods_per_year=252.0, nonfinite="raise")
        == float(np.var(ts, ddof=1)) / 252.0
    )
    with_nan = [1.0, math.nan, 2.0, 4.0, math.inf]
    assert trial_sharpe_variance(with_nan, periods_per_year=1.0, nonfinite="drop") == float(
        np.var(ts, ddof=1)
    )
    with pytest.raises(ValueError, match=">= 2 finite"):
        trial_sharpe_variance(with_nan, periods_per_year=1.0, nonfinite="raise")
    with pytest.raises(ValueError, match=">= 2 finite"):
        trial_sharpe_variance([1.0, math.nan], periods_per_year=1.0, nonfinite="drop")
    with pytest.raises(ValueError, match="periods_per_year"):
        trial_sharpe_variance(ts, periods_per_year=0.0, nonfinite="raise")


def test_deflated_sharpe_from_returns_calls_qik_formula() -> None:
    r = _returns()
    ts = np.array([0.5, 1.1, 1.9, -0.3, 0.8])
    inp = dsr_inputs(r, constant="raise")
    expected = deflated_sharpe(
        inp.sr,
        n_observations=inp.n_obs,
        n_trials=40,
        sharpe_variance=float(ts.var(ddof=1)) / DEFAULT_PERIODS_PER_YEAR,
        skew=inp.skew,
        kurtosis=inp.kurt,
    )
    assert deflated_sharpe_from_returns(r, 40, trial_sharpes=ts) == expected


def test_clamp_n() -> None:
    r = _returns()
    ts = [0.5, 1.1, 1.9]
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_from_returns(r, 0, trial_sharpes=ts)
    assert deflated_sharpe_from_returns(r, 0, trial_sharpes=ts, clamp_n=True) == (
        deflated_sharpe_from_returns(r, 1, trial_sharpes=ts)
    )


def test_deflated_sharpe_from_returns_rejects_constant() -> None:
    with pytest.raises(ValueError, match="undefined"):
        deflated_sharpe_from_returns(np.zeros(50), 10, trial_sharpes=[1.0, 2.0])


def _pnl_matrix(tz: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-03", periods=96, freq="B", tz=tz)
    m = pd.DataFrame(rng.normal(0, 1, (96, 5)), index=idx, columns=list("abcde"))
    m["flat"] = 0.0
    return m


def test_pbo_summary_drops_constant_and_names_it() -> None:
    s = pbo_summary(_pnl_matrix(), n_blocks=8, localize_tz=True)
    assert s.variants == 5
    assert s.omitted_constant_columns == 1
    assert s.omitted_names == ("flat",)
    assert s.combinations == 70
    assert 0.0 <= s.pbo <= 1.0
    assert "omitted_names" not in s.as_dict()
    assert s.as_dict(include_names=True)["omitted_names"] == ["flat"]
    assert pbo_summary(_pnl_matrix(), n_blocks=8, localize_tz=False) == s


def test_pbo_summary_tz_and_index_contract() -> None:
    tz = _pnl_matrix("America/New_York")
    stripped = pbo_summary(tz, n_blocks=8, localize_tz=True)
    assert stripped == pbo_summary(_pnl_matrix(), n_blocks=8, localize_tz=True)
    with pytest.raises(ValueError, match="timezone-naive"):
        pbo_summary(tz, n_blocks=8, localize_tz=False)
    plain = _pnl_matrix()
    plain.index = list(plain.index.strftime("%Y-%m-%d"))
    with pytest.raises(TypeError, match="date-indexed"):
        pbo_summary(plain, n_blocks=8, localize_tz=True)
    assert pbo_summary(plain, n_blocks=8, localize_tz=False) == stripped


def _paths(seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, (400, 6))


def _pr(x: object, **kw: object) -> NEffParticipation:
    """``n_eff_participation`` with the drop policies spelled out (they are required keywords)."""
    kw = {"constant": "drop", "nonfinite": "drop"} | kw
    return n_eff_participation(x, **kw)  # type: ignore[arg-type]


def test_participation_extremes() -> None:
    one = np.random.default_rng(0).normal(0, 1, 200)
    same = np.column_stack([one, one * 2.0, one + 5.0])
    assert _pr(same).n_eff == pytest.approx(1.0, abs=1e-9)
    big = np.random.default_rng(1).normal(0, 1, (20000, 4))
    assert _pr(big).n_eff == pytest.approx(4.0, rel=2e-2)


def test_participation_constant_and_nonfinite_columns() -> None:
    x = _paths()
    base = _pr(x)
    withbad = np.column_stack([x, np.zeros(len(x)), np.full(len(x), np.nan)])
    got = _pr(withbad)
    assert got.n_eff == base.n_eff
    assert (got.n_columns, got.n_dropped) == (6, 2)
    assert len(got.eigenvalues) == 6
    assert list(got.eigenvalues) == sorted(got.eigenvalues, reverse=True)
    with pytest.raises(ValueError, match="constant"):
        _pr(withbad, constant="raise")


def test_participation_keep_nonfinite_needs_nan_corr_zero() -> None:
    x = _paths()
    x[3, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        _pr(x, nonfinite="keep")
    got = _pr(x, nonfinite="keep", nan_corr="zero")
    assert got.n_columns == 6
    assert got.n_dropped == 0


def test_participation_min_columns() -> None:
    x = _paths()[:, :1]
    with pytest.raises(ValueError, match="at least 2"):
        _pr(x)
    assert _pr(x, min_columns=1).n_eff == 1.0
    z = np.zeros((10, 3))
    assert _pr(z, min_columns=0).n_eff == 0.0
    with pytest.raises(ValueError, match="2-D"):
        _pr(np.zeros(5))


def test_participation_policies_are_required() -> None:
    """Dropping NaN/inf paths lowers N_eff (laxer DSR): the caller must say so."""
    x = _paths()
    with pytest.raises(TypeError):
        n_eff_participation(x)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        n_eff_participation(x, constant="drop")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        n_eff_participation(x, nonfinite="drop")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="nonfinite must be"):
        n_eff_participation(x, constant="drop", nonfinite="skip")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="constant must be"):
        n_eff_participation(x, constant="skip", nonfinite="drop")  # type: ignore[arg-type]
    # the silent-drop reproducer: a NaN-poisoned path set reads lower than the clean one under 'drop'
    p = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 10)))
    q = p.copy()
    q.iloc[3, 4:] = np.nan
    assert _pr(p).n_eff > _pr(q).n_eff
    assert _pr(q).n_dropped == 6


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_participation_needs_three_days(rows: int) -> None:
    x = np.random.default_rng(0).normal(size=(rows, 4))
    with pytest.raises(ValueError, match=">= 3 days"):
        _pr(x)
    with pytest.raises(ValueError, match=">= 3 days"):
        _pr(x, min_columns=0)
    assert _pr(np.random.default_rng(0).normal(size=(3, 4))).n_columns == 4


def test_dsr_inputs_infinite_entry_message() -> None:
    r = _returns()
    r.iloc[4] = np.inf
    with pytest.raises(ValueError, match="infinite entry"):
        dsr_inputs(r, constant="raise")
    assert dsr_inputs(r, constant="none") is None
    with pytest.raises(ValueError, match="zero sample SD"):
        dsr_inputs(np.zeros(20), constant="raise")
    with pytest.raises(ValueError, match="fewer than 2"):
        dsr_inputs(np.array([0.01]), constant="raise")


def test_mean_corr_is_a_different_statistic() -> None:
    assert n_eff_mean_corr(1.0, 50) == 1
    assert n_eff_mean_corr(0.0, 50) == 50
    assert n_eff_mean_corr(0.5, 50) == 26  # ceil(25.5)
    assert n_eff_mean_corr(0.9, 10) == 2  # 0.9 + 0.1*10 = 1.9 (+ float noise) -> 2
    with pytest.raises(ValueError, match="finite"):
        n_eff_mean_corr(math.nan, 10)
    x = _paths()
    rho = (np.corrcoef(x, rowvar=False).sum() - 6) / 30
    assert _pr(x).n_eff != float(n_eff_mean_corr(rho, 6))
