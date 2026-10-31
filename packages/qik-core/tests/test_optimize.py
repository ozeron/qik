"""Contract tests: pairing, inherited edge, complete-family correction and safe IO."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qik.metrics import ann_sharpe
from qik.optimize import (
    circular_indices,
    incremental_max_test,
    load_sessions,
    normal_posterior,
    resampled_max_test,
    score_unit,
    unit_screen,
    validate_dates,
)


def test_loader_checks_all_dates_before_parsing_any_economics(tmp_path):
    path = tmp_path / "pair.csv"
    path.write_text("session_date,H0,pick\n2023-12-29,not-a-number,0\n2024-01-01,0,0\n")
    with pytest.raises(ValueError, match="sealed"):
        load_sessions(path)
    path.write_text("session_date,H0,pick\n2023-12-29,0,1\n2024-01-01,0,2\n")
    assert len(load_sessions(path, unsealed=True).dates) == 2
    with pytest.raises(ValueError):
        validate_dates(["20231229"])


def test_loader_does_not_drop_zero_days_or_duplicate_dates(tmp_path):
    path = tmp_path / "pair.csv"
    path.write_text("session_date,H0,pick\n2023-01-03,0,0\n2023-01-04,-1,2\n")
    assert load_sessions(path).column("H0").tolist() == [0, -1]
    path.write_text("session_date,H0,pick\n2023-01-03,0,0\n2023-01-03,-1,2\n")
    with pytest.raises(ValueError, match="unique sessions"):
        load_sessions(path)


def test_circular_blocks_wrap_and_truncate_without_changing_length():
    indices = circular_indices(17, 10, 199, 31)
    assert indices.shape == (199, 17)
    assert np.all(np.diff(indices[:, :10], axis=1) % 17 == 1)
    assert np.all(np.diff(indices[:, 10:], axis=1) % 17 == 1)
    assert (indices[:, :10] == 0).any()


def test_paired_constant_increment_preserves_mean_but_not_sharpe_uncertainty():
    baseline = np.random.default_rng(3).normal(10, 100, 200)
    score = score_unit(baseline, baseline + 5, draws=199)
    assert score.p_mean_ni == 1
    assert score.p_sharpe_positive == 1
    assert score.delta_sharpe == pytest.approx(ann_sharpe(baseline + 5) - ann_sharpe(baseline))
    with pytest.raises(ValueError, match="zero paired uncertainty"):
        score_unit(baseline, baseline, draws=99)


def test_maxstat_identical_variant_cannot_inherit_h0_edge():
    baseline = np.random.default_rng(7).normal(20, 100, 301)
    result = incremental_max_test(baseline, baseline[:, None], draws=499)
    assert result.effects[0] == 0
    assert result.adjusted_p[0] == 1


def test_full_family_correction_is_at_least_single_pick_and_column_invariant():
    rng = np.random.default_rng(15)
    baseline = rng.normal(10, 100, 401)
    family = baseline[:, None] + rng.normal(0, 20, (401, 12))
    pick = int(np.argmax([ann_sharpe(x) for x in family.T]))
    full = incremental_max_test(baseline, family, draws=499)
    alone = incremental_max_test(baseline, family[:, [pick]], draws=499)
    reversed_ = incremental_max_test(baseline, family[:, ::-1], draws=499)
    assert full.adjusted_p[pick] >= alone.adjusted_p[0]
    np.testing.assert_array_equal(full.adjusted_p, reversed_.adjusted_p[::-1])
    assert ((full.adjusted_p * 500).round() == full.adjusted_p * 500).all()


def test_maxstat_refuses_missing_replicates_or_unidentified_positive_effect():
    with pytest.raises(ValueError, match="finite full-family"):
        resampled_max_test(np.array([1]), np.array([[1.0], [np.nan]]), block=10, seed=1)
    with pytest.raises(ValueError, match="zero bootstrap variance"):
        resampled_max_test(np.array([1]), np.ones((10, 1)), block=10, seed=1)


def test_skeptical_prior_shrinks_and_does_not_claim_selection_adjustment():
    result = normal_posterior(0.229, 0.190, 0.20)
    assert 0 < result["mean"] < 0.229
    assert result["p_positive"] == pytest.approx(0.808, abs=0.002)


def test_route_intersection_formal_ni_and_predeclared_endpoint():
    rng = np.random.default_rng(11)
    base = rng.normal(10, 100, 300)
    score = score_unit(base, base + rng.normal(1, 5, 300), draws=99)
    repair = replace(
        score,
        ni_margin=1,
        es5_reduction=0.25,
        p_es5_lower=0.95,
        p_mean_ni=0.89,
        p_sharpe_ni=0.95,
        baseline={**score.baseline, "drawdown": -1000},
        candidate={**score.candidate, "drawdown": -1100},
    )
    assert unit_screen(repair, "R")
    assert not unit_screen(repair, "R", formal=True)
    assert not unit_screen(replace(repair, p_sharpe_ni=0.89), "R")
    assert not unit_screen(replace(repair, ni_margin=-1), "R")
    with pytest.raises(ValueError, match="parking"):
        unit_screen(repair, "R", endpoint="parking")
    assert unit_screen(replace(repair, p_mean_positive=0.99), "E")
