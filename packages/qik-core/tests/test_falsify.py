from __future__ import annotations

import numpy as np
import pytest

from qik import falsify as fz


def _pool(n_per_year: int = 400, years: tuple[int, ...] = (2020, 2021), seed: int = 1):
    rng = np.random.default_rng(seed)
    yr, sd, en, ex = [], [], [], []
    for k, y in enumerate(years):
        e = np.sort(rng.choice(5000, size=n_per_year, replace=False)) + k * 5000
        hold = rng.choice([1, 30], size=n_per_year)  # bimodal durations
        yr += [y] * n_per_year
        sd += list(rng.choice([-1, 1], size=n_per_year))
        en += list(e)
        ex += list(e + hold)
    return tuple(np.asarray(a) for a in (yr, sd, en, ex))


def _targets(yr: np.ndarray, sd: np.ndarray):
    ty = np.array([2020] * 12 + [2020] * 5 + [2021] * 9)
    ts = np.array([1] * 12 + [-1] * 5 + [1] * 5 + [-1] * 4)
    return ty, ts


def test_counts_preserved_every_draw_and_hash_stable() -> None:
    yr, sd, en, ex = _pool()
    ty, ts = _targets(yr, sd)
    seen: list[np.ndarray] = []

    def score(idx: np.ndarray) -> float:
        seen.append(idx)
        return float(len(idx))

    for overlap in (True, False):
        seen.clear()
        res = fz.random_entry_null(
            year=yr, side=sd, entry=en, exit=ex, target_year=ty, target_side=ts,
            score=score, observed=0.0, draws=200, seed=3, overlap=overlap,
        )  # fmt: skip
        assert res.count_assertions == 200 and len(seen) == 200
        for idx in seen:
            assert len(idx) == len(ty)
            for y, s in {(2020, 1), (2020, -1), (2021, 1), (2021, -1)}:
                want = int(((ty == y) & (ts == s)).sum())
                assert int(((yr[idx] == y) & (sd[idx] == s)).sum()) == want
    assert res.mode == "nonoverlap"
    h1 = fz.random_entry_null(
        year=yr, side=sd, entry=en, exit=ex, target_year=ty, target_side=ts,
        score=score, observed=0.0, draws=5, seed=99,
    )  # fmt: skip
    assert h1.counts_sha256 == res.counts_sha256  # depends on counts only, not seed or mode


def test_nonoverlap_histories_are_flat_to_flat() -> None:
    yr, sd, en, ex = _pool()
    sd = np.ones_like(sd)
    ty, ts = np.array([2020] * 20 + [2021] * 20), np.ones(40, dtype=int)
    seen: list[np.ndarray] = []
    fz.random_entry_null(
        year=yr, side=sd, entry=en, exit=ex, target_year=ty, target_side=ts,
        score=lambda i: seen.append(i) or 0.0, observed=0.0, draws=100, seed=5, overlap=False,
    )  # fmt: skip
    for idx in seen:
        assert (en[idx][1:] > ex[idx][:-1]).all()


def test_deterministic_by_seed() -> None:
    yr, sd, en, ex = _pool()
    ty, ts = _targets(yr, sd)

    def run(seed: int, overlap: bool) -> fz.NullResult:
        return fz.random_entry_null(
            year=yr, side=sd, entry=en, exit=ex, target_year=ty, target_side=ts,
            score=lambda i: float(en[i].sum() % 1013), observed=1.0, draws=50, seed=seed,
            overlap=overlap,
        )  # fmt: skip

    for ov in (True, False):
        assert np.array_equal(run(7, ov).stats, run(7, ov).stats)
        assert not np.array_equal(run(7, ov).stats, run(8, ov).stats)


def test_no_duration_bias_in_nonoverlap_packing() -> None:
    yr, sd, en, ex = _pool(n_per_year=800, years=(2020,))
    sd = np.ones_like(sd)
    pool_mean = float((ex - en).mean())
    ty, ts = np.array([2020] * 25), np.ones(25, dtype=int)
    res = fz.random_entry_null(
        year=yr, side=sd, entry=en, exit=ex, target_year=ty, target_side=ts,
        score=lambda i: float((ex[i] - en[i]).mean()), observed=0.0, draws=1000, seed=11,
        overlap=False,
    )  # fmt: skip
    assert abs(res.stats.mean() - pool_mean) / pool_mean < 0.04


def test_infeasible_count_raises() -> None:
    yr, sd, en, ex = _pool(n_per_year=20, years=(2020,))
    ty, ts = np.array([2020] * 15), np.array([1] * 15)
    with pytest.raises(ValueError):
        fz.random_entry_null(
            year=yr, side=np.ones_like(sd), entry=en, exit=ex + 1000, target_year=ty,
            target_side=ts, score=lambda i: 0.0, observed=0.0, draws=3, overlap=False,
        )  # fmt: skip


def test_percentile_tie_policies() -> None:
    null = [1.0, 2.0, 2.0, 2.0, 3.0, float("nan")]
    assert fz.percentile_of(2.0, null) == pytest.approx(50.0)  # (1 + 1.5) / 5
    assert fz.percentile_of(2.0, null, "weak") == pytest.approx(80.0)
    assert fz.percentile_of(2.0, null, "strict") == pytest.approx(20.0)
    assert fz.percentile_of(10.0, null) == 100.0
    with pytest.raises(ValueError):
        fz.percentile_of(1.0, [float("nan")])


def test_matched_removal_keeps_counts_per_stratum() -> None:
    rng = np.random.default_rng(0)
    pnl = rng.normal(size=300)
    strata = np.repeat([0, 1, 2], 100)
    kept = pnl > 0.2  # an informed filter
    seen: list[int] = []
    res = fz.matched_removal_permutation(
        kept=kept, strata=strata, score=lambda i: seen.append(len(i)) or float(pnl[i].mean()),
        draws=300, seed=2,
    )  # fmt: skip
    assert set(seen[:300]) == {int(kept.sum())}
    assert res.percentile > 99  # a PnL-based filter beats random removal
    assert res.mode == "removal"


def test_side_flip_mirrors() -> None:
    ret = np.array([1.0, -2.0, 3.0])
    r = fz.side_flip(lambda s: float((s * ret).sum()), np.ones(3))
    assert (r.observed, r.flipped, r.delta) == (2.0, -2.0, 4.0)


def test_circular_shift_preserves_per_year_counts_and_spacing() -> None:
    years = np.repeat([2020, 2021], 100)
    entries = np.array([5, 20, 90, 110, 150])
    for t in fz.circular_shift_entries(entries, years, n_tapes=20, seed=1):
        assert (years[t] == years[entries]).all()
        d0, d1 = np.diff(entries[:3]) % 100, np.diff(t[:3]) % 100
        assert (d0 == d1).all()
    a = list(fz.circular_shift_entries(entries, years, n_tapes=3, seed=4))
    b = list(fz.circular_shift_entries(entries, years, n_tapes=3, seed=4))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_random_direction_is_fair_pm1() -> None:
    tapes = np.array(list(fz.random_direction(np.ones(500), n_tapes=20, seed=1)))
    assert set(np.unique(tapes)) == {-1, 1}
    assert abs(tapes.mean()) < 0.05


def test_stationary_bootstrap_length_and_mean() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=1000)
    tapes = np.array(list(fz.stationary_bootstrap(r, n_tapes=300, seed=3)))
    assert tapes.shape == (300, 1000)
    assert abs(tapes.mean() - r.mean()) < 0.0005
    assert np.isin(tapes, r).all()
    # blocks: consecutive source indices are common (mean block ~ 10), unlike iid resampling
    pos = {v: i for i, v in enumerate(r)}
    t = tapes[0]
    cont = np.mean([(pos[b] - pos[a]) % 1000 == 1 for a, b in zip(t[:-1], t[1:], strict=True)])
    assert 0.5 < cont < 0.99
