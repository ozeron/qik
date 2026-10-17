"""Synthetic known-selector fixtures; no strategy return matrix is accessed."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from qik.metrics import pbo


def matrix(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2000-01-03", periods=len(values)),
        columns=[f"config-{i}" for i in range(values.shape[1])],
    )


def persistent() -> pd.DataFrame:
    noise = np.tile([-0.1, 0.1, -0.2, 0.2], 16)[:, None]
    return matrix(noise + np.array([0.1, 0.2, 0.3]))


def test_16_block_overfit_train_winner_loses_every_test_half() -> None:
    # Each hypothesis fits exactly one block; every train winner's special
    # block is absent from test, which rewards the complementary hypotheses.
    values = np.full((64, 16), -1.0)
    for block in range(16):
        values[block * 4 : (block + 1) * 4, block] = 16
    values += np.tile([-0.1, 0.1, -0.2, 0.2], 16)[:, None]
    result = pbo(matrix(values))
    assert result.n_splits == math.comb(16, 8) == 12870
    assert result.pbo == 1
    assert max(result.selection_logits) < 0
    assert max(result.test_relative_ranks) < 0.5


def test_persistent_winner_and_all_complementary_orientations() -> None:
    result = pbo(persistent())
    assert result.n_splits == 12870
    assert result.pbo == 0
    assert set(result.train_winners) == {"config-2"}
    assert result.test_relative_ranks == (0.75,) * 12870
    assert result.selection_logits == pytest.approx([math.log(3)] * 12870)


def test_exact_ties_are_deterministic_and_median_ties_fail() -> None:
    frame = persistent()
    frame.iloc[:, :] = frame.iloc[:, [0]].to_numpy()
    result = pbo(frame)
    assert result == pbo(frame)
    assert set(result.train_winners) == {"config-0"}
    assert result.test_relative_ranks == (0.5,) * 12870
    assert result.selection_logits == (0.0,) * 12870
    assert result.pbo == 1


def test_cached_statistics_match_direct_formula_and_contiguous_unequal_blocks() -> None:
    frame = matrix(np.random.default_rng(419).normal(0.1, 1, size=(19, 5)))
    result = pbo(frame, n_blocks=4)
    blocks = np.array_split(np.arange(len(frame)), 4)
    direct = []
    winners = []
    for combination in combinations(range(4), 2):
        train_idx = np.concatenate([blocks[i] for i in combination])
        test_idx = np.concatenate([blocks[i] for i in range(4) if i not in combination])
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        winner = (train.mean() / train.std(ddof=1)).idxmax()
        test_sr = test.mean() / test.std(ddof=1)
        rank = test_sr.rank(method="average")[winner] / (frame.shape[1] + 1)
        direct.append(math.log(rank / (1 - rank)))
        winners.append(winner)
    assert result.selection_logits == pytest.approx(direct, abs=1e-12)
    assert result.train_winners == tuple(winners)
    assert result.pbo == np.mean(np.array(direct) <= 0)
    assert result.block_bounds == tuple(
        (str(frame.index[b[0]].date()), str(frame.index[b[-1]].date()), len(b)) for b in blocks
    )
    assert [b[2] for b in result.block_bounds] == [5, 5, 5, 4]


def test_positive_per_configuration_scaling_preserves_selection() -> None:
    frame = matrix(np.random.default_rng(719).normal(0.1, 1, size=(65, 5)))
    before = pbo(frame)
    after = pbo(frame * [0.01, 1, 2, 100, 999])
    assert before.train_winners == after.train_winners
    assert before.selection_logits == after.selection_logits
    assert before.pbo == after.pbo


def test_explicit_constant_policy_preserves_fixed_population_and_reports_omissions() -> None:
    frame = persistent()
    frame["global-constant"] = 0.0
    frame["half-constant"] = np.r_[np.zeros(32), np.tile([-1, 1], 16)]
    with pytest.raises(ValueError, match="constant-column"):
        pbo(frame)
    result = pbo(frame, constant_policy="drop")
    assert result.columns == ("config-0", "config-1", "config-2")
    assert result.omitted_columns == ("global-constant", "half-constant")
    assert result.pbo == pbo(persistent()).pbo
    with pytest.raises(ValueError, match="fewer than two"):
        pbo(frame[["config-0", "global-constant"]], constant_policy="drop")


@pytest.mark.parametrize("blocks", [True, 0, 2, 3, 5, 16.5])
def test_invalid_block_count(blocks: int) -> None:
    with pytest.raises(ValueError, match="even integer"):
        pbo(persistent(), n_blocks=blocks)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pnl_never_imputed_even_with_explicit_drop(value: float) -> None:
    frame = persistent()
    frame.iloc[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        pbo(frame, constant_policy="drop")


@pytest.mark.parametrize(
    "kind", ["unsorted", "duplicate", "timezone", "intraday", "nat", "strings"]
)
def test_invalid_dates(kind: str) -> None:
    frame = persistent()
    if kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0]] * len(frame))
    elif kind == "timezone":
        frame.index = frame.index.tz_localize("UTC")
    elif kind == "intraday":
        frame.index += pd.Timedelta(hours=1)
    elif kind == "nat":
        frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    else:
        frame.index = frame.index.astype(str)
    with pytest.raises(ValueError, match="dates"):
        pbo(frame)


def test_shape_schema_and_policy_errors() -> None:
    frame = persistent()
    for invalid in [frame.iloc[:31], frame.iloc[:, :1]]:
        with pytest.raises(ValueError, match="observations per block"):
            pbo(invalid)
    for columns in [["same"] * 3, [1, 2, 3], ["", "b", "c"]]:
        invalid = frame.copy()
        invalid.columns = columns
        with pytest.raises(ValueError, match="configuration IDs"):
            pbo(invalid)
    with pytest.raises(ValueError, match="real numeric"):
        pbo(frame.astype(str))
    with pytest.raises(ValueError, match="real numeric"):
        pbo(frame.astype(complex))
    with pytest.raises(ValueError, match="DataFrame"):
        pbo(frame.to_numpy())
    with pytest.raises(ValueError, match="constant_policy"):
        pbo(frame, constant_policy="ignore")
