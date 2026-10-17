"""Exhaustive combinatorially symmetric cross-validation of a Sharpe selector."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PBOResult:
    """PBO and the complete deterministic split audit, in combination order."""

    pbo: float
    n_splits: int
    n_blocks: int
    n_observations: int
    columns: tuple[str, ...]
    omitted_columns: tuple[str, ...]
    block_bounds: tuple[tuple[str, str, int], ...]
    selection_logits: tuple[float, ...]
    test_relative_ranks: tuple[float, ...]
    train_winners: tuple[str, ...]
    constant_policy: str
    tie_policy: str = (
        "first input column wins train ties; average test rank; logit zero counts as failure"
    )


def _all_midnight(stamps: np.ndarray) -> bool:
    return bool(np.array_equal(stamps, stamps.astype("datetime64[D]").astype(stamps.dtype)))


def pbo(
    daily_pnl: pd.DataFrame,
    *,
    n_blocks: int = 16,
    constant_policy: Literal["raise", "drop"] = "raise",
) -> PBOResult:
    """Bailey et al. CSCV probability of backtest overfitting, using daily SR.

    Rows must be finite, date-aligned daily PnL on a common eligible calendar
    (including explicit flat-day zeros); columns are distinct configuration IDs.
    Require a sorted, unique, timezone-naive midnight DatetimeIndex, >=2 configs,
    an even block count >=4 and >=2 observations per block. Unequal lengths use
    ``array_split``: the first T % n_blocks contiguous blocks get one extra row.
    Every choose(n_blocks, n_blocks/2) combination is used, including both
    complementary orientations (16 blocks => 12870 splits). No random subset.

    Score = daily mean / sample SD, with no annualization. Train ties select the
    first original column. Ascending test ranks use average ranks for exact
    ties. Relative rank = rank/(n_configs+1), giving finite logits; PBO is the
    fraction log(relative_rank/(1-relative_rank)) <= 0 (ties at median fail).

    Constant or numerically constant columns have undefined SR. With ``raise``
    any config constant globally OR in ANY train/test half raises. Explicit
    ``drop`` omits each such config from the WHOLE experiment, preserving a
    fixed candidate population across splits; all omissions are returned. At
    least two must remain. Half variances <=64*eps*half_sum_squares after
    global centering/scaling are numerically constant. NaN/inf always raise;
    they are never dropped or filled. Nonfinite intermediate scores also raise.

    Block sums/squared sums are cached; all split statistics and ranks operate
    on arrays. The complete score matrices use O(choose(B,B/2)*n_configs) memory.
    Contiguous blocks preserve within-block dependence but do not establish
    stationarity, independence, causal alpha or an OS significance guarantee.
    """
    if (
        isinstance(n_blocks, bool)
        or not isinstance(n_blocks, Integral)
        or n_blocks < 4
        or n_blocks % 2
    ):
        raise ValueError("n_blocks must be an even integer >= 4")
    if constant_policy not in {"raise", "drop"}:
        raise ValueError("constant_policy must be 'raise' or 'drop'")
    if not isinstance(daily_pnl, pd.DataFrame) or daily_pnl.ndim != 2:
        raise ValueError("daily_pnl must be a date-indexed DataFrame")
    dates = daily_pnl.index
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.tz is not None
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not _all_midnight(dates.to_numpy())
    ):
        raise ValueError("index must contain sorted unique timezone-naive midnight dates")
    if len(daily_pnl) < 2 * n_blocks or daily_pnl.shape[1] < 2:
        raise ValueError("need >=2 configurations and >=2 observations per block")
    if not daily_pnl.columns.is_unique or any(
        not isinstance(c, str) or not c for c in daily_pnl.columns
    ):
        raise ValueError("configuration IDs must be unique nonempty strings")
    if any(
        not pd.api.types.is_numeric_dtype(d) or pd.api.types.is_complex_dtype(d)
        for d in daily_pnl.dtypes
    ):
        raise ValueError("daily PnL must be real numeric")
    x = daily_pnl.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("daily PnL must be finite; NaN/inf are never imputed")
    # Center/scale for stable sums of squares without changing any Sharpe.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center = x.mean(axis=0)
        centered = x - center
        scale = np.max(np.abs(centered), axis=0)
        global_constant = scale == 0
        safe_scale = np.where(global_constant, 1.0, scale)
        z = centered / safe_scale
        offset = center / safe_scale
    if not np.isfinite(z).all() or not np.isfinite(offset).all():
        raise ValueError("PnL magnitude exceeds stable CSCV arithmetic")
    blocks = np.array_split(np.arange(len(x)), n_blocks)
    sums = np.stack([z[b].sum(axis=0) for b in blocks])
    squares = np.stack([(z[b] ** 2).sum(axis=0) for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    split_blocks = np.array(list(combinations(range(n_blocks), n_blocks // 2)))
    membership = np.zeros((len(split_blocks), n_blocks))
    membership[np.arange(len(split_blocks))[:, None], split_blocks] = 1
    train_n = (membership @ sizes)[:, None]
    test_n = len(x) - train_n
    train_sum = membership @ sums
    train_sq = membership @ squares
    test_sum = sums.sum(axis=0) - train_sum
    test_sq = squares.sum(axis=0) - train_sq

    def stats(total: np.ndarray, sq: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ss = sq - total**2 / n
        constant = ss <= 64 * np.finfo(float).eps * sq
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = (offset + total / n) / np.sqrt(np.maximum(ss, 0) / (n - 1))
        return sr, constant

    train, train_constant = stats(train_sum, train_sq, train_n)
    test, test_constant = stats(test_sum, test_sq, test_n)
    omitted = global_constant | train_constant.any(axis=0) | test_constant.any(axis=0)
    names = np.asarray(daily_pnl.columns)
    if omitted.any() and constant_policy == "raise":
        raise ValueError(
            f"Undefined constant-column Sharpe in full/split data: {names[omitted].tolist()}"
        )
    if (~omitted).sum() < 2:
        raise ValueError("fewer than two configurations remain after constant-column exclusions")
    train, test, names_kept = train[:, ~omitted], test[:, ~omitted], names[~omitted]
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite split Sharpe; cannot rank configurations")
    winner = np.argmax(train, axis=1)
    winner_test = test[np.arange(len(test)), winner][:, None]
    average_rank = (test < winner_test).sum(axis=1) + ((test == winner_test).sum(axis=1) + 1) / 2
    relative = average_rank / (len(names_kept) + 1)
    logits = np.log(relative / (1 - relative))
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        n_splits=len(split_blocks),
        n_blocks=int(n_blocks),
        n_observations=len(x),
        columns=tuple(names_kept),
        omitted_columns=tuple(names[omitted]),
        block_bounds=tuple(
            (str(dates[b[0]].date()), str(dates[b[-1]].date()), len(b)) for b in blocks
        ),
        selection_logits=tuple(float(v) for v in logits),
        test_relative_ranks=tuple(float(v) for v in relative),
        train_winners=tuple(names_kept[winner]),
        constant_policy=constant_policy,
    )
