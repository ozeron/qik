"""Selection-statistics inputs shared by optimize/repro projects: DSR inputs, PBO summary, N_eff.

Pure functions. Every input is a parameter: nothing here reads a trial log or calls ``count_trials``
(the caller supplies ``n_trials`` and the logged trial Sharpes). The Sharpe/PBO formulas stay in
``qik.metrics`` (``deflated_sharpe``, ``pbo``); this module only builds their inputs.

Project copies of these helpers differ in small, deliberate ways, so each difference is a keyword:

* ``dsr_inputs(constant=)``: a constant series either raises or yields ``None``.
* ``trial_sharpe_variance(nonfinite=)``: non-finite trial Sharpes either raise or are dropped.
* ``deflated_sharpe_from_returns(clamp_n=)``: ``max(n_trials, 1)`` or ``n_trials`` as given.
* ``pbo_summary(localize_tz=)``: strict date-index contract with tz stripped, or coerce and normalize only.
* ``n_eff_participation(constant=, nonfinite=, nan_corr=, min_columns=)``: which columns count and what a
  NaN correlation does. ``constant`` and ``nonfinite`` have no default: dropping a column lowers N_eff,
  which deflates the DSR less, so the caller must choose it.

``n_eff_participation`` (eigenvalue participation ratio of the return-path correlation matrix) and
``n_eff_mean_corr`` (``ceil(rho + (1 - rho) * N)`` from the mean off-diagonal correlation) are DIFFERENT
statistics and give different numbers; keep them apart and say which one a report quotes.

Moments are z-score moments (``mean(z**3)``, ``mean(z**4)`` with ``z = (r - mean) / std(ddof=0)``).
``mean(c**3) / m2**1.5`` differs by an ulp; copies using it are not covered.

Not covered: P081 ``n_eff_from_paths`` answers ``float(k)`` (``None`` for ``k = 0``) with a note when it
has fewer than 3 rows or fewer than 2 usable paths; ``n_eff_participation`` raises instead.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd

from qik.metrics import DEFAULT_PERIODS_PER_YEAR, deflated_sharpe, pbo

__all__ = [
    "DsrInputs",
    "NEffParticipation",
    "PboSummary",
    "deflated_sharpe_from_returns",
    "dsr_inputs",
    "n_eff_mean_corr",
    "n_eff_participation",
    "pbo_summary",
    "trial_sharpe_variance",
]


@dataclass(frozen=True)
class DsrInputs:
    """Per-observation Sharpe, observation count and NON-excess moments of one return series."""

    sr: float
    n_obs: int
    skew: float
    kurt: float


@overload
def dsr_inputs(returns: Any, *, constant: Literal["raise"]) -> DsrInputs: ...
@overload
def dsr_inputs(returns: Any, *, constant: Literal["none"]) -> DsrInputs | None: ...
def dsr_inputs(returns: Any, *, constant: Literal["raise", "none"]) -> DsrInputs | None:
    """Sharpe (mean / sample SD, not annualized), n, z-score skew and raw kurtosis of ``returns``.

    NaN entries are dropped first (P079 instead calls a column with any NaN undefined: callers matching
    it must drop such columns themselves). A series with fewer than two observations, zero sample SD or a
    NaN sample SD (an infinite entry) has an undefined Sharpe: ``constant="raise"`` raises ``ValueError``,
    ``"none"`` returns ``None`` (as P079 does for a constant series). P076/P105/P097 raise for such
    a series too, from ``deflated_sharpe``.
    """
    if constant not in ("raise", "none"):
        raise ValueError("constant must be 'raise' or 'none'")
    r = pd.Series(returns).dropna().astype(float).to_numpy()
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    if not sd > 0.0:
        if constant == "raise":
            if len(r) < 2:
                why = "fewer than 2 observations"
            elif not np.isfinite(r).all():
                why = "an infinite entry"
            else:
                why = "zero sample SD"
            raise ValueError(f"Sharpe is undefined: returns have {why}")
        return None
    z = (r - r.mean()) / r.std(ddof=0)
    return DsrInputs(
        sr=float(r.mean() / r.std(ddof=1)),
        n_obs=len(r),
        skew=float((z**3).mean()),
        kurt=float((z**4).mean()),
    )


def trial_sharpe_variance(
    trial_sharpes: Sequence[float] | np.ndarray,
    *,
    periods_per_year: float,
    nonfinite: Literal["raise", "drop"],
) -> float:
    """Across-trial variance (ddof=1) of ``trial_sharpes`` in per-observation units.

    ``trial_sharpes`` are in the caller's units; the variance is divided by ``periods_per_year`` to
    convert annualized Sharpes (variance scales with the square of sqrt(periods)). Pass
    ``periods_per_year=1.0`` for Sharpes that are already per observation. Non-finite values raise
    or are dropped (``nonfinite``); fewer than two usable values always raise.
    """
    if nonfinite not in ("raise", "drop"):
        raise ValueError("nonfinite must be 'raise' or 'drop'")
    if not periods_per_year > 0:
        raise ValueError("periods_per_year must be positive")
    ts = np.asarray(list(trial_sharpes), dtype=float)
    finite = np.isfinite(ts)
    if nonfinite == "drop":
        ts = ts[finite]
    elif not finite.all():
        raise ValueError("trial_sharpes needs >= 2 finite values")
    if len(ts) < 2:
        raise ValueError("trial_sharpes needs >= 2 finite values")
    return float(ts.var(ddof=1)) / float(periods_per_year)


def deflated_sharpe_from_returns(
    returns: Any,
    n_trials: int,
    *,
    trial_sharpes: Sequence[float] | np.ndarray,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    nonfinite: Literal["raise", "drop"] = "raise",
    clamp_n: bool = False,
) -> float:
    """``qik.metrics.deflated_sharpe`` of ``returns`` at ``n_trials`` against ``trial_sharpes``.

    ``clamp_n`` applies ``max(int(n_trials), 1)``; without it ``n_trials`` goes through unchanged and
    ``deflated_sharpe`` rejects values below 1. A constant series raises.
    """
    inp = dsr_inputs(returns, constant="raise")
    variance = trial_sharpe_variance(
        trial_sharpes, periods_per_year=periods_per_year, nonfinite=nonfinite
    )
    return float(
        deflated_sharpe(
            inp.sr,
            n_observations=inp.n_obs,
            n_trials=max(int(n_trials), 1) if clamp_n else n_trials,
            sharpe_variance=variance,
            skew=inp.skew,
            kurtosis=inp.kurt,
        )
    )


@dataclass(frozen=True)
class PboSummary:
    """Headline PBO numbers plus the names of the constant columns that were dropped."""

    pbo: float
    variants: int
    omitted_constant_columns: int
    omitted_names: tuple[str, ...]
    combinations: int
    rows_used: int
    mean_oos_rank_of_winner: float
    median_logit: float

    def as_dict(self, *, include_names: bool = False) -> dict[str, Any]:
        """Report dict; ``omitted_names`` (a list) only with ``include_names``."""
        out: dict[str, Any] = {
            "pbo": self.pbo,
            "variants": self.variants,
            "omitted_constant_columns": self.omitted_constant_columns,
        }
        if include_names:
            out["omitted_names"] = list(self.omitted_names)
        out |= {
            "combinations": self.combinations,
            "rows_used": self.rows_used,
            "mean_oos_rank_of_winner": self.mean_oos_rank_of_winner,
            "median_logit": self.median_logit,
        }
        return out


def _midnight(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """``idx.normalize()`` (the type checker's pandas stubs do not declare ``DatetimeIndex.normalize``)."""
    normalized: pd.DatetimeIndex = cast("Any", idx).normalize()
    return normalized


def pbo_summary(m: pd.DataFrame, *, n_blocks: int = 16, localize_tz: bool) -> PboSummary:
    """``qik.metrics.pbo`` (CSCV) with constant columns dropped from the whole experiment.

    ``localize_tz=True`` is the strict contract: the index must be a ``DatetimeIndex`` (``TypeError``
    otherwise) and its timezone is stripped before normalizing to midnight. ``False`` coerces the index
    with ``pd.DatetimeIndex`` and only normalizes, so a tz-aware index is still rejected by ``pbo``.
    """
    x = m.copy()
    if localize_tz:
        if not isinstance(m.index, pd.DatetimeIndex):
            raise TypeError("pbo_summary needs a date-indexed return matrix")
        naive = m.index.tz_localize(None) if m.index.tz is not None else m.index
        x.index = _midnight(naive)
    else:
        x.index = _midnight(pd.DatetimeIndex(m.index))
    r = pbo(x, n_blocks=n_blocks, constant_policy="drop")
    return PboSummary(
        pbo=float(r.pbo),
        variants=len(r.columns),
        omitted_constant_columns=len(r.omitted_columns),
        omitted_names=tuple(str(c) for c in r.omitted_columns),
        combinations=int(r.n_splits),
        rows_used=int(r.n_observations),
        mean_oos_rank_of_winner=float(np.mean(r.test_relative_ranks)),
        median_logit=float(np.median(r.selection_logits)),
    )


@dataclass(frozen=True)
class NEffParticipation:
    """Participation-ratio N_eff, the columns it used and the eigenvalues (descending)."""

    n_eff: float
    n_columns: int
    n_dropped: int
    eigenvalues: tuple[float, ...]


def n_eff_participation(
    paths: pd.DataFrame | np.ndarray,
    *,
    constant: Literal["drop", "raise"],
    nonfinite: Literal["drop", "keep"],
    min_columns: int = 2,
    nan_corr: Literal["raise", "zero"] = "raise",
) -> NEffParticipation:
    """Participation ratio ``(sum lambda)^2 / sum lambda^2`` of the correlation matrix of return paths.

    ``paths`` is days x trials, one column per trial (identical columns lower it). It reads 1 for
    perfectly correlated paths and the column count for uncorrelated ones. Eigenvalues are clipped at 0.
    Fewer than 3 rows raise ``ValueError`` (two points always correlate at +-1).

    ``constant`` and ``nonfinite`` are required: every dropped column lowers N_eff, which deflates the
    DSR less, so silently dropping NaN/inf paths is the unsafe direction.

    * ``constant``: constant columns are dropped or raise. Constant means a population SD (``np.nanstd``
      over the matrix, axis 0) of exactly float zero, with no tolerance, so equal copies of a value are
      constant only if the reduction rounds to 0.0: ``np.full((10, 2), 0.1)`` gives 1.4e-17 per column
      and is kept, while ``DataFrame.std(ddof=0)`` gives 0.0 for it (P097 uses the latter, so it
      drops such a column where this function keeps it).
    * ``nonfinite``: columns with any non-finite value are dropped (they count in ``n_dropped``) or
      kept. A kept column still has NaN correlations, handled by ``nan_corr``; its constancy is then
      judged on its finite values only (as ``DataFrame.std`` does).
    * ``nan_corr``: ``"raise"`` on a non-finite correlation matrix or ``"zero"`` to set those entries to 0.
    * ``min_columns``: fewer usable columns raise ``ValueError``. At 1 the single column reads 1.0, at 0
      no columns read 0.0.
    """
    if constant not in ("drop", "raise"):
        raise ValueError("constant must be 'drop' or 'raise'")
    if nonfinite not in ("drop", "keep"):
        raise ValueError("nonfinite must be 'drop' or 'keep'")
    if nan_corr not in ("raise", "zero"):
        raise ValueError("nan_corr must be 'raise' or 'zero'")
    if min_columns < 0:
        raise ValueError("min_columns must be >= 0")
    x = np.asarray(paths, dtype=float)
    if x.ndim != 2:
        raise ValueError("paths must be a 2-D (days x trials) matrix")
    if x.shape[0] < 3:
        raise ValueError("n_eff needs >= 3 days")
    finite = np.isfinite(x).all(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(x, axis=0)
    moving = sd > 0
    if constant == "raise" and not moving[finite].all():
        raise ValueError("paths contain constant columns")
    keep = moving & finite if nonfinite == "drop" else moving
    k = int(keep.sum())
    dropped = int(x.shape[1] - k)
    if k < min_columns:
        raise ValueError(f"n_eff needs at least {min_columns} usable return paths, got {k}")
    if k == 0:
        return NEffParticipation(0.0, 0, dropped, ())
    if k == 1:
        return NEffParticipation(1.0, 1, dropped, (1.0,))
    corr = np.corrcoef(x[:, keep], rowvar=False)
    if nan_corr == "zero":
        corr = np.nan_to_num(corr, nan=0.0)
    elif not np.isfinite(corr).all():
        raise ValueError("correlation matrix is not finite")
    eig = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    total = float(eig.sum())
    ratio = float(total**2 / (eig**2).sum()) if total > 0 else math.nan
    return NEffParticipation(
        n_eff=ratio,
        n_columns=k,
        n_dropped=dropped,
        eigenvalues=tuple(float(v) for v in np.sort(eig)[::-1]),
    )


def n_eff_mean_corr(rho_bar: float, n: int) -> int:
    """``ceil(rho_bar + (1 - rho_bar) * n)``: N_eff from the mean off-diagonal correlation.

    Not the participation ratio. ``rho_bar = 1`` gives 1 and ``rho_bar = 0`` gives ``n``. The product
    is rounded to nine decimals before the ceiling so float noise cannot add a trial.
    """
    if not math.isfinite(rho_bar):
        raise ValueError("rho_bar must be finite")
    return math.ceil(round(rho_bar + (1.0 - rho_bar) * n, 9))
