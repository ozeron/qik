"""Risk / annualization helpers (Sharpe, Sortino, drawdown, CAGR, Calmar, tails)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_PERIODS_PER_YEAR = 252.0

# Carver AFTS: inv_cdf(0.99) / inv_cdf(0.70) ≈ inv_cdf(0.01) / inv_cdf(0.30).
_CARVER_TAIL_REF = 4.4395


def ann_sharpe(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    rf: float = 0.0,
    ddof: int = 1,
) -> float:
    """Annualized Sharpe from a return (or PnL) series.

    Default is sample std (``ddof=1``); ``ddof=0`` is the population variant some
    project reports use — pass it explicitly so the definition stays visible.
    """

    if len(returns) < 2:
        return float("nan")
    excess = returns.astype(float) - float(rf) / float(periods_per_year)
    mu = float(excess.mean())
    sig = float(excess.std(ddof=ddof))
    if sig <= 0.0:
        return float("nan")
    return (mu / sig) * float(np.sqrt(periods_per_year))


def ann_vol(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> float:
    """Annualized volatility (sample std, ddof=1)."""

    if len(returns) < 2:
        return float("nan")
    sig = float(returns.astype(float).std(ddof=1))
    if sig <= 0.0:
        return float("nan")
    return sig * float(np.sqrt(periods_per_year))


def ann_sortino(
    returns: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Annualized Sortino using downside deviation of excess returns < 0."""

    if len(returns) < 2:
        return float("nan")
    excess = returns.astype(float) - float(rf) / float(periods_per_year)
    mu = float(excess.mean())
    down = excess[excess < 0.0]
    if len(down) < 2:
        return float("nan")
    dsig = float(down.std(ddof=1))
    if dsig <= 0.0:
        return float("nan")
    return (mu / dsig) * float(np.sqrt(periods_per_year))


def return_skew(returns: pd.Series) -> float:
    """Sample skewness of a return series (pandas default)."""

    r = returns.dropna().astype(float)
    if len(r) < 3:
        return float("nan")
    return float(r.skew())


def carver_tail_ratios(returns: pd.Series) -> tuple[float, float]:
    """Carver AFTS relative lower/upper fat-tail ratios vs Normal.

    On demeaned returns:
      lower_raw = p01 / p30
      upper_raw = p99 / p70
      relative = raw / 4.4395

    Values > 1 ⇒ fatter than Gaussian on that side.
    """

    r = returns.dropna().astype(float)
    if len(r) < 30:
        return float("nan"), float("nan")
    demeaned = r - float(r.mean())
    p01 = float(demeaned.quantile(0.01))
    p30 = float(demeaned.quantile(0.30))
    p70 = float(demeaned.quantile(0.70))
    p99 = float(demeaned.quantile(0.99))
    if p30 == 0.0 or p70 == 0.0:
        return float("nan"), float("nan")
    lower_raw = p01 / p30
    upper_raw = p99 / p70
    if lower_raw <= 0.0 or upper_raw <= 0.0:
        return float("nan"), float("nan")
    return float(lower_raw / _CARVER_TAIL_REF), float(upper_raw / _CARVER_TAIL_REF)


def return_shape_stats(returns: pd.Series) -> dict[str, float]:
    """Skew + Carver lower/upper tail ratios for a return series."""

    lower, upper = carver_tail_ratios(returns)
    return {
        "skew": return_skew(returns),
        "lower_tail_ratio": lower,
        "upper_tail_ratio": upper,
    }


def max_drawdown_from_pnl(
    pnl: pd.Series,
    *,
    initial_capital: float,
) -> tuple[float, float]:
    """Max drawdown in dollars and fraction from a PnL stream (trade or daily)."""

    if pnl.empty:
        return 0.0, 0.0
    equity = float(initial_capital) + pnl.astype(float).cumsum()
    peak = equity.cummax()
    dd_usd = equity - peak
    dd_pct = equity / peak - 1.0
    return float(dd_usd.min()) if len(dd_usd) else 0.0, float(dd_pct.min()) if len(dd_pct) else 0.0


def max_drawdown_usd(
    pnl: pd.Series | np.ndarray | Sequence[float],
    *,
    start_at_zero: bool = False,
) -> float:
    """Max drawdown (≤ 0, dollars) of cumulative PnL with no capital base.

    ``start_at_zero=False``: the peak starts at the first cumulative value, so an
    opening losing streak is not a drawdown (same as :func:`max_drawdown_from_pnl`).
    ``start_at_zero=True``: a 0 is prepended, so an opening losing streak counts.
    Reports that show a positive magnitude use ``abs(max_drawdown_usd(...))``.
    """

    cum = pd.Series(np.asarray(pnl, dtype=float)).cumsum()
    if cum.empty:
        return 0.0
    if start_at_zero:
        cum = pd.concat([pd.Series([0.0]), cum], ignore_index=True)
    return float((cum - cum.cummax()).min())


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """Max fractional drawdown from an equity level series."""

    eq = equity.dropna().astype(float)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def cagr_from_equity(
    initial_capital: float,
    final_equity: float,
    *,
    years: float,
) -> float:
    """CAGR as a fraction (not percent). Returns nan if inputs invalid."""

    if years <= 0.0 or initial_capital <= 0.0 or final_equity <= 0.0:
        return float("nan")
    return float((final_equity / initial_capital) ** (1.0 / years) - 1.0)


def window_years(date_from: date, date_to: date) -> float:
    """Inclusive calendar span in years (365.25 day year)."""

    return max((pd.Timestamp(date_to) - pd.Timestamp(date_from)).days / 365.25, 1e-9)


def calmar(cagr: float, max_drawdown: float) -> float:
    """CAGR / |max drawdown|; 0 when drawdown is zero."""

    if max_drawdown == 0.0 or not np.isfinite(max_drawdown):
        return 0.0
    if not np.isfinite(cagr):
        return float("nan")
    return float(cagr / abs(max_drawdown))


__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "ann_sharpe",
    "ann_sortino",
    "ann_vol",
    "calmar",
    "carver_tail_ratios",
    "cagr_from_equity",
    "max_drawdown_from_equity",
    "max_drawdown_from_pnl",
    "max_drawdown_usd",
    "return_shape_stats",
    "return_skew",
    "window_years",
]
