"""Strategy summary: labelled Sharpe grids + blotter + window CAGR."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, TypedDict

import pandas as pd

from qik.metrics.returns import active_day_returns, session_returns
from qik.metrics.risk import (
    DEFAULT_PERIODS_PER_YEAR,
    ann_sharpe,
    ann_sortino,
    ann_vol,
    cagr_from_equity,
    max_drawdown_from_pnl,
    return_shape_stats,
    window_years,
)
from qik.metrics.schema import TradeColumns
from qik.metrics.trade_stats import trade_blotter_stats


class StrategySummary(TypedDict, total=False):
    """Canonical headline metrics.

    Primary Sharpe is **eligible tradable sessions** (flat days = 0) when an
    eligible grid is passed; active-day Sharpe is diagnostic only.
    """

    n_trades: int
    n_long: int
    n_short: int
    net_pnl_usd: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    win_rate: float
    expectancy: float
    avg_win: float
    avg_loss: float
    long_net_pnl_usd: float
    short_net_pnl_usd: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    cagr: float
    cagr_pct: float
    ann_vol: float
    sharpe_ann: float
    sharpe_session_ann: float
    sharpe_active_day_ann: float
    sortino_ann: float
    sortino_session_ann: float
    skew: float
    lower_tail_ratio: float
    upper_tail_ratio: float
    trades_per_year: float
    n_eligible_sessions: int
    n_active_sessions: int
    pct_sessions_traded: float
    window_years: float
    sharpe_definition: str
    periods_per_year: float


def _empty_summary() -> dict[str, Any]:
    return {
        "n_trades": 0,
        "n_long": 0,
        "n_short": 0,
        "net_pnl_usd": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": float("nan"),
        "win_rate": float("nan"),
        "expectancy": 0.0,
        "avg_win": float("nan"),
        "avg_loss": float("nan"),
        "long_net_pnl_usd": 0.0,
        "short_net_pnl_usd": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "cagr": float("nan"),
        "cagr_pct": float("nan"),
        "ann_vol": float("nan"),
        "sharpe_ann": float("nan"),
        "sharpe_session_ann": float("nan"),
        "sharpe_active_day_ann": float("nan"),
        "sortino_ann": float("nan"),
        "sortino_session_ann": float("nan"),
        "skew": float("nan"),
        "lower_tail_ratio": float("nan"),
        "upper_tail_ratio": float("nan"),
        "trades_per_year": float("nan"),
        "n_eligible_sessions": 0,
        "n_active_sessions": 0,
        "pct_sessions_traded": float("nan"),
        "periods_per_year": float(DEFAULT_PERIODS_PER_YEAR),
    }


def summarize_trades(
    trades: pd.DataFrame,
    *,
    eligible_sessions: Sequence[date | pd.Timestamp | str] | None = None,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    session_col: str | None = None,
    side_col: str | None = None,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Compute net/PF/win-rate style metrics from a trade blotter.

    When ``eligible_sessions`` is provided, ``sharpe_ann`` / ``sharpe_session_ann``
    use the eligible-tradable-session return series (zeros on flat sessions).
    Skew / Carver tail ratios are computed on that same primary series.
    ``sharpe_active_day_ann`` is always the trade-day-only diagnostic.
    """

    empty = _empty_summary()
    if trades is None or trades.empty:
        return empty

    try:
        blotter = trade_blotter_stats(
            trades,
            pnl_col=pnl_col,
            session_col=session_col,
            side_col=side_col,
        )
    except ValueError:
        return empty

    cols = TradeColumns.infer(
        trades,
        pnl_col=pnl_col or blotter.get("_pnl_col"),  # type: ignore[arg-type]
        session_col=session_col or blotter.get("_session_col"),  # type: ignore[arg-type]
        side_col=side_col,
    )
    pnl = pd.to_numeric(trades[cols.pnl], errors="coerce").fillna(0.0)
    max_dd_usd, max_dd_pct = max_drawdown_from_pnl(pnl, initial_capital=initial_capital)
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    avg_win = float(wins.mean()) if len(wins) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")

    active_rets = active_day_returns(
        trades,
        initial_capital=initial_capital,
        pnl_col=cols.pnl,
        session_col=cols.session,
    )
    sharpe_active = ann_sharpe(active_rets, periods_per_year=periods_per_year)
    sortino_active = ann_sortino(active_rets, periods_per_year=periods_per_year)
    n_active = int(active_rets.shape[0])

    sharpe_session = float("nan")
    sortino_session = float("nan")
    n_eligible = 0
    pct_traded = float("nan")
    primary_rets = active_rets
    if eligible_sessions is not None:
        session_rets = session_returns(
            trades,
            eligible_sessions,
            initial_capital=initial_capital,
            pnl_col=cols.pnl,
            session_col=cols.session,
        )
        n_eligible = int(len(session_rets))
        sharpe_session = ann_sharpe(session_rets, periods_per_year=periods_per_year)
        sortino_session = ann_sortino(session_rets, periods_per_year=periods_per_year)
        pct_traded = (100.0 * n_active / n_eligible) if n_eligible else float("nan")
        primary_rets = session_rets

    shape = return_shape_stats(primary_rets)
    vol = ann_vol(primary_rets, periods_per_year=periods_per_year)

    trades_per_year = float("nan")
    cagr = float("nan")
    n = int(len(trades))
    if n:
        dates = pd.to_datetime(trades[cols.session])
        span_days = max((dates.max() - dates.min()).days, 1)
        trades_per_year = n / (span_days / 365.25)
        years = max(span_days / 365.25, 1e-9)
        final_eq = float(initial_capital) + float(pnl.sum())
        cagr = cagr_from_equity(float(initial_capital), final_eq, years=years)

    # Primary headline Sharpe: eligible tradable sessions when available.
    sharpe_ann = sharpe_session if eligible_sessions is not None else sharpe_active
    sortino_ann = sortino_session if eligible_sessions is not None else sortino_active
    if eligible_sessions is not None:
        definition = (
            "eligible tradable sessions (flat=0), "
            f"rf=0, {int(periods_per_year)} periods/year; skew/tails on same series"
        )
    else:
        definition = (
            "active trade days only (diagnostic fallback), "
            f"rf=0, {int(periods_per_year)} periods/year; skew/tails on same series"
        )

    out: dict[str, Any] = {
        "n_trades": blotter["n_trades"],
        "n_long": blotter["n_long"],
        "n_short": blotter["n_short"],
        "net_pnl_usd": blotter["net_pnl_usd"],
        "gross_profit": blotter["gross_profit"],
        "gross_loss": blotter["gross_loss"],
        "profit_factor": blotter["profit_factor"],
        "win_rate": blotter["win_rate"],
        "expectancy": blotter["expectancy"],
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "long_net_pnl_usd": blotter["long_net_pnl_usd"],
        "short_net_pnl_usd": blotter["short_net_pnl_usd"],
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "cagr": float(cagr),
        "cagr_pct": float(cagr * 100.0) if cagr == cagr else float("nan"),
        "ann_vol": float(vol),
        "sharpe_ann": float(sharpe_ann),
        "sharpe_session_ann": float(sharpe_session),
        "sharpe_active_day_ann": float(sharpe_active),
        "sortino_ann": float(sortino_ann),
        "sortino_session_ann": float(sortino_session),
        "skew": float(shape["skew"]),
        "lower_tail_ratio": float(shape["lower_tail_ratio"]),
        "upper_tail_ratio": float(shape["upper_tail_ratio"]),
        "trades_per_year": float(trades_per_year),
        "n_eligible_sessions": n_eligible,
        "n_active_sessions": n_active,
        "pct_sessions_traded": float(pct_traded),
        "sharpe_definition": definition,
        "periods_per_year": float(periods_per_year),
    }
    return out


def summarize_window(
    trades: pd.DataFrame,
    *,
    date_from: date,
    date_to: date,
    eligible_sessions: Sequence[date | pd.Timestamp | str] | None = None,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    session_col: str | None = None,
    side_col: str | None = None,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Like summarize_trades but CAGR uses the declared calendar window."""

    summary = summarize_trades(
        trades,
        eligible_sessions=eligible_sessions,
        initial_capital=initial_capital,
        pnl_col=pnl_col,
        session_col=session_col,
        side_col=side_col,
        periods_per_year=periods_per_year,
    )
    if trades is None or trades.empty:
        return summary
    years = window_years(date_from, date_to)
    try:
        cols = TradeColumns.infer(trades, pnl_col=pnl_col, session_col=session_col)
        pnl = pd.to_numeric(trades[cols.pnl], errors="coerce").fillna(0.0)
    except ValueError:
        return summary
    final_eq = float(initial_capital) + float(pnl.sum())
    cagr = cagr_from_equity(float(initial_capital), final_eq, years=years)
    summary["cagr"] = float(cagr)
    summary["cagr_pct"] = float(cagr * 100.0) if cagr == cagr else float("nan")
    summary["window_years"] = float(years)
    return summary


def summarize_equity_curve(
    equity: pd.Series,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> dict[str, Any]:
    """Headline metrics for continuous equity-curve strategies."""

    from qik.metrics.returns import equity_curve_returns
    from qik.metrics.risk import calmar, max_drawdown_from_equity

    rets = equity_curve_returns(equity)
    eq = equity.dropna().astype(float)
    if len(eq) < 2:
        return _empty_summary()
    mdd = max_drawdown_from_equity(eq)
    n = len(rets)
    years = max(n / periods_per_year, 1e-9)
    cagr = cagr_from_equity(float(eq.iloc[0]), float(eq.iloc[-1]), years=years)
    sharpe = ann_sharpe(rets, periods_per_year=periods_per_year)
    sortino = ann_sortino(rets, periods_per_year=periods_per_year)
    shape = return_shape_stats(rets)
    out = _empty_summary()
    out.update(
        {
            "cagr": float(cagr),
            "cagr_pct": float(cagr * 100.0) if cagr == cagr else float("nan"),
            "ann_vol": float(ann_vol(rets, periods_per_year=periods_per_year)),
            "max_drawdown_pct": float(mdd),
            "sharpe_ann": float(sharpe),
            "sharpe_session_ann": float("nan"),
            "sharpe_active_day_ann": float("nan"),
            "sortino_ann": float(sortino),
            "sortino_session_ann": float("nan"),
            "skew": float(shape["skew"]),
            "lower_tail_ratio": float(shape["lower_tail_ratio"]),
            "upper_tail_ratio": float(shape["upper_tail_ratio"]),
            "calmar": calmar(float(cagr) if cagr == cagr else 0.0, float(mdd)),
            "n_active_sessions": int(n),
            "sharpe_definition": (
                f"equity-curve daily returns, rf=0, {int(periods_per_year)} periods/year; "
                "skew/tails on same series"
            ),
            "periods_per_year": float(periods_per_year),
        }
    )
    return out


__all__ = [
    "StrategySummary",
    "summarize_equity_curve",
    "summarize_trades",
    "summarize_window",
]
