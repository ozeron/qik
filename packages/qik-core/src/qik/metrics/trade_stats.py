"""Trade-level blotter statistics (PF, win rate, expectancy, side splits)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from qik.metrics.schema import TradeColumns


def profit_factor(pnl: pd.Series) -> float:
    """Gross wins / |gross losses|; inf if wins>0 and no losses; nan if empty."""

    if pnl.empty:
        return float("nan")
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    if gross_loss < 0.0:
        return gross_profit / abs(gross_loss)
    if gross_profit > 0.0:
        return float("inf")
    return float("nan")


def win_rate(pnl: pd.Series) -> float:
    if pnl.empty:
        return float("nan")
    return float((pnl > 0.0).mean())


def expectancy(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    return float(pnl.mean())


def gross_profit_loss(pnl: pd.Series) -> tuple[float, float]:
    wins = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    return (
        float(wins.sum()) if len(wins) else 0.0,
        float(losses.sum()) if len(losses) else 0.0,
    )


def trade_blotter_stats(
    trades: pd.DataFrame,
    *,
    pnl_col: str | None = None,
    session_col: str | None = None,
    side_col: str | None = None,
) -> dict[str, Any]:
    """Column-agnostic trade blotter scalars (no Sharpe / CAGR)."""

    empty: dict[str, Any] = {
        "n_trades": 0,
        "n_long": 0,
        "n_short": 0,
        "net_pnl_usd": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": float("nan"),
        "win_rate": float("nan"),
        "expectancy": 0.0,
        "long_net_pnl_usd": 0.0,
        "short_net_pnl_usd": 0.0,
    }
    if trades is None or trades.empty:
        return empty

    cols = TradeColumns.infer(
        trades,
        pnl_col=pnl_col,
        session_col=session_col,
        side_col=side_col,
    )
    pnl = pd.to_numeric(trades[cols.pnl], errors="coerce").fillna(0.0)
    gp, gl = gross_profit_loss(pnl)
    n = int(len(trades))
    n_long = 0
    n_short = 0
    long_net = 0.0
    short_net = 0.0
    if cols.side is not None:
        side = trades[cols.side].astype(str).str.lower()
        long_mask = side.isin(("long", "1", "buy"))
        short_mask = side.isin(("short", "-1", "sell"))
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        long_net = float(pnl[long_mask].sum())
        short_net = float(pnl[short_mask].sum())

    return {
        "n_trades": n,
        "n_long": n_long,
        "n_short": n_short,
        "net_pnl_usd": float(pnl.sum()),
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": float(profit_factor(pnl)),
        "win_rate": win_rate(pnl),
        "expectancy": expectancy(pnl),
        "long_net_pnl_usd": long_net,
        "short_net_pnl_usd": short_net,
        "_pnl_col": cols.pnl,
        "_session_col": cols.session,
        "_side_col": cols.side,
    }


__all__ = [
    "expectancy",
    "gross_profit_loss",
    "profit_factor",
    "trade_blotter_stats",
    "win_rate",
]
