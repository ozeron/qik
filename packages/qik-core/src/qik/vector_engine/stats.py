"""Per-config headline stats: ``qik.metrics.summarize_trades`` on each config's trades.

No metric is computed here; this module only shapes the trade table into what ``qik.metrics`` reads
(USD P&L, side, session date) and collects one row per config. The session of a trade is the date
of its exit in ``session_tz``. The eligible-session grid defaults to every session date in the bars,
so a config's flat sessions count as zero-return sessions in its Sharpe.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from qik.metrics import DEFAULT_PERIODS_PER_YEAR, summarize_trades

if TYPE_CHECKING:
    from .api import Result

__all__ = ["summarize"]


def _session_dates(ts: pd.DatetimeIndex | pd.Series, tz: str) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(ts)
    if stamps.tz is not None:
        stamps = stamps.tz_convert(tz)
    return pd.DatetimeIndex(pd.Series(stamps).dt.date)


def summarize(
    result: Result,
    *,
    eligible_sessions: Sequence[date | pd.Timestamp | str] | None = None,
    session_tz: str = "America/New_York",
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """One row per config (index ``config``, all ``n_configs`` rows, no-trade configs included)."""

    if "pnl_usd" not in result.trades.columns:
        raise ValueError("summary needs USD P&L: pass instrument= to simulate")
    if result.index is None:
        raise ValueError("summary needs session dates: pass Bars(index=...) to simulate")
    if result.initial_cash is None:
        raise ValueError("summary needs the capital returns are measured on: pass initial_cash=")
    sessions = (
        list(_session_dates(result.index, session_tz).unique())
        if eligible_sessions is None
        else eligible_sessions
    )
    trades = result.trades
    frame = pd.DataFrame(
        {
            "config": trades["config"].to_numpy(),
            "pnl_usd": trades["pnl_usd"].to_numpy(),
            "side": trades["side"].to_numpy(),
            "session_date": _session_dates(result.index[trades["exit_i"].to_numpy()], session_tz),
        }
    )
    ppy = DEFAULT_PERIODS_PER_YEAR if periods_per_year is None else periods_per_year
    by_config = dict(tuple(frame.groupby("config", sort=True)))
    empty = frame.iloc[:0]
    rows = [
        summarize_trades(
            by_config.get(j, empty),
            eligible_sessions=sessions,
            initial_capital=result.initial_cash,
            pnl_col="pnl_usd",
            session_col="session_date",
            side_col="side",
            periods_per_year=ppy,
        )
        for j in range(result.n_configs)
    ]
    out = pd.DataFrame(rows)
    out.index.name = "config"
    return out
