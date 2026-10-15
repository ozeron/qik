"""Return-series builders with explicit grid conventions.

Grids
-----
``eligible_session``
    Official for discrete RTH/session strategies. Index = eligible sessions;
    flat sessions get 0 return (weekends/holidays excluded when not in index).
``active_day``
    Diagnostic only — sessions with ≥1 trade.
``equity_curve``
    Continuous mark-to-market / daily long-flat strategies.
``calendar``
    Legacy QuantStats path — fill every calendar day between first and last
    activity (includes weekends). Labelled; not primary for freeze gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

import pandas as pd

from qik.metrics.schema import TradeColumns, resolve_col

ReturnGrid = Literal["eligible_session", "active_day", "equity_curve", "calendar"]

_PNL_DOLLAR = (
    "net_pnl_usd",
    "pnl_usd",
    "pnl_scaled_usd",
    "net_usd",
    "pnl",
    "profit",
)
_PNL_RET = ("net_ret", "return", "ret", "pnl_pct")
_EXIT = ("session_date", "exit_date", "exit_ts", "exit_time", "date")


def to_session_index(sessions: Sequence[date | pd.Timestamp | str]) -> pd.DatetimeIndex:
    """Normalize a sequence of session dates to a unique sorted DatetimeIndex."""

    if not sessions:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))).normalize().unique().sort_values()
    return pd.DatetimeIndex(idx)


def eligible_sessions_from_frame(
    frame: pd.DataFrame,
    *,
    session_col: str = "session_date",
) -> list[date]:
    """Sorted unique session dates from a feature or bar frame."""

    if frame.empty or session_col not in frame.columns:
        return []
    dates = pd.to_datetime(frame[session_col]).dt.normalize().drop_duplicates().sort_values()
    return [ts.date() for ts in dates]


def session_pnl(
    trades: pd.DataFrame,
    eligible_sessions: Sequence[date | pd.Timestamp | str],
    *,
    pnl_col: str | None = None,
    session_col: str | None = None,
) -> pd.Series:
    """Daily PnL on the eligible-session grid (missing sessions → 0)."""

    index = to_session_index(eligible_sessions)
    if index.empty:
        return pd.Series(dtype=float, name="session_pnl")
    if trades.empty:
        return pd.Series(0.0, index=index, name="session_pnl")
    cols = TradeColumns.infer(trades, pnl_col=pnl_col, session_col=session_col)
    frame = trades.copy()
    frame["_session"] = pd.to_datetime(frame[cols.session]).dt.normalize()
    daily = frame.groupby("_session", sort=True)[cols.pnl].sum()
    return daily.reindex(index, fill_value=0.0).astype(float).rename("session_pnl")


def _pnl_to_returns(daily_pnl: pd.Series, *, initial_capital: float) -> pd.Series:
    if daily_pnl.empty:
        return pd.Series(dtype=float, name="Strategy")
    equity = float(initial_capital) + daily_pnl.astype(float).cumsum()
    prior = equity.shift(1).fillna(float(initial_capital))
    rets = (daily_pnl.astype(float) / prior).astype(float)
    rets.index = pd.DatetimeIndex(rets.index).tz_localize(None)
    rets.name = "Strategy"
    return rets


def session_returns(
    trades: pd.DataFrame,
    eligible_sessions: Sequence[date | pd.Timestamp | str],
    *,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    session_col: str | None = None,
) -> pd.Series:
    """Simple returns on eligible sessions vs prior allocated equity."""

    pnl = session_pnl(
        trades,
        eligible_sessions,
        pnl_col=pnl_col,
        session_col=session_col,
    )
    return _pnl_to_returns(pnl, initial_capital=initial_capital)


def active_day_returns(
    trades: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    session_col: str | None = None,
) -> pd.Series:
    """Returns only on sessions with ≥1 trade (diagnostic; not for freeze gates)."""

    if trades.empty:
        return pd.Series(dtype=float, name="Strategy")
    cols = TradeColumns.infer(trades, pnl_col=pnl_col, session_col=session_col)
    frame = trades.copy()
    frame["_session"] = pd.to_datetime(frame[cols.session]).dt.normalize()
    daily = frame.groupby("_session", sort=True)[cols.pnl].sum().astype(float)
    return _pnl_to_returns(daily, initial_capital=initial_capital)


def equity_curve_returns(
    equity: pd.Series | pd.DataFrame,
    *,
    equity_col: str = "equity",
) -> pd.Series:
    """Daily simple returns from a mark-to-market equity curve."""

    if isinstance(equity, pd.DataFrame):
        if equity_col not in equity.columns:
            raise ValueError(f"equity_col {equity_col!r} not in {list(equity.columns)}")
        s = equity[equity_col].copy()
    else:
        s = equity.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    returns = s.pct_change().fillna(0.0)
    returns.name = "Strategy"
    return returns


def calendar_returns(
    trades: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    exit_col: str | None = None,
    mode: str | None = None,
) -> pd.Series:
    """Calendar-day returns (weekends zero-filled). Legacy QuantStats path."""

    if trades.empty:
        return pd.Series(dtype=float, name="Strategy")

    t = trades.copy()
    exit_name = resolve_col(t, _EXIT, exit_col)
    t["_exit"] = pd.to_datetime(t[exit_name]).dt.normalize()

    if mode is None:
        if pnl_col is not None:
            mode = "return" if pnl_col in _PNL_RET else "dollar"
        elif any(c in t.columns for c in _PNL_DOLLAR):
            mode = "dollar"
        elif any(c in t.columns for c in _PNL_RET):
            mode = "return"
        else:
            raise ValueError(f"cannot infer PnL column; pass pnl_col=. columns={list(t.columns)}")

    if mode == "dollar":
        col = resolve_col(t, _PNL_DOLLAR, pnl_col)
        daily_pnl = t.groupby("_exit")[col].sum().sort_index()
        full_idx = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D")
        daily_pnl = daily_pnl.reindex(full_idx, fill_value=0.0)
        return _pnl_to_returns(daily_pnl, initial_capital=initial_capital)

    if mode == "return":
        col = resolve_col(t, _PNL_RET, pnl_col)
        daily_ret = t.groupby("_exit")[col].apply(lambda s: float((1.0 + s).prod() - 1.0))
        daily_ret = daily_ret.sort_index()
        full_idx = pd.date_range(daily_ret.index.min(), daily_ret.index.max(), freq="D")
        daily_ret = daily_ret.reindex(full_idx, fill_value=0.0).astype(float)
        daily_ret.index = pd.DatetimeIndex(daily_ret.index).tz_localize(None)
        daily_ret.name = "Strategy"
        return daily_ret

    raise ValueError(f"mode must be 'dollar' or 'return', got {mode!r}")


def build_returns(
    trades: pd.DataFrame | None = None,
    *,
    grid: ReturnGrid = "eligible_session",
    eligible_sessions: Sequence[date | pd.Timestamp | str] | None = None,
    equity: pd.Series | pd.DataFrame | None = None,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    session_col: str | None = None,
    equity_col: str = "equity",
    mode: str | None = None,
) -> pd.Series:
    """Dispatch to the appropriate return-grid builder."""

    if grid == "eligible_session":
        if eligible_sessions is None:
            raise ValueError("eligible_sessions required for grid='eligible_session'")
        if trades is None:
            raise ValueError("trades required for grid='eligible_session'")
        return session_returns(
            trades,
            eligible_sessions,
            initial_capital=initial_capital,
            pnl_col=pnl_col,
            session_col=session_col,
        )
    if grid == "active_day":
        if trades is None:
            raise ValueError("trades required for grid='active_day'")
        return active_day_returns(
            trades,
            initial_capital=initial_capital,
            pnl_col=pnl_col,
            session_col=session_col,
        )
    if grid == "equity_curve":
        if equity is None:
            raise ValueError("equity required for grid='equity_curve'")
        return equity_curve_returns(equity, equity_col=equity_col)
    if grid == "calendar":
        if trades is None:
            raise ValueError("trades required for grid='calendar'")
        return calendar_returns(
            trades,
            initial_capital=initial_capital,
            pnl_col=pnl_col,
            exit_col=session_col,
            mode=mode,
        )
    raise ValueError(f"unknown grid={grid!r}")


__all__ = [
    "ReturnGrid",
    "active_day_returns",
    "build_returns",
    "calendar_returns",
    "eligible_sessions_from_frame",
    "equity_curve_returns",
    "session_pnl",
    "session_returns",
    "to_session_index",
]
