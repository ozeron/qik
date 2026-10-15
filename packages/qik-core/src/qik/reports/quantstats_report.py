"""Convert trade logs / equity curves to daily returns and QuantStats HTML tearsheets.

Shared QuantStats tearsheet helpers for strategy backtests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Common column aliases across project trade logs.
_EXIT_COLS = ("exit_ts", "exit_date", "exit_time", "date")
_PNL_DOLLAR_COLS = ("pnl_usd", "pnl_scaled_usd", "pnl", "profit")
_PNL_RET_COLS = ("net_ret", "return", "ret", "pnl_pct")


def _resolve_col(df: pd.DataFrame, candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"column {explicit!r} not in trade log; have {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"none of {candidates} found in trade log columns: {list(df.columns)}")


def trades_to_daily_returns(
    trades: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
    pnl_col: str | None = None,
    exit_col: str | None = None,
    mode: str | None = None,
    session_index: pd.DatetimeIndex | list | None = None,
) -> pd.Series:
    """
    Build a daily return series from a trade log (PnL attributed to exit date).

    Modes (auto-detected if ``mode`` is None):
      - ``dollar``: sum PnL dollars on exit day, convert to return vs prior equity
        (``pnl_usd``).
      - ``return``: compound fractional ``net_ret`` on exit day.

    Grid:
      - If ``session_index`` is provided, reindex to that eligible-session grid
        (flat sessions → 0). Preferred for discrete RTH strategies.
      - Otherwise calendar-fills every day between first and last activity
        (legacy; includes weekends).
    """
    if trades.empty:
        return pd.Series(dtype=float, name="Strategy")

    t = trades.copy()
    exit_name = _resolve_col(t, _EXIT_COLS, exit_col)
    t["_exit"] = pd.to_datetime(t[exit_name]).dt.normalize()

    if mode is None:
        if pnl_col is not None:
            mode = "return" if pnl_col in _PNL_RET_COLS else "dollar"
        elif any(c in t.columns for c in _PNL_DOLLAR_COLS):
            mode = "dollar"
        elif any(c in t.columns for c in _PNL_RET_COLS):
            mode = "return"
        else:
            raise ValueError(f"cannot infer PnL column; pass pnl_col=. columns={list(t.columns)}")

    def _grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if session_index is not None:
            return (
                pd.DatetimeIndex(pd.to_datetime(list(session_index)))
                .normalize()
                .unique()
                .sort_values()
            )
        return pd.date_range(index.min(), index.max(), freq="D")

    if mode == "dollar":
        col = _resolve_col(t, _PNL_DOLLAR_COLS, pnl_col)
        daily_pnl = t.groupby("_exit")[col].sum().sort_index()
        full_idx = _grid(daily_pnl.index)
        daily_pnl = daily_pnl.reindex(full_idx, fill_value=0.0)
        equity = initial_capital + daily_pnl.cumsum()
        prior_equity = equity.shift(1).fillna(initial_capital)
        returns = daily_pnl / prior_equity
    elif mode == "return":
        col = _resolve_col(t, _PNL_RET_COLS, pnl_col)
        # Compound full capital on each exit; multiple exits same day multiply.
        daily_ret = t.groupby("_exit")[col].apply(lambda s: float((1.0 + s).prod() - 1.0))
        daily_ret = daily_ret.sort_index()
        full_idx = _grid(daily_ret.index)
        daily_ret = daily_ret.reindex(full_idx, fill_value=0.0)
        returns = daily_ret
    else:
        raise ValueError(f"mode must be 'dollar' or 'return', got {mode!r}")

    returns.index = pd.DatetimeIndex(returns.index).tz_localize(None)
    returns.name = "Strategy"
    return returns


def equity_curve_to_daily_returns(
    equity: pd.Series | pd.DataFrame,
    *,
    equity_col: str = "equity",
) -> pd.Series:
    """Daily simple returns from a mark-to-market equity curve (trading-day index)."""
    if isinstance(equity, pd.DataFrame):
        if equity_col not in equity.columns:
            raise ValueError(f"equity_col {equity_col!r} not in {list(equity.columns)}")
        s = equity[equity_col].copy()
    else:
        s = equity.copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None)
    s = s.sort_index()
    # Collapse duplicate dates (keep last)
    s = s[~s.index.duplicated(keep="last")]
    returns = s.pct_change().fillna(0.0)
    returns.name = "Strategy"
    return returns


def generate_quantstats_html(
    returns: pd.Series,
    output: Path,
    *,
    title: str,
    benchmark: str | bool | None = None,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> Path:
    """Write QuantStats full HTML tearsheet to output."""
    import quantstats as qs

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    qs.extend_pandas()
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=str(output),
        title=title,
        rf=rf,
        periods_per_year=periods_per_year,
    )
    return output


def write_report_index(
    reports: list[tuple[str, Path]],
    output: Path,
    *,
    title: str = "Strategy Reports",
    subtitle: str = "QuantStats tearsheets generated from backtest trade logs.",
    meta: dict[str, str] | None = None,
) -> Path:
    """Simple index page linking to per-strategy QuantStats HTML reports."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = ""
    if meta:
        meta_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in meta.items())

    links = "".join(f'<li><a href="{path.name}">{label}</a></li>' for label, path in reports)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ font-size: 1.4rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }}
    th, td {{ text-align: left; padding: 0.35rem 0.75rem; border-bottom: 1px solid #ddd; }}
    th {{ color: #555; font-weight: 600; }}
    ul {{ line-height: 1.8; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  {"<table>" + meta_rows + "</table>" if meta_rows else ""}
  <h2>Reports</h2>
  <ul>{links}</ul>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output


def _as_float(value: Any) -> float:
    if hasattr(value, "iloc"):
        return float(value.iloc[-1])
    return float(value)


def summary_stats(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    *,
    initial_capital: float = 100_000.0,
) -> dict[str, str]:
    """Lightweight stats for index page meta table."""
    import quantstats as qs

    if returns.empty:
        return {"trades": "0", "total return": "n/a"}

    out: dict[str, str] = {}
    if trades is not None and len(trades):
        out["trades"] = str(len(trades))
        if "pnl_usd" in trades.columns:
            out["total PnL"] = f"${float(trades['pnl_usd'].sum()):,.0f}"
        elif "net_ret" in trades.columns:
            compound = float((1.0 + trades["net_ret"]).prod() - 1.0)
            out["total PnL"] = f"${initial_capital * compound:,.0f}"
            out["compound return"] = f"{compound:.1%}"

    out["cumulative return"] = f"{_as_float(qs.stats.comp(returns)):.1%}"
    out["sharpe"] = f"{_as_float(qs.stats.sharpe(returns)):.2f}"
    out["max drawdown"] = f"{_as_float(qs.stats.max_drawdown(returns)):.1%}"
    return out
