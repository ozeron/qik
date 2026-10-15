"""Closed-trade blotter contract shared by producers (book, projects) and prop.

One place for column names, units and the legacy-name mapping, so producers and the
eval engine cannot drift. Canonical unit-size export columns are ``BLOTTER_COLUMNS``;
``normalize`` reads any producer's frame (legacy names included) into the minimal
form prop consumes: ``trade_date``, ``pnl`` and, when present, ``mae_usd`` and
``exit_ts``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import pandas as pd

# Canonical export written by ``book.live.blotter`` (unit size, USD, net of costs).
BLOTTER_COLUMNS = (
    "entry_ts",
    "exit_ts",
    "trade_date",
    "side",
    "qty",
    "entry_px",
    "exit_px",
    "gross_usd",
    "cost_usd",
    "pnl_usd",
    "mae_usd",
)

# Legacy producers (projects/*) name the date and pnl columns differently. Order is
# the match priority and is frozen: changing it changes which column a blotter uses.
DATE_CANDIDATES = ("trade_date", "session_date", "date", "exit_date")
PNL_CANDIDATES = (
    "pnl",
    "pnl_usd",
    "net_pnl_usd",
    "realized_pnl",
    "net_pnl",
    "pnl_points",
)
MAE_COLUMN = "mae_usd"
EXIT_TS_COLUMN = "exit_ts"


def pick_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    raise ValueError(
        f"blotter missing {label} column; tried {list(candidates)}; have {list(columns)}"
    )


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``trade_date`` (date), ``pnl`` (float) [+ ``mae_usd`` <= 0, ``exit_ts``].

    ``frame.attrs['pnl_column']`` records which source column was used. A
    ``pnl_points`` source is **not** converted here (units live in
    ``book.prop.blotter.blotter_to_ledger(pnl_unit=, point_value=)``); it warns because a
    points blotter read as dollars silently shrinks every size and floor check.
    """
    if frame.empty:
        raise ValueError("blotter is empty")
    cols = list(frame.columns)
    date_col = pick_column(cols, DATE_CANDIDATES, "date")
    pnl_col = pick_column(cols, PNL_CANDIDATES, "pnl")
    if pnl_col.lower() == "pnl_points":
        warnings.warn(
            "blotter pnl column is 'pnl_points': pass pnl_unit='points' and point_value "
            "to blotter_to_ledger, or the points will be read as dollars",
            UserWarning,
            stacklevel=3,
        )
    out = frame[[date_col, pnl_col]].copy()
    out.columns = ["trade_date", "pnl"]
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    out["pnl"] = out["pnl"].astype(float)
    lower = {c.lower(): c for c in cols}
    if MAE_COLUMN in lower:
        out[MAE_COLUMN] = frame[lower[MAE_COLUMN]].astype(float).clip(upper=0.0)
    if EXIT_TS_COLUMN in lower:
        out[EXIT_TS_COLUMN] = pd.to_datetime(frame[lower[EXIT_TS_COLUMN]], utc=True)
        out = out.sort_values(EXIT_TS_COLUMN, kind="stable")
    out = out.reset_index(drop=True)
    out.attrs["pnl_column"] = pnl_col
    return out


def validate_export(frame: pd.DataFrame) -> None:
    """Producer-side check: a frame is a canonical unit-size export."""
    missing = [c for c in BLOTTER_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"blotter export missing columns {missing}")
    if (frame["mae_usd"] > 0).any():
        raise ValueError("mae_usd must be <= 0 (worst open-trade P&L)")


__all__ = [
    "BLOTTER_COLUMNS",
    "DATE_CANDIDATES",
    "EXIT_TS_COLUMN",
    "MAE_COLUMN",
    "PNL_CANDIDATES",
    "normalize",
    "pick_column",
    "validate_export",
]
