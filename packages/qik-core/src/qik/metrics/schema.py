"""Trade-log column contracts and alias resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_PNL_ALIASES = (
    "net_pnl_usd",
    "pnl_usd",
    "pnl_scaled_usd",
    "net_usd",
    "pnl",
    "profit",
)
_SESSION_ALIASES = (
    "session_date",
    "exit_date",
    "exit_ts",
    "exit_time",
    "date",
)
_SIDE_ALIASES = ("side", "direction")
_RET_ALIASES = ("net_ret", "return", "ret", "pnl_pct")


def resolve_col(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    explicit: str | None = None,
) -> str:
    """Pick the first matching column, or ``explicit`` if provided."""

    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"column {explicit!r} not in trade log; have {list(df.columns)}")
        return explicit
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"none of {candidates} found in trade log columns: {list(df.columns)}")


@dataclass(frozen=True)
class TradeColumns:
    """Resolved column names for a trade blotter."""

    pnl: str
    session: str
    side: str | None = None
    ret: str | None = None

    @classmethod
    def infer(
        cls,
        trades: pd.DataFrame,
        *,
        pnl_col: str | None = None,
        session_col: str | None = None,
        side_col: str | None = None,
        ret_col: str | None = None,
    ) -> TradeColumns:
        pnl = resolve_col(trades, _PNL_ALIASES, pnl_col)
        session = resolve_col(trades, _SESSION_ALIASES, session_col)
        side: str | None
        if side_col is not None:
            side = resolve_col(trades, _SIDE_ALIASES, side_col)
        else:
            side = next((c for c in _SIDE_ALIASES if c in trades.columns), None)
        ret: str | None
        if ret_col is not None:
            ret = resolve_col(trades, _RET_ALIASES, ret_col)
        else:
            ret = next((c for c in _RET_ALIASES if c in trades.columns), None)
        return cls(pnl=pnl, session=session, side=side, ret=ret)


__all__ = ["TradeColumns", "resolve_col"]
