"""Engine-independent export parsing and deterministic, one-to-one trade comparison.

Unknown export fields stay null. Timestamps retain their exported timezone (or
lack of one); callers must explicitly align clocks before comparing frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qik.metrics import profit_factor, win_rate
from qik.trial_storage import file_ref

COLUMNS = ("entry_ts", "exit_ts", "side", "qty", "entry_px", "exit_px", "pnl", "reason")


@dataclass(frozen=True)
class TradeFrame:
    """Canonical trades; source carries provenance, clock and P&L units.

    Row order is significant: matching follows the reference oracle's first-unused candidate
    policy. Missing qty/P&L/reason are permitted; an open trade has null exit clock
    and price. Entry clock and price are required.
    Extra input columns are retained. Side is +1 (long) or -1 (short).
    """

    trades: pd.DataFrame
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        frame = self.trades.copy(deep=True).reset_index(drop=True)
        required = {"entry_ts", "exit_ts", "side", "entry_px", "exit_px"}
        if missing := required - set(frame):
            raise ValueError(f"missing canonical trade fields: {sorted(missing)}")
        for name in COLUMNS:
            if name not in frame:
                frame[name] = None
        for name in ("entry_ts", "exit_ts"):
            frame[name] = pd.to_datetime(frame[name], errors="raise")
            if not pd.api.types.is_datetime64_any_dtype(frame[name]) or (
                name == "entry_ts" and frame[name].isna().any()
            ):
                raise ValueError(f"{name} needs one consistent clock and no nulls")
        if str(frame.entry_ts.dt.tz) != str(frame.exit_ts.dt.tz):
            raise ValueError("entry/exit clocks differ; align explicitly")
        if (frame.exit_ts < frame.entry_ts).any():
            raise ValueError("exit precedes entry")
        frame["side"] = frame.side.map(lambda value: {"long": 1, "short": -1}.get(value, value))
        if not frame.side.isin([1, -1]).all():
            raise ValueError("side must be long/short or +1/-1")
        for name in ("entry_px", "exit_px", "qty", "pnl"):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
            known = frame[name].dropna()
            if not np.isfinite(known.to_numpy(dtype=float)).all():
                raise ValueError(f"nonfinite {name}")
            if name == "entry_px" and frame[name].isna().any():
                raise ValueError(f"missing {name}")
        if not frame.exit_ts.isna().equals(frame.exit_px.isna()):
            raise ValueError("open trade requires both exit_ts and exit_px null")
        if (frame.qty.dropna() <= 0).any():
            raise ValueError("qty must be positive when exported")
        object.__setattr__(self, "trades", frame)
        object.__setattr__(self, "source", dict(self.source))


def _parse_list(path: Path, *, platform: str, utc: bool) -> TradeFrame:
    before = file_ref(path)
    raw = pd.read_csv(path)
    clocks = [name for name in ("Date/Time", "Date and time") if name in raw]
    if len(clocks) != 1:
        raise ValueError("expected one exported clock: Date/Time or Date and time")
    raw = raw.rename(columns={clocks[0]: "Date/Time"})
    required = {"Trade #", "Type", "Date/Time", "Price USD", "Signal"}
    if missing := required - set(raw):
        raise ValueError(f"unsupported {platform} export; missing {sorted(missing)}")
    if raw[list(required - {"Signal"})].isna().any().any():
        raise ValueError("null trade id, type, clock, price or signal")
    valid = raw.Type.isin(["Entry long", "Entry short", "Exit long", "Exit short"])
    if not valid.all():
        raise ValueError("unsupported trade type (open trades are not closed trades)")
    ent = raw[raw.Type.str.startswith("Entry")].set_index("Trade #")
    ext = raw[raw.Type.str.startswith("Exit")].set_index("Trade #")
    if not ent.index.is_unique or not ext.index.is_unique or set(ent.index) != set(ext.index):
        raise ValueError("expected exactly one entry and exit per trade id")
    ext = ext.reindex(ent.index)
    if not ent.Type.str.split().str[1].equals(ext.Type.str.split().str[1]):
        raise ValueError("entry/exit sides differ")
    frame = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(ent["Date/Time"], utc=utc),
            "exit_ts": pd.to_datetime(ext["Date/Time"], utc=utc),
            "side": ent.Type.str.split().str[1],
            "entry_px": ent["Price USD"],
            "exit_px": ext["Price USD"],
            "reason": ext.Signal,
            "trade_id": ent.index,
        }
    )
    # Never derive P&L from prices: point value, commissions and sizing are unknown.
    for aliases, canonical in (
        (("Position size (qty)", "Contracts"), "qty"),
        (("Net P&L USD", "Profit USD"), "pnl"),
    ):
        available = [name for name in aliases if name in raw]
        if len(available) > 1:
            raise ValueError(f"ambiguous exported {canonical} fields: {available}")
        if available:
            frame[canonical] = ext[available[0]]
    if file_ref(path) != before:
        raise ValueError("export changed while parsing")
    return TradeFrame(
        frame, {**before, "platform": platform, "pnl_unit": "USD" if "pnl" in frame else None}
    )


def parse_tradingview_list_of_trades(path: str | Path) -> TradeFrame:
    """English Strategy Tester CSV, USD price/P&L headers, explicit entry/exit rows.

    Naive exported wall times stay naive. Other currencies/locales must be
    explicitly adapted by the caller, not guessed from prices or the host zone.
    """
    return _parse_list(Path(path), platform="tradingview", utc=False)


def parse_pyne(path: str | Path) -> TradeFrame:
    """PyneCore list-of-trades CSV; Date/Time is the runtime's UTC clock."""
    return _parse_list(Path(path), platform="pyne", utc=True)


@dataclass(frozen=True)
class Tolerance:
    bars: int = 1
    bar_duration: pd.Timedelta = cast(pd.Timedelta, pd.Timedelta(minutes=1))
    time: pd.Timedelta = cast(pd.Timedelta, pd.Timedelta(0))
    ticks: float = 2
    tick_size: float = 0.25
    entry_price_required: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.bars, bool) or not isinstance(self.bars, int) or self.bars < 0:
            raise ValueError("bars must be a nonnegative integer")
        for name in ("bar_duration", "time"):
            value = pd.Timedelta(getattr(self, name))
            if (
                not isinstance(value, pd.Timedelta)
                or value < pd.Timedelta(0)
                or (name == "bar_duration" and value == pd.Timedelta(0))
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)
        if (
            not np.isfinite([self.ticks, self.tick_size]).all()
            or self.ticks < 0
            or self.tick_size <= 0
        ):
            raise ValueError("ticks must be nonnegative and tick_size positive")

    @property
    def clock(self) -> pd.Timedelta:
        """Bars converted to elapsed time, plus the explicit additional tolerance."""
        return self.bars * self.bar_duration + self.time


@dataclass(frozen=True)
class MatchReport:
    pairs: pd.DataFrame
    only_ours: pd.DataFrame
    only_theirs: pd.DataFrame
    deltas: pd.DataFrame
    counts: Mapping[str, int]
    verdict_inputs: Mapping[str, Any]
    sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _stats(frame: TradeFrame) -> dict[str, Any]:
    t = frame.trades
    known = not t.empty and t.pnl.notna().all()
    pf = profit_factor(t.pnl) if known else None
    window = frame.source.get("window")
    rate = None
    if window is not None:
        start, end = (pd.Timestamp(value) for value in window)
        if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp) or end < start:
            raise ValueError("source window must be ordered")
        if (
            not t.empty
            and ((t.entry_ts.dt.date < start.date()) | (t.entry_ts.dt.date > end.date())).any()
        ):
            raise ValueError("trade outside declared source window")
        rate = len(t) * 365.25 / ((end - start).days + 1)
    return {
        "trades": len(t),
        "win_rate": win_rate(t.pnl) if known else None,
        "profit_factor": pf if pf is not None and not np.isnan(pf) else None,
        "yearly_net": t.groupby(t.entry_ts.dt.year).pnl.sum().to_dict() if known else None,
        "pnl_unit": frame.source.get("pnl_unit"),
        "window": frame.source.get("window"),
        "trades_per_year": rate,
    }


def match_trades(ours: TradeFrame, theirs: TradeFrame, *, tol: Tolerance) -> MatchReport:
    """Pure first-unused matching by side, entry clock and optional entry price.

    Exit deltas never hide an entry pair. ``matched`` counts paired entries;
    ``within_tolerance`` additionally requires exit clock and both prices. Unknown
    qty/P&L/reason produce null deltas, not zero evidence. This reports gate
    inputs, never a MATCHED verdict: exposure windows, bootstrap PF CI and
    author-year coverage remain caller-owned.
    """
    a, b = ours.trades, theirs.trades
    if not a.empty and not b.empty and ((a.entry_ts.dt.tz is None) != (b.entry_ts.dt.tz is None)):
        raise ValueError("naive/aware clocks differ; align explicitly")
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for i, row in a.iterrows():
        candidates = b[(b.side == row.side) & ((b.entry_ts - row.entry_ts).abs() <= tol.clock)]
        if tol.entry_price_required:
            candidates = candidates[
                (candidates.entry_px - row.entry_px).abs() <= tol.ticks * tol.tick_size
            ]
        candidates = candidates[~candidates.index.isin(used)]
        if candidates.empty:
            continue
        j = int(candidates.index[0])
        used.add(j)
        other = b.loc[j]
        ds = {"ours": int(cast(int, i)), "theirs": j}
        exact_entry = row.entry_ts == other.entry_ts and row.entry_px == other.entry_px
        same_exit = (
            not pd.isna(row.exit_ts)
            and not pd.isna(other.exit_ts)
            and row.exit_ts == other.exit_ts
            and abs(row.exit_px - other.exit_px) < 1e-9
        )
        ok = (
            not pd.isna(row.exit_ts)
            and not pd.isna(other.exit_ts)
            and abs(row.exit_ts - other.exit_ts) <= tol.clock
        )
        for name in COLUMNS:
            x, y = row[name], other[name]
            unknown = pd.isna(x) or pd.isna(y)
            delta = None if unknown or name == "reason" else x - y
            if name.endswith("_ts"):
                delta = delta.total_seconds() if delta is not None else None
                within = None if unknown else abs(x - y) <= tol.clock
            elif name.endswith("_px"):
                within = None if unknown else abs(x - y) <= tol.ticks * tol.tick_size
                ok = ok and bool(within)
            else:
                within = None if unknown else bool(x == y)
            deltas.append(
                {
                    **ds,
                    "field": name,
                    "ours_value": x,
                    "theirs_value": y,
                    "delta": delta,
                    "within_tolerance": within,
                }
            )
        pairs.append(
            {
                **ds,
                "exact_entry": bool(exact_entry),
                "same_exit": bool(same_exit),
                "within_tolerance": bool(ok),
            }
        )
    paired = pd.DataFrame(
        pairs, columns=pd.Index(["ours", "theirs", "exact_entry", "same_exit", "within_tolerance"])
    )
    only_a = a.loc[~a.index.isin(paired.ours)].copy()
    only_b = b.loc[~b.index.isin(used)].copy()
    counts = {
        "ours": len(a),
        "theirs": len(b),
        "matched": len(pairs),
        "only_ours": len(only_a),
        "only_theirs": len(only_b),
        **{
            name: int(paired[name].sum())
            for name in ("exact_entry", "same_exit", "within_tolerance")
        },
    }
    return MatchReport(
        paired,
        only_a,
        only_b,
        pd.DataFrame(
            deltas,
            columns=pd.Index(
                [
                    "ours",
                    "theirs",
                    "field",
                    "ours_value",
                    "theirs_value",
                    "delta",
                    "within_tolerance",
                ]
            ),
        ),
        counts,
        {
            "ours": _stats(ours),
            "theirs": _stats(theirs),
            "trade_count_ratio": len(a) / len(b) if len(b) else None,
            "entry_match_fraction": len(pairs) / len(b) if len(b) else None,
        },
        {"ours": ours.source, "theirs": theirs.source},
    )
