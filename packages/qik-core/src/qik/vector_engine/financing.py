"""A25: opt-in, post-run carry on immutable fixed-size fills (no matcher hooks).

Signed rates are percentages of entry notional, in the result's USD currency:
positive credits, negative debits. Calendars, conversions and rates are caller data.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from .api import Result
from .sizing import Sizing

_DAY_NS = 86_400_000_000_000


def _clock(value: pd.Timestamp, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{name} must be a finite timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _index(values: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is None or idx.hasnans or not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError(f"{name} must be timezone-aware, increasing, unique and finite")
    return idx.tz_convert("UTC").as_unit("ns")


@dataclass(frozen=True)
class SideRates:
    """Signed percentages, separately for long and short; no annualization implicit."""

    long: float
    short: float

    def __post_init__(self) -> None:
        if not np.isfinite([self.long, self.short]).all():
            raise ValueError("side rates must be finite")


@dataclass(frozen=True)
class DatedRates:
    """Piecewise-constant side rates, effective at each aware table index instant.

    Columns are ``long`` and ``short``. Coverage is [first instant, until]. A
    rollover exactly at ``until`` is covered; elapsed intervals end there.
    """

    table: pd.DataFrame
    until: pd.Timestamp

    def __post_init__(self) -> None:
        table = self.table.copy(deep=True)
        table.index = _index(pd.DatetimeIndex(table.index), "dated rates")
        if table.empty or set(table.columns) != {"long", "short"}:
            raise ValueError("dated rates need nonempty long/short columns")
        if not np.isfinite(table.to_numpy(dtype=float)).all():
            raise ValueError("dated rates must be finite")
        until = _clock(self.until, "rates until")
        if until < table.index[-1]:
            raise ValueError("rates until precedes the last effective rate")
        object.__setattr__(self, "table", table.astype(float))
        object.__setattr__(self, "until", until)

    def check(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        if start < self.table.index[0] or end > self.until:
            raise ValueError("dated rates do not cover the holding interval")

    def at(self, stamp: pd.Timestamp, side: int) -> float:
        row = pd.DatetimeIndex(self.table.index).asi8.searchsorted(stamp.value, side="right") - 1
        return float(self.table.iloc[row]["long" if side == 1 else "short"])


@dataclass(frozen=True)
class RolloverEvents:
    """Explicit complete event calendar; entries at cutoff excluded, exit convention required.

    Rates are percent per event (supply annual rates / 365 yourself if needed).
    Multipliers include Wednesday x3 when that is the caller's convention.
    Coverage endpoints certify that no events were omitted, even in a gap.
    """

    times: pd.DatetimeIndex
    rates: SideRates | DatedRates
    multipliers: float | tuple[float, ...]
    coverage_start: pd.Timestamp
    coverage_end: pd.Timestamp
    include_exit: bool

    def __post_init__(self) -> None:
        times = _index(self.times, "rollover events")
        start = _clock(self.coverage_start, "coverage start")
        end = _clock(self.coverage_end, "coverage end")
        if end < start or (len(times) and (times[0] < start or times[-1] > end)):
            raise ValueError("rollover events outside calendar coverage")
        if not isinstance(self.include_exit, bool):
            raise ValueError("include_exit must explicitly be True or False")
        mult = np.broadcast_to(np.asarray(self.multipliers, dtype=float), (len(times),)).copy()
        if not np.isfinite(mult).all() or (mult < 0).any():
            raise ValueError("event multipliers must be finite and nonnegative")
        if not isinstance(self.rates, (SideRates, DatedRates)):
            raise TypeError("rates must be SideRates or DatedRates")
        if isinstance(self.rates, DatedRates):
            self.rates.check(start, end)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "multipliers", tuple(mult))
        object.__setattr__(self, "coverage_start", start)
        object.__setattr__(self, "coverage_end", end)


@dataclass(frozen=True)
class ElapsedDays:
    """Signed percent/day x entry notional x actual UTC seconds/86400; close posting.

    Weekends and fractional days accrue. Open inventory posts through valuation_end
    without liquidation. Dated rates split the integral at each effective instant.
    """

    rates: SideRates | DatedRates

    def __post_init__(self) -> None:
        if not isinstance(self.rates, (SideRates, DatedRates)):
            raise TypeError("rates must be SideRates or DatedRates")


@dataclass(frozen=True)
class CarryResult:
    """Research-only valuation, never a V2 execution policy.

    ``result`` retains all source bytes when every carry amount is zero. Otherwise
    pnl_usd includes closed carry and carry_usd is added. The curve is settled P&L
    plus chronological carry postings, not a price mark-to-market curve. Terminal
    carry is already in the curve; do not add it a second time.
    """

    result: Result
    trade_carry: pd.Series
    ledger: pd.DataFrame
    equity: pd.DataFrame
    terminal_open_carry: pd.Series

    @property
    def deployable(self) -> bool:
        return False


def _postings(
    policy: RolloverEvents | ElapsedDays,
    start: pd.Timestamp,
    end: pd.Timestamp,
    side: int,
    notional: float,
    terminal: bool,
) -> list[tuple[pd.Timestamp, float]]:
    rates = policy.rates
    if isinstance(rates, DatedRates):
        rates.check(start, end)

    def rate_at(t: pd.Timestamp) -> float:
        return (
            rates.at(t, side)
            if isinstance(rates, DatedRates)
            else (rates.long if side == 1 else rates.short)
        )

    if isinstance(policy, RolloverEvents):
        if start < policy.coverage_start or end > policy.coverage_end:
            raise ValueError("rollover calendar does not cover the holding interval")
        lo = policy.times.asi8.searchsorted(start.value, side="right")
        hi = policy.times.asi8.searchsorted(
            end.value, side="right" if terminal or policy.include_exit else "left"
        )
        multipliers = cast(tuple[float, ...], policy.multipliers)
        return [
            (
                policy.times[i],
                notional * rate_at(policy.times[i]) / 100 * multipliers[i],
            )
            for i in range(lo, hi)
        ]
    cuts = [start]
    if isinstance(rates, DatedRates):
        cuts += list(rates.table.index[(rates.table.index > start) & (rates.table.index < end)])
    cuts.append(end)
    amount = sum(
        notional * rate_at(a) / 100 * (b.value - a.value) / _DAY_NS for a, b in zip(cuts, cuts[1:])
    )
    return [(end, amount)]


def apply_financing(
    result: Result,
    policy: RolloverEvents | ElapsedDays,
    *,
    sizing: Sizing,
    contract_size: float,
    valuation_end: pd.Timestamp,
) -> CarryResult:
    """Value fixed quantities after matching; never feed carry back to Decisions.

    Supply the actual run's sizing and contract multiplier, in USD per price unit.
    All financial feedback is refused, even on an empty run or with zero rates.
    Callers must not have read equity/P&L to choose their strategy's decisions.
    Source results and State are never mutated. Reapplying to a financed result
    refuses, preventing double booking.
    """
    if sizing.contracts is None:
        raise ValueError(
            "D1 financing requires fixed Sizing.contracts; equity-dependent sizing such as "
            "risk_fraction needs D2 (the in-loop account); other computed sizing is unsupported"
        )
    if not isinstance(policy, (RolloverEvents, ElapsedDays)):
        raise TypeError("financing policy must be RolloverEvents or ElapsedDays")
    if "carry_usd" in result.trades or "_financing_applied" in result.__dict__:
        raise ValueError("financing already applied; use the original result")
    if not np.isfinite(contract_size) or contract_size <= 0:
        raise ValueError("contract_size must be finite and positive in USD per price unit")
    horizon = _clock(valuation_end, "valuation_end")
    if result.index is None:
        raise ValueError("financing requires result.index with aware clocks")
    source = result
    source_index = result.index
    result = copy(result)
    # Pandas clock operations can populate/infer metadata even on read-only data.
    # Work on copies so zero financing preserves the complete source serialization.
    result.index = source_index.copy(deep=True)
    result.trades = result.trades.copy(deep=True)
    result.open_positions = result.open_positions.copy(deep=True)
    clocks = _index(result.index, "result.index")
    if len(clocks) and horizon < clocks[-1]:
        raise ValueError("valuation_end precedes the last result clock")
    if result.initial_cash is None or not np.isfinite(result.initial_cash):
        raise ValueError("financing requires finite initial_cash for the equity curve")
    rows: list[dict[str, object]] = []
    carries = np.zeros(len(result.trades))
    terminal_carry = np.zeros(result.n_configs)
    for terminal, table in ((False, result.trades), (True, result.open_positions)):
        for row_id, (_, row) in enumerate(table.iterrows()):
            config = int(row["config"])
            if (
                config != row["config"]
                or not 0 <= config < result.n_configs
                or row["side"] not in (-1, 1)
            ):
                raise ValueError("invalid exposure config or side")
            qty = float(row["contracts"])
            if qty != float(sizing.contracts):
                raise ValueError("result quantity does not match the supplied fixed sizing")
            if not np.isfinite(row["entry_px"]) or row["entry_px"] <= 0:
                raise ValueError("financing requires finite positive entry prices")
            start = _clock(
                row["entry_time"] if "entry_time" in row else clocks[int(row["entry_i"])],
                "entry_time",
            )
            end = horizon if terminal else _clock(row["exit_time"], "exit_time")
            if end < start or end > horizon:
                raise ValueError("invalid holding interval or exit beyond valuation_end")
            postings = _postings(
                policy, start, end, row["side"], qty * contract_size * row["entry_px"], terminal
            )
            amount = sum(v for _, v in postings)
            if not np.isfinite(amount):
                raise ValueError("nonfinite carry amount")
            if terminal:
                terminal_carry[config] += amount
            else:
                carries[row_id] = amount
            for stamp, value in postings:
                rows.append(
                    dict(
                        config=config,
                        position_id=row_id,
                        terminal=terminal,
                        time=stamp,
                        side=int(row["side"]),
                        contracts=qty,
                        entry_px=float(row["entry_px"]),
                        carry_usd=value,
                    )
                )
    ledger = pd.DataFrame(
        rows,
        columns=pd.Index(
            [
                "config",
                "position_id",
                "terminal",
                "time",
                "side",
                "contracts",
                "entry_px",
                "carry_usd",
            ]
        ),
    )
    ledger = ledger.astype(
        {
            "config": "int64",
            "position_id": "int64",
            "terminal": "bool",
            "side": "int64",
            "contracts": "float64",
            "entry_px": "float64",
            "carry_usd": "float64",
        }
    )
    ledger["time"] = pd.to_datetime(ledger.time, utc=True)
    if not ledger.empty:
        ledger = ledger.sort_values(
            ["time", "config", "terminal", "position_id"], kind="stable"
        ).reset_index(drop=True)
    curve_times = clocks.union(pd.DatetimeIndex([horizon]))
    if not ledger.empty:
        curve_times = curve_times.union(pd.DatetimeIndex(ledger.time))
    if not result.trades.empty:
        curve_times = curve_times.union(pd.DatetimeIndex(result.trades.exit_time).tz_convert("UTC"))
    curve_times = curve_times.unique().sort_values()
    curve_rows = []
    for config in range(result.n_configs):
        trades = result.trades[result.trades.config == config]
        base = pd.Series(trades.pnl_usd.to_numpy(), index=pd.DatetimeIndex(trades.exit_time))
        base = base.groupby(level=0).sum().reindex(curve_times, fill_value=0).cumsum()
        carry = ledger[ledger.config == config]
        amounts = pd.Series(
            carry.carry_usd.to_numpy(dtype=float), index=pd.DatetimeIndex(carry.time, tz="UTC")
        )
        amounts = amounts.groupby(level=0).sum().reindex(curve_times, fill_value=0).cumsum()
        curve_rows.append(
            pd.DataFrame(
                dict(
                    config=config,
                    time=curve_times,
                    net_usd=base.to_numpy() + amounts.to_numpy(),
                    carry_usd=amounts.to_numpy(),
                    equity_usd=result.initial_cash + base.to_numpy() + amounts.to_numpy(),
                )
            )
        )
    equity = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    out = source
    if (ledger.carry_usd != 0).any():
        out = copy(result)
        out.trades = result.trades.copy(deep=True)
        out.open_positions = result.open_positions.copy(deep=True)
        if "_states" in result.__dict__:  # shallow copy would alias the mutable state records
            out.states = result.states.copy()
        out.trades["carry_usd"] = carries
        out.trades["pnl_usd"] = out.trades.pnl_usd + carries
        if "equity_usd" in out.trades:
            keys = pd.MultiIndex.from_arrays([out.trades.config, out.trades.exit_time])
            out.trades["equity_usd"] = (
                equity.set_index(["config", "time"]).equity_usd.reindex(keys).to_numpy()
            )
        out.research_only = True
        out.__dict__["_financing_applied"] = True
    return CarryResult(
        out,
        pd.Series(carries, index=result.trades.index, name="carry_usd"),
        ledger,
        equity,
        pd.Series(
            terminal_carry,
            name="terminal_open_carry",
            index=pd.RangeIndex(result.n_configs, name="config"),
        ),
    )
