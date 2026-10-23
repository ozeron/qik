"""High-level API of the vector engine: validated inputs in, one trade table out.

    bars = Bars(open, high, low, close, index=ts)
    sig = Signals(entry_side, exit, stop=levels, target=levels)      # (n,) or (n, k) arrays
    res = simulate(bars, sig, instrument=Instrument(tick=0.25, point_value=50.0),
                   costs=Costs(commission_per_side=2.5, slippage_ticks=1),
                   sizing=Sizing(risk_fraction=0.01), initial_cash=100_000)
    res.trades        # one row per closed trade, all k configs, column ``config``
    res.summary()     # one row per config: qik.metrics headline stats

Everything is explicit: nothing is rounded, clipped or defaulted for you. Inputs the engine cannot
execute the way Nautilus would (levels off the tick grid, a stop at or through the entry close,
non-finite prices, inconsistent OHLC) raise ``ValueError`` naming the config and the bar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import date
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd

from . import kernel as K
from . import sizing as S
from ._checks import is_int
from .sizing import Sizing

__all__ = [
    "SIGNAL_FIELDS",
    "Bars",
    "Compliance",
    "Costs",
    "Instrument",
    "FillModel",
    "FillTiming",
    "MarketFill",
    "Profile",
    "Result",
    "Rules",
    "Signals",
    "Sizing",
    "require_nautilus",
    "simulate",
]

REASON_NAMES = {
    K.REASON_SIGNAL: "signal",
    K.REASON_STOP: "stop",
    K.REASON_TARGET: "target",
    K.REASON_STOP_GAP: "stop_gap",
    K.REASON_PROTECT_FAIL: "protect_fail",
}


class Profile(StrEnum):
    NAUTILUS = "nautilus"
    TRADINGVIEW = "tradingview"


class Compliance(StrEnum):
    NAUTILUS = "nautilus"
    REPRO_TRADINGVIEW = "repro:tradingview"


class FillModel(StrEnum):
    DEFAULT = "default"
    ONE_TICK = "one_tick"


class FillTiming(StrEnum):
    CLOSE = "close"
    NEXT_OPEN = "next_open"


# Backwards-compatible spelling, including enum identity checks.
MarketFill = FillTiming


@dataclass(frozen=True)
class Instrument:
    tick: float
    point_value: float

    def __post_init__(self) -> None:
        if not (self.tick > 0 and self.point_value > 0):
            raise ValueError("tick and point_value must be positive")


@dataclass(frozen=True)
class Costs:
    """Applied after execution: every fill moves ``slippage_ticks`` against us, then commission."""

    commission_per_side: float = 0.0
    slippage_ticks: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_per_side", "slippage_ticks"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and non-negative, got {v!r}")


@dataclass(frozen=True)
class Rules:
    """Execution switches (docs/vector-engine.md, A6, A9, A13, A16-A20 and repro-only R1).

    ``stale_book`` decides what happens when a new entry's stop lands inside the quote a same-bar
    limit fill left stale in Nautilus' L1 book: ``"error"`` (default) refuses the input,
    ``"nautilus"`` reproduces Nautilus: entry fills, stop rejected, linked target cancelled, the
    position runs with no levels. The second is a venue-simulation artifact, not market behaviour.

    NEXT_OPEN is available only under Profile.TRADINGVIEW, for external-author reproduction;
    it requires timezone-aware Bars.index and excludes limit-then-market entries.
    ``latency_bars=1`` delays each command until the next bar's settlement. ``ONE_TICK``
    changes venue liquidity and needs Instrument.tick. Neither combines with NEXT_OPEN.
    """

    adaptive_high_low_ordering: bool = False  # Nautilus venue flag of the same name
    reenter_on_exit_bar: bool = True  # may a new entry follow an exit on the same bar's close
    stale_book: Literal["error", "nautilus"] = "error"
    limit_chase_bars: int = 1  # LimitThenMarket.chase_after_bars for Signals.entry_limit (A15)
    # a limit entry's stop at or through its fill bar's close: "error", or "flatten" as V2 does
    protect_rejected: Literal["error", "flatten"] = "error"
    profile: Profile = Profile.NAUTILUS
    market_fill: MarketFill = MarketFill.CLOSE
    entry_fill: FillTiming | None = None
    exit_fill: FillTiming | None = None
    latency_bars: Literal[0, 1] = 0
    fill_model: FillModel = FillModel.DEFAULT
    protection: Literal["callback", "oto"] = "callback"
    target_order: Literal["limit", "mit"] = "limit"
    amend_rejected: Literal["error", "keep"] = "error"
    stop_update_mode: Literal["any", "ratchet"] = "any"
    stop_entry_in_market: Literal["error", "skip"] = "error"

    def __post_init__(self) -> None:
        for name in ("entry_fill", "exit_fill"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FillTiming):
                raise ValueError(f"{name} must be a FillTiming enum")
        next_open = FillTiming.NEXT_OPEN in (self.effective_entry_fill, self.effective_exit_fill)
        for name, choices in (
            ("protection", ("callback", "oto")),
            ("target_order", ("limit", "mit")),
            ("amend_rejected", ("error", "keep")),
            ("stop_update_mode", ("any", "ratchet")),
            ("stop_entry_in_market", ("error", "skip")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        if not is_int(self.latency_bars) or self.latency_bars not in (0, 1):
            raise ValueError("latency_bars must be the integer 0 or 1")
        if not isinstance(self.fill_model, FillModel):
            raise ValueError("fill_model must be a FillModel enum")
        if next_open and (self.latency_bars or self.fill_model is not FillModel.DEFAULT):
            raise ValueError("NEXT_OPEN cannot mix with latency_bars or a venue fill_model")
        if not isinstance(self.profile, Profile):
            raise ValueError(f"profile must be a Profile enum, got {self.profile!r}")
        if not isinstance(self.market_fill, MarketFill):
            raise ValueError(f"market_fill must be a MarketFill enum, got {self.market_fill!r}")
        if next_open and self.profile is not Profile.TRADINGVIEW:
            raise ValueError("MarketFill.NEXT_OPEN requires Profile.TRADINGVIEW (repro only)")
        if self.stale_book not in ("error", "nautilus"):
            raise ValueError(f"stale_book must be 'error' or 'nautilus', got {self.stale_book!r}")
        if self.protect_rejected not in ("error", "flatten"):
            raise ValueError(
                f"protect_rejected must be 'error' or 'flatten', got {self.protect_rejected!r}"
            )
        if not is_int(self.limit_chase_bars) or self.limit_chase_bars < 1:
            raise ValueError(
                f"limit_chase_bars must be an integer >= 1, got {self.limit_chase_bars!r}"
            )

    @property
    def effective_entry_fill(self) -> FillTiming:
        return self.market_fill if self.entry_fill is None else self.entry_fill

    @property
    def effective_exit_fill(self) -> FillTiming:
        return self.market_fill if self.exit_fill is None else self.exit_fill


@dataclass(frozen=True)
class Bars:
    """OHLC arrays, one row per bar, in time order. Prices must be finite, positive and consistent.

    ``volume`` is needed only to size above one contract (the liquidity check, ``sizing.py``).
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    index: pd.DatetimeIndex | None = None
    volume: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(a, dtype=np.float64) for a in (self.open, self.high, self.low, self.close)
        ]
        n = arrays[0].shape[0]
        if any(a.ndim != 1 or a.shape[0] != n for a in arrays) or n == 0:
            raise ValueError("open/high/low/close must be equal-length non-empty 1-D arrays")
        o, h, lo, c = arrays
        if not all(np.isfinite(a).all() for a in arrays) or not all((a > 0).all() for a in arrays):
            raise ValueError("prices must be finite and positive")
        bad = np.flatnonzero((h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (h < lo))
        if bad.size:
            raise ValueError(
                f"inconsistent OHLC at bar {int(bad[0])}: need low <= open,close <= high"
            )
        if self.index is not None and len(self.index) != n:
            raise ValueError("index length must equal the number of bars")
        if self.volume is not None:
            v = np.ascontiguousarray(self.volume, dtype=np.float64)
            if v.shape != (n,) or not np.isfinite(v).all() or (v < 0).any():
                raise ValueError(
                    "volume must be a finite, non-negative array with one value per bar"
                )
            object.__setattr__(self, "volume", v)
        for name, a in zip(("open", "high", "low", "close"), arrays, strict=True):
            object.__setattr__(self, name, np.ascontiguousarray(a))

    def __len__(self) -> int:
        return int(self.open.shape[0])


def _as_2d(a: Any, n: int, dtype: Any, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(f"{name} must have shape (n,) or (n, k) with n={n}, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class Signals:
    """What the strategy decides, one column per config.

    ``entry_side[i]`` (+1 long, -1 short, 0 none) is a market entry decided at bar ``i``'s close,
    taken only when flat. ``exit[i]`` decides a market exit there while in a position. Markets fill
    at that close by default, or the next open under repro-only R1. ``stop[i]`` and
    ``target[i]`` are absolute price levels placed with the entry decided on bar ``i`` (NaN = none): a
    resting stop-market and a resting limit, live from bar ``i + 1``, cancelling each other (R1:
    live during that bar after the entry open, with point offsets measured from that open).

    ``entry_limit[i]`` turns that entry into the book's ``LimitThenMarket``: a resting limit at that
    price, chased with a market order at the close ``Rules.limit_chase_bars`` bars later if still
    unfilled (A15). ``stop_points`` / ``target_points`` give a level as a distance from the entry
    instead (``Stop.points``): from the limit price when the limit fills, from the fill otherwise.
    Each leg is either a level or points, never both. A limit entry needs a stop (V2 sends no
    protection without one) and ``Bars.index`` (a limit still working when the New York date
    changes is refused).

    ``entry_stop`` is a DAY stop-market entry (A18). ``stop_update`` / ``target_update`` amend
    existing legs at this close (A19); NaN keeps the level. ``breakeven_after_points`` and
    ``breakeven_offset_points`` are fixed at entry and request a one-shot stop amendment (A19b).
    """

    entry_side: np.ndarray
    exit: np.ndarray
    stop: np.ndarray | None = None
    target: np.ndarray | None = None
    entry_limit: np.ndarray | None = None
    stop_points: np.ndarray | None = None
    target_points: np.ndarray | None = None
    entry_stop: np.ndarray | None = None
    stop_update: np.ndarray | None = None
    target_update: np.ndarray | None = None
    breakeven_after_points: np.ndarray | None = None
    breakeven_offset_points: np.ndarray | None = None


SIGNAL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Signals))


@dataclass
class Result:
    trades: pd.DataFrame  # closed trades, all configs
    open_positions: pd.DataFrame  # one row per config still in a position after the last bar
    n_bars: int
    n_configs: int
    index: pd.DatetimeIndex | None = None  # the bars' timestamps
    initial_cash: float | None = None
    compliance: Compliance = Compliance.NAUTILUS
    research_only: bool = False

    @property
    def deployable(self) -> bool:
        """V2 has no stop-entry/amend/MIT API; compliant research can still be undeployable."""

        return self.compliance is Compliance.NAUTILUS and not self.research_only

    def summary(
        self,
        *,
        eligible_sessions: Sequence[date | pd.Timestamp | str] | None = None,
        session_tz: str = "America/New_York",
        periods_per_year: float | None = None,
    ) -> pd.DataFrame:
        """Headline stats per config from ``qik.metrics.summarize_trades`` (``stats.py``)."""

        from .stats import summarize

        out = summarize(
            self,
            eligible_sessions=eligible_sessions,
            session_tz=session_tz,
            periods_per_year=periods_per_year,
        )
        out["compliance"] = self.compliance.value
        out["deployable"] = self.deployable
        return out


def require_nautilus(res: Result) -> None:
    """Refuse reproduction results at a research/selection boundary."""

    if res.compliance is not Compliance.NAUTILUS:
        raise ValueError(f"Nautilus-compliant result required; got {res.compliance}")


def _check_tick(levels: np.ndarray, rows: np.ndarray, tick: float, name: str) -> None:
    fin = rows & np.isfinite(levels)
    if not fin.any():
        return
    steps = levels[fin] / tick
    off = np.abs(steps - np.round(steps)) > 1e-6
    if off.any():
        where = np.argwhere(fin)[np.flatnonzero(off)[0]]
        raise ValueError(
            f"{name} level {levels[tuple(where)]!r} (bar {int(where[0])}, config {int(where[1])}) "
            f"is off the {tick} tick grid; round it in the strategy"
        )


def simulate(
    bars: Bars,
    signals: Signals,
    *,
    instrument: Instrument | None = None,
    rules: Rules = Rules(),
    costs: Costs | None = None,
    sizing: Sizing | None = None,
    initial_cash: float | None = None,
) -> Result:
    """Run every config under the selected execution profile; defaults are Nautilus-compatible."""

    if rules.fill_model is FillModel.ONE_TICK and instrument is None:
        raise ValueError("FillModel.ONE_TICK needs Instrument(tick=..., point_value=...)")
    n = len(bars)
    side = _as_2d(signals.entry_side, n, np.int8, "entry_side")
    k = side.shape[1]
    exit_ = _as_2d(signals.exit, n, np.bool_, "exit")
    nan = np.full((n, k), np.nan)

    def opt(a: np.ndarray | None, name: str) -> np.ndarray:
        return nan if a is None else _as_2d(a, n, np.float64, name)

    stop = opt(signals.stop, "stop")
    target = opt(signals.target, "target")
    limit = opt(signals.entry_limit, "entry_limit")
    stop_pts = opt(signals.stop_points, "stop_points")
    target_pts = opt(signals.target_points, "target_points")
    slot_names = (
        "entry_stop",
        "stop_update",
        "target_update",
        "breakeven_after_points",
        "breakeven_offset_points",
    )
    slot_arrays = tuple(opt(getattr(signals, name), name) for name in slot_names)
    trigger, sup, tup, after, offset = slot_arrays
    slot_used = tuple(a is not nan and bool(np.isfinite(a).any()) for a in slot_arrays)
    slots = rules.target_order == "mit" or rules.protection == "oto" or any(slot_used)
    levels = (stop, target, limit, stop_pts, target_pts, *slot_arrays)
    next_open = rules.effective_entry_fill is FillTiming.NEXT_OPEN
    exit_next_open = rules.effective_exit_fill is FillTiming.NEXT_OPEN
    if next_open and np.isfinite(limit).any():
        raise ValueError("NEXT_OPEN does not support entry_limit / limit-then-market entries")
    if not all(a.shape[1] == k for a in (exit_, *levels)):
        raise ValueError("all signal arrays must have the same number of configs")
    if not np.isin(side, (-1, 0, 1)).all():
        raise ValueError("entry_side must be -1, 0 or +1")
    if any(np.isinf(a).any() for a in levels if a is not nan):
        raise ValueError("levels must be finite or NaN")
    entries = side != 0
    if (
        slot_used[0]
        and limit is not nan
        and np.any(entries & np.isfinite(trigger) & np.isfinite(limit))
    ):
        raise ValueError("an entry cannot specify both entry_stop and entry_limit")
    for a, name in zip(slot_arrays[:4], slot_names[:4], strict=True):
        if a is not nan and np.any(a <= 0):
            raise ValueError(f"{name} must be positive or NaN")
    if slot_used[4] and np.any(entries & np.isfinite(offset) & np.isnan(after)):
        raise ValueError("breakeven_offset_points needs breakeven_after_points at the entry")
    if slot_used[3] and np.any(entries & np.isfinite(after) & np.isnan(stop) & np.isnan(stop_pts)):
        raise ValueError("breakeven_after_points needs an entry stop")
    if slots and rules.protection == "oto":
        if np.isfinite(limit).any():
            raise ValueError("OTO order slots do not support entry_limit / chase; use callback")
        if np.any(entries & (np.isfinite(stop_pts) | np.isfinite(target_pts))):
            raise ValueError(
                "OTO order slots need absolute protection; use callback for point offsets"
            )
    for lvl, pts, leg in ((stop, stop_pts, "stop"), (target, target_pts, "target")):
        both = np.argwhere(entries & np.isfinite(lvl) & np.isfinite(pts))
        if both.size:
            i, j = (int(x) for x in both[0])
            raise ValueError(f"bar {i}, config {j}: {leg} is given as a level and as points")
        neg = np.argwhere(entries & (pts <= 0))
        if neg.size:
            i, j = (int(x) for x in neg[0])
            raise ValueError(f"bar {i}, config {j}: {leg}_points must be positive")
    if costs is not None and instrument is None:
        raise ValueError("costs need an instrument (tick, point_value)")
    if instrument is not None and costs is None:
        raise ValueError("USD P&L needs costs: pass costs=Costs(...) (Costs() for none)")
    if sizing is not None and instrument is None:
        raise ValueError("sizing needs an instrument (point_value)")
    if initial_cash is not None and not (np.isfinite(initial_cash) and initial_cash > 0):
        raise ValueError(f"initial_cash must be positive, got {initial_cash!r}")
    if sizing is not None and sizing.risk_fraction is not None and initial_cash is None:
        raise ValueError("Sizing(risk_fraction=...) needs initial_cash")
    if instrument is not None:
        for a, name in zip(
            levels,
            ("stop", "target", "entry_limit", "stop_points", "target_points", *slot_names),
            strict=True,
        ):
            if a is nan:
                continue
            rows = np.ones_like(entries) if name in ("stop_update", "target_update") else entries
            _check_tick(a, rows, instrument.tick, name)

    new_day = np.zeros(n, np.bool_)
    if next_open or exit_next_open:
        new_day = _new_york_days(bars, "NEXT_OPEN")
    elif np.isfinite(limit).any():
        new_day = _limit_entry_checks(bars, entries, limit, stop, stop_pts)
    elif slot_used[0]:
        new_day = _new_york_days(bars, "stop entries")
    t = np.ascontiguousarray
    slot_data: tuple[np.ndarray | None, ...] = (None,) * 5
    if slots:
        absent = t(nan.T)
        slot_data = tuple(absent if a is nan else t(a.T) for a in slot_arrays)
    out = K.run_columns(
        bars.open, bars.high, bars.low, bars.close,
        t(side.T), t(exit_.T), t(stop.T), t(target.T), t(limit.T), t(stop_pts.T), t(target_pts.T),
        new_day, int(rules.limit_chase_bars), rules.adaptive_high_low_ordering,
        rules.reenter_on_exit_bar, rules.stale_book == "nautilus",
        rules.protect_rejected == "flatten", next_open,
        rules.latency_bars, rules.fill_model is FillModel.ONE_TICK,
        0.0 if instrument is None else instrument.tick,
        entry_trigger=slot_data[0], stop_updates=slot_data[1], target_updates=slot_data[2],
        be_after=slot_data[3], be_offset=slot_data[4],
        mit=rules.target_order == "mit", oto=rules.protection == "oto",
        amend_keep=rules.amend_rejected == "keep", ratchet=rules.stop_update_mode == "ratchet",
        stop_skip=rules.stop_entry_in_market == "skip",
        exit_next_open=exit_next_open,
    )  # fmt: skip
    (counts, _, errs, err_bar, final_side, final_entry, final_px, final_stop,
     e_i, x_i, sd, e_px, x_px, why, trade_stop) = out  # fmt: skip
    bad = np.flatnonzero(errs)
    if bad.size:
        j = int(bad[0])
        eb = int(err_bar[j])
        price = float(bars.open[eb] if next_open else bars.close[eb])
        _raise_kernel_error(int(errs[j]), eb, j, price, next_open=next_open)
    cfg = np.repeat(np.arange(k, dtype=np.int64), counts)
    trades = pd.DataFrame(
        {
            "config": cfg,
            "entry_i": e_i,
            "exit_i": x_i,
            "side": sd.astype(np.int64),
            "entry_px": e_px,
            "exit_px": x_px,
            "reason": pd.Categorical(
                [REASON_NAMES[int(r)] for r in why], categories=list(REASON_NAMES.values())
            ),
            "stop": trade_stop,
        }
    )
    trades["gross_points"] = trades["side"] * (trades["exit_px"] - trades["entry_px"])
    if bars.index is not None:
        trades["entry_time"] = bars.index[trades["entry_i"].to_numpy()]
        trades["exit_time"] = bars.index[trades["exit_i"].to_numpy()]
    held = np.flatnonzero(final_side != 0)
    open_positions = pd.DataFrame(
        {
            "config": held.astype(np.int64),
            "side": final_side[held].astype(np.int64),
            "entry_i": final_entry[held].astype(np.int64),
            "entry_px": final_px[held],
            "stop": final_stop[held],
        }
    )
    if instrument is not None and costs is not None:
        c = costs
        per_contract_cost = 2.0 * (
            c.commission_per_side + c.slippage_ticks * instrument.tick * instrument.point_value
        )
        net_1 = trades["gross_points"].to_numpy() * instrument.point_value - per_contract_cost
        contracts = np.ones(len(trades), np.int64)
        open_contracts = np.ones(len(open_positions), np.int64)
        if sizing is not None:
            pv = instrument.point_value
            risk = np.abs(e_px - trade_stop) * pv
            o_cfg = open_positions["config"].to_numpy()
            o_entry = open_positions["entry_i"].to_numpy()
            o_risk = np.abs(open_positions["entry_px"].to_numpy() - final_stop[held]) * pv
            cash = float(initial_cash) if initial_cash is not None else np.nan  # risk_usd: unused
            contracts, final_equity, err, row = S.size_trades(
                cfg, net_1, risk, sizing.mode, sizing.amount, sizing.cap, cash, k
            )
            if err:
                _sizing_error(err, int(cfg[row]), int(e_i[row]))
            open_contracts = S.size_open(
                o_cfg, final_equity, o_risk, sizing.mode, sizing.amount, sizing.cap
            )
            if (open_contracts < 0).any():
                r = int(np.flatnonzero(open_contracts < 0)[0])
                _sizing_error(int(-open_contracts[r]), int(o_cfg[r]), int(o_entry[r]))
        _check_liquidity(bars, e_i, x_i, cfg, contracts)
        _check_liquidity(
            bars,
            open_positions["entry_i"].to_numpy(),
            None,
            open_positions["config"].to_numpy(),
            open_contracts,
        )
        trades["contracts"] = contracts
        trades["gross_usd"] = trades["gross_points"] * instrument.point_value * contracts
        trades["pnl_usd"] = net_1 * contracts
        if initial_cash is not None:
            trades["equity_usd"] = initial_cash + trades.groupby("config")["pnl_usd"].cumsum()
            blown = np.flatnonzero(trades["equity_usd"].to_numpy() <= 0)
            if blown.size:
                r = int(blown[0])
                raise ValueError(
                    f"config {int(cfg[r])} runs out of cash at bar {int(x_i[r])} (equity "
                    f"{trades['equity_usd'].iat[r]:.2f}): Nautilus would deny every later order; "
                    f"raise initial_cash or size smaller"
                )
        open_positions["contracts"] = open_contracts
    compliance = (
        Compliance.NAUTILUS if rules.profile is Profile.NAUTILUS else Compliance.REPRO_TRADINGVIEW
    )
    return Result(trades, open_positions, n, k, bars.index, initial_cash, compliance, slots)


def _limit_entry_checks(
    bars: Bars,
    entries: np.ndarray,
    limit: np.ndarray,
    stop: np.ndarray,
    stop_pts: np.ndarray,
) -> np.ndarray:
    """Refuse limit entries V2 would treat differently; return the bars starting a New York date."""

    rows = entries & np.isfinite(limit)
    naked = np.argwhere(rows & np.isnan(stop) & np.isnan(stop_pts))
    if naked.size:
        i, j = (int(x) for x in naked[0])
        raise ValueError(
            f"bar {i}, config {j}: a limit entry needs a stop: V2 sends no protection at all "
            f"without one (not even the target), so the engine cannot know what it holds"
        )
    return _new_york_days(bars, "limit entries")


def _new_york_days(bars: Bars, rule: str) -> np.ndarray:
    """Shared session boundary calculation for pending limits and next-open market orders."""

    if bars.index is None:
        raise ValueError(f"{rule} need timezone-aware Bars(index=...) to detect New York dates")
    stamps = pd.DatetimeIndex(bars.index)
    if stamps.tz is None:
        raise ValueError(f"Bars.index must be timezone-aware for {rule}")
    days = pd.Series(stamps.tz_convert("America/New_York")).dt.date.to_numpy()
    new_day = np.zeros(len(bars), np.bool_)
    new_day[1:] = days[1:] != days[:-1]
    return new_day


_KERNEL_ERRORS = {
    K.ERR_STOP_REPLACE_LATENCY: "stop entry replacement at bar {bar}, config {j} with latency is not pinned: cancel the old entry and wait before replacing it",
    K.ERR_STOP_ENTRY_IN_MARKET: "stop entry at bar {bar}, config {j} is in the market at submission",
    K.ERR_MIT_IN_MARKET: "MIT target at bar {bar}, config {j} is touched at submission",
    K.ERR_AMEND_REJECTED: "amendment at bar {bar}, config {j} rejected: trigger is in the market or conflicts with breakeven",
    K.ERR_AMEND_NO_ORDER: "amendment at bar {bar}, config {j} has no working order in that slot",
    K.ERR_NEXT_OPEN_SESSION_END: (
        "NEXT_OPEN decision at bar {bar}, config {j} is at session end: the next bar is absent "
        "or starts a new New York date; no cross-session market fills (docs/vector-engine.md R1)"
    ),
    K.ERR_PROTECT_REJECTED: (
        "stop for the limit entry filled at bar {bar}, config {j} is at or through that bar's "
        "close ({close}): the protection goes out after the bar, Nautilus rejects the stop and the "
        "position runs naked (docs/vector-engine.md A15). Widen the stop, place the limit further "
        "away, or take V2's answer with Rules(protect_rejected='flatten') (exit at that close)"
    ),
    K.ERR_PENDING_NEW_DAY: (
        "the DAY limit/stop entry of config {j} is still working at bar {bar}, the first bar of a new New "
        "York date: pending DAY entries across this boundary are not modelled "
        "(docs/vector-engine.md A15/A18). Exit (cancel) it before the session ends"
    ),
    K.ERR_STALE_BOOK: (
        "stop at bar {bar}, config {j} sits inside the quote left stale by a limit fill earlier in "
        "the same bar: Nautilus would reject it and cancel the linked target "
        "(docs/vector-engine.md A13). Move the stop, skip same-bar re-entry "
        "(Rules(reenter_on_exit_bar=False)), or opt in with Rules(stale_book='nautilus')"
    ),
    K.ERR_STOP_IN_MARKET: (
        "stop level at bar {bar}, config {j} is not beyond the entry close ({close}): Nautilus "
        "would reject a stop there and fill a limit there immediately; the engine refuses both"
    ),
    K.ERR_TARGET_IN_MARKET: (
        "target level at bar {bar}, config {j} is not beyond the entry close ({close}): Nautilus "
        "would reject a stop there and fill a limit there immediately; the engine refuses both"
    ),
}


def _raise_kernel_error(
    code: int, bar: int, j: int, close: float, *, next_open: bool = False
) -> None:
    if next_open and code in (K.ERR_STOP_IN_MARKET, K.ERR_TARGET_IN_MARKET):
        leg = "stop" if code == K.ERR_STOP_IN_MARKET else "target"
        raise ValueError(
            f"{leg} level at bar {bar}, config {j} is not beyond the entry fill at the open "
            f"({close}): R1 refuses already-marketable protection on the entry bar"
        )
    template = _KERNEL_ERRORS.get(code)
    if template is None:
        raise RuntimeError(f"kernel returned unknown error code {code} (bar {bar}, config {j})")
    raise ValueError(template.format(bar=bar, j=j, close=close))


def _sizing_error(err: int, config: int, bar: int) -> None:
    if err == S.ERR_NO_STOP:
        raise ValueError(
            f"entry at bar {bar}, config {config} has no stop: risk sizing needs a stop on every entry"
        )
    raise ValueError(
        f"sizing gives 0 contracts for the entry at bar {bar}, config {config}: Nautilus would not "
        f"enter, which changes every later trade of the config; raise the risk, set fixed "
        f"contracts, or tighten the stop"
    )


def _check_liquidity(
    bars: Bars,
    entry_i: np.ndarray,
    exit_i: np.ndarray | None,
    cfg: np.ndarray,
    contracts: np.ndarray,
) -> None:
    """Refuse sizes above what Nautilus fills at one price on the entry / exit bar (A14)."""

    if contracts.size == 0 or contracts.max() <= 1:
        return
    if bars.volume is None:
        raise ValueError(
            "sizing above 1 contract needs Bars(volume=...): Nautilus fills against it"
        )
    liq = S.point_liquidity(bars.volume)
    for where, idx in (("entry", entry_i), ("exit", exit_i)):
        if idx is None:
            continue
        over = np.flatnonzero(contracts > liq[idx])
        if over.size:
            r = int(over[0])
            raise ValueError(
                f"{int(contracts[r])} contracts at the {where} bar {int(idx[r])}, config {int(cfg[r])} "
                f"exceed the bar's liquidity ({int(liq[idx[r]])} = max(floor(volume/4), 1)): "
                f"Nautilus would fill the rest one tick worse (docs/vector-engine.md A14)"
            )
