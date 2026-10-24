"""High-level API of the vector engine: validated inputs in, one trade table out.

    bars = Bars(open, high, low, close, index=ts)
    sig = Signals(entry_side, exit, stop=levels, target=levels)      # (n,) or (n, k) arrays
    res = simulate(bars, sig, instrument=Instrument(tick=0.25, point_value=50.0),
                   costs=Costs(commission_per_side=2.5, slippage_ticks=1),
                   sizing=Sizing(risk_fraction=0.01), initial_cash=100_000)
    res.trades        # one row per closed trade, all k configs, column ``config``
    res.summary()     # one row per config: qik.metrics headline stats

Defaults never round or clip inputs; explicit repro rounding is opt-in. Inputs the engine cannot
execute the way Nautilus would (levels off the tick grid, a stop at or through the entry close,
non-finite prices, inconsistent OHLC) raise ``ValueError`` naming the config and the bar.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from datetime import date
from enum import StrEnum
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from . import account as A
from . import kernel as K
from . import sizing as S
from ._checks import is_int
from .sizing import Sizing

__all__ = [
    "SIGNAL_FIELDS",
    "Bars",
    "BracketAnchor",
    "BracketPolicy",
    "Compliance",
    "Costs",
    "Instrument",
    "FillModel",
    "FillTiming",
    "Flatten",
    "SessionBoundary",
    "TargetGap",
    "TickRound",
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
    REPRO = "repro"


class Compliance(StrEnum):
    NAUTILUS = "nautilus"
    REPRO_TRADINGVIEW = "repro:tradingview"


class _ReproCompliance(str):
    @property
    def value(self) -> str:
        return str(self)


class FillModel(StrEnum):
    DEFAULT = "default"
    ONE_TICK = "one_tick"


class FillTiming(StrEnum):
    CLOSE = "close"
    NEXT_OPEN = "next_open"


# Backwards-compatible spelling, including enum identity checks.
MarketFill = FillTiming


class BracketPolicy(StrEnum):
    STOP_FIRST = "stop_first"
    ADVERSE_FIRST = "adverse_first"


class BracketAnchor(StrEnum):
    SIGNAL = "signal"
    FILL = "fill"


class Flatten(StrEnum):
    NEXT_OPEN = "next_open"
    LAST_BAR_CLOSE = "last_bar_close"
    SESSION_END = "session_end"


class SessionBoundary(StrEnum):
    REFUSE = "refuse"
    NEXT_OBSERVED = "next_observed"
    LAST_BAR_CLOSE = "last_bar_close"


class TargetGap(StrEnum):
    AT_LIMIT = "at_limit"
    GAP_IMPROVED = "gap_improved"


class TickRound(StrEnum):
    NONE = "none"
    NEAREST = "nearest"
    UP = "up"


@dataclass(frozen=True)
class Instrument:
    tick: float
    point_value: float
    lot_step: float = field(default=1.0, kw_only=True)
    min_qty: float = field(default=1.0, kw_only=True)

    def __post_init__(self) -> None:
        if not (
            np.isfinite(self.tick)
            and np.isfinite(self.point_value)
            and self.tick > 0
            and self.point_value > 0
        ):
            raise ValueError("tick and point_value must be positive")
        for name in ("lot_step", "min_qty"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Instrument.{name} must be finite and positive")


@dataclass(frozen=True)
class Costs:
    """Post-fill valuation: adverse ticks, commission and per-side notional basis points."""

    commission_per_side: float = 0.0
    slippage_ticks: float = 0.0
    proportional_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_per_side", "slippage_ticks", "proportional_bps"):
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

    NEXT_OPEN is available under Profile.TRADINGVIEW or Profile.REPRO for author reproduction;
    it requires timezone-aware Bars.index and excludes limit-then-market entries.
    ``latency_bars=1`` delays each command until the next bar's settlement. ``ONE_TICK``
    changes venue liquidity and needs Instrument.tick. Neither combines with NEXT_OPEN.

    ``stop_combine="tighter"`` permits a level plus points on the same entry: long
    max(level, fill - points), short min(level, fill + points). ``target_combine``
    independently selects the nearer target (long min, short max). Combined offsets
    always use the actual fill, regardless of bracket_anchor; selection precedes
    tick rounding. Missing legs retain their existing anchor semantics. Marketable
    levels retain the existing rejection/fill policy. Both default to ``"refuse"``.

    ``amend_timing="close_submit"`` validates and settles stop/target amendments
    at the completed decision close, independently of the entry fill clock. It
    requires zero command latency. The default retains next-open slot timing.
    ``entry_bar_decision="completed_close"`` lets a next-open market entry that
    closes intrabar schedule a fresh entry at this close for the following open.
    ``reenter_on_exit_bar`` still gates that entry; resting-entry callbacks keep A15.
    """

    adaptive_high_low_ordering: bool = False  # Nautilus venue flag of the same name
    reenter_on_exit_bar: bool = True  # may a new entry follow an exit on the same bar's close
    stale_book: Literal["error", "nautilus"] = "error"
    limit_chase_bars: int = 1  # LimitThenMarket.chase_after_bars for Signals.entry_limit (A15)
    # a limit entry's stop at or through its fill bar's close: "error", or "flatten" as V2 does
    protect_rejected: Literal["error", "flatten"] = "error"
    profile: Profile = Profile.NAUTILUS
    author_platform: str = field(default="tradingview", kw_only=True)
    market_fill: MarketFill = MarketFill.CLOSE
    entry_fill: FillTiming | None = field(default=None, kw_only=True)
    exit_fill: FillTiming | None = field(default=None, kw_only=True)
    bracket_policy: BracketPolicy = field(default=BracketPolicy.STOP_FIRST, kw_only=True)
    bracket_anchor: BracketAnchor | None = field(default=None, kw_only=True)
    flatten: Flatten = field(default=Flatten.NEXT_OPEN, kw_only=True)
    session_boundary: SessionBoundary = field(default=SessionBoundary.REFUSE, kw_only=True)
    target_gap: TargetGap = field(default=TargetGap.AT_LIMIT, kw_only=True)
    tick_round: TickRound = field(default=TickRound.NONE, kw_only=True)
    latency_bars: Literal[0, 1] = 0
    fill_model: FillModel = FillModel.DEFAULT
    protection: Literal["callback", "oto"] = "callback"
    target_order: Literal["limit", "mit"] = "limit"
    amend_rejected: Literal["error", "keep"] = "error"
    amend_timing: Literal["next_open", "close_submit"] = field(default="next_open", kw_only=True)
    entry_bar_decision: Literal["legacy", "completed_close"] = field(default="legacy", kw_only=True)
    stop_update_mode: Literal["any", "ratchet"] = "any"
    stop_combine: Literal["refuse", "tighter"] = field(default="refuse", kw_only=True)
    target_combine: Literal["refuse", "tighter"] = field(default="refuse", kw_only=True)
    stop_entry_in_market: Literal["error", "skip"] = "error"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.author_platform, str)
            or re.fullmatch(r"[a-z][a-z0-9_-]*", self.author_platform) is None
        ):
            raise ValueError("author_platform must be a nonempty lowercase platform identifier")
        if self.profile is not Profile.REPRO and self.author_platform != "tradingview":
            raise ValueError(
                "custom author_platform requires Profile.REPRO; TRADINGVIEW is its preset"
            )
        for name in ("entry_fill", "exit_fill"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, FillTiming):
                raise ValueError(f"{name} must be a FillTiming enum")
        if not isinstance(self.bracket_policy, BracketPolicy):
            raise ValueError("bracket_policy must be a BracketPolicy enum")
        if self.bracket_anchor is not None and not isinstance(self.bracket_anchor, BracketAnchor):
            raise ValueError("bracket_anchor must be a BracketAnchor enum")
        if self.profile is Profile.NAUTILUS and (
            self.bracket_policy is not BracketPolicy.STOP_FIRST or self.bracket_anchor is not None
        ):
            raise ValueError("non-default bracket rules require a repro profile")
        for name, enum in (
            ("target_gap", TargetGap),
            ("tick_round", TickRound),
            ("flatten", Flatten),
            ("session_boundary", SessionBoundary),
        ):
            if not isinstance(getattr(self, name), enum):
                raise ValueError(f"{name} must be a {enum.__name__} enum")
        if self.profile is Profile.NAUTILUS and (
            self.target_gap is not TargetGap.AT_LIMIT
            or self.tick_round is not TickRound.NONE
            or self.flatten is not Flatten.NEXT_OPEN
            or self.session_boundary is not SessionBoundary.REFUSE
        ):
            raise ValueError("non-default target_gap/tick_round require a repro profile")
        next_open = FillTiming.NEXT_OPEN in (self.effective_entry_fill, self.effective_exit_fill)
        for name, choices in (
            ("stop_combine", ("refuse", "tighter")),
            ("target_combine", ("refuse", "tighter")),
            ("protection", ("callback", "oto")),
            ("target_order", ("limit", "mit")),
            ("amend_rejected", ("error", "keep")),
            ("amend_timing", ("next_open", "close_submit")),
            ("entry_bar_decision", ("legacy", "completed_close")),
            ("stop_update_mode", ("any", "ratchet")),
            ("stop_entry_in_market", ("error", "skip")),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        if not is_int(self.latency_bars) or self.latency_bars not in (0, 1):
            raise ValueError("latency_bars must be the integer 0 or 1")
        if self.amend_timing == "close_submit" and self.latency_bars:
            raise ValueError("close_submit requires latency_bars=0")
        if not isinstance(self.fill_model, FillModel):
            raise ValueError("fill_model must be a FillModel enum")
        if next_open and (self.latency_bars or self.fill_model is not FillModel.DEFAULT):
            raise ValueError("NEXT_OPEN cannot mix with latency_bars or a venue fill_model")
        if not isinstance(self.profile, Profile):
            raise ValueError(f"profile must be a Profile enum, got {self.profile!r}")
        if not isinstance(self.market_fill, MarketFill):
            raise ValueError(f"market_fill must be a MarketFill enum, got {self.market_fill!r}")
        if (next_open or self.market_fill is MarketFill.NEXT_OPEN) and self.profile not in (
            Profile.TRADINGVIEW,
            Profile.REPRO,
        ):
            raise ValueError(
                "MarketFill.NEXT_OPEN requires Profile.TRADINGVIEW or Profile.REPRO (repro only)"
            )
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

    session_labels: np.ndarray | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.session_labels is not None:
            labels = np.asarray(self.session_labels)
            if labels.shape != (len(self.open),) or pd.isna(labels).any():
                raise ValueError("session_labels must have one nonmissing label per bar")
            object.__setattr__(self, "session_labels", labels)
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
    flatten: np.ndarray | None = None
    exit_long: np.ndarray | None = None
    exit_short: np.ndarray | None = None
    entry_limit_ioc: np.ndarray | None = None


SIGNAL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Signals))


@dataclass
class Result[S = Any]:
    """Engine output; typed strategy runs additionally expose final state records."""

    trades: pd.DataFrame  # closed trades, all configs
    open_positions: pd.DataFrame  # one row per config still in a position after the last bar
    n_bars: int
    n_configs: int
    index: pd.DatetimeIndex | None = None  # the bars' timestamps
    initial_cash: float | None = None
    compliance: Compliance | _ReproCompliance = Compliance.NAUTILUS
    research_only: bool = False

    @property
    def states(self) -> np.recarray:
        """Final strategy states; present only on results produced by typed run."""
        if "_states" not in self.__dict__:
            raise AttributeError("states are available only for typed strategy results")
        return self.__dict__["_states"]

    @states.setter
    def states(self, value: np.recarray) -> None:
        self.__dict__["_states"] = value

    def state(self, i: int = 0) -> S:
        """Return a mutable typed view of final state i."""
        return cast(S, self.states[i])

    @property
    def deployable(self) -> bool:
        """Nautilus compliance plus a V2-supported execution and sizing contract."""

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
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
) -> Result:
    """Run every config under the selected execution profile."""
    return _simulate(
        bars,
        signals,
        instrument=instrument,
        rules=rules,
        costs=costs,
        sizing=sizing,
        initial_cash=initial_cash,
    )


def _simulate(
    bars: Bars,
    signals: Signals,
    *,
    instrument: Instrument | None = None,
    rules: Rules = Rules(),
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    _run_columns: Any = None,
) -> Result:
    """Run every config under the selected execution profile; defaults are Nautilus-compatible."""

    if bars.session_labels is not None and rules.profile is Profile.NAUTILUS:
        raise ValueError("custom session labels require a repro profile")
    if (
        signals.flatten is not None
        and np.asarray(signals.flatten).any()
        and rules.profile is Profile.NAUTILUS
    ):
        raise ValueError("flatten intents require a repro profile")
    if rules.fill_model is FillModel.ONE_TICK and instrument is None:
        raise ValueError("FillModel.ONE_TICK needs Instrument(tick=..., point_value=...)")
    n = len(bars)
    side = _as_2d(signals.entry_side, n, np.int8, "entry_side")
    k = side.shape[1]

    def exit_mask(a: np.ndarray | None, name: str) -> np.ndarray:
        return (
            np.broadcast_to(np.zeros((n, 1), np.bool_), (n, k))
            if a is None
            else _as_2d(a, n, np.bool_, name)
        )

    exit_long = exit_mask(signals.exit_long, "exit_long")
    exit_short = exit_mask(signals.exit_short, "exit_short")
    flatten = exit_mask(signals.flatten, "flatten")
    flatten_used = signals.flatten is not None and bool(flatten.any())
    exit_ = _as_2d(signals.exit, n, np.bool_, "exit")
    nan = np.full((n, k), np.nan)

    def opt(a: np.ndarray | None, name: str) -> np.ndarray:
        return nan if a is None else _as_2d(a, n, np.float64, name)

    stop = opt(signals.stop, "stop")
    target = opt(signals.target, "target")
    limit = opt(signals.entry_limit, "entry_limit")
    ioc = opt(signals.entry_limit_ioc, "entry_limit_ioc")
    ioc_used = bool(np.isfinite(ioc).any())
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
    levels = (stop, target, limit, stop_pts, target_pts, *slot_arrays, ioc)
    next_open = rules.effective_entry_fill is FillTiming.NEXT_OPEN
    exit_next_open = rules.effective_exit_fill is FillTiming.NEXT_OPEN
    if next_open and np.isfinite(limit).any():
        raise ValueError("NEXT_OPEN does not support entry_limit / limit-then-market entries")
    if not all(a.shape[1] == k for a in (exit_, exit_long, exit_short, flatten, *levels)):
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
    if np.any(entries & np.isfinite(ioc) & (np.isfinite(limit) | np.isfinite(trigger))):
        raise ValueError("entry_limit_ioc, entry_limit and entry_stop are mutually exclusive")
    if np.any(ioc <= 0):
        raise ValueError("entry_limit_ioc must be positive or NaN")
    if ioc_used and rules.protection == "oto":
        raise ValueError("entry_limit_ioc does not support OTO protection; use callback")
    if ioc_used and rules.fill_model is FillModel.ONE_TICK:
        raise ValueError("entry_limit_ioc ONE_TICK execution is unproved")
    if instrument is not None:
        _check_tick(ioc, entries, instrument.tick, "entry_limit_ioc")
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
        if rules.bracket_anchor is not BracketAnchor.FILL and np.any(
            entries
            & (
                (np.isfinite(stop_pts) & ~(np.isfinite(stop) & (rules.stop_combine == "tighter")))
                | (
                    np.isfinite(target_pts)
                    & ~(np.isfinite(target) & (rules.target_combine == "tighter"))
                )
            )
        ):
            raise ValueError(
                "OTO order slots need absolute protection; use callback for point offsets"
            )
    for lvl, pts, leg in ((stop, stop_pts, "stop"), (target, target_pts, "target")):
        both = np.argwhere(entries & np.isfinite(lvl) & np.isfinite(pts))
        if both.size and getattr(rules, f"{leg}_combine") == "refuse":
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
    if (
        sizing is None
        and instrument is not None
        and (
            instrument.min_qty > 1
            or abs(1 / instrument.lot_step - round(1 / instrument.lot_step)) > 1e-7
        )
    ):
        raise ValueError("default one-unit sizing is outside the instrument quantity grid")
    if sizing is not None and instrument is not None:
        if (
            sizing.min_qty < instrument.min_qty
            or abs(
                sizing.lot_step / instrument.lot_step - round(sizing.lot_step / instrument.lot_step)
            )
            > 1e-7
            or sizing.lot_step < instrument.lot_step
        ):
            raise ValueError("Sizing lot_step/min_qty must respect the instrument quantity grid")
    if initial_cash is not None and not (np.isfinite(initial_cash) and initial_cash > 0):
        raise ValueError(f"initial_cash must be positive, got {initial_cash!r}")
    if sizing is not None and sizing.risk_fraction is not None and initial_cash is None:
        raise ValueError("Sizing(risk_fraction=...) needs initial_cash")
    if rules.tick_round is not TickRound.NONE and instrument is None:
        raise ValueError("tick_round needs Instrument.tick")
    if instrument is not None and rules.tick_round is TickRound.NONE:
        for a, name in zip(
            levels,
            (
                "stop",
                "target",
                "entry_limit",
                "stop_points",
                "target_points",
                *slot_names,
                "entry_limit_ioc",
            ),
            strict=True,
        ):
            if a is nan:
                continue
            rows = np.ones_like(entries) if name in ("stop_update", "target_update") else entries
            _check_tick(a, rows, instrument.tick, name)

    new_day = np.zeros(n, np.bool_)
    if np.isfinite(limit).any():
        new_day = _limit_entry_checks(bars, entries, limit, stop, stop_pts)
    elif next_open or exit_next_open or ioc_used:
        new_day = _new_york_days(bars, "NEXT_OPEN")
    elif slot_used[0]:
        new_day = _new_york_days(bars, "stop entries")
    if ioc_used:
        new_day = _new_york_days(bars, "entry_limit_ioc")
    session_days = new_day.copy()
    needs_sessions = rules.flatten is Flatten.SESSION_END or (
        rules.flatten is Flatten.NEXT_OPEN and flatten_used
    )
    if bars.session_labels is not None:
        session_days = (
            np.r_[False, bars.session_labels[1:] != bars.session_labels[:-1]] if n else session_days
        )
    elif needs_sessions:
        session_days = _new_york_days(bars, "flatten")
    if bars.session_labels is not None and (next_open or exit_next_open or ioc_used):
        new_day = session_days.copy()
    ends = np.r_[session_days[1:], True] if n else np.zeros(0, bool)
    force_close: np.ndarray | None = None
    force_open: np.ndarray | None = None
    if (
        flatten_used
        or rules.flatten is not Flatten.NEXT_OPEN
        or (rules.session_boundary is SessionBoundary.LAST_BAR_CLOSE and exit_next_open)
    ):
        force_close = np.zeros((n, k), np.int8)
        force_open = np.zeros((n, k), bool)
        if rules.session_boundary is SessionBoundary.LAST_BAR_CLOSE and exit_next_open:
            force_close |= (ends[:, None] & exit_).astype(np.int8)
            force_close |= 2 * (ends[:, None] & exit_long).astype(np.int8)
            force_close |= 4 * (ends[:, None] & exit_short).astype(np.int8)
            exit_ = exit_ & ~ends[:, None]
            exit_long = exit_long & ~ends[:, None]
            exit_short = exit_short & ~ends[:, None]
        if rules.flatten is Flatten.SESSION_END:
            force_close |= np.broadcast_to(ends[:, None], (n, k))
        elif rules.flatten is Flatten.LAST_BAR_CLOSE:
            force_close |= flatten
        elif flatten_used:
            for i in np.flatnonzero(flatten.any(axis=1)):
                boundary = i + 1 == n or (
                    session_days[i + 1]
                    and rules.session_boundary is not SessionBoundary.NEXT_OBSERVED
                )
                if boundary:
                    if rules.session_boundary is not SessionBoundary.LAST_BAR_CLOSE:
                        force_close[i] |= 8 * flatten[i].astype(np.int8)
                        continue
                    force_close[i] |= flatten[i]
                else:
                    force_open[i + 1] |= flatten[i]
    t = np.ascontiguousarray
    slot_data: tuple[np.ndarray | None, ...] = (None,) * 5
    if slots:
        absent = t(nan.T)
        slot_data = tuple(absent if a is nan else t(a.T) for a in slot_arrays)
    run_columns = K.run_columns if _run_columns is None else _run_columns
    out = run_columns(
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
        amend_close=rules.amend_timing == "close_submit",
        completed_close=rules.entry_bar_decision == "completed_close",
        stop_skip=rules.stop_entry_in_market == "skip",
        exit_next_open=exit_next_open,
        target_gap=rules.target_gap is TargetGap.GAP_IMPROVED,
        tick_round=tuple(TickRound).index(rules.tick_round),
        exit_long=t(exit_long.T) if signals.exit_long is not None and exit_long.any() else None,
        exit_short=t(exit_short.T) if signals.exit_short is not None and exit_short.any() else None,
        force_close=t(force_close.T) if force_close is not None and force_close.any() else None,
        force_open=t(force_open.T) if force_open is not None and force_open.any() else None,
        entry_limit_ioc=t(ioc.T) if signals.entry_limit_ioc is not None else None,
        cross_session=rules.session_boundary is SessionBoundary.NEXT_OBSERVED,
        adverse_first=rules.bracket_policy is BracketPolicy.ADVERSE_FIRST,
        combine=(rules.stop_combine == "tighter") + 2 * (rules.target_combine == "tighter"),
        bracket_anchor=0 if rules.bracket_anchor is None else (1 if rules.bracket_anchor is BracketAnchor.SIGNAL else 2),
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
        fixed_cost = getattr(A.fixed_cost, "py_func", A.fixed_cost)
        per_contract_cost = fixed_cost(
            c.commission_per_side, c.slippage_ticks, instrument.tick, instrument.point_value
        )
        # Keep NumPy warning/errstate behavior for host arrays; compiled cores use
        # the dispatchers for the same policy bodies.
        fill_cost = getattr(A.fill_cost, "py_func", A.fill_cost)
        settle = getattr(A.settle, "py_func", A.settle)
        proportional = fill_cost(e_px, x_px, instrument.point_value, c.proportional_bps)
        net_1 = cast(
            np.ndarray,
            settle(
                trades["gross_points"].to_numpy(),
                instrument.point_value,
                per_contract_cost,
                proportional,
            ),
        )
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
                cfg,
                net_1,
                risk,
                sizing.mode,
                sizing.amount,
                sizing.cap,
                cash,
                k,
                sizing.lot_step,
                sizing.min_qty,
                sizing.min_qty_policy == "clamp",
            )
            if err:
                _sizing_error(err, int(cfg[row]), int(e_i[row]))
            open_contracts = S.size_open(
                o_cfg,
                final_equity,
                o_risk,
                sizing.mode,
                sizing.amount,
                sizing.cap,
                sizing.lot_step,
                sizing.min_qty,
                sizing.min_qty_policy == "clamp",
            )
            if (open_contracts < 0).any():
                r = int(np.flatnonzero(open_contracts < 0)[0])
                _sizing_error(int(-open_contracts[r]), int(o_cfg[r]), int(o_entry[r]))
        if sizing is not None and sizing.lot_step == 1.0 and sizing.min_qty == 1.0:
            contracts = contracts.astype(np.int64)
            open_contracts = open_contracts.astype(np.int64)
        _check_liquidity(bars, e_i, x_i, cfg, contracts, instrument)
        _check_liquidity(
            bars,
            open_positions["entry_i"].to_numpy(),
            None,
            open_positions["config"].to_numpy(),
            open_contracts,
            instrument,
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
    compliance: Compliance | _ReproCompliance
    if rules.profile is Profile.NAUTILUS:
        compliance = Compliance.NAUTILUS
    elif rules.author_platform == "tradingview":
        compliance = Compliance.REPRO_TRADINGVIEW
    else:
        compliance = _ReproCompliance(f"repro:{rules.author_platform}")
    research_only = ioc_used or (
        slots
        and (
            rules.protection == "oto"
            or rules.adaptive_high_low_ordering
            or not rules.reenter_on_exit_bar
            or bool(np.any(entries & np.isnan(stop) & np.isnan(stop_pts)))
            or (sizing is not None and sizing.contracts is None)
        )
    )
    return Result(trades, open_positions, n, k, bars.index, initial_cash, compliance, research_only)


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

    if bars.session_labels is not None:
        labels = bars.session_labels
        return np.r_[False, labels[1:] != labels[:-1]] if len(bars) else np.zeros(0, bool)
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
    instrument: Instrument,
) -> None:
    """Refuse sizes above what Nautilus fills at one price on the entry / exit bar (A14)."""

    if contracts.size == 0 or contracts.max() <= 1:
        return
    if bars.volume is None:
        raise ValueError(
            "sizing above 1 contract needs Bars(volume=...): Nautilus fills against it"
        )
    liq = S.point_liquidity(bars.volume, instrument.lot_step, instrument.min_qty)
    for where, idx in (("entry", entry_i), ("exit", exit_i)):
        if idx is None:
            continue
        over = np.flatnonzero(contracts > liq[idx])
        if over.size:
            r = int(over[0])
            raise ValueError(
                f"{contracts[r]:g} contracts at the {where} bar {int(idx[r])}, config {int(cfg[r])} "
                f"exceed the bar's liquidity ({int(liq[idx[r]])} = max(floor(volume/4), 1)): "
                f"Nautilus would fill the rest one tick worse (docs/vector-engine.md A14)"
            )
