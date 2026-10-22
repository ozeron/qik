"""High-level API of the vector engine: validated inputs in, one trade table out.

    bars = Bars(open, high, low, close, index=ts)
    sig = Signals(entry_side, exit, stop=levels, target=levels)      # (n,) or (n, k) arrays
    res = simulate(bars, sig, instrument=Instrument(tick=0.25, point_value=50.0),
                   costs=Costs(commission_per_side=2.5, slippage_ticks=1))
    res.trades        # one row per closed trade, all k configs, column ``config``

Everything is explicit: nothing is rounded, clipped or defaulted for you. Inputs the engine cannot
execute the way Nautilus would (levels off the tick grid, a stop at or through the entry close,
non-finite prices, inconsistent OHLC) raise ``ValueError`` naming the config and the bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import kernel as K

__all__ = ["Bars", "Costs", "Instrument", "Result", "Rules", "Signals", "simulate"]

REASON_NAMES = {
    K.REASON_SIGNAL: "signal",
    K.REASON_STOP: "stop",
    K.REASON_TARGET: "target",
    K.REASON_STOP_GAP: "stop_gap",
}


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


@dataclass(frozen=True)
class Rules:
    """Execution switches (docs/vector-engine.md, A6, A9 and A13).

    ``stale_book`` decides what happens when a new entry's stop lands inside the quote a same-bar
    limit fill left stale in Nautilus' L1 book: ``"error"`` (default) refuses the input,
    ``"nautilus"`` reproduces Nautilus: entry fills, stop rejected, linked target cancelled, the
    position runs with no levels. The second is a venue-simulation artifact, not market behaviour.
    """

    adaptive_high_low_ordering: bool = False  # Nautilus venue flag of the same name
    reenter_on_exit_bar: bool = True  # may a new entry follow an exit on the same bar's close
    stale_book: str = "error"

    def __post_init__(self) -> None:
        if self.stale_book not in ("error", "nautilus"):
            raise ValueError(f"stale_book must be 'error' or 'nautilus', got {self.stale_book!r}")


@dataclass(frozen=True)
class Bars:
    """OHLC arrays, one row per bar, in time order. Prices must be finite, positive and consistent."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    index: pd.DatetimeIndex | None = None

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

    ``entry_side[i]`` (+1 long, -1 short, 0 none) is a market entry at bar ``i``'s close, taken only
    when flat. ``exit[i]`` is a market exit at bar ``i``'s close while in a position. ``stop[i]`` and
    ``target[i]`` are absolute price levels placed with the entry made on bar ``i`` (NaN = none): a
    resting stop-market and a resting limit, live from bar ``i + 1``, cancelling each other.
    """

    entry_side: np.ndarray
    exit: np.ndarray
    stop: np.ndarray | None = None
    target: np.ndarray | None = None


@dataclass
class Result:
    trades: pd.DataFrame  # closed trades, all configs
    open_positions: pd.DataFrame  # one row per config still in a position after the last bar
    n_bars: int
    n_configs: int


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
) -> Result:
    """Run every config of ``signals`` over ``bars`` under the Nautilus-compatible rules."""

    n = len(bars)
    side = _as_2d(signals.entry_side, n, np.int8, "entry_side")
    k = side.shape[1]
    exit_ = _as_2d(signals.exit, n, np.bool_, "exit")
    nan = np.full((n, k), np.nan)
    stop = nan if signals.stop is None else _as_2d(signals.stop, n, np.float64, "stop")
    target = nan if signals.target is None else _as_2d(signals.target, n, np.float64, "target")
    if not (exit_.shape[1] == stop.shape[1] == target.shape[1] == k):
        raise ValueError("all signal arrays must have the same number of configs")
    if not np.isin(side, (-1, 0, 1)).all():
        raise ValueError("entry_side must be -1, 0 or +1")
    if np.isinf(stop).any() or np.isinf(target).any():
        raise ValueError("levels must be finite or NaN")
    if costs is not None and instrument is None:
        raise ValueError("costs need an instrument (tick, point_value)")
    if instrument is not None:
        _check_tick(stop, side != 0, instrument.tick, "stop")
        _check_tick(target, side != 0, instrument.tick, "target")

    t = np.ascontiguousarray
    out = K.run_columns(
        bars.open, bars.high, bars.low, bars.close,
        t(side.T), t(exit_.T), t(stop.T), t(target.T),
        rules.adaptive_high_low_ordering, rules.reenter_on_exit_bar, rules.stale_book == "nautilus",
    )  # fmt: skip
    counts, offsets, errs, err_bar, final_side, final_entry, e_i, x_i, sd, e_px, x_px, why = out
    bad = np.flatnonzero(errs)
    if bad.size:
        j = int(bad[0])
        if errs[j] == K.ERR_STALE_BOOK:
            raise ValueError(
                f"stop at bar {int(err_bar[j])}, config {j} sits inside the quote left stale by a limit "
                f"fill earlier in the same bar: Nautilus would reject it and cancel the linked target "
                f"(docs/vector-engine.md A13). Move the stop, skip same-bar re-entry "
                f"(Rules(reenter_on_exit_bar=False)), or opt in with Rules(stale_book='nautilus')"
            )
        kind = "stop" if errs[j] == K.ERR_STOP_IN_MARKET else "target"
        raise ValueError(
            f"{kind} level at bar {int(err_bar[j])}, config {j} is not beyond the entry close "
            f"({bars.close[int(err_bar[j])]}): Nautilus would reject a stop there and fill a "
            f"limit there immediately; the engine refuses both"
        )
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
        }
    )
    trades["gross_points"] = trades["side"] * (trades["exit_px"] - trades["entry_px"])
    if bars.index is not None:
        trades["entry_time"] = bars.index[trades["entry_i"].to_numpy()]
        trades["exit_time"] = bars.index[trades["exit_i"].to_numpy()]
    if instrument is not None:
        trades["gross_usd"] = trades["gross_points"] * instrument.point_value
        c = costs or Costs()
        per_trade = 2.0 * (
            c.commission_per_side + c.slippage_ticks * instrument.tick * instrument.point_value
        )
        trades["pnl_usd"] = trades["gross_usd"] - per_trade
    held = np.flatnonzero(final_side != 0)
    open_positions = pd.DataFrame(
        {
            "config": held.astype(np.int64),
            "side": final_side[held].astype(np.int64),
            "entry_i": final_entry[held].astype(np.int64),
            "entry_px": bars.close[final_entry[held]],
        }
    )
    return Result(trades, open_positions, n, k)
