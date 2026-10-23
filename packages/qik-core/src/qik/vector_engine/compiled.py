"""Prototype compiled strategy runner; matching stays in kernel's three advance functions.

Cores declare signal_fields so dispatch cannot change after observing a future decision.
The default declaration supports all fields and therefore chooses the slots body.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import accumulate, product
from typing import Any, NamedTuple

import numba as nb
import numpy as np
from numba.extending import overload
from numba.typed import List

from . import kernel as K
from .api import (
    SIGNAL_FIELDS,
    Bars,
    BracketAnchor,
    BracketPolicy,
    Costs,
    FillModel,
    FillTiming,
    Flatten,
    Instrument,
    Result,
    Rules,
    SessionBoundary,
    Signals,
    Sizing,
    TargetGap,
    TickRound,
    _new_york_days,
    _simulate,
)


def _initial_core_state(init_state: Any, cores: Any, j: int | np.integer[Any]) -> Any:
    return init_state.copy() if cores is None else cores[j]


@overload(_initial_core_state)
def _initial_core_state_type(init_state, cores, j):
    # A class instance and an array must never enter the same Numba phi node.
    if isinstance(cores, nb.types.NoneType):

        def array_state(init_state, cores, j):
            return init_state.copy()

        return array_state

    def class_state(init_state, cores, j):
        return cores[j]

    return class_state


@nb.njit(parallel=True, cache=False)
def _run(
    step,
    init_state,
    params_grid,
    feats,
    o,
    h,
    lo,
    c,
    volume,
    days,
    ends,
    rules,
    economic,
    initial_cash,
    slots,
    cores=None,
    masks=None,
    session_days=None,
):
    k, n = len(params_grid), len(c)
    # A bar can close existing protection and a new same-bar bracket: at most two trades.
    width = 2 * n
    counts = np.empty(k, np.int64)
    meta = np.empty((k, 6))
    a0s, a1s = np.empty((k, width), np.int64), np.empty((k, width), np.int64)
    a2s, a5s = np.empty((k, width), np.int8), np.empty((k, width), np.int8)
    a3s, a4s, a6s = np.empty((k, width)), np.empty((k, width)), np.empty((k, width))
    sides_all, exits_all = np.zeros((k, n), np.int8), np.zeros((k, n), np.bool_)
    levels_all = np.full((k, 10, n), np.nan)
    bools_all = np.zeros((k, 3, n), np.bool_)
    decisions_all = np.full((k, 15, n), np.nan)
    for column in nb.prange(k):  # ty: ignore[not-iterable]  (Numba ships prange as a class)
        j = np.int64(column)
        sides, exits, levels = sides_all[j], exits_all[j], levels_all[j]
        long_exits, short_exits = bools_all[j, 1], bools_all[j, 2]
        force_close, force_open = np.zeros(n, np.int8), np.zeros(n, np.bool_)
        if masks is not None:
            force_close[:] = masks[j, 0]
            force_open[:] = masks[j, 1]
        view = np.full(12, np.nan)
        view[K.V_LAST_EXIT_I] = -1
        decision = np.empty(15)
        account = np.array([0.0, initial_cash])
        core = K.CoreContext(
            step,
            _initial_core_state(init_state, cores, j),
            params_grid[j],
            feats,
            volume,
            view,
            decision,
            sides,
            exits,
            levels,
            long_exits,
            short_exits,
            force_close,
            force_open,
            days if session_days is None else session_days,
            ends,
            economic,
            account,
            int(rules[20]),
            int(rules[21]),
            rules[9],
            int(rules[13]),
            decisions_all[j],
            bool(rules[6]),
        )
        a0, a1, a2, a3, a4, a5, a6 = a0s[j], a1s[j], a2s[j], a3s[j], a4s[j], a5s[j], a6s[j]
        slip = rules[9] if rules[8] else 0.0
        if slots:
            result = K._slots_column(
                o,
                h,
                lo,
                c,
                sides,
                exits,
                levels[0],
                levels[1],
                levels[2],
                levels[3],
                levels[4],
                days,
                int(rules[0]),
                bool(rules[1]),
                bool(rules[2]),
                bool(rules[3]),
                bool(rules[4]),
                bool(rules[5]),
                bool(rules[6]),
                int(rules[7]),
                slip,
                levels[5],
                levels[6],
                levels[7],
                levels[8],
                levels[9],
                bool(rules[15]),
                bool(rules[16]),
                bool(rules[17]),
                bool(rules[18]),
                bool(rules[19]),
                bool(rules[10]),
                int(rules[11]),
                bool(rules[12]),
                int(rules[13]),
                rules[9],
                long_exits,
                short_exits,
                force_close,
                force_open,
                bool(rules[14]),
                False,
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                0,
                core,
            )
        else:
            result = K._column(
                o,
                h,
                lo,
                c,
                sides,
                exits,
                levels[0],
                levels[1],
                levels[2],
                levels[3],
                levels[4],
                days,
                int(rules[0]),
                bool(rules[1]),
                bool(rules[2]),
                bool(rules[3]),
                bool(rules[4]),
                bool(rules[5]),
                bool(rules[6]),
                int(rules[7]),
                bool(rules[8]),
                rules[9],
                bool(rules[10]),
                int(rules[11]),
                bool(rules[12]),
                int(rules[13]),
                rules[9],
                long_exits,
                short_exits,
                force_close,
                force_open,
                bool(rules[14]),
                False,
                a0,
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                0,
                core,
            )
        counts[j] = result[0]
        meta[j, 0] = result[1]
        meta[j, 1] = result[2]
        meta[j, 2] = result[3]
        meta[j, 3] = result[4]
        meta[j, 4] = result[5]
        meta[j, 5] = result[6]
        # Preserve original decisions rather than the boundary-transformed engine masks.
        bools_all[j, 0] = force_open
    return (
        counts,
        meta,
        a0s,
        a1s,
        a2s,
        a3s,
        a4s,
        a5s,
        a6s,
        sides_all,
        exits_all,
        levels_all,
        bools_all,
        decisions_all,
    )


def _run_arrays(
    step: Callable[..., None],
    init_state: np.ndarray,
    params_grid: np.ndarray,
    feats: np.ndarray,
    bars: Bars,
    rules: Rules = Rules(),
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    *,
    initial_cash: float | None = None,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    _cores: Any = None,
    cash_accumulation: str = "batch",
    _capture_signals: list[Signals] | None = None,
) -> list[Result]:
    """Run independent compiled cores in parallel, returning one normal Result per config.

    signal_fields is a declaration of possible outputs, not a scan of future decisions.
    Stateful feedback is post-resting-fill, before close decisions and their settlement.
    """
    if cash_accumulation not in ("batch", "sequential"):
        raise ValueError("cash_accumulation must be batch or sequential")
    fields = set(signal_fields)
    if not fields <= set(SIGNAL_FIELDS):
        raise ValueError("unknown signal_fields")
    state, params = (np.ascontiguousarray(x, dtype=np.float64) for x in (init_state, params_grid))
    features = np.ascontiguousarray(feats)
    if state.ndim != 1 or params.ndim != 2 or features.ndim != 2 or len(params) == 0:
        raise ValueError("run needs 1D state, nonempty 2D params, 2D features")
    if len(features) != len(bars):
        raise ValueError("features must have one row per bar")
    slots = (
        rules.target_order == "mit"
        or rules.protection == "oto"
        or bool(
            fields
            & {
                "entry_stop",
                "stop_update",
                "target_update",
                "breakeven_after_points",
                "breakeven_offset_points",
            }
        )
    )
    dates_needed = bool(fields & {"entry_limit", "entry_stop", "flatten"}) or (
        rules.effective_entry_fill is FillTiming.NEXT_OPEN
        or rules.effective_exit_fill is FillTiming.NEXT_OPEN
    )
    days = (
        _new_york_days(bars, "compiled run")
        if dates_needed and bars.index is not None
        else np.zeros(len(bars), bool)
    )
    if bars.session_labels is not None:
        session_days = np.r_[False, bars.session_labels[1:] != bars.session_labels[:-1]]
        if (
            rules.effective_entry_fill is FillTiming.NEXT_OPEN
            or rules.effective_exit_fill is FillTiming.NEXT_OPEN
        ):
            days = session_days.copy()
    elif rules.flatten is Flatten.SESSION_END:
        session_days = _new_york_days(bars, "flatten")
    else:
        session_days = days
    ends = np.r_[session_days[1:], True]

    tick = instrument.tick if instrument else 0.0
    rc = np.array(
        [
            rules.limit_chase_bars,
            rules.adaptive_high_low_ordering,
            rules.reenter_on_exit_bar,
            rules.stale_book == "nautilus",
            rules.protect_rejected == "flatten",
            rules.effective_entry_fill is FillTiming.NEXT_OPEN,
            rules.effective_exit_fill is FillTiming.NEXT_OPEN,
            rules.latency_bars,
            rules.fill_model is FillModel.ONE_TICK,
            tick,
            rules.bracket_policy is BracketPolicy.ADVERSE_FIRST,
            0
            if rules.bracket_anchor is None
            else (1 if rules.bracket_anchor is BracketAnchor.SIGNAL else 2),
            rules.target_gap is TargetGap.GAP_IMPROVED,
            tuple(TickRound).index(rules.tick_round),
            rules.session_boundary is SessionBoundary.NEXT_OBSERVED,
            rules.target_order == "mit",
            rules.protection == "oto",
            rules.amend_rejected == "keep",
            rules.stop_update_mode == "ratchet",
            rules.stop_entry_in_market == "skip",
            tuple(Flatten).index(rules.flatten),
            tuple(SessionBoundary).index(rules.session_boundary),
        ],
        float,
    )
    pv = instrument.point_value if instrument else 1.0
    friction = 2 * (costs.commission_per_side + costs.slippage_ticks * tick * pv) if costs else 0.0
    economic = np.array(
        [
            pv,
            friction,
            costs.proportional_bps if costs else 0.0,
            sizing.mode if sizing else 0,
            sizing.amount if sizing else 1.0,
            sizing.cap if sizing else 0.0,
            sizing.lot_step if sizing else 1.0,
            sizing.min_qty if sizing else 1.0,
        ]
    )
    raw = _run(
        step,
        state,
        params,
        features,
        bars.open,
        bars.high,
        bars.low,
        bars.close,
        bars.volume if bars.volume is not None else np.zeros(len(bars)),
        days,
        ends,
        rc,
        economic,
        float(initial_cash) if initial_cash is not None else np.nan,
        slots,
        _cores,
        None,
        session_days,
    )
    counts, meta, *tail = raw
    arrays = tail[:7]
    sides, exits, levels, booleans, decisions = tail[7:]
    results = []
    for j, cnt in enumerate(counts):
        # Validation and accounting use the existing API; the matcher is not run again.
        sig = Signals(
            decisions[j, 0].astype(np.int8),
            decisions[j, 1].astype(bool),
            **{
                name: decisions[j, q].astype(bool)
                if name in ("flatten", "exit_long", "exit_short")
                else decisions[j, q]
                for q, name in enumerate(SIGNAL_FIELDS)
                if q >= 2 and name in fields
            },
        )
        for q, name in enumerate(SIGNAL_FIELDS):
            if name in fields or q < 2:
                continue
            active = decisions[j, q].any() if q >= 12 else np.isfinite(decisions[j, q]).any()
            if active:
                raise ValueError(f"core emitted undeclared field {name}")
        if _capture_signals is not None:
            _capture_signals.append(sig)
        values = (
            np.array([cnt]),
            np.array([0, cnt]),
            *(np.array([meta[j, q]], dtype=np.int64 if q < 4 else float) for q in range(6)),
            *(a[j, :cnt] for a in arrays),
        )

        def already_matched(*args: Any, _values: tuple = values, **kwargs: Any) -> tuple:
            return _values

        results.append(
            _simulate(
                bars,
                sig,
                rules=rules,
                instrument=instrument,
                costs=costs,
                sizing=sizing,
                initial_cash=initial_cash,
                _run_columns=already_matched,
            )
        )
        if cash_accumulation == "sequential" and initial_cash is not None:
            frame = results[-1].trades
            if "pnl_usd" in frame:
                frame["equity_usd"] = np.fromiter(
                    accumulate(frame["pnl_usd"], initial=initial_cash), dtype=float
                )[1:]
    return results


# Numeric time is UTC epoch nanoseconds; no pandas objects cross the compiled boundary.


class CompiledBar(NamedTuple):
    i: int
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class Position(NamedTuple):
    side: int
    qty: float
    entry_i: int
    entry_px: float
    stop: float
    target: float
    working_side: int
    working_level: float
    last_exit_i: int
    last_exit_px: float
    last_exit_reason: int
    equity: float


@nb.njit(cache=False)
def _jitclass_step(instance, params, i, feats, o, h, lo, c, v, view, out):
    bar = CompiledBar(i, int(feats[i, 0]), o, h, lo, c, v)
    pos = Position(
        int(view[0]),
        view[1],
        int(view[2]),
        view[3],
        view[4],
        view[5],
        int(view[6]),
        view[7],
        int(view[8]),
        view[9],
        -1 if np.isnan(view[10]) else int(view[10]),
        view[11],
    )
    decision = instance.step(bar, pos)
    out[0] = decision[0]
    out[1] = decision[1]
    out[2] = decision[2]
    out[3] = decision[3]
    out[4] = decision[4]
    out[5] = decision[5]
    out[6] = decision[6]
    out[7] = decision[7]
    out[8] = decision[8]
    out[9] = decision[9]
    out[10] = decision[10]
    out[11] = decision[11]
    out[12] = decision[12]
    out[13] = decision[13]
    out[14] = decision[14]


def run(
    CoreClass: Any,
    bars: Bars,
    *,
    grid: dict[str, Sequence[Any]] | None = None,
    features: dict[str, np.ndarray] | None = None,
    rules: Rules = Rules(),
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> list[Result]:
    """Instantiate one core per Cartesian grid row and call step(bar, pos) in prange.

    CoreClass is a jitclass (or a plain class with NUMBA_DISABLE_JIT=1). Constructors
    receive grid parameters and immutable feature arrays as named arguments. step
    returns a 15-field numeric NamedTuple in Decision/SIGNAL_FIELDS order.
    """
    axes = {} if grid is None else grid
    supplied = {} if features is None else features
    if set(axes) & set(supplied):
        raise ValueError("grid and features names overlap")
    rows = list(product(*axes.values()))
    if not rows:
        raise ValueError("grid axes must be nonempty")
    for name, feature in supplied.items():
        if not isinstance(feature, np.ndarray) or feature.ndim == 0 or len(feature) != len(bars):
            raise ValueError(f"feature {name} must be an array with one row per bar")
    instances = [CoreClass(**dict(zip(axes, row, strict=True)), **supplied) for row in rows]
    cores = instances if getattr(nb.config, "DISABLE_JIT") else List(instances)
    clock = np.full((len(bars), 1), -1, dtype=np.int64)
    if bars.index is not None:
        clock[:, 0] = bars.index.as_unit("ns").asi8
    return _run_arrays(
        _jitclass_step,
        np.empty(0),
        np.zeros((len(rows), 0)),
        clock,
        bars,
        rules=rules,
        instrument=instrument,
        costs=costs,
        sizing=sizing,
        initial_cash=initial_cash,
        signal_fields=signal_fields,
        _cores=cores,
        cash_accumulation=cash_accumulation,
    )
