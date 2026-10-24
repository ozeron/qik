"""Typed strategy runner; matching stays in kernel's three advance functions.

Cores declare signal_fields so dispatch cannot change after observing a future decision.
The default declaration supports all fields and therefore chooses the slots body.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, product
from pathlib import Path
from types import FunctionType
from typing import Any, NamedTuple, cast, get_type_hints
from typing import overload as typing_overload

import numba as nb
import numpy as np
import pandas as pd
from numba.core.errors import NumbaError
from numba.extending import overload
from numba.typed import List

from . import kernel as K
from .api import (
    ONE_TICK_SLOTS,
    SIGNAL_FIELDS,
    Bars,
    BracketAnchor,
    BracketPolicy,
    Costs,
    FillModel,
    FillTiming,
    Flatten,
    Instrument,
    Rules,
    SessionBoundary,
    Signals,
    Sizing,
    TargetGap,
    TickRound,
    _day_order_dates,
    _new_york_days,
    _raise_kernel_error,
    _simulate,
)
from .api import (
    Result as EngineResult,
)
from .core import Core
from .schema import Features, Params, Scalar, State, StrategySpec
from .types import Bar, Decision
from .types import Position as AuthorPosition

_legacy_warning_sites: set[tuple[str, int]] = set()


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
    expiry_days=None,
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
    levels_all = np.full((k, 11, n), np.nan)
    bools_all = np.zeros((k, 3, n), np.bool_)
    decisions_all = np.full((k, 16, n), np.nan)
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
        decision = np.empty(16)
        account = np.array([0.0, initial_cash, 0.0])
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
        combine = int(rules[22]) if len(rules) > 22 else 0
        arm_stop = bool(rules[25])
        day_cancel = bool(rules[26])
        if slots:
            result = K._compiled_slots_column(
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
                levels[10],
                combine=int(rules[22]),
                amend_close=bool(rules[23]),
                completed_close=bool(rules[24]),
                arm_stop=arm_stop,
                day_cancel=day_cancel,
                expiry_days=expiry_days,
            )
        else:
            result = K._compiled_column(
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
                levels[10],
                combine,
                arm_stop,
                day_cancel,
                expiry_days,
            )
        # Negative counts carry sizing refusals out of prange without exceptions.
        counts[j] = -int(account[2]) if account[2] else result[0]
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
    step: Callable[..., None] | None,
    init_state: np.ndarray,
    params_grid: np.ndarray,
    feats: Any,
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
    _bound_runner: Any = None,
    _final_states: list[np.recarray] | None = None,
) -> list[EngineResult]:
    """Run independent compiled cores in parallel, returning one normal Result per config.

    signal_fields is a declaration of possible outputs, not a scan of future decisions.
    Stateful feedback is post-resting-fill, before close decisions and their settlement.
    """
    if cash_accumulation not in ("batch", "sequential"):
        raise ValueError("cash_accumulation must be batch or sequential")
    fields = set(signal_fields)
    if not fields <= set(SIGNAL_FIELDS):
        raise ValueError("unknown signal_fields")
    if _bound_runner is None:
        state, params = (
            np.ascontiguousarray(x, dtype=np.float64) for x in (init_state, params_grid)
        )
        features = np.ascontiguousarray(feats)
    else:
        state, params, features = init_state, params_grid, feats
    if _bound_runner is None and (
        state.ndim != 1 or params.ndim != 2 or features.ndim != 2 or len(params) == 0
    ):
        raise ValueError("run needs 1D state, nonempty 2D params, 2D features")
    if _bound_runner is None and len(features) != len(bars):
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
    if (
        slots
        and rules.fill_model is FillModel.ONE_TICK
        and rules.effective_entry_fill is FillTiming.NEXT_OPEN
    ):
        raise ValueError(ONE_TICK_SLOTS)
    dates_needed = (
        bool(fields & {"entry_limit", "entry_stop", "entry_limit_ioc"})
        or ("flatten" in fields and rules.flatten is Flatten.NEXT_OPEN)
        or (
            rules.effective_entry_fill is FillTiming.NEXT_OPEN
            or rules.effective_exit_fill is FillTiming.NEXT_OPEN
        )
    )
    days = (
        _new_york_days(bars, "compiled run")
        if dates_needed
        and (
            bars.session_labels is not None
            or (bars.index is not None and bars.index.tz is not None)
        )
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
            # CoreContext's stable ABI is REFUSE=0, LAST_BAR_CLOSE=1,
            # NEXT_OBSERVED=2, unlike the public enum's declaration order.
            {
                SessionBoundary.REFUSE: 0,
                SessionBoundary.LAST_BAR_CLOSE: 1,
                SessionBoundary.NEXT_OBSERVED: 2,
            }[rules.session_boundary],
            (rules.stop_combine == "tighter") + 2 * (rules.target_combine == "tighter"),
            rules.amend_timing == "close_submit",
            rules.entry_bar_decision == "completed_close",
            rules.arm_stop == "next_bar",
            rules.day_order_expiry == "cancel",
        ],
        float,
    )
    pv = instrument.point_value if instrument else 1.0
    from .account import fixed_cost

    fixed_cost = getattr(fixed_cost, "py_func", fixed_cost)
    # Preserve the original integer factor (batch accounting used 2.0).
    friction = (
        fixed_cost(costs.commission_per_side, costs.slippage_ticks, tick, pv, 2) if costs else 0.0
    )
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
            sizing.min_qty_policy == "clamp" if sizing else False,
        ]
    )
    runner: Any = _run if _bound_runner is None else _bound_runner
    callback_args = (step,) if _bound_runner is None else ()
    raw = runner(
        *callback_args,
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
        _day_order_dates(bars) if rules.day_order_expiry == "cancel" else None,
    )
    if _bound_runner is not None:
        if _final_states is not None:
            _final_states.append(raw[-1].view(np.recarray))
        raw = raw[:-1]
    counts, meta, *tail = raw
    arrays = tail[:7]
    sides, exits, levels, booleans, decisions = tail[7:]
    results = []
    for j, cnt in enumerate(counts):
        # Unvisited rows contain NaNs, not boolean decisions. Surface the matcher
        # refusal first, retaining complete terminal captures for causality checks.
        if meta[j, 0] and (_capture_signals is None or np.isnan(decisions[j, 0]).any()):
            bar = int(meta[j, 1])
            next_open = rules.effective_entry_fill is FillTiming.NEXT_OPEN
            price = float(bars.open[bar] if next_open else bars.close[bar])
            _raise_kernel_error(int(meta[j, 0]), bar, j, price, next_open=next_open)
        if cnt < 0:
            phase = "entry" if cnt == -1 else "open"
            raise ValueError(f"compiled core: {phase} sizing refused for config {j}")
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
            active = (
                decisions[j, q].any() if q in (12, 13, 14) else np.isfinite(decisions[j, q]).any()
            )
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


def _decision_ioc(decision: Any) -> float:
    """Read the optional IOC field from either numeric decision ABI."""
    return decision[15] if len(decision) > 15 else np.nan


@overload(_decision_ioc)
def _decision_ioc_type(decision: Any) -> Callable[[Any], float] | None:
    if not isinstance(decision, nb.types.BaseTuple):
        return None
    # Numba type-checks both arms of a conditional index before folding len().
    # Select an implementation before indexing a legacy 15-field tuple.
    if len(decision) > 15:

        def with_ioc(decision: Any) -> float:
            return decision[15]

        return with_ioc

    def without_ioc(decision: Any) -> float:
        return np.nan

    return without_ioc


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
    out[15] = _decision_ioc(decision)


def _run_jitclass(
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
) -> list[EngineResult]:
    """Instantiate one core per Cartesian grid row and call step(bar, pos) in prange.

    CoreClass is a jitclass (or a plain class with NUMBA_DISABLE_JIT=1). Constructors
    receive grid parameters and immutable feature arrays as named arguments. step
    returns a 16-field numeric NamedTuple in Decision/SIGNAL_FIELDS order.
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


# Public record/function API. The jitclass prototype remains private.

NAUTILUS = Rules()


Result = EngineResult


@dataclass(frozen=True)
class GridResult[S: State]:
    """Ordered Cartesian results and parameters, without a selection policy."""

    results: tuple[Result[S], ...]
    params: pd.DataFrame
    states: np.recarray

    def __len__(self) -> int:
        """Number of configurations."""
        return len(self.results)

    def __getitem__(self, i: int) -> Result[S]:
        """Access one configuration by its stable row index."""
        return self.results[i]

    def state(self, i: int) -> S:
        """Return a mutable view of configuration i's final state."""
        return cast(S, self.states[i])


def _bound(step: Any) -> Any:
    from .bound import write_bound_module

    module_name, name = step.__module__, step.__name__
    if module_name == "__main__" or "<locals>" in step.__qualname__:
        raise ValueError("strategy must be defined at module scope in an importable module")
    directory = Path(os.environ.get("QIK_VECTOR_CACHE", Path.home() / ".cache/qik/vector_engine"))
    directory.mkdir(parents=True, exist_ok=True)
    # Generate to a process-owned draft, then publish at a content-addressed path.
    draft = directory / f"draft_{os.getpid()}_{name}.py"
    try:
        write_bound_module(draft, module_name, name, records=True)
        source = draft.read_bytes()
    finally:
        draft.unlink(missing_ok=True)
    key = hashlib.sha256(source).hexdigest()[:24]
    bound_name = f"_qik_bound_{key}"
    path = directory / f"{bound_name}.py"
    if not path.exists():
        temp = directory / f"{bound_name}.{os.getpid()}.tmp"
        temp.write_bytes(source)
        temp.replace(path)
    if bound_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(bound_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[bound_name] = module
        spec.loader.exec_module(module)
    return sys.modules[bound_name]


def _python_runner(module: Any) -> Callable[..., Any]:
    # Rebind every shared helper in a private namespace. No dispatcher or global
    # config is mutated: jit=False and JIT runs can coexist in one process.
    from . import account, protection
    from . import sizing as sizing_module

    namespace = dict(vars(account))
    namespace.update(vars(protection))
    namespace.update(vars(sizing_module))
    namespace.update(vars(K))
    namespace.update(vars(module))
    originals = {name: getattr(value, "py_func", value) for name, value in namespace.items()}
    for name, value in originals.items():
        if name == "_strategy":
            namespace[name] = value
        elif isinstance(value, FunctionType) and value.__module__ in (
            K.__name__,
            module.__name__,
            sizing_module.__name__,
            account.__name__,
            protection.__name__,
        ):
            namespace[name] = FunctionType(
                value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__
            )
    return namespace["run"]


def _record_run(
    step: Any,
    bars: Bars,
    *,
    grid: Mapping[str, Sequence[Scalar]] | None,
    features: Features | Mapping[str, np.ndarray] | None,
    jit: bool,
    _capture_signals: list[Signals] | None = None,
    **kwargs: Any,
) -> Result[Any] | GridResult[Any]:
    spec: StrategySpec = step.__strategy__
    params = spec.params.from_grid({} if grid is None else grid)
    if not len(params):
        raise ValueError("grid axes must be nonempty")
    supplied = {} if features is None else features
    spec.features.validate_values(supplied, len(bars))
    names = tuple(get_type_hints(spec.features))
    if isinstance(supplied, Mapping):
        if unknown := set(supplied).difference(names):
            raise ValueError(f"unknown features: {sorted(unknown)}")
        values = spec.features(*(np.ascontiguousarray(supplied[n]) for n in names))
    else:
        values = spec.features(*(np.ascontiguousarray(getattr(supplied, n)) for n in names))
    clock = (
        np.full(len(bars), -1, np.int64) if bars.index is None else bars.index.as_unit("ns").asi8
    )
    states = spec.state.new(len(params))
    # Numba refuses zero-width Records; an unused internal byte admits stateless
    # strategies without leaking a synthetic field into the public state schema.
    internal_states = (
        states
        if states.dtype.names
        else np.zeros(len(params), dtype=[("_empty", "i1")]).view(np.recarray)
    )
    internal_params = (
        params
        if params.dtype.names
        else np.zeros(len(params), dtype=[("_empty", "i1")]).view(np.recarray)
    )
    module = _bound(step)
    runner = module.run if jit or getattr(nb.config, "DISABLE_JIT") else _python_runner(module)
    final: list[np.recarray] = []
    try:
        results = _run_arrays(
            None,
            internal_states,
            internal_params,
            (clock, *(getattr(values, name) for name in names)),
            bars,
            _bound_runner=runner,
            _final_states=final,
            _capture_signals=_capture_signals,
            **kwargs,
        )
    except NumbaError as error:
        raise TypeError(
            f"strategy {step.__module__}.{step.__name__}: compilation failed; "
            "check declared state/parameter fields, feature dtypes and non-jitable calls. "
            f"{error}"
        ) from error
    if not states.dtype.names:
        final[0] = states
    for i, result in enumerate(results):
        result.states = cast(np.recarray, final[0][i : i + 1])
    typed = tuple(results)
    if grid is None:
        return typed[0]
    table = (
        pd.DataFrame.from_records(params)
        if params.dtype.names
        else pd.DataFrame(index=range(len(params)))
    )
    return GridResult(typed, table, final[0])


@typing_overload
def run[S: State, P: Params, F: Features](
    step: Callable[[S, P, F, Bar, AuthorPosition], Decision],
    bars: Bars,
    *,
    grid: None = None,
    features: Features | Mapping[str, np.ndarray] | None = None,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> Result[S]: ...


@typing_overload
def run[S: State, P: Params, F: Features](
    step: Callable[[S, P, F, Bar, AuthorPosition], Decision],
    bars: Bars,
    *,
    grid: Mapping[str, Sequence[Scalar]],
    features: Features | Mapping[str, np.ndarray] | None = None,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> GridResult[S]: ...


@typing_overload
def run[S: State, P: Params, F: Features](
    step: Callable[[S, P, F, Bar, AuthorPosition], Decision],
    bars: Bars,
    *,
    grid: Mapping[str, Sequence[Scalar]] | None,
    features: Features | Mapping[str, np.ndarray] | None = None,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> Result[S] | GridResult[S]: ...


@typing_overload
def run(
    step: Signals | Callable[..., Core],
    bars: Bars,
    *,
    grid: None = None,
    features: Features | Mapping[str, np.ndarray] | None = None,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> Result[State]: ...


@typing_overload
def run(
    step: Callable[..., Core],
    bars: Bars,
    *,
    grid: Mapping[str, Sequence[Scalar]],
    features: Features | Mapping[str, np.ndarray] | None = None,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> list[Result[State]]: ...


def run(
    step: object,
    bars: Bars,
    *,
    grid: Mapping[str, Sequence[Scalar]] | None = None,
    features: Features | Mapping[str, np.ndarray] | None = None,
    rules: Rules = NAUTILUS,
    instrument: Instrument | None = None,
    costs: Costs | None = None,
    sizing: Sizing[int] | Sizing[float] | None = None,
    initial_cash: float | None = None,
    jit: bool = True,
    signal_fields: Sequence[str] = SIGNAL_FIELDS,
    cash_accumulation: str = "batch",
) -> Result[State] | GridResult[State] | list[Result[State]]:
    """Execute a typed strategy through the cacheable, strategy-bound shared engine.

    No grid returns Result[S]; an explicit grid (including {}) returns GridResult[S].
    Legacy stateless Core classes and Signals remain supported with one deprecation
    warning per source call site per process.
    """
    kwargs: dict[str, Any] = dict(
        rules=rules, instrument=instrument, costs=costs, sizing=sizing, initial_cash=initial_cash
    )
    if hasattr(step, "__strategy__"):
        return _record_run(
            step,
            bars,
            grid=grid,
            features=features,
            jit=jit,
            signal_fields=signal_fields,
            cash_accumulation=cash_accumulation,
            **kwargs,
        )
    caller = sys._getframe(1)
    warning_site = (caller.f_code.co_filename, caller.f_lineno)
    del caller
    # Numba/pandas can invalidate Python's warning registry by changing filters.
    # Keep the public once-per-call-site contract independent of those changes.
    if warning_site not in _legacy_warning_sites:
        warnings.warn(
            "Core/Signals API is deprecated; use @strategy and run",
            DeprecationWarning,
            stacklevel=2,
        )
        _legacy_warning_sites.add(warning_site)
    if hasattr(step, "class_type"):
        return _run_jitclass(
            step,
            bars,
            grid=None if grid is None else dict(grid),
            features=cast(dict[str, np.ndarray] | None, features),
            signal_fields=signal_fields,
            cash_accumulation=cash_accumulation,
            **kwargs,
        )
    from .core import Decision as LegacyDecision
    from .core import signals_from_core

    if isinstance(step, Signals):
        if grid is not None or features is not None:
            raise ValueError("Signals already contains configurations; grid/features are invalid")
        # Signals and the bound driver share kernel advance functions. Keep the
        # array ABI here to preserve per-column optional-field dispatch exactly.
        from .api import simulate

        return simulate(bars, step, **kwargs)
    axes = {} if grid is None else grid
    supplied = {} if features is None else dict(cast(Mapping[str, np.ndarray], features))
    rows = list(product(*axes.values()))
    if not rows:
        raise ValueError("grid axes must be nonempty")
    results = []
    from .api import simulate

    for row in rows:
        config: dict[str, object] = dict(zip(axes, row, strict=True))
        config.update(supplied)

        class LegacyView:
            def __init__(self, instance: Any) -> None:
                self.instance = instance

            def step(self, bar: Any) -> LegacyDecision:
                decision = self.instance.step(bar)
                return LegacyDecision(*decision) if isinstance(decision, Decision) else decision

        def make(config=config):
            return LegacyView(cast(Callable[..., Core], step)(**config))

        results.append(simulate(bars, signals_from_core(make, bars), **kwargs))
    return results[0] if grid is None else results
