"""Typed strategy runner; matching stays in kernel's three advance functions.

Cores declare signal_fields so dispatch cannot change after observing a future decision.
The default declaration supports all fields and therefore chooses the slots body.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from types import FunctionType
from typing import Any, NamedTuple, cast, get_type_hints
from typing import overload as typing_overload

import numba as nb
import numpy as np
import pandas as pd
from numba.core.errors import NumbaError

from . import kernel as K
from ._columns import DecisionColumns
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
    Sizing,
    TargetGap,
    TickRound,
    _day_order_dates,
    _new_york_days,
    _one_tick_clock_slots,
    _raise_kernel_error,
    _sizing_error,
)
from .api import (
    Result as EngineResult,
)
from .schema import Features, Params, Scalar, State, StrategySpec
from .straddle import PAIR_FIELDS
from .types import Bar, Decision
from .types import Position as AuthorPosition

RULES_WIDTH = 29


def _run_columns_typed(
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
    cash_accumulation: str = "batch",
    _capture_columns: list[DecisionColumns] | None = None,
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
    state, params, features = init_state, params_grid, feats
    pair_declared = bool(fields & set(PAIR_FIELDS))
    slots = (
        rules.exit_and_rearm
        or pair_declared
        or (
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
                    "entry_from_open",
                    "stop_fraction",
                    "target_fraction",
                }
            )
        )
        or _one_tick_clock_slots(rules)
    )
    if (
        slots
        and rules.fill_model is FillModel.ONE_TICK
        and rules.effective_entry_fill is FillTiming.NEXT_OPEN
        and (rules.protection == "oto" or rules.target_order == "mit")
    ):
        raise ValueError(ONE_TICK_SLOTS)
    dates_needed = (
        bool(fields & {"entry_limit", "entry_stop", "entry_limit_ioc"})
        or pair_declared
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
            rules.order_lifetime == "next_bar",
            rules.exit_and_rearm,
        ],
        float,
    )
    assert len(rc) == RULES_WIDTH, "compiled Rules builder/reader width mismatch"
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
    raw = _bound_runner(
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
        None,
        np.empty((0, 2, 0), np.int8),
        np.ascontiguousarray(session_days, dtype=np.bool_),
        _day_order_dates(bars) if rules.day_order_expiry == "cancel" else np.empty(0, np.bool_),
        pair_declared,
    )
    if _bound_runner is not None:
        if _final_states is not None:
            _final_states.append(raw[-1].view(np.recarray))
        raw = raw[:-1]
    counts, meta, *tail = raw
    arrays = tail[:7]
    sides, exits, levels, booleans, sizing_risks, decisions = tail[7:]
    # Refusals carried out of prange; translate before any row's arrays are read.
    if (counts == -8).any():
        raise ValueError("exit_and_rearm flatten requires SESSION_END or LAST_BAR_CLOSE")
    results = []
    for j, cnt in enumerate(counts):
        if cnt == -3:
            raise ValueError("straddle arm_stop=next_bar composition is unproved")
        if cnt == -4:
            raise ValueError("straddle day_order_expiry=cancel composition is unproved")
        if cnt == -5:
            raise ValueError("straddle ONE_TICK execution is unproved")
        if cnt == -6:
            raise ValueError("straddle ONE_TICK targets execution is unproved")
        if cnt == -7:
            raise ValueError("straddle ONE_TICK IOC execution is unproved")
        # Unvisited rows contain NaNs, not boolean decisions. Surface the matcher
        # refusal first, retaining complete terminal captures for causality checks.
        if meta[j, 0] and (_capture_columns is None or np.isnan(decisions[j, 0]).any()):
            bar = int(meta[j, 1])
            next_open = rules.effective_entry_fill is FillTiming.NEXT_OPEN
            price = float(bars.open[bar] if next_open else bars.close[bar])
            _raise_kernel_error(int(meta[j, 0]), bar, j, price, next_open=next_open)
        if cnt < 0:
            _sizing_error(int(meta[j, 7]), j, int(meta[j, 6]))
        # Validation and accounting use the existing API; the matcher is not run again.
        sig = DecisionColumns(
            decisions[j, 0].astype(np.int8),
            decisions[j, 1].astype(bool),
            **{
                name: decisions[j, q].astype(bool)
                if name in ("flatten", "exit_long", "exit_short", "entry_from_open")
                else decisions[j, q]
                for q, name in enumerate(SIGNAL_FIELDS)
                if q >= 2 and name in fields
            },
        )
        for q, name in enumerate(SIGNAL_FIELDS):
            if name in fields or q < 2:
                continue
            active = (
                decisions[j, q].any()
                if q in (12, 13, 14, 28)
                else np.isfinite(decisions[j, q]).any()
            )
            if active:
                raise ValueError(f"core emitted undeclared field {name}")
        if _capture_columns is not None:
            _capture_columns.append(sig)
        values = (
            np.array([cnt]),
            np.array([0, cnt]),
            *(np.array([meta[j, q]], dtype=np.int64 if q < 4 else float) for q in range(6)),
            *(a[j, :cnt] for a in arrays),
        )

        def already_matched(*args: Any, _values: tuple = values, **kwargs: Any) -> tuple:
            if "sizing_risks" in kwargs:
                kwargs["sizing_risks"][0] = sizing_risks[j]
            return _values

        results.append(
            sig.evaluate(
                bars,
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
    working_long: float = np.nan
    working_short: float = np.nan


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


# Sources and schema classes are immutable during a process; a new process still
# regenerates the content-addressed module and validates its complete disk identity.
_bound_modules: dict[tuple[Any, ...], Any] = {}


def _bound(step: Any) -> Any:
    from .bound import write_bound_module

    schema: StrategySpec = step.__strategy__
    directory = Path(os.environ.get("QIK_VECTOR_CACHE", Path.home() / ".cache/qik/vector_engine"))
    key = (step, schema.state, schema.params, schema.features, True, directory.resolve())
    if key in _bound_modules:
        return _bound_modules[key]
    module_name, name = step.__module__, step.__name__
    if module_name == "__main__" or "<locals>" in step.__qualname__:
        raise ValueError("strategy must be defined at module scope in an importable module")
    directory.mkdir(parents=True, exist_ok=True)
    # Generate to a process-owned draft, then publish at a content-addressed path.
    draft = directory / f"draft_{os.getpid()}_{name}.py"
    try:
        write_bound_module(draft, module_name, name, records=True)
        source = draft.read_bytes()
    finally:
        draft.unlink(missing_ok=True)
    digest = hashlib.sha256(source).hexdigest()[:24]
    bound_name = f"_qik_bound_{digest}"
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
    _bound_modules[key] = sys.modules[bound_name]
    return _bound_modules[key]


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
    _capture_columns: list[DecisionColumns] | None = None,
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
        results = _run_columns_typed(
            internal_states,
            internal_params,
            (clock, *(getattr(values, name) for name in names)),
            bars,
            _bound_runner=runner,
            _final_states=final,
            _capture_columns=_capture_columns,
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
) -> Result[State] | GridResult[State]:
    """Execute a typed strategy through the cacheable, strategy-bound shared engine.

    No grid returns Result[S]; an explicit grid (including {}) returns GridResult[S].
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
    raise TypeError("run requires a @strategy-decorated function")
