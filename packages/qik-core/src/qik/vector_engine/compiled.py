"""Typed strategy runner; matching stays in kernel's three advance functions.

Cores declare signal_fields so dispatch cannot change after observing a future decision.
The default declaration supports all fields and therefore chooses the slots body.
"""

from __future__ import annotations

import os
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
from .types import Bar, Decision, Position

RULES_WIDTH = K.RULES_WIDTH


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
    rc = K.rule_settings(
        chase=rules.limit_chase_bars,
        adaptive=rules.adaptive_high_low_ordering,
        reenter=rules.reenter_on_exit_bar,
        stale_drop=rules.stale_book == "nautilus",
        protect_flatten=rules.protect_rejected == "flatten",
        next_open=rules.effective_entry_fill is FillTiming.NEXT_OPEN,
        exit_next_open=rules.effective_exit_fill is FillTiming.NEXT_OPEN,
        latency_bars=rules.latency_bars,
        one_tick=rules.fill_model is FillModel.ONE_TICK,
        tick=tick,
        adverse_first=rules.bracket_policy is BracketPolicy.ADVERSE_FIRST,
        bracket_anchor=0
        if rules.bracket_anchor is None
        else 1
        if rules.bracket_anchor is BracketAnchor.SIGNAL
        else 2,
        target_gap=rules.target_gap is TargetGap.GAP_IMPROVED,
        tick_round=tuple(TickRound).index(rules.tick_round),
        cross_session=rules.session_boundary is SessionBoundary.NEXT_OBSERVED,
        mit=rules.target_order == "mit",
        oto=rules.protection == "oto",
        amend_keep=rules.amend_rejected == "keep",
        ratchet=rules.stop_update_mode == "ratchet",
        stop_skip=rules.stop_entry_in_market == "skip",
        flatten=tuple(Flatten).index(rules.flatten),
        boundary={
            SessionBoundary.REFUSE: 0,
            SessionBoundary.LAST_BAR_CLOSE: 1,
            SessionBoundary.NEXT_OBSERVED: 2,
        }[rules.session_boundary],
        combine=(rules.stop_combine == "tighter") + 2 * (rules.target_combine == "tighter"),
        amend_close=rules.amend_timing == "close_submit",
        completed_close=rules.entry_bar_decision == "completed_close",
        arm_stop=rules.arm_stop == "next_bar",
        day_cancel=rules.day_order_expiry == "cancel",
        next_bar=rules.order_lifetime == "next_bar",
        exit_and_rearm=rules.exit_and_rearm,
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
    k, n = len(params), len(bars)
    declared = frozenset(SIGNAL_FIELDS.index(name) for name in fields)
    rounding = rules.tick_round is not TickRound.NONE
    own_risk = bool(slots and rc[8] and rc[5])
    full = _layout(frozenset(range(len(SIGNAL_FIELDS))), declared, rounding, own_risk)
    compact = _layout(_stored_fields(declared), declared, rounding, own_risk)
    capacity = _initial_capacity(n)
    blank = np.full(n, np.nan)
    # Unstored straddle legs: the batch path's absent block (every pair lookup misses).
    blank_pairs = np.full((12, 0), np.nan)
    volume = bars.volume if bars.volume is not None else np.zeros(n)
    expiry_days = (
        _day_order_dates(bars) if rules.day_order_expiry == "cancel" else np.empty(0, np.bool_)
    )
    shared = _Buffers()

    def launch(
        index: np.ndarray, layout: _Layout, width: int, buffers: _Buffers
    ) -> tuple[list[_Config], np.recarray]:
        store, trades = buffers.take(len(index), layout.nrows, n, width)
        raw = _bound_runner(
            state[index],
            params[index],
            features,
            bars.open,
            bars.high,
            bars.low,
            bars.close,
            volume,
            days,
            ends,
            rc,
            economic,
            float(initial_cash) if initial_cash is not None else np.nan,
            slots,
            layout.context,
            layout.level_rows,
            store,
            trades,
            blank,
            blank_pairs,
            layout.risk_row,
            None,
            np.empty((0, 2, 0), np.int8),
            np.ascontiguousarray(session_days, dtype=np.bool_),
            expiry_days,
            pair_declared,
        )
        counts = raw[0]
        # Scan the whole batch before touching any potentially incomplete row.
        if np.any(counts == -K.ACCOUNT_INVALID_SIDE):
            raise ValueError("compiled core: side must be -1, 0, 1")
        meta = raw[1]
        configs = [
            _Config(
                int(counts[p]),
                meta[p],
                tuple(a[p, : max(0, min(int(counts[p]), width))] for a in trades),
                store[p],
                layout,
                width,
            )
            for p in range(len(index))
        ]
        return configs, raw[-1].view(np.recarray)

    def redo(config: _Config) -> tuple[bool, bool] | None:
        """Whether a config reruns, as (store every field, at the 2n trade bound)."""
        # -10: the compact layout dropped an undeclared field the core emitted.
        dropped = config.count == -10
        overflow = config.count == -9 or config.count > config.width
        if not (dropped or overflow):
            return None
        return dropped or config.layout is full, overflow or config.width == 2 * n

    def finalize(index: np.ndarray) -> tuple[list[_Config], np.recarray]:
        """Run one config batch to its final per-config outputs, rerunning where needed."""
        try:
            return settle(index)
        except Exception as error:
            if compact is full and capacity == 2 * n:
                raise
            # A config stopped early (dropped field, full trade buffer) may raise later
            # than this exception's config: the reference run decides which surfaces.
            try:
                launch(index, full, 2 * n, _Buffers())
            except Exception as reference:  # noqa: BLE001  (the reference's own refusal)
                raise reference from None
            raise error

    def settle(index: np.ndarray) -> tuple[list[_Config], np.recarray]:
        configs, finals = launch(index, compact, capacity, shared)
        if compact is not full and any(config.meta[8] for config in configs):
            # An undeclared field was emitted: rerun with every field stored, as the
            # matcher acted on it before, so the same refusal surfaces first.
            configs, finals = launch(index, full, capacity, _Buffers())
        # Reruns compose (a full-storage rerun can still overflow): repeat until final.
        while pending := {p: plan for p, config in enumerate(configs) if (plan := redo(config))}:
            for every, bound in set(pending.values()):
                group = [p for p, plan in pending.items() if plan == (every, bound)]
                rerun, rerun_finals = launch(
                    index[group],
                    full if every else compact,
                    2 * n if bound else capacity,
                    _Buffers(),
                )
                for p, config in zip(group, rerun, strict=True):
                    # A bar records at most two trades: 2n is a proven bound, never refilled.
                    assert not (every and bound) or redo(config) is None, "trade buffer bound"
                    configs[p] = config
                finals[group] = rerun_finals
        return configs, finals

    results: list[EngineResult] = []
    captured: list[DecisionColumns] = []
    finals: list[np.recarray] = []
    failure: Exception | None = None
    rearm_refused = False
    lo = 0
    while lo < k:
        width = compact.nrows * 8 * n + 6 * n + capacity * 42
        hi = min(k, lo + max(nb.get_num_threads(), _BATCH_BYTES // width))
        configs, batch_finals = finalize(np.arange(lo, hi))
        finals.append(batch_finals)
        # Today's precedence across all configs: a compiled exception (raised above),
        # then the exit_and_rearm refusal, then each config's refusal in config order.
        rearm_refused |= any(config.count == -8 for config in configs)
        if failure is None and not rearm_refused:
            try:
                for j, config in enumerate(configs, start=lo):
                    results.append(
                        _matched_result(
                            j,
                            config,
                            bars,
                            rules,
                            fields,
                            blank,
                            captured if _capture_columns is not None else None,
                            cash_accumulation=cash_accumulation,
                            instrument=instrument,
                            costs=costs,
                            sizing=sizing,
                            initial_cash=initial_cash,
                        )
                    )
            except Exception as error:  # noqa: BLE001  (deferred, re-raised below)
                failure = error
        busiest = max((config.count for config in configs), default=0)
        capacity = min(2 * n, max(capacity, 2 * busiest))
        lo = hi
    if _final_states is not None:
        _final_states.append(np.concatenate(finals).view(np.recarray))
    if rearm_refused:
        raise ValueError("exit_and_rearm flatten requires SESSION_END or LAST_BAR_CLOSE")
    if _capture_columns is not None:
        _capture_columns.extend(captured)
    if failure is not None:
        raise failure
    return results


# Batches bound peak memory by bars x batch, not bars x configs.
_BATCH_BYTES = 256 << 20


def _initial_capacity(n: int) -> int:
    """Trade rows per config before a rerun; a bar records at most two trades (2n)."""
    return min(2 * n, n // 4 + 16)


# Level row -> the decision field it carries: protection and slot levels, IOC, straddle
# legs, open-relative flag and fill fractions (the matcher's columnar inputs).
_LEVEL_FIELDS = (*range(2, 12), 15, *range(16, 28), 28, 29, 30)
_FROM_OPEN = SIGNAL_FIELDS.index("entry_from_open")
_PAIR_INDICES = frozenset(SIGNAL_FIELDS.index(name) for name in PAIR_FIELDS)
_BOOL_FIELDS = ("flatten", "exit_long", "exit_short", "entry_from_open")


@dataclass(frozen=True)
class _Layout:
    """Which decision fields and level rows one config stores, and where (-1: not stored)."""

    rows: np.ndarray
    level_rows: np.ndarray
    context: np.ndarray  # CoreContext.layout as int64 (kernel._layout_context)
    nrows: int
    risk_row: int


def _stored_fields(declared: frozenset[int]) -> frozenset[int]:
    # Straddle legs are one contiguous matcher block: store all twelve or none. The
    # open-relative flag defaults to 0, not NaN, so it cannot read the shared NaN row.
    pairs = _PAIR_INDICES if declared & _PAIR_INDICES else frozenset()
    return declared | {0, 1, _FROM_OPEN} | pairs


def _layout(
    stored: frozenset[int], declared: frozenset[int], rounding: bool, own_risk: bool
) -> _Layout:
    """Rows for stored fields; a level row of its own only where _core_decision transforms it.

    Undeclared fields read one shared all-NaN row. The twelve straddle legs are one
    contiguous matcher block: a transformed leg owns its block row and the leg's raw
    decision moves out of the block.
    """
    from_open = _FROM_OPEN in stored

    def transformed(level: int) -> bool:
        # _core_decision rounds these (TickRound) and shifts open-relative triggers.
        return (level in (2, 12, 18) and rounding) or (
            level in (5, 11, 17) and (rounding or from_open)
        )

    rows = np.full(len(SIGNAL_FIELDS), -1, np.int64)
    level_rows = np.full(len(_LEVEL_FIELDS), -1, np.int64)
    own = np.full(len(_LEVEL_FIELDS), -1, np.int64)
    nrows = 0

    def take() -> int:
        nonlocal nrows
        nrows += 1
        return nrows - 1

    legs = [(11 + p, q) for p, q in enumerate(sorted(_PAIR_INDICES))]
    for q in sorted(stored - _PAIR_INDICES):
        rows[q] = take()
    if stored & _PAIR_INDICES:
        for level, q in legs:
            if transformed(level):
                rows[q] = take()
        for level, q in legs:
            if transformed(level):
                own[level] = take()
            else:
                rows[q] = take()
    extra = nrows
    for level, q in enumerate(_LEVEL_FIELDS):
        if rows[q] >= 0 and own[level] < 0 and transformed(level):
            own[level] = take()
        level_rows[level] = own[level] if own[level] >= 0 else rows[q]
    risk_row = take() if own_risk else -1
    pairs = sorted(_PAIR_INDICES)
    context = np.array(
        [
            sum(1 << q for q in stored),
            sum(1 << q for q in range(len(SIGNAL_FIELDS)) if q < 2 or q in declared),
            sum(1 << p for p, q in enumerate(pairs) if rows[q] >= 0 and own[11 + p] >= 0),
            sum(1 << level for level in (2, 5) if own[level] >= 0),
            level_rows[11] if rows[pairs[0]] >= 0 else -1,
            extra,
        ],
        np.int64,
    )
    return _Layout(rows, level_rows, context, nrows, risk_row)


class _Config(NamedTuple):
    """One config's matcher outputs, views into a batch's buffers."""

    count: int
    meta: np.ndarray
    trades: tuple[np.ndarray, ...]
    levels: np.ndarray
    layout: _Layout
    width: int


class _Buffers:
    """Per-batch store and trade buffers, reused while their shape fits."""

    def __init__(self) -> None:
        self.key: tuple[int, int, int] | None = None
        self.size = 0

    def take(
        self, m: int, nrows: int, n: int, width: int
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if self.key != (nrows, n, width) or self.size < m:
            self.key, self.size = (nrows, n, width), m
            self.store = np.empty((m, nrows, n))
            self.trades = tuple(
                np.empty((m, width), dtype)
                for dtype in (
                    np.int64,
                    np.int64,
                    np.int8,
                    np.float64,
                    np.float64,
                    np.int8,
                    np.float64,
                )
            )
        return self.store[:m], tuple(a[:m] for a in self.trades)


def _matched_result(
    j: int,
    config: _Config,
    bars: Bars,
    rules: Rules,
    fields: set[str],
    blank: np.ndarray,
    captured: list[DecisionColumns] | None,
    *,
    cash_accumulation: str,
    instrument: Instrument | None,
    costs: Costs | None,
    sizing: Sizing[int] | Sizing[float] | None,
    initial_cash: float | None,
) -> EngineResult:
    """Validate and account one matched config through the existing API, without rematching."""
    cnt, meta = config.count, config.meta
    if cnt == -11:
        raise RuntimeError(
            "fixed-ABI strategy callback raised; run refused (use jit=False for the original exception)"
        )
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
    rows = config.layout.rows

    def decision(q: int) -> np.ndarray:
        row = config.levels[rows[q]]
        # Buffers are reused by the next batch: captured columns own their arrays.
        return row.copy() if captured is not None else row

    # Unvisited rows contain NaNs, not boolean decisions. Surface the matcher
    # refusal first, retaining complete terminal captures for causality checks.
    if meta[0] and (captured is None or np.isnan(config.levels[rows[0]]).any()):
        bar = int(meta[1])
        next_open = rules.effective_entry_fill is FillTiming.NEXT_OPEN
        price = float(bars.open[bar] if next_open else bars.close[bar])
        _raise_kernel_error(int(meta[0]), bar, j, price, next_open=next_open)
    if cnt < 0:
        _sizing_error(int(meta[7]), j, int(meta[6]))
    # Validation and accounting use the existing API; the matcher is not run again.
    sig = DecisionColumns(
        decision(0).astype(np.int8),
        decision(1).astype(bool),
        **{
            name: decision(q).astype(bool) if name in _BOOL_FIELDS else decision(q)
            for q, name in enumerate(SIGNAL_FIELDS)
            if q >= 2 and name in fields
        },
    )
    emitted = int(meta[8])
    for q, name in enumerate(SIGNAL_FIELDS):
        if q >= 2 and name not in fields and emitted >> q & 1:
            raise ValueError(f"core emitted undeclared field {name}")
    if captured is not None:
        captured.append(sig)
    risk_row = config.layout.risk_row
    risks = config.levels[risk_row] if risk_row >= 0 else blank
    values = (
        np.array([cnt]),
        np.array([0, cnt]),
        *(np.array([meta[q]], dtype=np.int64 if q < 4 else float) for q in range(6)),
        *(a.copy() for a in config.trades),
    )

    def already_matched(*args: Any, **kwargs: Any) -> tuple:
        if "sizing_risks" in kwargs:
            kwargs["sizing_risks"][0] = risks
        return values

    result = sig.evaluate(
        bars,
        rules=rules,
        instrument=instrument,
        costs=costs,
        sizing=sizing,
        initial_cash=initial_cash,
        _run_columns=already_matched,
    )
    if cash_accumulation == "sequential" and initial_cash is not None:
        frame = result.trades
        if "pnl_usd" in frame:
            frame["equity_usd"] = np.fromiter(
                accumulate(frame["pnl_usd"], initial=initial_cash), dtype=float
            )[1:]
    return result


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
_python_modules: dict[tuple[Any, ...], Any] = {}


def _python_module(step: Any) -> Any:
    from .codegen import write_python_module

    schema: StrategySpec = step.__strategy__
    directory = Path(os.environ.get("QIK_VECTOR_CACHE", Path.home() / ".cache/qik/vector_engine"))
    key = (step, schema.state, schema.params, schema.features, True, directory.resolve())
    if key in _python_modules:
        return _python_modules[key]
    module_name, name = step.__module__, step.__name__
    if module_name == "__main__" or "<locals>" in step.__qualname__:
        raise ValueError("strategy must be defined at module scope in an importable module")
    directory.mkdir(parents=True, exist_ok=True)
    # Generate to a process-owned draft, then publish at a content-addressed path.
    draft = directory / f"draft_{os.getpid()}_{name}.py"
    try:
        write_python_module(draft, module_name, name)
        source = draft.read_text()
    finally:
        draft.unlink(missing_ok=True)
    from .fixed import _load

    module = _load(source, directory, "python")
    _python_modules[key] = module
    return module


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
        values = spec.features(*(supplied[n] for n in names))
    else:
        values = spec.features(*(getattr(supplied, n) for n in names))
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
    final: list[np.recarray] = []
    try:
        if jit and not getattr(nb.config, "DISABLE_JIT"):
            from .fixed import runner as fixed_runner

            runner = fixed_runner(step).run
        else:
            runner = _python_runner(_python_module(step))
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
    step: Callable[[S, P, F, Bar, Position], Decision],
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
    step: Callable[[S, P, F, Bar, Position], Decision],
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
    step: Callable[[S, P, F, Bar, Position], Decision],
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
    """Execute a typed strategy through the shared fixed-ABI engine.

    Python execution (``jit=False`` or NUMBA_DISABLE_JIT=1) uses the
    undecorated Python matcher as an independent execution oracle.

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
