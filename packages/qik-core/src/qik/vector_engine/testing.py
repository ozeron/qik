"""Deterministic test tapes and core assertions; no lake or execution-rule substitutes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ._testing_replay import (  # noqa: F401 -- public testing helpers
    SIGNAL_FIELDS,
    ReplayFeatures,
    ReplayParams,
    ReplayState,
    captured_decisions,
    declared_signal_fields,
    prefix_bars,
    replay_decisions,
    replay_features,
    replay_step,
    signal_rows,
)
from .api import Bars, Result
from .compiled import GridResult
from .types import Decision as RecordDecision


def minute_bars(
    start: str | pd.Timestamp = "2023-01-10 09:30",
    periods: int = 30,
    *,
    price: float = 100.0,
    step: float = 0.0,
    open_offset: float = 0.0,
    wick: float = 0.0,
    volume: float | None = 100.0,
    tz: str | None = None,
    halt_hours: Sequence[int] = (),
    session_roll_hour: int = 0,
    gap_at: int | None = None,
    gap_points: float = 0.0,
    ohlc: Sequence[tuple[float, float, float, float]] | None = None,
) -> pd.DataFrame:
    """Build minute bars, with optional trend, gap, halt and explicit OHLC rows.

    ``periods`` counts clock minutes before removing ``halt_hours``. Prices advance
    only on retained bars. Session labels use the local wall date, rolling to the
    following date at ``session_roll_hour`` (0 means calendar dates). A timezone
    preserves actual elapsed minutes through DST; it never invents missing hours.
    Volume can be omitted explicitly for equivalence with legacy OHLC fixtures.
    """
    if periods < 1 or wick < 0 or not 0 <= session_roll_hour <= 23:
        raise ValueError("positive periods, nonnegative wick and roll hour 0..23 required")
    if any(not 0 <= hour <= 23 for hour in halt_hours):
        raise ValueError("halt hours must lie in 0..23")
    index = pd.date_range(start, periods=periods, freq="min", tz=tz)
    index = index[~pd.Series(index).dt.hour.isin(halt_hours).to_numpy()]
    if index.empty:
        raise ValueError("halt removed every bar")
    n = len(index)
    if gap_at is None and gap_points != 0:
        raise ValueError("gap_points requires gap_at")
    if gap_at is not None and not 1 <= gap_at < n:
        raise ValueError("gap_at must be inside the retained tape")
    if ohlc is not None:
        if len(ohlc) != n or gap_at is not None or step or open_offset or wick:
            raise ValueError("explicit OHLC requires one row per bar and no price transforms")
        prices = np.asarray(ohlc, dtype=float)
        if prices.shape != (n, 4):
            raise ValueError("OHLC rows must have four prices")
        opens, highs, lows, closes = prices.T
    else:
        closes = float(price) + float(step) * np.arange(n)
        if gap_at is not None:
            closes[gap_at:] += gap_points
        opens = closes + open_offset
        highs = np.maximum(opens, closes) + wick
        lows = np.minimum(opens, closes) - wick
    # Validate through the public engine input contract, without changing prices.
    Bars(
        opens,
        highs,
        lows,
        closes,
        index=index,
        volume=None if volume is None else np.full(n, volume),
    )
    sessions = [
        ts.date() + timedelta(days=int(bool(session_roll_hour) and ts.hour >= session_roll_hour))
        for ts in index
    ]
    frame = pd.DataFrame(
        {
            "datetime": index,
            "session_date": sessions,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )
    if volume is not None:
        frame["volume"] = float(volume)
    return frame


@dataclass(frozen=True)
class DecisionTape:
    """One configuration's emitted decisions and its declared signal fields."""

    decisions: tuple[RecordDecision, ...]
    signal_fields: tuple[str, ...]


def capture_decisions(
    step: Callable[..., Any], bars: Bars, **run_kwargs: Any
) -> tuple[Result[Any] | GridResult[Any], tuple[DecisionTape, ...]]:
    """Run a typed strategy once and record each configuration's decision tape.

    The tape is what the strategy emitted on its own run, including position
    feedback from these rules. ``replay_decisions`` executes it unchanged under
    other rules; re-running the strategy there could emit different decisions.
    """
    from .compiled import _record_run

    captured: list[Any] = []
    result = _record_run(
        step,
        bars,
        grid=run_kwargs.pop("grid", None),
        features=run_kwargs.pop("features", None),
        jit=run_kwargs.pop("jit", True),
        _capture_signals=captured,
        **run_kwargs,
    )
    tapes = tuple(
        DecisionTape(captured_decisions(column), declared_signal_fields(column))
        for column in captured
    )
    return result, tapes


def assert_trades_equal(a: pd.DataFrame, b: pd.DataFrame, fields: Sequence[str]) -> None:
    """Exact ordered field comparison, ignoring storage dtype and frame index only.

    No sorting, rounding, timezone stripping or tolerance: normalize deliberately at
    the fixture boundary. NaNs compare equal; categorical storage is ignored.
    """
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("fields must be nonempty and unique")
    pd.testing.assert_frame_equal(
        a.loc[:, list(fields)].reset_index(drop=True),
        b.loc[:, list(fields)].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
        check_categorical=False,
    )
    for field in fields:
        if pd.api.types.is_datetime64_any_dtype(a[field]) or pd.api.types.is_datetime64_any_dtype(
            b[field]
        ):
            pd.testing.assert_series_equal(
                a[field].reset_index(drop=True),
                b[field].reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
            )


def assert_causal_feature(
    fn_builder: Callable[[pd.DataFrame], np.ndarray], bars: pd.DataFrame, n: int
) -> None:
    """Assert exact prefix stability (including NaN) for a feature builder."""
    if not 0 < n <= len(bars):
        raise ValueError("n must select a nonempty prefix")
    prefix = np.asarray(fn_builder(bars.iloc[:n].copy()))
    full = np.asarray(fn_builder(bars.copy()))
    if prefix.shape != (n,) or full.shape != (len(bars),):
        raise AssertionError("feature builder must return one value per base bar")
    np.testing.assert_array_equal(prefix, full[:n], err_msg="feature looks ahead")


def assert_python_jit_equal(step: Callable[..., Any], bars: Bars, **run_kwargs: Any) -> None:
    """Compare every trade field and final state against a fresh JIT-disabled process.

    The strategy must be a module-level function. Existing PYTHONPATH and sys.path
    are retained so test fixtures and importable user modules resolve identically.
    """
    import os
    import pickle
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import numba as nb

    from .compiled import run

    if getattr(nb.config, "DISABLE_JIT"):
        raise ValueError("assert_python_jit_equal requires a JIT-enabled parent process")
    kwargs = dict(run_kwargs)
    kwargs.pop("jit", None)
    compiled = run(step, bars, jit=True, **kwargs)
    with tempfile.TemporaryDirectory(prefix="qik-python-jit-") as directory:
        payload = Path(directory) / "input.pkl"
        output = Path(directory) / "output.pkl"
        payload.write_bytes(
            pickle.dumps((step.__module__, getattr(step, "__name__"), bars, kwargs))
        )
        script = """
import importlib, pickle, sys
from pathlib import Path
from qik.vector_engine import run
name, attr, bars, kwargs = pickle.loads(Path(sys.argv[1]).read_bytes())
result = run(getattr(importlib.import_module(name), attr), bars, **kwargs)
Path(sys.argv[2]).write_bytes(pickle.dumps(result))
"""
        env = dict(os.environ, NUMBA_DISABLE_JIT="1")
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
        process = subprocess.run(
            [sys.executable, "-c", script, str(payload), str(output)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode:
            raise AssertionError(
                f"JIT-disabled subprocess failed:\n{process.stdout}\n{process.stderr}"
            )
        python = pickle.loads(output.read_bytes())
    left = compiled.results if isinstance(compiled, GridResult) else (compiled,)
    right = python.results if isinstance(python, GridResult) else (python,)
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        pd.testing.assert_frame_equal(a.trades, b.trades, check_exact=True)
        pd.testing.assert_frame_equal(a.open_positions, b.open_positions, check_exact=True)
        assert a.states.dtype == b.states.dtype
        assert a.states.tobytes() == b.states.tobytes(), "final state bytes differ"


def assert_causal(step: Callable[..., Any], bars: Bars, n: int, **kw: Any) -> None:
    """Require the prefix run's decisions to equal the full run's first n decisions.

    Pass features as a callable ``features(bars)`` to also check the feature
    construction, including HTF alignment. Precomputed arrays can only test the
    strategy's use of those arrays; their construction cannot be inferred.
    """
    from collections.abc import Mapping

    from .compiled import _record_run

    if not 0 < n <= len(bars):
        raise ValueError("n must lie in 1..len(bars)")
    features = kw.pop("features", None)
    grid = kw.pop("grid", None)
    jit = kw.pop("jit", True)
    axes: list[Any] = [None]
    if grid is not None:
        params = getattr(step, "__strategy__").params.from_grid(grid)
        if not len(params):
            raise ValueError("grid axes must be nonempty")
        axes = [{name: [row[name].item()] for name in (params.dtype.names or ())} for row in params]
    captures = []
    for tape in (bars, prefix_bars(bars, n)):
        if callable(features):
            values = features(tape)
        elif isinstance(features, Mapping):
            values = {name: array[: len(tape)] for name, array in features.items()}
        elif features is None:
            values = None
        else:
            values = type(features)(*(array[: len(tape)] for array in features))
        signals = []
        for config in axes:
            captured = []
            try:
                _record_run(
                    step,
                    tape,
                    grid=config,
                    features=values,
                    jit=jit,
                    _capture_signals=captured,
                    **kw,
                )
            except ValueError as error:
                # A terminal NEXT_OPEN decision has no prefix settlement bar.
                # Capture one config at a time so every original decision is
                # available even when Result validation refuses that settlement.
                if not (len(tape) < len(bars) and captured and "is at session end" in str(error)):
                    raise
            signals.extend(captured)
        captures.append(signals)
    for full, prefix in zip(*captures, strict=True):
        for a, b in zip(signal_rows(full, n), signal_rows(prefix, n), strict=True):
            np.testing.assert_array_equal(a, b, err_msg="strategy is not prefix-causal")
