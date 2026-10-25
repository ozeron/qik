"""Opt-in minimum risk quantity and compiled refusal diagnostics."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from _typed_scripts import replay, script
from hypothesis import given
from hypothesis import strategies as st

from qik.vector_engine import (
    HOLD,
    Bar,
    Bars,
    BracketAnchor,
    Costs,
    Decision,
    Features,
    FillTiming,
    GridResult,
    Instrument,
    Params,
    Position,
    Profile,
    Result,
    Rules,
    Sizing,
    State,
    TickRound,
    enter,
    exit,
    run,
    strategy,
)
from qik.vector_engine.account import quantity
from qik.vector_engine.reference import quantity_reference

QuantityFunction = Callable[[int, float, float, float, float, float, float, bool], float]
# A plain function under NUMBA_DISABLE_JIT.
python_quantity: QuantityFunction = getattr(quantity, "py_func", quantity)
jit_quantity: QuantityFunction = quantity


class EmptyState(State):
    pass


class RiskParams(Params):
    distance: float = 26.22411448753229


class EmptyFeatures(Features):
    pass


@strategy(state=EmptyState, params=RiskParams, features=EmptyFeatures)
def risk_step(s: EmptyState, p: RiskParams, f: EmptyFeatures, bar: Bar, pos: Position) -> Decision:
    if bar.i == 0:
        return enter(1, stop_points=p.distance)
    if bar.i == 1:
        return exit()
    return HOLD


def zero_contracts(bar: int) -> str:
    """Exact sizing refusal for a column whose entry rounds to zero contracts."""
    return (
        f"sizing gives 0 contracts for the entry at bar {bar}, config 0: Nautilus would not "
        "enter, which changes every later trade of the config; raise the risk, set fixed "
        "contracts, or tighten the stop"
    )


@pytest.fixture(scope="module", autouse=True)
def warm_quantity() -> None:
    quantity(1, 500.0, 50.0, 1000.0, 20.0, 1.0, 1.0, True)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("clamp", [False, True])
@given(
    budget=st.floats(0.01, 1e6, allow_nan=False),
    risk=st.floats(0.01, 1e6, allow_nan=False),
    lot=st.sampled_from([0.01, 0.1, 0.25, 1.0]),
    units=st.integers(1, 20),
)
def test_quantity_oracle(
    mode: int, clamp: bool, budget: float, risk: float, lot: float, units: int
) -> None:
    args = (
        mode,
        budget if mode == 1 else 0.01,
        100 * lot,
        budget * 100,
        risk,
        lot,
        units * lot,
        clamp,
    )
    expected = quantity_reference(*args)
    assert python_quantity(*args) == expected
    assert jit_quantity(*args) == expected


@pytest.mark.parametrize("risk", [np.nan, 0.0, -1.0])
def test_clamp_does_not_admit_invalid_stop(risk: float) -> None:
    for fn in (python_quantity, jit_quantity, quantity_reference):
        assert fn(1, 500.0, 50.0, 1000.0, risk, 1.0, 1.0, True) == -2


@pytest.mark.parametrize("lot", [0.01, 0.25, 1.0])
@pytest.mark.parametrize("offset", [-2e-9, 0.0, 2e-9])
def test_rounding_boundary(lot: float, offset: float) -> None:
    budget = (7 + offset) * lot * 20
    expected = (6 if offset < -1e-9 else 7) * lot
    for fn in (python_quantity, jit_quantity, quantity_reference):
        assert fn(1, budget, 0.0, 1000.0, 20.0, lot, lot, True) == expected


def test_batch_clamp_matches_floor_to_one_projects() -> None:
    # P025 qty_for_stop and P051 _qty_for_risk: max(1, min(max_qty, floor(risk / (stop * pv)))).
    n = 2000
    rng = np.random.default_rng(11)
    c = (16000 + np.cumsum(rng.integers(-3, 4, n))) * 0.25
    o = np.r_[c[0], c[:-1]]
    bars = Bars(
        o,
        np.maximum(o, c) + 1,
        np.minimum(o, c) - 1,
        c,
        index=pd.date_range("2020-01-02 14:30", periods=n, freq="5min", tz="UTC"),
        volume=np.full(n, 1e6),
    )
    side = np.where(rng.random(n) < 0.05, rng.choice([-1, 1], n), 0).astype(np.int8)
    stop = bars.close - side * rng.integers(2, 120, n) * 0.25
    signals = script(side, np.zeros(n, bool), stop=stop, target=bars.close + side * 8 * 0.25)
    instrument = Instrument(tick=0.25, point_value=20)

    def sized(policy: Literal["refuse", "clamp"]) -> Result:
        return replay(
            bars,
            signals,
            instrument=instrument,
            costs=Costs(),
            sizing=Sizing(risk_usd=300, max_contracts=4, min_qty_policy=policy),
        )

    with pytest.raises(ValueError) as refused:
        sized("refuse")
    assert str(refused.value) == zero_contracts(28)
    clamped = sized("clamp")
    for frame in (clamped.trades, clamped.open_positions):
        risk = (frame.entry_px - frame.stop).abs() * 20
        expected = np.maximum(1, np.minimum(4, np.floor(300 / risk)))
        assert frame.contracts.tolist() == expected.tolist()
    assert (clamped.trades.contracts == 1).any() and (clamped.trades.contracts == 4).any()


def test_policy_validation() -> None:
    assert Sizing(risk_usd=500).min_qty_policy == "refuse"
    with pytest.raises(ValueError, match="min_qty_policy"):
        Sizing(risk_usd=500, min_qty_policy="bad")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="multiple"):
        Sizing(risk_usd=500, min_qty=0.3, lot_step=0.25, min_qty_policy="clamp")


@pytest.mark.filterwarnings("error:invalid value encountered in cast:RuntimeWarning")
@pytest.mark.parametrize("distance,phase", [(26.22411448753229, "open"), (0.5, "entry")])
def test_python_jit_refusal_and_clamp(distance: float, phase: str) -> None:
    # Large distance leaves the position open at the next observation; a small
    # stop closes on that bar and exercises settlement sizing instead.
    x = np.full(4, 8890.75)
    bars = Bars(
        x,
        x + 1,
        x - 1,
        x,
        volume=np.full(4, 10000.0),
        index=pd.date_range("2020-01-08 14:59", periods=4, freq="min", tz="UTC"),
    )

    def execute(sizing: Sizing[int], jit: bool) -> GridResult[EmptyState]:
        return run(
            risk_step,
            bars,
            features=EmptyFeatures(),
            grid={"distance": [distance]},
            signal_fields=("entry_side", "exit", "stop_points"),
            rules=Rules(
                profile=Profile.REPRO,
                entry_fill=FillTiming.NEXT_OPEN,
                exit_fill=FillTiming.NEXT_OPEN,
                bracket_anchor=BracketAnchor.FILL,
                tick_round=TickRound.NEAREST,
            ),
            instrument=Instrument(tick=0.25, point_value=20),
            costs=Costs(),
            sizing=sizing,
            jit=jit,
        )

    errors = []
    results = []
    for jit in (False, True):
        with pytest.raises(ValueError) as exc:
            execute(Sizing(risk_usd=1), jit)
        errors.append(str(exc.value))
        results.append(execute(Sizing(risk_usd=1, min_qty_policy="clamp"), jit))
    # Entry- and open-phase sizing both refuse with the entry bar (depends on t3/shim3-f1a).
    assert errors == [zero_contracts(1)] * 2

    pd.testing.assert_frame_equal(results[0][0].trades, results[1][0].trades, check_exact=True)
    assert results[0][0].trades.contracts.tolist() == [1]
