"""A24: stop-free notional sizing, the entry-notional fee basis and the refusals around them.

Every engine result is compared with ``_notional_oracle`` (exact rationals, a bar-by-bar cash
ledger, no engine imports) and, where the number is simple enough to do by hand, with a literal in
the test. Existing sizing modes are pinned unchanged against the untouched ``reference``.
"""

from __future__ import annotations

import ast
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import _notional_oracle as O  # noqa: N812 -- the oracle's short name reads better in fixtures
import numpy as np
import pandas as pd
import pytest
from _notional_strategies import Column, Seen, probe
from _typed_scripts import replay, script

from qik.vector_engine import (
    Bars,
    Costs,
    FillTiming,
    Instrument,
    Profile,
    Rules,
    Sizing,
    run,
)
from qik.vector_engine.account import MODE_NOTIONAL, quantity
from qik.vector_engine.reference import quantity_reference
from qik.vector_engine.testing import assert_python_jit_equal

Rows = list[tuple[float, float, float, float]]
python_quantity: Any = getattr(quantity, "py_func", quantity)


# -- engine side of a scenario ---------------------------------------------------------------


def _sizing(rule: O.Rule) -> Sizing[float]:
    return Sizing(
        notional_usd=rule.notional,
        affordability=rule.affordability,
        lot_step=rule.lot_step,
        min_qty=rule.min_qty,
        max_contracts=rule.cap,
    )


def _costs(rule: O.Rule) -> Costs:
    return Costs(rule.commission_per_side, rule.slippage_ticks, rule.bps, fee_basis=rule.fee_basis)


def _instrument(rule: O.Rule) -> Instrument:
    return Instrument(rule.tick, rule.point_value, lot_step=rule.lot_step, min_qty=rule.min_qty)


def _bars(
    rows: Rows, *, tz: str = "UTC", start: str = "2024-03-04 14:30", freq: str = "min"
) -> Bars:
    o, h, lo, c = (np.array([r[i] for r in rows], float) for i in range(4))
    index = pd.date_range(start, periods=len(rows), freq=freq, tz=tz)
    return Bars(o, h, lo, c, index=index, volume=np.full(len(rows), 1e9))


def _script(rows: Rows, attempts: list[O.Attempt]) -> Any:
    n = len(rows)
    side, leave, stop = np.zeros(n, np.int64), np.zeros(n, bool), np.full(n, np.nan)
    for a in attempts:
        side[a.entry_bar] = a.side
        if a.exit_bar is not None:
            leave[a.exit_bar] = True
        if a.stop is not None:
            stop[a.entry_bar] = a.stop
    return script(side, leave, stop=stop if np.isfinite(stop).any() else None)


def engine(
    rows: Rows, attempts: list[O.Attempt], rule: O.Rule, *, jit: bool = False, **kw: Any
) -> Any:
    next_open = any(a.fill == "next_open" for a in attempts)
    rules = (
        Rules(profile=Profile.REPRO, entry_fill=FillTiming.NEXT_OPEN, exit_fill=FillTiming.CLOSE)
        if next_open
        else Rules()
    )
    bars = (
        _bars(rows, tz="America/New_York", start="2024-03-04 09:30") if next_open else _bars(rows)
    )
    return replay(
        bars,
        _script(rows, attempts),
        rules=kw.pop("rules", rules),
        instrument=_instrument(rule),
        costs=_costs(rule),
        sizing=_sizing(rule),
        initial_cash=rule.cash,
        jit=jit,
        **kw,
    )


def _exact(x: float) -> Fraction:
    return O.q(x)


def _close(a: float, b: Fraction, tol: float = 1e-9) -> bool:
    return abs(a - float(b)) <= tol * max(1.0, abs(float(b)))


def _same_lots(got: float, want: Fraction, lot: float) -> bool:
    """The same number of lot steps: a step like 0.01 is not a float64, so the product drifts."""
    return abs(got - float(want)) <= 1e-6 * lot


def assert_matches(res: Any, ledger: O.Ledger, lot: float = 1.0) -> None:
    """Engine Result == oracle ledger: lots and prices exact, money to float64 rounding."""
    t = res.trades
    assert len(t) == len(ledger.closed)
    for row, want in zip(t.itertuples(), ledger.closed, strict=True):
        assert _same_lots(row.contracts, want.qty, lot)
        assert _exact(row.entry_px) == want.entry.price
        assert _exact(row.exit_px) == want.exit.price
        assert row.entry_i == want.entry.bar and row.exit_i == want.exit.bar
        assert _close(row.gross_usd, want.gross), (row.gross_usd, want.gross)
        assert _close(row.pnl_usd, want.net), (row.pnl_usd, want.net)
        assert _close(row.gross_usd - row.pnl_usd, want.fees), "fee allocation"
        if want.equity_after is not None:
            assert _close(row.equity_usd, want.equity_after, 1e-9)
    o = res.open_positions
    if ledger.open is None:
        assert len(o) == 0
    else:
        assert len(o) == 1
        assert _same_lots(o.contracts.iloc[0], ledger.open.qty, lot)
        assert _exact(o.entry_px.iloc[0]) == ledger.open.entry.price


def expect(res_fn: Any, ledger: O.Ledger, lot: float = 1.0) -> Any:
    """Run the engine; a refusing ledger must refuse with its own text, otherwise match."""
    if ledger.refused is None:
        res = res_fn()
        assert_matches(res, ledger, lot)
        return res
    key = {"zero": "0 contracts", "unaffordable": "unaffordable"}[ledger.refused.why]
    with pytest.raises(ValueError, match=key):
        res_fn()
    return None


def both(rows: Rows, attempts: list[O.Attempt], rule: O.Rule, **kw: Any) -> O.Ledger:
    ledger = O.run(rows, attempts, rule)
    expect(lambda: engine(rows, attempts, rule, **kw), ledger, rule.lot_step)
    return ledger


# -- hand fixtures ---------------------------------------------------------------------------

FLAT: Rows = [(100.0, 100.0, 100.0, 100.0)] * 6


def test_hand_numbers_close_fill_entry_notional_fee() -> None:
    """N=10,000 at 100 is 100 shares; 2.5 bps per side on the entry notional is 5.00 round trip."""
    rows: Rows = [(100.0, 100.0, 100.0, 100.0)] * 2 + [(101.0, 101.0, 101.0, 101.0)] * 4
    attempts = [O.Attempt(entry_bar=0, exit_bar=3)]
    rule = O.Rule(notional=10_000, bps=2.5, fee_basis="entry_notional")
    ledger = both(rows, attempts, rule)
    res = engine(rows, attempts, rule)
    assert res.trades.contracts.tolist() == [100]
    assert res.trades.pnl_usd.tolist() == pytest.approx([100.0 - 5.0])  # 100 gross, 5.00 fees
    assert float(ledger.closed[0].net) == 95.0
    assert res.deployable is False  # notional sizing and the fee basis are research-only


def test_default_fee_basis_is_the_established_both_fill_notional() -> None:
    rows: Rows = [(100.0, 100.0, 100.0, 100.0)] * 2 + [(110.0, 110.0, 110.0, 110.0)] * 4
    attempts = [O.Attempt(entry_bar=0, exit_bar=3)]
    fill = O.Rule(notional=10_000, bps=10)
    res = engine(rows, attempts, fill)
    # (100 + 110) * 100 shares * 10 bps = 21.00 of fees, not 20.00
    assert res.trades.gross_usd.iloc[0] - res.trades.pnl_usd.iloc[0] == pytest.approx(21.0)
    assert Costs().fee_basis == "fill_notional"
    both(rows, attempts, fill)
    both(rows, attempts, O.Rule(notional=10_000, bps=10, fee_basis="entry_notional"))


def test_price_gap_sizes_from_the_fill_not_the_signal_close() -> None:
    """NEXT_OPEN: the signal closes at 100, the order fills 5 % higher at the open of the next bar."""
    rows: Rows = [
        (100.0, 100.5, 99.5, 100.0),
        (105.0, 106.0, 104.0, 105.5),
        (105.5, 106.0, 105.0, 105.5),
        (105.5, 106.0, 105.0, 105.5),
    ]
    attempts = [O.Attempt(entry_bar=0, fill="next_open", exit_bar=2)]
    rule = O.Rule(notional=10_000, bps=5, fee_basis="entry_notional")
    ledger = both(rows, attempts, rule)
    assert ledger.closed[0].qty == 95  # floor(10000 / 105), not 100 from the signal close
    res = engine(rows, attempts, rule)
    assert res.trades.contracts.tolist() == [95] and res.trades.entry_px.tolist() == [105.0]
    # entry_notional: 2 * 95 * 105 * 5 bps; fill_notional would charge the 105.5 exit instead
    assert res.trades.gross_usd.iloc[0] - res.trades.pnl_usd.iloc[0] == pytest.approx(9.975)
    both(rows, attempts, O.Rule(notional=10_000, bps=5))


def test_stop_gapped_through_exits_at_the_open_and_the_basis_sees_it() -> None:
    rows: Rows = [
        (100.0, 100.0, 100.0, 100.0),  # long entry at the close, stop 98
        (95.0, 96.0, 94.0, 95.5),  # opens through the stop: fills at the open, 95
        (95.5, 96.0, 95.0, 95.5),
    ]
    attempts = [O.Attempt(entry_bar=0, stop=98.0)]
    for basis in ("fill_notional", "entry_notional"):
        ledger = both(rows, attempts, O.Rule(notional=10_000, bps=10, fee_basis=basis))
        c = ledger.closed[0]
        assert c.exit.price == 95 and float(c.net) == pytest.approx(-500 - c.fees)
    res = engine(rows, attempts, O.Rule(notional=10_000, bps=10, fee_basis="fill_notional"))
    assert res.trades.reason.tolist() == ["stop_gap"]
    assert res.trades.gross_usd.iloc[0] - res.trades.pnl_usd.iloc[0] == pytest.approx(19.5)


def test_short_side_and_point_value_not_one() -> None:
    rows: Rows = [
        (16000.0, 16000.25, 16000.0, 16000.25),
        (16000.25, 16001.0, 15990.0, 15995.0),
        (15995.0, 15996.0, 15990.0, 15992.5),
        (15992.5, 15993.0, 15990.0, 15991.0),
    ]
    attempts = [O.Attempt(entry_bar=0, side=-1, exit_bar=2)]
    rule = O.Rule(
        notional=1_000_000,
        point_value=20.0,
        tick=0.25,
        commission_per_side=2.0,
        slippage_ticks=1,
        bps=0.5,
        fee_basis="entry_notional",
    )
    ledger = both(rows, attempts, rule)
    assert ledger.closed[0].qty == 3  # floor(1e6 / (16000.25 * 20)) = 3.12
    for pv, want in ((0.5, 124), (2.0, 31), (50.0, 1)):  # floor(1e6 / (16000.25 * pv))
        ledger = both(rows, attempts, O.Rule(notional=1_000_000, point_value=pv, tick=0.25))
        assert ledger.closed[0].qty == want


def test_no_stop_anywhere_where_risk_sizing_refuses() -> None:
    rows: Rows = [(100.0, 100.0, 100.0, 100.0)] * 4
    attempts = [O.Attempt(entry_bar=0, exit_bar=2)]
    both(rows, attempts, O.Rule(notional=10_000))
    with pytest.raises(ValueError, match="no stop"):
        replay(
            _bars(rows),
            _script(rows, attempts),
            instrument=Instrument(0.25, 1.0),
            costs=Costs(),
            sizing=Sizing(risk_usd=100.0),
        )


def test_loss_then_next_entry_keeps_the_notional_and_the_cash_boundary_is_exact() -> None:
    """Loss -1,050; the next entry (80 x 125 = 10,000) is not reduced and needs 10,031.25 of cash."""
    rows: Rows = [
        (100.0, 100.0, 100.0, 100.0),
        (95.0, 95.0, 95.0, 95.0),
        (90.0, 90.0, 90.0, 90.0),
        (85.0, 85.0, 85.0, 85.0),
        (80.0, 80.0, 80.0, 80.0),
        (82.0, 82.0, 82.0, 82.0),
        (84.0, 84.0, 84.0, 84.0),
        (84.0, 84.0, 84.0, 84.0),
    ]
    attempts = [O.Attempt(0, exit_bar=2), O.Attempt(4, exit_bar=6)]
    base = {
        "notional": 10_000.0,
        "commission_per_side": 0.25,
        "tick": 0.25,
        "affordability": "refuse",
    }
    ok = both(rows, attempts, O.Rule(cash=11_081.25, **base))
    assert [float(c.qty) for c in ok.closed] == [100, 125]  # constant N: no resize after a loss
    assert float(ok.closed[0].net) == -1050.0 and float(ok.closed[0].equity_after) == 10_031.25
    short = both(rows, attempts, O.Rule(cash=11_081.0, **base))  # 0.25 short of the second entry
    assert short.refused is not None and short.refused.why == "unaffordable"
    both(rows, attempts, O.Rule(cash=11_081.0, **{**base, "affordability": "unchecked"}))


def test_a_decimal_cost_equal_to_the_cash_is_affordable_and_a_cent_short_is_not() -> None:
    """3 x 100.01 = 300.03 exactly, though float64 gives 300.03000000000003 (review r1)."""
    rows: Rows = [(100.01, 100.01, 100.01, 100.01)] * 4
    attempts = [O.Attempt(0, exit_bar=2)]
    base = {"notional": 300.03, "affordability": "refuse"}
    ok = both(rows, attempts, O.Rule(cash=300.03, **base))
    assert [float(c.qty) for c in ok.closed] == [3]
    short = both(rows, attempts, O.Rule(cash=300.02, **base))
    assert short.refused is not None and short.refused.why == "unaffordable"


def test_open_terminal_position_is_sized_from_its_entry_and_marked_at_the_last_close() -> None:
    rows: Rows = [
        (100.0, 100.0, 100.0, 100.0),
        (102.0, 103.0, 101.0, 102.0),
        (104.0, 105.0, 103.0, 104.0),
    ]
    attempts = [O.Attempt(entry_bar=1)]
    rule = O.Rule(notional=10_000, commission_per_side=0.25, tick=0.25, bps=4, cash=50_000.0)
    ledger = both(rows, attempts, O.Rule(**{**rule.__dict__, "affordability": "refuse"}))
    op = ledger.open
    assert op is not None and op.qty == 98  # floor(10000 / 102)
    res = engine(rows, attempts, O.Rule(**{**rule.__dict__, "affordability": "refuse"}))
    assert len(res.trades) == 0 and res.open_positions.contracts.tolist() == [98]
    # mark = 98 * (104 - 102) - entry fill: 98 * (0.25 + 102 * 4 bps)
    assert float(op.terminal_mark) == pytest.approx(196 - 98 * (0.25 + 102 * 0.0004))
    assert res.deployable is False


def test_empty_run_is_still_research_only() -> None:
    rows: Rows = [(100.0, 100.0, 100.0, 100.0)] * 4
    rule = O.Rule(notional=10_000, cash=50_000.0, affordability="refuse")
    res = engine(rows, [], rule)
    assert len(res.trades) == 0 and len(res.open_positions) == 0
    assert res.deployable is False
    assert res.summary()["deployable"].tolist() == [False]


# -- independent symbol books -----------------------------------------------------------------


def test_independent_symbol_books_do_not_share_cash_or_sizes() -> None:
    a: Rows = [(50.0, 50.0, 50.0, 50.0), (45.0, 45.0, 45.0, 45.0), (40.0, 40.0, 40.0, 40.0)] * 2
    b: Rows = [(400.0, 400.0, 400.0, 400.0), (404.0, 404.0, 404.0, 404.0)] * 3
    ra = O.Rule(notional=10_000, cash=10_500.0, affordability="refuse", commission_per_side=0.25)
    rb = O.Rule(notional=10_000, cash=10_500.0, affordability="refuse", commission_per_side=0.25)
    att_a, att_b = [O.Attempt(0, exit_bar=2)], [O.Attempt(0, exit_bar=1), O.Attempt(2, exit_bar=3)]
    la, lb = both(a, att_a, ra), both(b, att_b, rb)
    ea, eb = engine(a, att_a, ra), engine(b, att_b, rb)
    # A loses 1,000+; B's first entry (25 shares = 10,000) would be refused had they shared cash.
    assert la.closed[0].net < -1000 and [float(c.qty) for c in lb.closed] == [25, 25]
    pooled = ea.trades.pnl_usd.sum() + eb.trades.pnl_usd.sum()
    assert pooled == pytest.approx(float(sum(c.net for c in la.closed + lb.closed)), abs=1e-9)


def test_a_grid_of_configs_is_isolated_per_config() -> None:
    rows: Rows = [(100.0 + (i % 3), 101.0 + (i % 3), 99.0, 100.0 + (i % 3)) for i in range(40)]
    bars = _bars(rows)
    lane0 = np.zeros(40, np.int64)
    lane0[[2, 20]] = 1
    lane1 = np.zeros(40, np.int64)
    lane1[[5, 25]] = 1
    leave = np.zeros(40, np.int64)
    leave[[9, 30]] = 1
    kw = dict(
        instrument=Instrument(0.01, 1.0),
        costs=Costs(proportional_bps=3, fee_basis="entry_notional"),
        sizing=Sizing(notional_usd=7_000.0, affordability="refuse"),
        initial_cash=20_000.0,
        signal_fields=("entry_side", "exit"),
        jit=False,
    )
    both_cfg = run(probe, bars, grid={"lane": [0, 1]}, features=Column(lane0, leave, lane1), **kw)
    for j, lane in enumerate((lane0, lane1)):
        alone = run(probe, bars, grid={"lane": [j]}, features=Column(lane, leave, lane1), **kw)
        for name in ("trades", "open_positions"):
            got, want = getattr(both_cfg[j], name), getattr(alone[0], name)
            pd.testing.assert_frame_equal(got.drop(columns="config"), want.drop(columns="config"))


# -- fee allocation ----------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["fill_notional", "entry_notional"])
@pytest.mark.parametrize(
    "fees",
    [
        {"commission_per_side": 0.5},
        {"slippage_ticks": 2.0},
        {"bps": 7.5},
        {"commission_per_side": 0.5, "slippage_ticks": 1.0, "bps": 7.5},
    ],
)
def test_exact_fee_allocation_per_fill(basis: str, fees: dict[str, float]) -> None:
    rows: Rows = [(80.0, 80.0, 80.0, 80.0)] * 2 + [(88.0, 88.0, 88.0, 88.0)] * 4
    attempts = [O.Attempt(entry_bar=0, exit_bar=3)]
    rule = O.Rule(notional=8_000, tick=0.25, fee_basis=basis, **fees)  # type: ignore[arg-type]
    ledger = both(rows, attempts, rule)
    c = ledger.closed[0]
    fixed = (0.5 if "commission_per_side" in fees else 0) + 0.25 * fees.get("slippage_ticks", 0)
    bps = fees.get("bps", 0.0) / 1e4
    entry_fee = 100 * (fixed + 80 * bps)
    exit_fee = 100 * (fixed + (80 if basis == "entry_notional" else 88) * bps)
    assert float(c.entry.fee) == pytest.approx(entry_fee) and float(c.exit.fee) == pytest.approx(
        exit_fee
    )
    assert float(c.fees) == pytest.approx(entry_fee + exit_fee)


def test_entry_notional_fee_ignores_the_exit_price() -> None:
    fees = []
    for exit_px in (60.0, 80.0, 120.0):
        rows: Rows = [(80.0, 80.0, 80.0, 80.0), (exit_px,) * 4, (exit_px,) * 4]
        res = engine(
            rows,
            [O.Attempt(0, exit_bar=1)],
            O.Rule(notional=8_000, bps=9, fee_basis="entry_notional"),
        )
        fees.append(res.trades.gross_usd.iloc[0] - res.trades.pnl_usd.iloc[0])
    assert fees == pytest.approx([2 * 100 * 80 * 9 / 1e4] * 3)


# -- grid, minimum, cap -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("notional", "px", "lot", "minq", "cap", "want"),
    [
        (
            1_000.0,
            333.33,
            0.01,
            0.01,
            None,
            "3",
        ),  # fractional lots: floor(3.0000300003 / .01) * .01
        (100.0, 50.0, 1.0, 2.0, None, "2"),  # exactly the minimum
        (99.99, 50.0, 1.0, 2.0, None, "zero"),  # one step below the minimum refuses
        (10_000.0, 100.0, 1.0, 1.0, 40, "40"),  # an explicit cap only lowers the notional
        (50.0, 100.0, 1.0, 1.0, None, "zero"),  # N below one unit
        (0.3, 0.1, 1.0, 1.0, None, "3"),  # 0.3 / 0.1 = 2.9999999999999996 in float64
        (1.1, 0.1, 1.0, 1.0, None, "11"),  # 1.1 / 0.1 = 11.000000000000002
        (0.2999999, 0.1, 1.0, 1.0, None, "2"),  # 1e-7 below: the 1e-9 epsilon must not round up
    ],
)
def test_grid_minimum_cap_and_near_integer_quotients(
    notional: float, px: float, lot: float, minq: float, cap: int | None, want: str
) -> None:
    rows: Rows = [(px, px, px, px)] * 4
    rule = O.Rule(notional=notional, tick=0.01, lot_step=lot, min_qty=minq, cap=cap)
    ledger = both(rows, [O.Attempt(0, exit_bar=2)], rule)
    if want == "zero":
        assert ledger.refused is not None and ledger.refused.why == "zero"
    else:
        assert ledger.closed[0].qty == Fraction(want)


def test_clamp_is_refused_with_a_fixed_notional() -> None:
    with pytest.raises(ValueError, match="clamp"):
        Sizing(notional_usd=1_000.0, affordability="unchecked", min_qty_policy="clamp")


# -- validation ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"notional_usd": 1_000.0, "contracts": 1, "affordability": "unchecked"},
        {"notional_usd": 1_000.0, "risk_usd": 10.0, "affordability": "unchecked"},
        {"notional_usd": 1_000.0, "risk_fraction": 0.1, "affordability": "unchecked"},
        {"notional_usd": 1_000.0},  # the policy must be stated
        {"notional_usd": 1_000.0, "affordability": "maybe"},
        {"notional_usd": 0.0, "affordability": "unchecked"},
        {"notional_usd": -5.0, "affordability": "unchecked"},
        {"notional_usd": math.nan, "affordability": "unchecked"},
        {"notional_usd": math.inf, "affordability": "unchecked"},
        {"notional_usd": True, "affordability": "unchecked"},
        {"contracts": 1, "affordability": "refuse"},  # a policy without notional
        {"risk_usd": 10.0, "affordability": "unchecked"},
        {},
    ],
)
def test_sizing_mode_validation(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Sizing(**bad)


def test_existing_sizing_constructors_are_unchanged() -> None:
    assert Sizing(contracts=2).affordability is None and Sizing(contracts=2).check_cash is False
    assert Sizing(risk_usd=100.0, max_contracts=3).notional_usd is None
    s = Sizing(notional_usd=500.0, affordability="refuse")
    assert (s.mode, s.amount, s.check_cash) == (MODE_NOTIONAL, 500.0, True)


def test_fee_basis_validation_and_run_requirements() -> None:
    with pytest.raises(ValueError, match="fee_basis"):
        Costs(fee_basis="entry")  # type: ignore[arg-type]
    rows: Rows = [(100.0, 100.0, 100.0, 100.0)] * 4
    sig = _script(rows, [O.Attempt(0, exit_bar=2)])
    refuse = Sizing(notional_usd=1_000.0, affordability="refuse")
    with pytest.raises(ValueError, match="needs initial_cash"):
        replay(_bars(rows), sig, instrument=Instrument(0.01, 1.0), costs=Costs(), sizing=refuse)
    with pytest.raises(ValueError, match="instrument"):
        replay(_bars(rows), sig, sizing=Sizing(notional_usd=1_000.0, affordability="unchecked"))
    with pytest.raises(ValueError, match="grid"):  # sizing must respect the instrument grid
        replay(
            _bars(rows),
            sig,
            instrument=Instrument(0.01, 1.0, lot_step=1.0),
            costs=Costs(),
            sizing=Sizing(notional_usd=1_000.0, affordability="unchecked", lot_step=0.5),
        )


# -- the quantity rule itself ---------------------------------------------------------------------


def _old_quantity(
    mode: int, amount: float, cap: float, equity: float, risk: float, lot: float, minq: float,
    clamp: bool,
) -> float:  # fmt: skip
    """``account.quantity`` as it was before A24, restated verbatim."""
    if mode == 0:
        return amount
    if not (risk > 0.0):
        return -2.0
    budget = amount if mode == 1 else amount * equity
    if not (budget > 0.0):
        return -1.0
    n = np.floor(budget / risk / lot + 1e-9) * lot
    if cap > 0 and n > cap:
        n = cap
    if clamp:
        n = max(minq, n)
    if n < minq - 1e-10:
        return -1.0
    return float(n)


def test_existing_modes_are_bit_identical_to_the_pre_change_rule_and_the_reference() -> None:
    rng = np.random.default_rng(24)
    specials = [math.nan, math.inf, -math.inf, 0.0, -1.0, 1e-12, 1e18]
    for _ in range(4000):
        mode = int(rng.integers(0, 3))

        def pick() -> float:
            if rng.random() < 0.2:
                return float(rng.choice(specials))
            return float(rng.uniform(0.01, 500))

        args = (
            mode,
            pick(),
            float(rng.choice([0.0, 3.0, 40.0])),
            pick() * 100,
            pick(),
            float(rng.choice([1.0, 0.5, 0.01])),
            float(rng.choice([1.0, 0.5, 0.01, 2.0])),
            bool(rng.random() < 0.3),
        )
        want = _old_quantity(*args)
        got = python_quantity(*args)
        assert (math.isnan(got) and math.isnan(want)) or got == want, (args, got, want)
        if all(math.isfinite(float(a)) for a in args):  # the reference's own edge cases differ
            assert quantity_reference(*args) == want, args


def test_notional_quantity_equals_the_exact_oracle_over_random_inputs() -> None:
    rng = np.random.default_rng(2401)
    for _ in range(3000):
        px = float(rng.integers(1, 400_000)) / 100  # two-decimal prices
        pv = float(rng.choice([1.0, 2.0, 5.0, 20.0, 0.5]))
        notional = float(rng.integers(1, 2_000_000)) / 100
        lot = float(rng.choice([1.0, 0.5, 0.25, 0.01]))
        minq = float(rng.choice([lot, 2 * lot, 1.0])) if lot <= 1 else 1.0
        cap = float(rng.choice([0, 3, 100])) if rng.random() < 0.3 else 0.0
        cash = float(rng.integers(1000, 3_000_000)) / 100
        comm = float(rng.choice([0.0, 0.25, 1.0]))
        bps = float(rng.choice([0.0, 2.5, 10.0]))
        check = bool(rng.random() < 0.5)
        rule = O.Rule(
            notional=notional,
            point_value=pv,
            tick=0.25,
            lot_step=lot,
            min_qty=minq,
            cap=cap or None,
            commission_per_side=comm,
            bps=bps,
            cash=cash,
            affordability="refuse" if check else "unchecked",
        )
        want = O.quantity(rule, O.q(px), O.q(cash))
        unit = px * pv
        got = python_quantity(
            MODE_NOTIONAL, notional, cap, cash, math.nan, lot, minq, False,
            unit, comm + unit * bps / 10000, check,
        )  # fmt: skip
        if isinstance(want, str):
            assert got == {"zero": -1.0, "unaffordable": -3.0}[want], (rule, px, got, want)
        else:
            assert _same_lots(got, want, lot), (rule, px, got, want)


# -- independent random differential -------------------------------------------------------------


def _random_case(seed: int) -> tuple[Rows, list[O.Attempt], O.Rule]:
    rng = np.random.default_rng(seed)
    n = 70
    px = 100.0 + np.cumsum(rng.integers(-6, 7, n)) * 0.25
    px = np.maximum(px, 20.0)
    rows: Rows = []
    for i in range(n):
        o = px[i - 1] if i else px[0]
        c = px[i]
        rows.append(
            (o, max(o, c) + rng.integers(0, 5) * 0.25, min(o, c) - rng.integers(0, 5) * 0.25, c)
        )
    rule = O.Rule(
        notional=float(rng.choice([10_000.0, 25_000.0, 12_345.67, 60_000.0])),
        point_value=float(rng.choice([1.0, 2.0, 20.0])),
        tick=0.25,
        lot_step=float(rng.choice([1.0, 1.0, 0.25])),
        min_qty=float(rng.choice([1.0, 1.0, 0.25])),
        commission_per_side=float(rng.choice([0.0, 0.25, 1.0])),
        slippage_ticks=float(rng.choice([0.0, 1.0])),
        bps=float(rng.choice([0.0, 2.5, 10.0])),
        fee_basis=str(rng.choice(["fill_notional", "entry_notional"])),  # type: ignore[arg-type]
        cash=float(rng.choice([30_000.0, 15_000.0, 70_000.0, 1_000_000.0])),
        affordability=str(rng.choice(["refuse", "unchecked"])),  # type: ignore[arg-type]
    )
    attempts: list[O.Attempt] = []
    bar = int(rng.integers(0, 4))
    while bar < n - 4:
        side = int(rng.choice([-1, 1]))
        close = rows[bar][3]
        stop = None
        if rng.random() < 0.5:
            stop = close - side * float(rng.integers(3, 30)) * 0.25
        a = O.Attempt(bar, side, "close", int(min(n - 1, bar + rng.integers(2, 15))), stop)
        _, _, leave = O._fills(rows, a)  # noqa: SLF001 -- the oracle's own fill, to place the next entry
        assert leave is not None
        if leave[0] != a.exit_bar:  # the stop got there first: no stray signal exit
            a = O.Attempt(bar, side, "close", None, stop)
        attempts.append(a)
        bar = leave[0] + 1 + int(rng.integers(0, 5))
    return rows, attempts, rule


@pytest.mark.parametrize("seed", range(60))
def test_random_scenarios_match_the_oracle(seed: int) -> None:
    rows, attempts, rule = _random_case(seed)
    ledger = O.run(rows, attempts, rule)
    expect(lambda: engine(rows, attempts, rule), ledger, rule.lot_step)


# -- the in-column mirror, Python == JIT ---------------------------------------------------------


def test_the_strategy_sees_the_quantity_and_balance_the_result_reports() -> None:
    rows: Rows = [(100.0 + (i % 5) * 0.25,) * 4 for i in range(60)]
    bars = _bars(rows)
    side, leave, other = np.zeros(60, np.int64), np.zeros(60, np.int64), np.zeros(60, np.int64)
    side[[2, 14, 33, 50]] = 1
    leave[[8, 20, 40]] = 1  # the last entry stays open
    rule = O.Rule(notional=9_000.0, bps=4, tick=0.25, cash=30_000.0, affordability="refuse")
    result = run(
        probe,
        bars,
        features=Column(side, leave, other),
        instrument=_instrument(rule),
        costs=_costs(rule),
        sizing=_sizing(rule),
        initial_cash=rule.cash,
        signal_fields=("entry_side", "exit"),
        jit=False,
    )
    seen: Seen = result.state(0)
    assert seen.entries == 4
    sized = result.trades.contracts.sum() + result.open_positions.contracts.sum()
    assert seen.qty_sum == sized
    # the balance the view shows is the balance the Result accumulates (A24 mirror == pass)
    assert seen.last_equity == pytest.approx(rule.cash + result.trades.pnl_usd.sum(), abs=1e-9)


def test_an_unaffordable_entry_refuses_in_the_column_with_the_same_text() -> None:
    rows: Rows = [(100.0,) * 4] * 12
    bars = _bars(rows)
    side, leave, other = np.zeros(12, np.int64), np.zeros(12, np.int64), np.zeros(12, np.int64)
    side[2] = 1
    leave[8] = 1
    for cash, refused in ((10_049.75, True), (10_050.0, False)):
        # 100 shares at 100 need 10,000 + 100 * 0.5 commission on the entry fill
        rule = O.Rule(notional=10_000.0, cash=cash, affordability="refuse", commission_per_side=0.5)

        def call(r: O.Rule = rule) -> Any:
            return run(
                probe,
                bars,
                features=Column(side, leave, other),
                instrument=_instrument(r),
                costs=_costs(r),
                sizing=_sizing(r),
                initial_cash=r.cash,
                signal_fields=("entry_side", "exit"),
                jit=False,
            )

        if refused:
            with pytest.raises(ValueError, match="unaffordable"):
                call()
        else:
            assert call().trades.contracts.tolist() == [100]


@pytest.mark.engine_jit
def test_notional_python_jit_parity_and_refusal() -> None:
    rows: Rows = [(100.0 + (i % 7) * 0.25,) * 4 for i in range(80)]
    bars = _bars(rows)
    side, leave, other = np.zeros(80, np.int64), np.zeros(80, np.int64), np.zeros(80, np.int64)
    side[[3, 20, 41, 60]] = 1
    leave[[10, 30, 50]] = 1
    rule = O.Rule(
        notional=8_000.0, bps=3, commission_per_side=0.25, tick=0.25, cash=25_000.0,
        affordability="refuse", fee_basis="entry_notional",
    )  # fmt: skip
    kw: dict[str, Any] = {
        "features": Column(side, leave, other),
        "instrument": _instrument(rule),
        "costs": _costs(rule),
        "sizing": _sizing(rule),
        "initial_cash": rule.cash,
        "signal_fields": ("entry_side", "exit"),
    }
    assert_python_jit_equal(probe, bars, **kw)
    python = run(probe, bars, jit=False, **kw)
    compiled = run(probe, bars, jit=True, **kw)
    pd.testing.assert_frame_equal(python.trades, compiled.trades, check_exact=True)
    pd.testing.assert_frame_equal(python.open_positions, compiled.open_positions, check_exact=True)
    kw["initial_cash"] = 7_999.0
    for jit in (False, True):
        with pytest.raises(ValueError, match="unaffordable"):
            run(probe, bars, jit=jit, **kw)


# -- the oracle really is independent -------------------------------------------------------------


def test_oracle_imports_nothing_from_the_engine() -> None:
    tree = ast.parse(Path(O.__file__).read_text())
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [None])
    }
    assert imported <= {"__future__", "dataclasses", "decimal", "fractions", "typing"}
