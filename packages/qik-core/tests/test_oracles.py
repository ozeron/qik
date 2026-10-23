from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from qik.oracles import (
    Tolerance,
    TradeFrame,
    match_trades,
    parse_pyne,
    parse_tradingview_list_of_trades,
)


def trades() -> TradeFrame:
    t = pd.DataFrame(
        {
            "entry_ts": pd.to_datetime(["2023-01-02 10:00", "2023-01-03 10:00"]),
            "exit_ts": pd.to_datetime(["2023-01-02 11:00", "2023-01-03 11:00"]),
            "side": [1, -1],
            "entry_px": [100.0, 101.0],
            "exit_px": [102.0, 103.0],
            "pnl": [3.0, -2.0],
            "qty": [1.0, 1.0],
            "reason": ["tp", "sl"],
        }
    )
    return TradeFrame(t, {"window": ["2023-01-01", "2023-12-31"], "pnl_unit": "USD"})


@pytest.mark.parametrize("parser", [parse_pyne, parse_tradingview_list_of_trades])
def test_export_parser_preserves_only_supplied_values(tmp_path: Path, parser: object) -> None:
    path = tmp_path / "trades.csv"
    path.write_text(
        "Trade #,Type,Date/Time,Signal,Price USD\n1,Exit long,2023-01-02 11:00:00,tp,102\n1,Entry long,2023-01-02 10:00:00,le,100\n"
    )
    got = parser(path)  # type: ignore[operator]
    assert len(got.trades) == 1
    assert got.trades.qty.isna().all() and got.trades.pnl.isna().all()
    assert got.trades.reason.tolist() == ["tp"]
    assert got.source["sha256"]
    assert str(got.trades.entry_ts.dt.tz) == ("UTC" if parser is parse_pyne else "None")
    path.write_text(path.read_text().replace("1,Exit long", "2,Exit long"))
    with pytest.raises(ValueError, match="one entry and exit"):
        parser(path)  # type: ignore[operator]


def test_export_pnl_qty_and_side_validation(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    path.write_text(
        "Trade #,Type,Date/Time,Signal,Price USD,Profit USD,Contracts\n1,Entry short,2023-01-02 10:00,se,100,0,2\n1,Exit short,2023-01-02 11:00,tp,99,3,2\n"
    )
    got = parse_pyne(path)
    assert got.trades.side.tolist() == [-1]
    assert got.trades.pnl.tolist() == [3] and got.trades.qty.tolist() == [2]
    path.write_text(path.read_text().replace("Exit short", "Exit long"))
    with pytest.raises(ValueError, match="sides differ"):
        parse_pyne(path)


def test_pairing_is_one_to_one_with_exit_diagnostics_and_no_mutation() -> None:
    a = trades()
    original = a.trades.copy(deep=True)
    b = a.trades.copy()
    b.loc[0, "entry_ts"] += pd.Timedelta(minutes=1)
    b.loc[0, "entry_px"] += 0.5
    b.loc[0, "exit_px"] += 0.75
    b = pd.concat([b, b.iloc[:1]], ignore_index=True)
    report = match_trades(a, TradeFrame(b, a.source), tol=Tolerance())
    assert report.counts == {
        "ours": 2,
        "theirs": 3,
        "matched": 2,
        "only_ours": 0,
        "only_theirs": 1,
        "exact_entry": 1,
        "same_exit": 1,
        "within_tolerance": 1,
    }
    assert report.verdict_inputs["ours"]["profit_factor"] == 1.5
    assert report.verdict_inputs["ours"]["win_rate"] == 0.5
    assert report.verdict_inputs["ours"]["trades_per_year"] == pytest.approx(2 * 365.25 / 365)
    pd.testing.assert_frame_equal(a.trades, original)
    assert report.deltas.query("field == 'exit_px'").delta.tolist() == [-0.75, 0.0]


def test_empty_unknown_and_clock_mismatch() -> None:
    a = trades()
    empty = TradeFrame(a.trades.iloc[:0], {})
    report = match_trades(empty, a, tol=Tolerance())
    assert report.counts["matched"] == 0 and report.counts["only_theirs"] == 2
    assert report.verdict_inputs["ours"]["profit_factor"] is None
    unknown = a.trades.drop(columns=["pnl", "qty", "reason"])
    report = match_trades(TradeFrame(unknown, {}), a, tol=Tolerance())
    assert report.verdict_inputs["ours"]["win_rate"] is None
    assert report.deltas.query("field == 'pnl'").delta.isna().all()
    aware = a.trades.copy()
    for name in ("entry_ts", "exit_ts"):
        aware[name] = aware[name].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="align explicitly"):
        match_trades(a, TradeFrame(aware, {}), tol=Tolerance())


@pytest.mark.parametrize(
    "kwargs", [{"bars": -1}, {"bars": 1.5}, {"ticks": -1}, {"tick_size": 0}, {"time": "NaT"}]
)
def test_invalid_tolerances(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Tolerance(**kwargs)


def test_current_tradingview_headers(tmp_path: Path) -> None:
    path = tmp_path / "tv.csv"
    path.write_text(
        "Trade #,Type,Date and time,Signal,Price USD,Position size (qty),Net P&L USD\n1,Exit long,2023-01-02 11:00,tp,102,2,3.5\n1,Entry long,2023-01-02 10:00,le,100,2,0\n"
    )
    got = parse_tradingview_list_of_trades(path)
    assert got.trades.qty.tolist() == [2] and got.trades.pnl.tolist() == [3.5]


def test_oracles_import_without_vector_engine() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qik.vector_engine' or fullname.startswith('qik.vector_engine.'):
            raise AssertionError('oracle imported vector_engine: ' + fullname)

sys.meta_path.insert(0, NoEngine())
import qik.oracles
assert not any(name.startswith('qik.vector_engine') for name in sys.modules)
""",
        ],
        check=True,
    )
