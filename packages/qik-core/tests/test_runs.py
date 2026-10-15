from __future__ import annotations

from pathlib import Path

import pandas as pd

from qik.runs import RunRecord, params_hash, read_runs, write_run


def test_params_hash_ignores_key_order() -> None:
    assert params_hash({"a": 1, "b": 2.0}) == params_hash({"b": 2.0, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    rec = RunRecord(
        project="demo",
        params={"long_stop_pts": 70.0},
        metrics={"trades": 798, "net_usd": 9896.5},
        window={"from": "2025-10-01", "to": "2026-09-30"},
        git_sha="abc123",
    )
    trades = pd.DataFrame({"pnl": [10.0, -5.0]})
    out = write_run(tmp_path, rec, trades=trades)

    assert out == tmp_path / "demo" / rec.run_id
    assert rec.run_id.endswith(rec.params_hash[:8])
    assert pd.read_parquet(out / "trades.parquet")["pnl"].tolist() == [10.0, -5.0]

    runs = read_runs(tmp_path, "demo")
    assert runs.loc[0, "run_id"] == rec.run_id
    assert runs.loc[0, "p.long_stop_pts"] == 70.0
    assert runs.loc[0, "net_usd"] == 9896.5
