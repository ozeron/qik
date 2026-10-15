"""Run records: one directory per backtest run, the same layout for research sweeps and live days.

    <root>/<project>/<run_id>/
        meta.json      run_id, parent_run_id, project, created_at, git_sha, params, params_hash,
                       window, source, extra
        metrics.json   headline numbers (trades, net_usd, ...)
        trades.parquet one row per round trip (optional)
        equity.parquet equity curve (optional)

``run_id`` is ``<UTC timestamp>-<params_hash[:8]>`` so records sort by time and repeats of the
same params are easy to spot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def params_hash(params: dict[str, Any]) -> str:
    """Stable hash of a parameter dict (key order does not matter)."""
    blob = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class RunRecord:
    project: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    window: dict[str, str] = field(default_factory=dict)
    source: str = ""
    git_sha: str = ""
    parent_run_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def params_hash(self) -> str:
        return params_hash(self.params)

    @property
    def run_id(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%S%fZ}-{self.params_hash[:8]}"

    def meta(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "project": self.project,
            "created_at": self.created_at.isoformat(),
            "git_sha": self.git_sha,
            "params": self.params,
            "params_hash": self.params_hash,
            "window": self.window,
            "source": self.source,
            "extra": self.extra,
        }


def write_run(
    root: Path,
    record: RunRecord,
    *,
    trades: pd.DataFrame | None = None,
    equity: pd.DataFrame | None = None,
) -> Path:
    """Write ``record`` under ``root/<project>/<run_id>/`` and return that directory."""
    out = Path(root) / record.project / record.run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "meta.json").write_text(json.dumps(record.meta(), indent=2, default=str) + "\n")
    (out / "metrics.json").write_text(json.dumps(record.metrics, indent=2, default=str) + "\n")
    if trades is not None:
        trades.to_parquet(out / "trades.parquet", index=False)
    if equity is not None:
        equity.to_parquet(out / "equity.parquet")
    return out


def read_runs(root: Path, project: str) -> pd.DataFrame:
    """One row per run: run_id, params (flattened as ``p.<name>``) and metrics."""
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((Path(root) / project).glob("*/meta.json")):
        meta = json.loads(run_dir.read_text())
        metrics = json.loads((run_dir.parent / "metrics.json").read_text())
        row: dict[str, Any] = {"run_id": meta["run_id"], "created_at": meta["created_at"]}
        row.update({f"p.{k}": v for k, v in meta.get("params", {}).items()})
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
