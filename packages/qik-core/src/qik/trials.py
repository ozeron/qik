"""Trial log: every scored configuration is one JSON line, so the multiple-testing N is counted, not remembered.

An agent that tries fifty configurations and reports the winner has run fifty trials, even when
only the winner reaches a notes table. Call ``log_trial`` from the one function that scores a
configuration, so exploratory and debug runs are counted too; deflate with ``count_trials().unique``.

    <path>  one JSON object per line: ts, project, stage, params_hash, params, metrics

The path is ``$QIK_TRIAL_LOG`` when set, else ``results/trial_log.jsonl`` under the current directory.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qik.runs import params_hash
from qik.trial_storage import (
    ParameterStore,
    append_record,
    compact_enabled,
    dump_row,
    external_path,
    iter_trials,
)

TRIAL_LOG_ENV = "QIK_TRIAL_LOG"


def trial_log_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(TRIAL_LOG_ENV) or Path("results") / "trial_log.jsonl")


def log_trial(
    params: dict[str, Any],
    metrics: dict[str, Any],
    *,
    project: str,
    stage: str,
    path: str | Path | None = None,
) -> str:
    """Append one scored configuration; returns its params hash."""
    h = params_hash(params)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),  # noqa: TID251 (log wall time, not strategy time)
        "project": project,
        "stage": stage,
        "params_hash": h,
        "params": params,
        "metrics": metrics,
    }
    p = trial_log_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    compact = compact_enabled(p)
    if compact:  # objects first and durable; the invocation line is what makes them count
        row["params_ref"] = ParameterStore(p).put(params)
        del row["params"]
    # Exclusive flock; the compact partial-tail check is atomic with this append.
    append_record(p, dump_row(row), strict=compact)
    return h


@dataclass(frozen=True)
class TrialCount:
    calls: int  # every scoring call, repeats included
    unique: int  # distinct params: the N for deflation


def count_trials(
    path: str | Path | None = None, *, project: str | None = None, stage: str | None = None
) -> TrialCount:
    p = trial_log_path(path)
    if not p.exists() and not external_path(p).exists():
        return TrialCount(0, 0)
    calls = 0
    # Exact union, bounded RAM even for exhaustive windowed validation logs.
    with tempfile.TemporaryDirectory(prefix="qik-trial-count-") as folder:
        with sqlite3.connect(str(Path(folder) / "identities.sqlite")) as db:
            db.execute("CREATE TABLE identities (hash TEXT PRIMARY KEY) WITHOUT ROWID")
            for row in iter_trials(p):
                if (project and row["project"] != project) or (stage and row["stage"] != stage):
                    continue
                calls += 1
                db.execute("INSERT OR IGNORE INTO identities VALUES (?)", (row["params_hash"],))
            unique = db.execute("SELECT count(*) FROM identities").fetchone()[0]
    return TrialCount(calls, unique)
