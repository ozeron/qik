"""Append-only writer for the shared trial ledger (``notes/trials.csv``).

One CSV, header ``date,project,stage,variant_id,n_trades,sharpe_is,is_selected,note``, many
writers. The file is merged with ``merge=union`` and a frozen project checks that it still
starts with its prior bytes, so this module only ever appends: existing bytes are never
rewritten, sorted, re-quoted or "repaired". It refuses instead of fixing.

The header line is bare. Dialect of NEW rows (fixed): ``csv.QUOTE_NONNUMERIC``, ``\\n`` line terminator, UTF-8.
``n_trades`` and ``is_selected`` are bare integers; ``date``, ``project``, ``stage``,
``variant_id``, ``sharpe_is`` (``"{:.4f}"``, or ``""`` for NaN / None) and ``note`` are quoted.
Existing rows may use CRLF or LF; they are accepted as they are, new rows are always LF.

Differences from the legacy per-project writers (parsed fields are what matters, not bytes):

* P076 (``screen_run.append_trials``): same dialect and same parsed fields; NaN sharpe is
  ``""``. It dedupes on ``variant_id`` among ``stage == "optimize"`` rows only; here the
  id is unique across the whole file.
* P097 (``screen.append_trials_csv``): hand-quoted, so the note's ``"`` is doubled exactly as
  csv does; no NaN guard (would write ``"nan"``); date from ``time.strftime``. Parsed fields equal.
* P105 (``screen.append_trials_csv``): QUOTE_NONNUMERIC with the platform default ``\\r\\n``
  terminator, writes ``"nan"`` for a NaN sharpe, repairs a missing trailing newline and writes the
  header into an empty file. Here NaN is ``""``, a missing trailing newline raises, and the header
  is written only when the file is new or empty.

``when`` is required (the caller owns the clock). ``path`` is always an argument: qik knows no
repo-root constants. Concurrent writers are serialised by ``qik.trial_storage.append_record``
(advisory ``flock`` on the file, newline check made under that lock, one append per call, so a
batch is never interleaved). Dedupe reads the file before taking that lock, so two writers
adding the same NEW ``variant_id`` at the same moment can both succeed; creating a missing
ledger concurrently can likewise write two headers. Neither happens when ids are owned by one
project, which is the contract of ``variant_id``.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qik.trial_storage import append_record

HEADER: tuple[str, ...] = (
    "date",
    "project",
    "stage",
    "variant_id",
    "n_trades",
    "sharpe_is",
    "is_selected",
    "note",
)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_REAL_LEDGER = ("notes", "trials.csv")


class LedgerError(ValueError):
    """The ledger or the rows to append are not acceptable; nothing was written."""


@dataclass(frozen=True)
class LedgerRow:
    stage: str
    variant_id: str
    n_trades: int
    sharpe_is: float | None
    is_selected: bool
    note: str


@dataclass(frozen=True)
class LedgerReport:
    added: int
    skipped: int


def _sharpe_text(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    if math.isinf(value):
        raise LedgerError("sharpe_is must be finite, NaN or None")
    return f"{value:.4f}"


def _fields(row: LedgerRow, project: str, when: str) -> list[str | int]:
    return [
        when,
        project,
        row.stage,
        row.variant_id,
        int(row.n_trades),
        _sharpe_text(row.sharpe_is),
        int(row.is_selected),
        row.note,
    ]


def _guard(path: Path, allow_real_ledger: bool) -> None:
    """Refuse to touch the real ``notes/trials.csv`` from a test or with a temp trial log."""
    if allow_real_ledger or path.parts[-2:] != _REAL_LEDGER:
        return
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        raise LedgerError(f"refusing to write {path} under pytest (allow_real_ledger=True)")
    log = os.environ.get("QIK_TRIAL_LOG")
    if log and Path(log).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise LedgerError(
            f"refusing to write {path} while QIK_TRIAL_LOG={log} is a temp log "
            "(allow_real_ledger=True)"
        )


def _read_existing(path: Path) -> dict[str, list[list[str]]]:
    """Validate header and tail; return the parsed rows keyed by ``variant_id``."""
    with path.open("rb") as stream:
        data = stream.read()
    if not data.endswith(b"\n"):
        raise LedgerError(f"ledger tail is not newline-terminated: {path} (not repaired)")
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    if tuple(next(reader)) != HEADER:
        raise LedgerError(f"unexpected ledger header in {path}")
    by_id: dict[str, list[list[str]]] = {}
    for fields in reader:
        if not fields:
            continue
        if len(fields) != len(HEADER):
            raise LedgerError(f"malformed ledger row in {path}: {fields!r}")
        by_id.setdefault(fields[3], []).append(fields)
    return by_id


def _same_row(existing: Sequence[str], new: Sequence[str | int]) -> bool:
    """Everything but the date (a re-run on another day is the same trial)."""
    return list(existing[1:]) == [str(v) for v in new[1:]]


def append_ledger(
    path: Path,
    rows: Sequence[LedgerRow],
    *,
    project: str,
    when: str,
    dedupe: Literal["variant_id", "none"] = "variant_id",
    allow_real_ledger: bool = False,
) -> LedgerReport:
    """Append ``rows`` to the ledger at ``path`` and return how many were added / skipped.

    ``dedupe="variant_id"`` skips a row whose ``variant_id`` is already in the file (or
    earlier in ``rows``) with identical content, and raises ``LedgerError`` when the same id
    comes with different content. ``dedupe="none"`` appends every row. A missing or empty file
    gets the header first. Raises (writing nothing) on an unexpected header, a tail that is
    not newline-terminated, a malformed ``when`` / ``project`` or a guarded real ledger.
    """
    if not _DATE.match(when):
        raise LedgerError(f"when must be YYYY-MM-DD, got {when!r}")
    if not project:
        raise LedgerError("project must be non-empty")
    _guard(path, allow_real_ledger)
    fresh = not path.exists() or path.stat().st_size == 0
    seen = {} if fresh else _read_existing(path)
    new: list[list[str | int]] = []
    skipped = 0
    for row in rows:
        fields = _fields(row, project, when)
        if dedupe == "variant_id":
            known = seen.get(row.variant_id, [])
            if known:
                if not any(_same_row(old, fields) for old in known):
                    raise LedgerError(
                        f"variant_id already in ledger with other content: {row.variant_id}"
                    )
                skipped += 1
                continue
            seen[row.variant_id] = [[str(v) for v in fields]]
        new.append(fields)
    if not new:
        return LedgerReport(0, skipped)
    out = io.StringIO(newline="")
    writer = csv.writer(out, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
    if fresh:
        out.write(",".join(HEADER) + "\n")  # bare, like the header of the real ledger
    writer.writerows(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        append_record(path, out.getvalue().encode("utf-8"), strict=True)
    except ValueError as exc:  # strict append: a writer left a partial line since we looked
        raise LedgerError(f"ledger tail is not newline-terminated: {path} (not repaired)") from exc
    return LedgerReport(len(new), skipped)
