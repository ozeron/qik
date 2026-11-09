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

* writer A (``append_trials``): same dialect and same parsed fields; NaN sharpe is
  ``""``. It dedupes on ``variant_id`` among ``stage == "optimize"`` rows only; here the
  id is unique across the whole file.
* writer B (``append_trials_csv``, hand-quoted): hand-quoted, so the note's ``"`` is doubled exactly as
  csv does; no NaN guard (would write ``"nan"``); date from ``time.strftime``. Parsed fields equal.
* writer C (``append_trials_csv``, csv module): QUOTE_NONNUMERIC with the platform default ``\\r\\n``
  terminator, writes ``"nan"`` for a NaN sharpe, repairs a missing trailing newline and writes the
  header into an empty file. Here NaN is ``""``, a missing trailing newline raises, and the header
  is written only when the file is new or empty.

``when`` is required (the caller owns the clock). ``path`` is always an argument: qik knows no
repo-root constants. Concurrent writers are serialised by an advisory ``flock`` on the file: the
read, the header decision, the dedupe and the single append of a call all happen under that one
lock, so two writers adding the same NEW ``variant_id`` add it once and a missing ledger gets one
header. A call never interleaves with another's batch.
"""

from __future__ import annotations

import csv
import fcntl
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

from qik.trial_storage import sync_directory

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


def _is_real_ledger(path: Path) -> bool:
    """True when ``path``, as spelled or after symlinks / cwd resolution, is ``notes/trials.csv``."""
    for candidate in (Path(os.path.abspath(path)), path.resolve()):
        if tuple(part.lower() for part in candidate.parts[-2:]) == _REAL_LEDGER:
            return True
    return False


def _is_temp(path: Path) -> bool:
    roots = {Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve(), Path("/private/tmp")}
    return any(path.resolve().is_relative_to(root) for root in roots)


def _guard(path: Path, allow_real_ledger: bool) -> None:
    """Refuse to touch the real ``notes/trials.csv`` from a test or with a temp trial log.

    The ledger is recognised by resolved name: the last two parts ``notes/trials.csv``, compared
    case-insensitively, on the path as given (made absolute against the cwd) and on the path
    after symlinks are followed.
    """
    if allow_real_ledger or not _is_real_ledger(path):
        return
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        raise LedgerError(f"refusing to write {path} under pytest (allow_real_ledger=True)")
    log = os.environ.get("QIK_TRIAL_LOG")
    if log and _is_temp(Path(log)):
        raise LedgerError(
            f"refusing to write {path} while QIK_TRIAL_LOG={log} is a temp log "
            "(allow_real_ledger=True)"
        )


def _read_existing(path: Path, data: bytes) -> dict[str, list[list[str]]]:
    """Validate header and tail of ``data`` (the bytes of ``path``); return rows by ``variant_id``."""
    if not data.endswith(b"\n"):
        raise LedgerError(f"ledger tail is not newline-terminated: {path} (not repaired)")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerError(f"ledger is not UTF-8: {path}") from exc
    reader = csv.reader(io.StringIO(text, newline=""))
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


_SHARPE = 5  # column of sharpe_is
_SHARPE_TOL = 1e-4  # new rows are written with 4 decimals; older rows kept more or fewer


def _sharpe_value(text: str) -> float | None:
    try:
        value = float(text) if text.strip() else math.nan
    except ValueError:
        return None
    return value


def _same_sharpe(old: str, new: str) -> bool:
    a, b = _sharpe_value(old), _sharpe_value(new)
    if a is None or b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= _SHARPE_TOL


def _same_row(existing: Sequence[str], new: Sequence[str | int]) -> bool:
    """Everything but the date (a re-run on another day is the same trial).

    ``sharpe_is`` is compared as a number (the file has 0.54, -0.661459817 and ``nan`` next to
    4-decimal values; empty and ``nan`` are the same), every other column as text.
    """
    fresh = [str(v) for v in new]
    if len(existing) != len(fresh):
        return False
    return all(
        _same_sharpe(old, value) if i == _SHARPE else old == value
        for i, (old, value) in enumerate(zip(existing, fresh, strict=True))
        if i != 0
    )


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
    fielded = [(row, _fields(row, project, when)) for row in rows]
    if not fielded:
        return LedgerReport(0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a+b") as handle:  # O_APPEND: every write lands at the end
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # read, dedupe, header and write: one lock
        handle.seek(0)
        data = handle.read()
        fresh = not data
        seen = {} if fresh else _read_existing(path, data)
        new: list[list[str | int]] = []
        skipped = 0
        for row, fields in fielded:
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
        handle.write(out.getvalue().encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        sync_directory(path.parent)
    return LedgerReport(len(new), skipped)
