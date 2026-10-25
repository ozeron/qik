"""The ledger writer only appends, refuses what it cannot append safely, and loses no rows."""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from qik.ledger import HEADER, LedgerError, LedgerReport, LedgerRow, append_ledger

HEADER_LINE = ",".join(HEADER)
WHEN = "2026-10-21"
NOTE = 'atom "x" cand (fam): net $1,234, n 55'


def row(i: int = 1, *, sharpe: float | None = 0.12345, note: str = NOTE) -> LedgerRow:
    return LedgerRow("optimize", f"P900-c{i}", 55, sharpe, False, note)


def parse(path: Path) -> list[list[str]]:
    with path.open(newline="") as stream:
        return list(csv.reader(stream))


def seed(path: Path, *lines: str, eol: str = "\n") -> bytes:
    data = (eol.join([HEADER_LINE, *lines]) + eol).encode()
    path.write_bytes(data)
    return data


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_append_keeps_existing_bytes(tmp_path: Path, eol: str) -> None:
    path = tmp_path / "trials.csv"
    old = seed(
        path,
        '2026-10-16,P064,repro,P064-H0,983,-0.8162,0,"H0, base ""q"""',
        '"2026-10-16","P065","optimize","P065-a",3,"0.1000",1,"multi\nline"',
        eol=eol,
    )
    report = append_ledger(path, [row(1), row(2)], project="P900", when=WHEN)
    new = path.read_bytes()
    assert report == LedgerReport(2, 0)
    assert hashlib.sha256(old).digest() == hashlib.sha256(new[: len(old)]).digest()
    tail = new[len(old) :]
    assert tail.count(b"\r") == 0 and tail.endswith(b"\n")
    assert len(parse(path)) == 1 + 2 + 2


def test_row_format(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_ledger(
        path, [row(1), row(2, sharpe=float("nan")), row(3, sharpe=None)], project="P900", when=WHEN
    )
    raw = path.read_text()
    assert raw.startswith(HEADER_LINE + "\n")
    assert '"2026-10-21","P900","optimize","P900-c1",55,"0.1235",0,"atom ""x"" cand' in raw
    got = parse(path)
    assert got[1][5] == "0.1235"
    assert got[2][5] == "" and got[3][5] == ""
    assert got[1][6] == "0"


def test_header_only_written_for_new_or_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    path.write_bytes(b"")
    append_ledger(path, [row(1)], project="P900", when=WHEN)
    append_ledger(path, [row(2)], project="P900", when=WHEN)
    assert [r[0] for r in parse(path)].count("date") == 1


def test_dedupe_skips_same_and_raises_on_change(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    append_ledger(path, [row(1), row(2)], project="P900", when=WHEN)
    before = path.read_bytes()
    again = append_ledger(
        path, [row(1), row(2), row(3), row(3)], project="P900", when="2026-10-22"
    )
    assert again == LedgerReport(1, 3)  # other date is still the same trial
    with pytest.raises(LedgerError, match="other content"):
        append_ledger(path, [row(1, note="changed")], project="P900", when=WHEN)
    with pytest.raises(LedgerError, match="other content"):
        append_ledger(path, [row(9), row(9, sharpe=0.5)], project="P900", when=WHEN)
    assert path.read_bytes().startswith(before)
    assert len(parse(path)) == 1 + 3
    assert append_ledger(path, [row(1)], project="P900", when=WHEN, dedupe="none").added == 1


def test_dedupe_compares_sharpe_numerically(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    data = seed(
        path,
        '"2026-10-16","P900","optimize","P900-c1",55,0.54,0,"n"',
        '"2026-10-16","P900","optimize","P900-c2",55,-0.661459817,0,"n"',
        '"2026-10-16","P900","optimize","P900-c3",55,nan,0,"n"',
        '"2026-10-16","P900","optimize","P900-c4",55,"",0,"n"',
    )

    def same(i: int, sharpe: float | None) -> LedgerReport:
        r = LedgerRow("optimize", f"P900-c{i}", 55, sharpe, False, "n")
        return append_ledger(path, [r], project="P900", when=WHEN)

    nan = float("nan")
    assert same(1, 0.54) == LedgerReport(0, 1)
    assert same(2, -0.661459817) == LedgerReport(0, 1)
    assert same(3, nan) == LedgerReport(0, 1) and same(3, None) == LedgerReport(0, 1)
    assert same(4, None) == LedgerReport(0, 1)
    for i, sharpe in ((1, 0.55), (2, -0.67), (3, 0.1), (4, 0.0), (1, None)):
        with pytest.raises(LedgerError, match="other content"):
            same(i, sharpe)
    assert path.read_bytes() == data


def test_unexpected_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    path.write_bytes(b"date,project,stage,variant_id,n_trades,sharpe,is_selected,note\n")
    before = path.read_bytes()
    with pytest.raises(LedgerError, match="header"):
        append_ledger(path, [row(1)], project="P900", when=WHEN)
    assert path.read_bytes() == before


def test_bom_raises_and_leaves_file_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    path.write_bytes(b"\xef\xbb\xbf" + (HEADER_LINE + "\n").encode())
    before = path.read_bytes()
    with pytest.raises(LedgerError, match="header"):
        append_ledger(path, [row(1)], project="P900", when=WHEN)
    assert path.read_bytes() == before


def test_non_newline_tail_raises_and_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    path.write_bytes((HEADER_LINE + '\n2026-10-16,P064,repro,P064-H0,1,0.1,0,"n"').encode())
    before = path.read_bytes()
    with pytest.raises(LedgerError, match="newline"):
        append_ledger(path, [row(1)], project="P900", when=WHEN)
    assert path.read_bytes() == before
    with pytest.raises(LedgerError, match="newline"):
        append_ledger(path, [row(1)], project="P900", when=WHEN, dedupe="none")


def test_bad_arguments(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    with pytest.raises(LedgerError, match="when"):
        append_ledger(path, [row(1)], project="P900", when="today")
    with pytest.raises(LedgerError, match="project"):
        append_ledger(path, [row(1)], project="", when=WHEN)
    with pytest.raises(LedgerError, match="finite"):
        append_ledger(path, [row(1, sharpe=float("inf"))], project="P900", when=WHEN)
    assert not path.exists()


def test_real_ledger_guard_under_pytest(tmp_path: Path) -> None:
    path = tmp_path / "notes" / "trials.csv"
    with pytest.raises(LedgerError, match="pytest"):
        append_ledger(path, [row(1)], project="P900", when=WHEN)
    assert not path.exists()
    assert (
        append_ledger(path, [row(1)], project="P900", when=WHEN, allow_real_ledger=True).added == 1
    )
    other = tmp_path / "other" / "ledger.csv"  # only the real name is guarded
    assert append_ledger(other, [row(1)], project="P900", when=WHEN).added == 1


def test_temp_trial_log_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    monkeypatch.delitem(sys.modules, "pytest")
    path = tmp_path / "notes" / "trials.csv"
    for log in (
        Path(tempfile.gettempdir()) / "x.jsonl",
        Path("/tmp/x.jsonl"),
        Path("/private/tmp/x.jsonl"),
    ):
        monkeypatch.setenv("QIK_TRIAL_LOG", str(log))
        with pytest.raises(LedgerError, match="temp log"):
            append_ledger(path, [row(1)], project="P900", when=WHEN)
    monkeypatch.delenv("QIK_TRIAL_LOG")
    assert append_ledger(path, [row(1)], project="P900", when=WHEN).added == 1


def test_real_ledger_guard_matches_resolved_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes = tmp_path / "notes"
    notes.mkdir()
    seed(notes / "trials.csv")
    before = (notes / "trials.csv").read_bytes()
    link = tmp_path / "elsewhere.csv"
    link.symlink_to(notes / "trials.csv")
    monkeypatch.chdir(notes)
    for path in (link, Path("trials.csv"), tmp_path / "Notes" / "Trials.csv"):
        with pytest.raises(LedgerError, match="pytest"):
            append_ledger(path, [row(1)], project="P900", when=WHEN)
    assert (notes / "trials.csv").read_bytes() == before


_WORKER = """
import sys
from pathlib import Path
from qik.ledger import LedgerRow, append_ledger

path, tag, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for i in range(count):
    note = "x" * 3000
    append_ledger(path, [LedgerRow("optimize", f"{tag}-{i}", i, 0.5, False, note),
                   LedgerRow("optimize", f"{tag}-{i}b", i, 0.5, False, note)],
                  project=tag, when="2026-10-21")
"""


def test_concurrent_processes_lose_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    seed(path)
    count = 40
    procs = [
        subprocess.Popen([sys.executable, "-c", _WORKER, str(path), tag, str(count)])
        for tag in ("PA", "PB", "PC")
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    got = parse(path)
    assert got[0] == list(HEADER)
    ids = [r[3] for r in got[1:]]
    assert len(ids) == len(set(ids)) == 3 * count * 2
    assert all(len(r) == len(HEADER) for r in got)


_RACE = """
import sys, time
from pathlib import Path
import qik.ledger as L

path, start = Path(sys.argv[1]), float(sys.argv[2])
orig = L._read_existing

def slow(*args):  # widen the window between reading the file and writing
    result = orig(*args)
    time.sleep(0.3)
    return result

L._read_existing = slow
while time.time() < start:
    pass
print(L.append_ledger(path, [L.LedgerRow("optimize", "MYX-new", 5, 0.5, False, "n")],
                      project="MYX", when="2026-10-21"))
"""


def _race(path: Path) -> list[str]:
    start = str(time.time() + 3)  # noqa: TID251 (barrier; time for both to import qik)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _RACE, str(path), start], stdout=subprocess.PIPE, text=True
        )
        for _ in range(2)
    ]
    outs = [p.communicate(timeout=120)[0].strip() for p in procs]
    assert [p.returncode for p in procs] == [0, 0]
    return outs


def test_race_same_new_id_adds_exactly_once(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    seed(path)
    outs = _race(path)
    assert sorted(outs) == ["LedgerReport(added=0, skipped=1)", "LedgerReport(added=1, skipped=0)"]
    got = parse(path)
    assert [r[3] for r in got[1:]] == ["MYX-new"]


def test_race_creating_missing_ledger_writes_one_header(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "trials.csv"
    outs = _race(path)
    assert sorted(outs) == ["LedgerReport(added=0, skipped=1)", "LedgerReport(added=1, skipped=0)"]
    got = parse(path)
    assert got[0] == list(HEADER) and [r[3] for r in got[1:]] == ["MYX-new"]


# ---- parsed fields equal what the legacy project writers produce ---------------------------------
# The writers' code is copied here on purpose: frozen projects are neither imported nor edited.


def _p076(path: Path, r: dict) -> None:  # screen_run.py: append_trials
    with path.open("a", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        sh = r["sharpe"]
        w.writerow(
            [
                WHEN,
                "P076",
                "optimize",
                f"P076-{r['cell']}",
                int(r["n"]),
                f"{sh:.4f}" if sh == sh else "",
                0,
                r["note"],
            ]
        )


def _p097(path: Path, r: dict) -> None:  # screen.py: append_trials_csv

    def q(s: str) -> str:
        return '"' + s.replace('"', '""') + '"'

    with path.open("a") as fh:
        fh.write(
            f"{q(WHEN)},{q('P097')},{q('optimize')},{q('P097-' + r['cell'])},{r['n']},"
            f"{q(f'{r["sharpe"]:.4f}')},0,{q(r['note'])}\n"
        )


def _p105(path: Path, r: dict) -> None:  # screen.py: append_trials_csv
    with path.open("a", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_NONNUMERIC)
        w.writerows(
            [
                [
                    WHEN,
                    "P105",
                    "optimize",
                    f"P105-{r['cell']}",
                    int(r["n"]),
                    f"{r['sharpe']:.4f}",
                    0,
                    r["note"],
                ]
            ]
        )


@pytest.mark.parametrize(
    ("project", "legacy"), [("P076", _p076), ("P097", _p097), ("P105", _p105)]
)
def test_parsed_fields_equal_legacy_writers(tmp_path: Path, project: str, legacy) -> None:
    sample = {
        "cell": "A1-L2",
        "n": 222,
        "sharpe": 1.234567,
        "note": 'A6 R3.A6 L2 "60": net $16,275, n 222, PF 1.17\nline',
    }
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    for path in (old, new):
        path.write_text(HEADER_LINE + "\n")
    legacy(old, sample)
    mine = LedgerRow(
        "optimize", f"{project}-{sample['cell']}", 222, 1.234567, False, sample["note"]
    )
    append_ledger(new, [mine], project=project, when=WHEN)
    assert parse(new) == parse(old)
    # and the ledger header/typed read-back agrees through DictReader too
    with new.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["n_trades"] == "222"


def test_nan_sharpe_matches_p076_not_p105(tmp_path: Path) -> None:
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    old.write_text(HEADER_LINE + "\n")
    _p076(old, {"cell": "c", "n": 0, "sharpe": float("nan"), "note": "n"})
    append_ledger(
        new,
        [LedgerRow("optimize", "P076-c", 0, float("nan"), False, "n")],
        project="P076",
        when=WHEN,
    )
    assert parse(new) == parse(old)
    assert new.read_text().endswith(',0,"",0,"n"\n')
