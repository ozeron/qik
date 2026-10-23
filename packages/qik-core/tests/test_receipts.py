from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from qik.receipts import Pins, diff_fields, write_receipt


def pins() -> Pins:
    return Pins("a" * 40, "b" * 64, "c" * 64, ("2023-01-01", "2023-12-31"), "d" * 64)


def test_field_diffs_are_absolute_and_do_not_drop_unpaired_rows() -> None:
    a = pd.DataFrame({"px": [100.0, 200.0], "reason": ["tp", None]})
    b = pd.DataFrame({"px": [100.01, 201.0, 300.0], "reason": ["tp", None, "sl"]})
    got = diff_fields(a, b, fields=["px", "reason"], tol={"px": 0.02})
    assert (got.equal_rows, got.changed_rows, got.old_rows, got.new_rows) == (1, 2, 2, 3)
    assert got.changes.field.tolist() == ["px", "__row__"]
    assert not got.equal
    assert diff_fields(a, a, fields=["reason"], tol=0).equal
    with pytest.raises(ValueError, match="missing"):
        diff_fields(a, b, fields=["missing"], tol=0)
    with pytest.raises(ValueError, match="nonfinite"):
        diff_fields(pd.DataFrame({"px": [float("inf")]}), a, fields=["px"], tol=0)


def test_receipt_is_exact_atomic_no_overwrite_even_concurrently(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    old = tmp_path / "old.csv"
    new = tmp_path / "new.csv"
    old.write_text("px,reason\n100,tp\n")
    new.write_text("px,reason\n101,tp\n")

    def publish() -> str:
        try:
            receipt = write_receipt(path, old=old, new=new, pins=pins(), fields=["px", "reason"])
            assert receipt.diff.changed_rows == 1
            return "published"
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: publish(), range(2))) == ["published", "refused"]
    data = json.loads(path.read_text())
    assert data["verdict"] == "DIFF" and data["changes"][0]["delta"] == 1
    assert data["old"]["sha256"] != data["new"]["sha256"]
    assert data["pins"]["tape_sha256"] == pins().tape_sha256
    assert list(tmp_path.glob(".qik-receipt-*")) == []


def test_dataframe_receipt_and_missing_pins(tmp_path: Path) -> None:
    a = pd.DataFrame({"value": [1.0, None]})
    got = write_receipt(tmp_path / "same.json", old=a, new=a, pins=pins(), fields=["value"])
    assert got.diff.equal and got.diff.equal_rows == 2
    with pytest.raises(ValueError, match="source_git_sha"):
        Pins("short", "b" * 64, "c" * 64, ("2023-01-01", "2023-12-31"), "d" * 64)


def test_exact_float_and_large_integer_evidence_is_lossless(tmp_path: Path) -> None:
    a = pd.DataFrame({"value": [1.0000000000000002]})
    b = pd.DataFrame({"value": [1.0000000000000004]})
    receipt = write_receipt(tmp_path / "float.json", old=a, new=b, pins=pins(), fields=["value"])
    value = json.loads(receipt.path.read_text())
    assert value["old"]["sha256"] != value["new"]["sha256"]
    assert value["changes"][0]["old"] == a.value.iloc[0]
    assert value["changes"][0]["new"] == b.value.iloc[0]
    c = pd.DataFrame({"value": [2**60]})
    d = pd.DataFrame({"value": [2**60 + 1]})
    assert diff_fields(c, d, fields=["value"], tol=0).changed_rows == 1


@pytest.mark.parametrize("other", [[], [None], [float("inf")]])
@pytest.mark.parametrize("reverse", [False, True])
def test_nonfinite_cells_are_refused_even_without_known_counterpart(
    other: list, reverse: bool
) -> None:
    a = pd.DataFrame({"px": [float("inf")]})
    b = pd.DataFrame({"px": other})
    old, new = (b, a) if reverse else (a, b)
    with pytest.raises(ValueError, match="nonfinite"):
        diff_fields(old, new, fields=["px"], tol=0)
