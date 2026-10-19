"""Physical representation changes must preserve every scientific invocation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from qik.trial_storage import (
    TEMP_PREFIX,
    ParameterStore,
    SegmentStore,
    assert_disjoint_logs,
    atomic_json,
    canonical,
    clean_stale_temps,
    compact_log,
    compact_trial_logging,
    external_ledger_floor,
    external_manifest,
    external_path,
    file_ref,
    hash_matches,
    iter_trials,
    ordered_hash,
    publish_external_manifest,
    reconcile_partial_tail,
    verify_file,
)
from qik.trials import count_trials, log_trial, params_hash

EDGE_PARAMS: list[dict[Any, Any]] = [
    {},
    {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "negzero": -0.0},
    {"tuple": (1, (2, 3)), "list": [1, [2, 3]]},
    {2: "a", 10: "b", 1: "c"},  # numeric keys: 2 < 10 as ints, "10" < "2" as strings
    {"nested": {"k": {"deep": [{"x": None}, True, False]}, "n": {5: [1], 20: [2], 3: [0]}}},
    {"unicode": 'ünï   \x00 日本語 😀 "q" \\ \n\t', "ключ": [1]},
    {"when": date(2019, 1, 2), "stamp": datetime(2020, 3, 4, 5, 6, 7), "path": Path("a/b")},
    {"big": 10**30, "small": 1e-300, "float": 0.1 + 0.2, "int": -7},
    {1.5: "float-key", 2.5: "another"},
    {True: "bool-key"},
    {None: "none-key"},
    {"window": {"fit": [str(date(2019, 1, d)) for d in range(1, 29)]}, "config": {"x": 1}},
    {"empty_str": "", "empty": {}, "list": [], "nested_empty": {"a": {}}},
]


def _text(row: dict[str, Any]) -> bytes:
    return canonical(row)  # NaN-safe equality: compare serialisations


def _legacy_log(path: Path, params_list: list[dict[Any, Any]], repeats: int = 2) -> None:
    scopes = [("P1", "optimize"), ("P1", "debug"), ("P2", "optimize")]
    for round_ in range(repeats):
        for number, params in enumerate(params_list):
            project, stage = scopes[(number + round_) % len(scopes)]
            log_trial(
                params,
                {"round": round_, "cache_hit": bool(round_), "nan": float("nan")},
                project=project,
                stage=stage,
                path=path,
            )


def _compact_log(path: Path, number: int, tag: str) -> None:
    with compact_trial_logging(path):
        for i in range(number):
            log_trial(
                {"window": {"fit": [tag] * 20}, "config": {"i": i, "tag": tag}},
                {"n": i},
                project="P079",
                stage="optimize",
                path=path,
            )


def test_roundtrip_mixed_repeats_and_default_legacy(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    params = {"window": {"fit": [date(2019, 1, 2)] * 100}, "configuration": {"x": 2}}
    h = log_trial(params, {"cache_hit": False}, project="P", stage="s", path=path)
    with compact_trial_logging(path):
        for hit in (False, True, True):
            assert (
                log_trial(
                    params, {"cache_hit": hit, "error": None}, project="P", stage="s", path=path
                )
                == h
            )
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert "params" in raw[0] and "params_ref" in raw[-1]
    expanded = list(iter_trials(path, verify_legacy=True))
    assert len(expanded) == 4
    assert all(params_hash(row["params"]) == h for row in expanded)
    assert [row["metrics"]["cache_hit"] for row in expanded] == [False, False, True, True]
    assert count_trials(path).calls == 4 and count_trials(path).unique == 1
    assert ParameterStore(path).segments.count() == 2
    # Context exit does not activate compact storage for another legacy call.
    log_trial({"x": 3}, {}, project="P", stage="s", path=path)
    assert "params" in json.loads(path.read_text().splitlines()[-1])


# ----------------------------------------------------------------- golden equivalence


def test_legacy_to_compact_is_golden_equivalent(tmp_path: Path) -> None:
    legacy, compact = tmp_path / "legacy.jsonl", tmp_path / "compact.jsonl"
    _legacy_log(legacy, EDGE_PARAMS)
    assert compact_log(legacy, compact) == 2 * len(EDGE_PARAMS)
    assert all("params_ref" in json.loads(line) for line in compact.read_text().splitlines())
    old = list(iter_trials(legacy, verify_legacy=True))
    new = list(iter_trials(compact, verify_legacy=True))
    assert [_text(r) for r in old] == [_text(r) for r in new]
    for project, stage in [
        (None, None),
        ("P1", None),
        ("P2", None),
        ("P1", "optimize"),
        ("P1", "debug"),
        ("P2", "optimize"),
        ("P2", "debug"),
        ("none", None),
    ]:
        assert count_trials(legacy, project=project, stage=stage) == count_trials(
            compact, project=project, stage=stage
        ), (project, stage)
    total = count_trials(compact)
    assert total.calls == 2 * len(EDGE_PARAMS)
    assert total.unique == len({params_hash(p) for p in EDGE_PARAMS})


def test_native_compact_and_converted_logs_share_object_identities(tmp_path: Path) -> None:
    """Converting a legacy log yields the same content addresses as writing compact natively."""
    legacy, converted, native = (tmp_path / n for n in ("l.jsonl", "c.jsonl", "n.jsonl"))
    _legacy_log(legacy, EDGE_PARAMS, repeats=1)
    compact_log(legacy, converted)
    with compact_trial_logging(native):
        for number, params in enumerate(EDGE_PARAMS):
            scopes = [("P1", "optimize"), ("P1", "debug"), ("P2", "optimize")]
            project, stage = scopes[number % len(scopes)]
            log_trial(params, {}, project=project, stage=stage, path=native)
    refs = lambda p: [json.loads(line)["params_ref"] for line in p.read_text().splitlines()]  # noqa: E731
    assert refs(converted) == refs(native)
    assert {d for d, _ in ParameterStore(converted).segments.iter_objects()} == {
        d for d, _ in ParameterStore(native).segments.iter_objects()
    }


@pytest.mark.parametrize("params", EDGE_PARAMS, ids=lambda p: repr(p)[:30])
def test_edge_params_round_trip_and_hash_verify(tmp_path: Path, params: dict[Any, Any]) -> None:
    path = tmp_path / "calls.jsonl"
    with compact_trial_logging(path):
        h = log_trial(params, {}, project="P", stage="s", path=path)
    (row,) = iter_trials(path, verify_legacy=True)
    assert row["params_hash"] == h
    assert hash_matches(row["params"], h)
    # identical to what a legacy log would have parsed back
    legacy = tmp_path / "legacy.jsonl"
    log_trial(params, {}, project="P", stage="s", path=legacy)
    (old,) = iter_trials(legacy, verify_legacy=True)
    assert canonical(old["params"]) == canonical(row["params"])
    assert count_trials(path).unique == 1


def test_numeric_keys_need_order_preserving_hash(tmp_path: Path) -> None:
    """{2,10,1}: re-sorting the parsed (string-keyed) dict gives a different hash."""
    params = {2: "a", 10: "b", 1: "c"}
    parsed = json.loads(json.dumps(params, sort_keys=True))
    assert params_hash(parsed) != params_hash(params)
    assert ordered_hash(parsed) == params_hash(params)


def test_unreferenced_extra_object_does_not_change_counts(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    ParameterStore(path).put({"orphan": 1})
    assert count_trials(path).calls == 0


@pytest.mark.parametrize("damage", ["missing", "corrupt", "hash", "version", "ambiguous"])
def test_count_fails_closed_even_when_scope_filtered(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "calls.jsonl"
    with compact_trial_logging(path):
        log_trial({"a": [1, 2]}, {}, project="P", stage="s", path=path)
    row = json.loads(path.read_text())
    (shard,) = ParameterStore(path).segments.segment_files()
    if damage == "missing":
        shard.unlink()
    elif damage == "corrupt":
        shard.write_bytes(shard.read_bytes().replace(b" [1,2]", b" [2,1]"))
    elif damage == "hash":
        row["params_hash"] = "0" * 64
        path.write_text(json.dumps(row) + "\n")
    elif damage == "version":
        row["params_ref"]["version"] = 99
        path.write_text(json.dumps(row) + "\n")
    else:
        row["params"] = {"a": [1, 2]}
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        count_trials(path, project="different-project")


def test_restart_orphan_object_is_idempotent_and_repeated_calls_are_not_deduped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.jsonl"
    params = {"window": {"fit": ["2019-01-02"]}, "config": {"x": 1}}
    refs = ParameterStore(path).put(params)  # interrupted after object publish
    store = ParameterStore(path).segments
    before = {p: p.read_bytes() for p in store.segment_files()}
    assert ParameterStore(path).put(params) == refs
    for _attempt in range(2):
        with compact_trial_logging(path):
            log_trial(
                params,
                {"status": "unsupported", "error": "same error"},
                project="P",
                stage="s",
                path=path,
            )
    assert before == {p: p.read_bytes() for p in before}
    assert count_trials(path).calls == 2 and count_trials(path).unique == 1


def test_generic_legacy_count_still_accepts_complete_JSON_without_final_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.jsonl"
    params = {"x": 1}
    path.write_text(
        json.dumps(
            {
                "project": "P",
                "stage": "s",
                "params": params,
                "params_hash": params_hash(params),
                "metrics": {},
            }
        )
    )
    assert count_trials(path).calls == count_trials(path).unique == 1


# ------------------------------------------------------------------ partial tail / reconcile


def test_partial_invocation_preserved_and_next_write_refused(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b'{"ts":"partial')
    with pytest.raises(ValueError, match="partial trial invocation"):
        list(iter_trials(path))
    with compact_trial_logging(path), pytest.raises(ValueError, match="partial trial invocation"):
        log_trial({"x": 1}, {}, project="P", stage="s", path=path)
    assert path.read_bytes() == b'{"ts":"partial'


def test_error_message_names_a_real_reconciliation_function(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b'{"ts":"partial')
    with compact_trial_logging(path), pytest.raises(ValueError, match="reconcile_partial_tail"):
        log_trial({"x": 1}, {}, project="P", stage="s", path=path)


def test_reconcile_quarantines_fragment_and_keeps_complete_rows(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    _compact_log(path, 3, "a")
    good = path.read_bytes()
    path.write_bytes(good + b'{"ts":"2026-01-01T00:00:00+00:00","project":"P079","stage"')
    report = reconcile_partial_tail(path)
    assert report.action == "quarantined" and report.quarantine is not None
    assert report.quarantine.read_bytes().startswith(b'{"ts"')
    assert path.read_bytes() == good
    assert reconcile_partial_tail(path).action == "clean"
    assert count_trials(path).calls == 3
    with compact_trial_logging(path):
        log_trial({"x": 1}, {}, project="P079", stage="optimize", path=path)
    assert count_trials(path).calls == 4


def test_reconcile_seals_a_complete_record_so_the_call_stays_counted(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    _compact_log(path, 2, "a")
    path.write_bytes(path.read_bytes()[:-1])  # killed after the body, before the newline
    with pytest.raises(ValueError, match="partial trial invocation"):
        count_trials(path)
    assert reconcile_partial_tail(path).action == "sealed"
    assert count_trials(path).calls == 2
    assert path.read_bytes().endswith(b"}\n")


def test_reconcile_refuses_nothing_on_a_clean_or_empty_log(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b"")
    assert reconcile_partial_tail(path).action == "clean"
    _compact_log(path, 1, "a")
    before = path.read_bytes()
    assert reconcile_partial_tail(path).action == "clean"
    assert path.read_bytes() == before


def test_reconcile_with_a_single_line_log(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_bytes(b"not json at all")
    report = reconcile_partial_tail(path)
    assert report.action == "quarantined" and path.read_bytes() == b""
    assert report.quarantine is not None and report.quarantine.read_bytes() == b"not json at all"


# ------------------------------------------------------------------ locking / concurrency

WORKER = """
import sys
from pathlib import Path
from qik.trial_storage import compact_trial_logging
from qik.trials import log_trial
path, who, number = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with compact_trial_logging(path):
    for i in range(number):
        params = {
            "window": {"fit": ["2019-01-02"] * 50},
            "config": {"i": i % 20, "who": who if i % 2 else "shared"},
        }
        log_trial(params, {"n": i}, project="P", stage="s", path=path)
"""


def _run_worker(code: str, *args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", code, *args])


def test_concurrent_writers_lose_nothing(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    names, number = ["a", "b", "c", "d"], 40
    procs = [_run_worker(WORKER, str(path), name, str(number)) for name in names]
    assert [proc.wait(timeout=120) for proc in procs] == [0] * len(names)
    rows = list(iter_trials(path, verify_legacy=True))
    assert len(rows) == len(names) * number
    expected = {(i % 20, n if i % 2 else "shared") for n in names for i in range(number)}
    assert count_trials(path).calls == len(names) * number
    assert count_trials(path).unique == len(expected)
    store = ParameterStore(path).segments
    raw_records = sum(len(p.read_bytes().splitlines()) for p in store.segment_files())
    assert raw_records == store.count() == 1 + len(expected)  # shared window stored once
    assert len(store.segment_files()) <= 256


def test_log_lock_excludes_another_writer_and_check_is_atomic_with_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calls.jsonl"
    _compact_log(path, 1, "a")
    outcome: list[str] = []

    def writer() -> None:
        try:
            with compact_trial_logging(path):
                log_trial({"late": 1}, {}, project="P079", stage="optimize", path=path)
            outcome.append("wrote")
        except ValueError as exc:
            outcome.append(str(exc))

    holder = path.open("a+b")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    thread = threading.Thread(target=writer)
    thread.start()
    thread.join(timeout=0.3)
    assert thread.is_alive(), "writer must block on the log lock"
    holder.write(b'{"ts":"torn')  # another writer dies mid-append while holding the lock
    holder.flush()
    fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    holder.close()
    thread.join(timeout=30)
    assert outcome and "partial trial invocation" in outcome[0]
    assert path.read_bytes().endswith(b'{"ts":"torn')  # not merged into, not removed


def test_writer_killed_holding_the_lock_mid_line(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    _compact_log(path, 2, "a")
    code = (
        "import fcntl, os, signal, sys\n"
        "h = open(sys.argv[1], 'a+b')\n"
        "fcntl.flock(h.fileno(), fcntl.LOCK_EX)\n"
        'h.write(b\'{"ts":"2026-\'); h.flush()\n'
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    assert _run_worker(code, str(path)).wait(timeout=60) == -9
    with compact_trial_logging(path):
        with pytest.raises(ValueError, match="partial trial invocation"):
            log_trial({"x": 1}, {}, project="P079", stage="optimize", path=path)
        with pytest.raises(ValueError, match="partial trial invocation"):
            count_trials(path)
        assert reconcile_partial_tail(path).action == "quarantined"
        log_trial({"x": 1}, {}, project="P079", stage="optimize", path=path)
    assert count_trials(path).calls == 3


def test_writer_killed_between_object_and_invocation(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    code = (
        "import os, signal, sys\n"
        "from pathlib import Path\n"
        "from qik.trial_storage import ParameterStore\n"
        "ParameterStore(Path(sys.argv[1])).put({'config': {'x': 1}, 'window': {'fit': [1]}})\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    assert _run_worker(code, str(path)).wait(timeout=60) == -9
    assert not path.exists() and count_trials(path).calls == 0
    with compact_trial_logging(path):
        log_trial(
            {"config": {"x": 1}, "window": {"fit": [1]}}, {}, project="P", stage="s", path=path
        )
    assert count_trials(path).calls == 1
    assert ParameterStore(path).segments.count() == 2  # orphans reused, not duplicated


def test_writer_killed_mid_object_record_is_healed_by_next_put(tmp_path: Path) -> None:
    store = SegmentStore(tmp_path / "objs")
    payload = canonical({"k": [1, 2, 3]})
    digest = store.put(payload)
    shard = store.shard_path(digest)
    shard.unlink()
    code = (
        "import fcntl, os, signal, sys\n"
        "h = open(sys.argv[1], 'a+b')\n"
        "fcntl.flock(h.fileno(), fcntl.LOCK_EX)\n"
        "h.write(sys.argv[2].encode() + b' {\"k\":[1,'); h.flush()\n"
        "os.kill(os.getpid(), signal.SIGKILL)\n"
    )
    assert _run_worker(code, str(shard), digest).wait(timeout=60) == -9
    fresh = SegmentStore(tmp_path / "objs")
    with pytest.raises(ValueError, match="corrupt trial object"):
        fresh.get(digest)  # torn record is never trusted
    assert fresh.put(payload) == digest
    assert SegmentStore(tmp_path / "objs").get(digest) == payload
    assert shard.read_bytes().count(b"\n") == 2  # torn line terminated, then the real record


# ------------------------------------------------------------------ object packing


def test_objects_are_packed_into_few_segment_files(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with compact_trial_logging(path):
        for i in range(600):
            log_trial(
                {"window": {"fit": ["2019-01-02"] * 30}, "config": {"i": i}},
                {"n": i},
                project="P",
                stage="s",
                path=path,
            )
    store = ParameterStore(path).segments
    files = [p for p in store.root.rglob("*") if p.is_file()]
    assert len(files) == len(store.segment_files()) <= 256
    assert store.count() == 601  # 600 configs + 1 shared window
    assert count_trials(path).calls == 600 and count_trials(path).unique == 600


def test_segment_store_is_content_addressed_and_deterministic(tmp_path: Path) -> None:
    items = [canonical({"i": i, "pad": "x" * (i % 7)}) for i in range(300)]
    first, second = SegmentStore(tmp_path / "a"), SegmentStore(tmp_path / "b")
    digests = first.put_many(items)
    assert second.put_many(list(reversed(items))) == list(reversed(digests))
    assert second.put_many(items) == digests  # idempotent: nothing appended
    files_a = {p.name: p.read_bytes() for p in first.segment_files()}
    files_b = {p.name: p.read_bytes() for p in second.segment_files()}
    assert files_a == files_b
    assert all(first.get(d) == i for d, i in zip(digests, items, strict=True))
    assert [d for d, _ in first.iter_objects()] == sorted(set(digests), key=lambda d: (d[:2], d))
    with pytest.raises(ValueError):
        first.put(b"two\nlines")
    with pytest.raises(ValueError):
        first.get("../" + "0" * 61)
    with pytest.raises(FileNotFoundError):
        first.get("0" * 64)


def test_segment_store_reads_see_other_writers_and_digest_text_in_payload(tmp_path: Path) -> None:
    root = tmp_path / "objs"
    reader, writer = SegmentStore(root), SegmentStore(root)
    one = writer.put(canonical({"a": 1}))
    assert reader.get(one) == canonical({"a": 1})  # fills the reader's cache
    # a payload that merely MENTIONS another digest must never be mistaken for its record
    mention = writer.put(canonical({"ref": one}))
    other = writer.put(canonical({"b": 2}))
    assert reader.get(other) == canonical({"b": 2})
    assert reader.get(mention) == canonical({"ref": one})


def test_segment_cache_is_bounded(tmp_path: Path) -> None:
    store = SegmentStore(tmp_path / "objs", cache_bytes=1)
    digests = store.put_many([canonical({"i": i}) for i in range(100)])
    assert all(store.get(d) for d in digests)
    assert len(store._cache) == 1  # noqa: SLF001


def test_reads_one_file_per_object_layout_written_by_older_code(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    params = {"window": {"fit": [1]}, "config": {"x": 1}}
    root = path.with_name(path.name + ".objects")
    refs = {}
    for key, value in params.items():
        data = canonical(value)
        digest = hashlib.sha256(data).hexdigest()
        (root / digest[:2]).mkdir(parents=True, exist_ok=True)
        (root / digest[:2] / (digest + ".json")).write_bytes(data)
        refs[key] = digest
    row = {
        "project": "P",
        "stage": "s",
        "params_hash": params_hash(params),
        "metrics": {},
        "params_ref": {"version": 1, "objects": refs},
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n")
    assert count_trials(path).calls == 1
    assert ParameterStore(path).put(params) == row["params_ref"]  # no duplicate appended
    assert not ParameterStore(path).segments.segment_files()


def test_clean_stale_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    with compact_trial_logging(path):
        log_trial({"a": 1}, {}, project="P", stage="s", path=path)
    store = ParameterStore(path).segments
    old = 0  # the epoch: older than any threshold
    stale = [
        store.root / (TEMP_PREFIX + "new-style"),
        store.root / "ab" / "tmpq1w2e3",  # one-file-per-object layout temp
        tmp_path / (TEMP_PREFIX + "manifest"),
    ]
    (store.root / "ab").mkdir()
    fresh = store.root / (TEMP_PREFIX + "recent")
    for temp in (*stale, fresh):
        temp.write_bytes(b"x")
    for temp in stale:
        os.utime(temp, (old, old))
    keep_json = store.root / "ab" / ("0" * 64 + ".json")
    keep_json.write_bytes(b"{}")
    os.utime(keep_json, (old, old))
    assert set(store.clean_stale_temps()) == set(stale[:2])
    assert clean_stale_temps(tmp_path) == [stale[2]]
    assert fresh.exists() and keep_json.exists() and store.count() == 1


def test_atomic_json_leaves_no_temp_and_replaces_whole(tmp_path: Path) -> None:
    target = tmp_path / "m.json"
    atomic_json(target, {"b": 1, "a": 2})
    atomic_json(target, {"a": 3})
    assert json.loads(target.read_text()) == {"a": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


# ------------------------------------------------------------------ external manifests


def _ledger(path: Path, rows: int = 3) -> Path:
    header = "date,project,stage,variant_id,n_trades,sharpe_is,is_selected,note\n"
    path.write_text(header + "".join(f"d,P,s,v{i},1,0.1,0,n\n" for i in range(rows)))
    return path


def _peer(path: Path) -> Path:
    inline = {"a": 1}
    stored = {"b": [2, 3], "window": {"fit": ["x"]}}
    refs = ParameterStore(path).put(stored)
    lines = ["project,stage,note\n"]
    for note in (
        {"params_hash": params_hash(inline), "scientific": inline},
        {"params_hash": params_hash(stored), "scientific_ref": refs},
    ):
        lines.append('P079,optimize,"' + json.dumps(note).replace('"', '""') + '"\n')
    path.write_text("".join(lines))
    return path


def _world(tmp_path: Path) -> dict[str, Path]:
    base, one, two = tmp_path / "t.jsonl", tmp_path / "src1.jsonl", tmp_path / "src2.jsonl"
    _compact_log(base, 3, "base")
    _compact_log(one, 4, "one")
    _compact_log(two, 5, "two")
    return {
        "base": base,
        "one": one,
        "two": two,
        "ledger": _ledger(tmp_path / "ledger.csv"),
        "peer": _peer(tmp_path / "peer.csv"),
    }


def test_external_manifest_counts_union_with_relative_paths_and_moves(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    w = _world(tmp_path / "a")
    value = publish_external_manifest(
        w["base"],
        [w["one"], w["two"]],
        ledger=w["ledger"],
        additional_ledger_rows=7,
        peer_ledgers=[w["peer"]],
    )
    assert count_trials(w["base"]).calls == 3 + 4 + 5
    assert count_trials(w["base"]).unique == 12
    assert external_ledger_floor(w["base"], w["ledger"]) == 7
    manifest_text = external_path(w["base"]).read_text()
    manifest = json.loads(manifest_text)
    refs = [
        manifest["base"],
        *manifest["sources"],
        *manifest["peer_ledgers"],
        manifest["accounting"]["ledger"],
        manifest["index"],
    ]
    assert all(not Path(r["path"]).is_absolute() for r in refs)
    assert str(tmp_path) not in manifest_text
    assert value["sources"] == manifest["sources"]
    # republishing identical content is idempotent
    publish_external_manifest(
        w["base"],
        [w["one"], w["two"]],
        ledger=w["ledger"],
        additional_ledger_rows=7,
        peer_ledgers=[w["peer"]],
    )
    # the whole tree can move: nothing in the manifest is bound to the old location
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "a", moved)
    shutil.rmtree(tmp_path / "a")
    assert count_trials(moved / "t.jsonl").calls == 12
    assert external_ledger_floor(moved / "t.jsonl", moved / "ledger.csv") == 7


def test_external_manifest_with_legacy_absolute_paths_still_verifies(tmp_path: Path) -> None:
    w = _world(tmp_path)
    ledger = file_ref(w["ledger"])
    value = {
        "version": 1,
        "base": file_ref(w["base"]),
        "sources": [file_ref(w["one"])],
        "peer_ledgers": [file_ref(w["peer"])],
        "accounting": {"ledger": ledger, "additional_ledger_rows": 2},
    }
    index = tmp_path / "idx.json"
    atomic_json(index, value)
    atomic_json(external_path(w["base"]), {**value, "index": file_ref(index)})
    assert Path(json.loads(external_path(w["base"]).read_text())["base"]["path"]).is_absolute()
    assert count_trials(w["base"]).calls == 7
    assert external_ledger_floor(w["base"], w["ledger"]) == 2


def test_external_manifest_absent_means_no_floor_and_no_extras(tmp_path: Path) -> None:
    w = _world(tmp_path)
    assert external_manifest(w["base"]) is None
    assert external_ledger_floor(w["base"], w["ledger"]) == 0
    assert count_trials(w["base"]).calls == 3


def test_external_floor_bound_to_its_ledger(tmp_path: Path) -> None:
    w = _world(tmp_path)
    publish_external_manifest(w["base"], [w["one"]], ledger=w["ledger"], additional_ledger_rows=1)
    other = _ledger(tmp_path / "other.csv", rows=4)
    with pytest.raises(ValueError, match="different ledger"):
        external_ledger_floor(w["base"], other)


@pytest.mark.parametrize("tamper", ["source", "base", "ledger", "index", "peer", "peer_identity"])
def test_external_manifest_fails_closed_when_anything_changes(tmp_path: Path, tamper: str) -> None:
    w = _world(tmp_path)
    publish_external_manifest(
        w["base"],
        [w["one"]],
        ledger=w["ledger"],
        additional_ledger_rows=1,
        peer_ledgers=[w["peer"]],
    )
    target = {
        "source": w["one"],
        "base": w["base"],
        "ledger": w["ledger"],
        "peer": w["peer"],
        "index": w["base"].with_name("t.jsonl.external.index.json"),
        "peer_identity": w["peer"],
    }[tamper]
    if tamper == "peer_identity":
        # a ledger edit that keeps bytes... but the manifest pins bytes, so re-pin to isolate identity
        text = target.read_text().replace('{""a"": 1}', '{""a"": 2}')
        assert text != target.read_text()
        target.write_text(text)
        manifest = json.loads(external_path(w["base"]).read_text())
        manifest["peer_ledgers"] = [file_ref(w["peer"], w["base"].parent)]
        value = {k: v for k, v in manifest.items() if k != "index"}
        index = w["base"].with_name("t.jsonl.external.index.json")
        atomic_json(index, value)
        atomic_json(external_path(w["base"]), {**value, "index": file_ref(index, w["base"].parent)})
    else:
        with target.open("ab") as stream:
            stream.write(b" " if tamper in {"ledger", "peer", "index"} else b"\n")
    with pytest.raises(ValueError):
        count_trials(w["base"])


def test_external_sources_overlapping_the_base_are_rejected(tmp_path: Path) -> None:
    w = _world(tmp_path)
    base = w["base"]
    kwargs: dict[str, Any] = {"ledger": w["ledger"], "additional_ledger_rows": 0}
    copy = tmp_path / "copy.jsonl"
    shutil.copy(base, copy)
    symlink = tmp_path / "link.jsonl"
    symlink.symlink_to(base)
    hardlink = tmp_path / "hard.jsonl"
    os.link(base, hardlink)
    prefix = tmp_path / "prefix.jsonl"
    prefix.write_bytes(b"".join(base.read_bytes().splitlines(keepends=True)[:2]))
    extended = tmp_path / "extended.jsonl"
    extended.write_bytes(base.read_bytes() + w["one"].read_bytes())
    for overlap in (base, copy, symlink, hardlink, prefix, extended):
        with pytest.raises(ValueError, match="overlaps|duplicate"):
            publish_external_manifest(base, [overlap], **kwargs)
    with pytest.raises(ValueError, match="overlaps|duplicate"):
        publish_external_manifest(base, [w["one"], w["one"]], **kwargs)
    with pytest.raises(ValueError, match="overlaps"):  # two sources sharing an invocation
        shared = tmp_path / "shared.jsonl"
        shutil.copy(w["one"], shared)
        publish_external_manifest(base, [w["one"], shared], **kwargs)
    assert not external_path(base).exists()
    # disjoint logs are accepted, and calls are exactly the sum
    publish_external_manifest(base, [w["one"], w["two"]], **kwargs)
    assert count_trials(base).calls == 3 + 4 + 5


def test_hand_written_manifest_cannot_smuggle_an_overlapping_source(tmp_path: Path) -> None:
    w = _world(tmp_path)
    copy = tmp_path / "copy.jsonl"
    shutil.copy(w["base"], copy)
    root = tmp_path
    value = {
        "version": 1,
        "base": file_ref(w["base"], root),
        "sources": [file_ref(copy, root)],
        "peer_ledgers": [],
        "accounting": {"ledger": file_ref(w["ledger"], root), "additional_ledger_rows": 0},
    }
    index = tmp_path / "idx.json"
    atomic_json(index, value)
    atomic_json(external_path(w["base"]), {**value, "index": file_ref(index, root)})
    with pytest.raises(ValueError, match="overlaps"):
        count_trials(w["base"])


def test_nested_manifest_and_partial_ledger_are_rejected(tmp_path: Path) -> None:
    w = _world(tmp_path)
    publish_external_manifest(w["one"], [w["two"]], ledger=w["ledger"], additional_ledger_rows=0)
    with pytest.raises(ValueError, match="nested"):
        publish_external_manifest(
            w["base"], [w["one"]], ledger=w["ledger"], additional_ledger_rows=0
        )
    partial = tmp_path / "partial.csv"
    partial.write_text(w["ledger"].read_text().rstrip("\n"))
    with pytest.raises(ValueError, match="partial external ledger"):
        publish_external_manifest(w["base"], [w["two"]], ledger=partial, additional_ledger_rows=0)


def test_assert_disjoint_logs_ignores_blank_lines_and_in_file_repeats(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"x":1}\n\n{"x":1}\n')  # a log may repeat its own line
    b.write_text('{"x":2}\n\n')
    assert_disjoint_logs(a, [b])
    verify_file(file_ref(a))
