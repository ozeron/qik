"""Opt-in, lossless trial storage. Hashes always identify the expanded parameters.

Objects precede invocation appends. Orphan objects after interruption are harmless.
Writers to one log are serialised by an advisory ``flock`` on the log itself; the partial
trailing line check is made under that lock, atomically with the append. A partial
invocation line fails closed and is never silently removed: ``reconcile_partial_tail`` is
the explicit way out (seal a complete record, or quarantine the fragment to a sidecar).
External manifests reference CLOSED, mutually disjoint logs, never copied invocations.

Parameter objects are content addressed (sha256 of their canonical JSON) and packed into at
most 256 append-only shard files, so a long run does not create one file per trial.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qik.runs import params_hash

_COMPACT: ContextVar[frozenset[Path]] = ContextVar("compact_trial_paths", default=frozenset())
_SHA = re.compile(r"[0-9a-f]{64}\Z")
TEMP_PREFIX = ".qik-tmp-"
_LEGACY_TEMP_PREFIX = "tmp"  # tempfile default, used by the one-file-per-object layout
SEGMENT_SUFFIX = ".seg"
_REQUIRED_ROW_KEYS = frozenset({"project", "stage", "params_hash"})
_DISJOINT_OK: set[tuple[str, ...]] = set()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()


def _preserved(value: Any) -> bytes:
    """Canonical text of a value parsed from JSON: key order kept (it was written sorted)."""
    return json.dumps(value, default=str, separators=(",", ":")).encode()


_REF_MARK = "\u0000qik-params-ref\u0000"


def dump_row(row: dict[str, Any]) -> bytes:
    """One newline-terminated log line, keys sorted exactly like the legacy writer, except
    ``params_ref.objects``, which keeps the order params_hash saw (numeric keys numerically,
    not as strings): re-sorting it would break hash verification for non-string keys."""
    ref = row.get("params_ref")
    if ref is None:
        return (json.dumps(row, sort_keys=True, default=str) + "\n").encode()
    shell = json.dumps({**row, "params_ref": _REF_MARK}, sort_keys=True, default=str)
    token = json.dumps(_REF_MARK)
    if shell.count(token) != 1:
        raise ValueError("trial row collides with the params_ref marker")
    ref_text = json.dumps({"objects": ref["objects"], "version": ref["version"]}, default=str)
    return (shell.replace(token, ref_text) + "\n").encode()


def ordered_hash(params: Any) -> str:
    """Hash of params as loaded from JSON, keeping file order (no re-sort).

    ``params_hash`` sorts keys, which reorders numeric-origin keys ("2" < "10" as strings,
    2 < 10 as ints). A row parsed back from its own log keeps the order it was written in.
    """
    return hashlib.sha256(
        json.dumps(params, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def hash_matches(params: Any, expected: str) -> bool:
    return expected in (params_hash(params), ordered_hash(params))


def file_ref(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Fingerprint a file. ``root`` stores the path relative to it, else absolute."""
    resolved = path.resolve()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"trial storage source changed while hashing: {path}")
        size = after.st_size
    shown = str(resolved) if root is None else os.path.relpath(resolved, root.resolve())
    return {"path": shown, "bytes": size, "sha256": sha}


def ref_path(ref: dict[str, Any], root: Path | None = None) -> Path:
    path = Path(ref["path"])
    if path.is_absolute():
        return path.resolve()
    if root is None:
        raise ValueError(f"relative trial storage reference needs a root: {path}")
    return (root / path).resolve()


def verify_file(ref: dict[str, Any], root: Path | None = None) -> Path:
    path = ref_path(ref, root)
    fresh = file_ref(path)
    if (fresh["bytes"], fresh["sha256"]) != (ref["bytes"], ref["sha256"]):
        raise ValueError(f"trial storage reference changed: {path}")
    return path


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete manifest atomically; prior readers see the prior version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=TEMP_PREFIX, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temp, path)
        sync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _remove_old(candidates: Iterable[Path], older_than: float) -> list[Path]:
    cutoff = time.time() - older_than  # noqa: TID251 (file age, not strategy time)
    removed = []
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(path)
        except FileNotFoundError:
            continue  # a concurrent cleaner or the owner finished first
    return removed


def clean_stale_temps(directory: Path, *, older_than: float = 3600.0) -> list[Path]:
    """Remove ``atomic_json`` temp files that a killed writer left in ``directory``."""
    if not directory.is_dir():
        return []
    return _remove_old(directory.glob(TEMP_PREFIX + "*"), older_than)


@contextmanager
def compact_trial_logging(path: Path) -> Iterator[None]:
    """Only this explicit path/context opts in; generic qik defaults stay legacy."""
    token = _COMPACT.set(_COMPACT.get() | {path.resolve()})
    try:
        yield
    finally:
        _COMPACT.reset(token)


def compact_enabled(path: Path) -> bool:
    return path.resolve() in _COMPACT.get()


# --------------------------------------------------------------------------- log appends


def partial_tail_error(path: Path) -> ValueError:
    return ValueError(
        f"partial trial invocation retained: {path}; inspect it, then call "
        "qik.trial_storage.reconcile_partial_tail(path) to seal or quarantine the fragment"
    )


def _ends_with_newline(descriptor: int) -> bool:
    size = os.fstat(descriptor).st_size
    return size == 0 or os.pread(descriptor, 1, size - 1) == b"\n"


def append_record(path: Path, line: bytes, *, strict: bool) -> None:
    """Append one newline-terminated record while holding an exclusive advisory lock.

    ``strict`` (compact logs) also refuses a partial trailing line, checked under the
    same lock as the write, and makes the append durable.
    """
    if not line.endswith(b"\n"):
        raise ValueError("trial record must be newline terminated")
    created = not path.exists()
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # released when the handle closes
        if strict and not _ends_with_newline(handle.fileno()):
            raise partial_tail_error(path)
        handle.write(line)
        handle.flush()
        if strict:
            os.fsync(handle.fileno())
    if strict and created:
        sync_directory(path.parent)


@dataclass(frozen=True)
class TailReport:
    action: Literal["clean", "sealed", "quarantined"]
    fragment_bytes: int = 0
    quarantine: Path | None = None


def reconcile_partial_tail(path: Path) -> TailReport:
    """Resolve a trailing line without newline (a writer killed mid-append), explicitly.

    A complete, verifiable invocation record is sealed by appending the newline, so the call
    stays counted (conservative for the multiple-testing N). Anything else is a fragment: it
    is saved byte-for-byte to ``<log>.partial-<sha12>`` and then cut from the log.
    """
    with path.open("r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        descriptor = handle.fileno()
        if _ends_with_newline(descriptor):
            return TailReport("clean")
        size = os.fstat(descriptor).st_size
        start = size
        while start > 0:  # find the start of the last line, reading backwards
            chunk_start = max(0, start - 65536)
            chunk = os.pread(descriptor, start - chunk_start, chunk_start)
            cut = chunk.rfind(b"\n")
            if cut != -1:
                start = chunk_start + cut + 1
                break
            start = chunk_start
        fragment = os.pread(descriptor, size - start, start)
        if _is_complete_record(fragment, path):
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(descriptor)
            return TailReport("sealed", len(fragment))
        sha = hashlib.sha256(fragment).hexdigest()[:12]
        sidecar = path.with_name(f"{path.name}.partial-{sha}")
        if not sidecar.exists() or sidecar.read_bytes() != fragment:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=TEMP_PREFIX, delete=False
            ) as stream:
                temp = Path(stream.name)
                stream.write(fragment)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, sidecar)
            sync_directory(path.parent)
        handle.truncate(start)
        handle.flush()
        os.fsync(descriptor)
        return TailReport("quarantined", len(fragment), sidecar)


def _is_complete_record(fragment: bytes, path: Path) -> bool:
    try:
        row = json.loads(fragment)
        if not isinstance(row, dict) or not _REQUIRED_ROW_KEYS <= row.keys():
            return False
        expand_record(row, path, verify_legacy=True)
    except (ValueError, KeyError, TypeError, OSError):
        return False
    return True


# --------------------------------------------------------------------------- object store


class SegmentStore:
    """Content-addressed single-line objects packed into <= 256 append-only shard files.

    Record: ``<sha256 hex of payload> <payload>\\n`` in ``<root>/<digest[:2]>.seg``. A record
    is trusted only if its payload hashes to its digest, so a torn or garbage line is
    ignored on read and healed (newline-terminated) by the next append. The digest is the
    identity; shards are only a physical location. Reads keep a byte-bounded shard cache.
    """

    def __init__(self, root: Path, *, cache_bytes: int = 128 << 20) -> None:
        self.root = root
        self._limit = cache_bytes
        self._cache: OrderedDict[str, tuple[tuple[int, int, int], bytes]] = OrderedDict()
        self._cached = 0

    @staticmethod
    def _check(digest: str) -> str:
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("invalid trial object digest")
        return digest

    def shard_path(self, digest: str) -> Path:
        return self.root / (self._check(digest)[:2] + SEGMENT_SUFFIX)

    def _legacy_path(self, digest: str) -> Path:
        return self.root / digest[:2] / (digest + ".json")  # one-file-per-object layout

    @staticmethod
    def _records(data: bytes, digest: str) -> Iterator[tuple[bytes, bool]]:
        needle = digest.encode() + b" "
        start = 0
        while (at := data.find(needle, start)) != -1:
            start = at + 1
            if at and data[at - 1] != 10:  # not at a line start: payload text, not a record
                continue
            end = data.find(b"\n", at)
            if end == -1:
                yield b"", False  # unterminated fragment
                continue
            payload = data[at + len(needle) : end]
            yield payload, hashlib.sha256(payload).hexdigest() == digest

    def _read_shard(self, name: str) -> bytes:
        path = self.root / (name + SEGMENT_SUFFIX)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return b""
        key = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        hit = self._cache.get(name)
        if hit is not None and hit[0] == key:
            self._cache.move_to_end(name)
            return hit[1]
        data = path.read_bytes()
        if hit is not None:
            self._cached -= len(hit[1])
        self._cache[name] = (key, data)
        self._cached += len(data)
        while self._cached > self._limit and len(self._cache) > 1:
            _, (_, dropped) = self._cache.popitem(last=False)
            self._cached -= len(dropped)
        return data

    def get(self, digest: str) -> bytes:
        corrupt = False
        for payload, ok in self._records(self._read_shard(self._check(digest)[:2]), digest):
            if ok:
                return payload
            corrupt = True
        legacy = self._legacy_path(digest)
        if legacy.exists():
            payload = legacy.read_bytes()
            if hashlib.sha256(payload).hexdigest() == digest:
                return payload
            corrupt = True
        if corrupt:
            raise ValueError(f"corrupt trial object: {digest}")
        raise FileNotFoundError(f"missing trial object: {digest}")

    def put(self, data: bytes) -> str:
        return self.put_many([data])[0]

    def put_many(self, items: Iterable[bytes]) -> list[str]:
        """Store objects, one lock and fsync per touched shard; return their digests in order.

        Within a shard new records are written sorted by digest, so bulk output is
        deterministic regardless of input order.
        """
        digests: list[str] = []
        by_shard: dict[str, dict[str, bytes]] = {}
        for data in items:
            if b"\n" in data:
                raise ValueError("trial object must be a single line")
            digest = hashlib.sha256(data).hexdigest()
            digests.append(digest)
            by_shard.setdefault(digest[:2], {})[digest] = data
        for name in sorted(by_shard):
            self._append_missing(name, by_shard[name])
        return digests

    def _legacy_has(self, digest: str, data: bytes) -> bool:
        legacy = self._legacy_path(digest)
        return legacy.exists() and legacy.read_bytes() == data

    def _append_missing(self, name: str, wanted: dict[str, bytes]) -> None:
        wanted = {d: b for d, b in wanted.items() if not self._legacy_has(d, b)}
        if not wanted:
            return
        root_existed = self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (name + SEGMENT_SUFFIX)
        created = not path.exists()
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            data = handle.read()
            if len(wanted) > 8:
                present = self._verified_digests(data)
                missing = {d: b for d, b in sorted(wanted.items()) if d not in present}
            else:
                missing = {
                    d: b
                    for d, b in sorted(wanted.items())
                    if not any(ok for _, ok in self._records(data, d))
                }
            if missing:
                out = b"".join(d.encode() + b" " + b + b"\n" for d, b in missing.items())
                if data and not data.endswith(b"\n"):
                    out = b"\n" + out  # heal a torn tail; the fragment never verifies
                handle.write(out)
                handle.flush()
                os.fsync(handle.fileno())
        if created:
            sync_directory(self.root)
            if not root_existed:
                sync_directory(self.root.parent)

    @staticmethod
    def _verified_digests(data: bytes) -> set[str]:
        present = set()
        for line in data.split(b"\n"):
            digest, _, payload = line.partition(b" ")
            if len(digest) == 64 and hashlib.sha256(payload).hexdigest().encode() == digest:
                present.add(digest.decode())
        return present

    def iter_objects(self) -> Iterator[tuple[str, bytes]]:
        """Every verified (digest, payload), shard by shard, in digest order; skips garbage."""
        for shard in sorted(self.root.glob("*" + SEGMENT_SUFFIX)):
            seen: set[str] = set()
            records = []
            for line in shard.read_bytes().split(b"\n"):
                digest, _, payload = line.partition(b" ")
                text = digest.decode("ascii", "replace")
                if (
                    len(digest) == 64
                    and text not in seen
                    and hashlib.sha256(payload).hexdigest() == text
                ):
                    seen.add(text)
                    records.append((text, payload))
            yield from sorted(records)

    def count(self) -> int:
        return sum(1 for _ in self.iter_objects())

    def segment_files(self) -> list[Path]:
        return sorted(self.root.glob("*" + SEGMENT_SUFFIX))

    def clean_stale_temps(self, *, older_than: float = 3600.0) -> list[Path]:
        """Remove temp files from killed writers (current and one-file-per-object layouts)."""
        if not self.root.is_dir():
            return []
        found = list(self.root.glob(TEMP_PREFIX + "*"))
        for folder in self.root.iterdir():
            if folder.is_dir():
                found += [
                    p
                    for p in folder.iterdir()
                    if p.name.startswith((TEMP_PREFIX, _LEGACY_TEMP_PREFIX)) and p.suffix != ".json"
                ]
        return _remove_old(found, older_than)


class ParameterStore:
    """Deduplicate top-level values, especially windows, execution and cells.

    No global or unbounded payload cache. Each object is verified when used.
    A reference expands to exactly the JSON representation used by params_hash.
    """

    def __init__(self, log: Path) -> None:
        self.root = log.with_name(log.name + ".objects")
        self.segments = SegmentStore(self.root)

    def put(self, params: dict[str, Any], *, preserve_order: bool = False) -> dict[str, Any]:
        """Store each top-level value; ``preserve_order`` keeps the dict's own key order
        (for params parsed back from a log, whose order is already the hashed order)."""
        if preserve_order:
            items = list(params.items())
            texts = [_preserved(value) for _, value in items]
        else:
            items = sorted(params.items(), key=lambda item: item[0])
            texts = [canonical(value) for _, value in items]
        digests = self.segments.put_many(texts)
        # Original key objects on purpose: the row dump keeps this order, which is the order
        # params_hash saw (numeric keys numerically); strings only appear once parsed back.
        refs = {key: digest for (key, _), digest in zip(items, digests, strict=True)}
        return {"version": 1, "objects": refs}

    def expand(self, ref: dict[str, Any]) -> dict[str, Any]:
        if set(ref) != {"version", "objects"} or ref["version"] != 1:
            raise ValueError("unsupported trial parameter reference")
        return {
            key: json.loads(self.segments.get(digest)) for key, digest in ref["objects"].items()
        }


def expand_record(
    row: dict[str, Any],
    path: Path,
    *,
    verify_legacy: bool = False,
    store: ParameterStore | None = None,
) -> dict[str, Any]:
    if "params_ref" in row:
        if "params" in row:
            raise ValueError("ambiguous compact trial parameters")
        row = dict(row)
        row["params"] = (store or ParameterStore(path)).expand(row.pop("params_ref"))
        verify_legacy = True
    if verify_legacy and not hash_matches(row["params"], row["params_hash"]):
        raise ValueError("expanded trial params_hash mismatch")
    return row


def compact_log(src: Path, dst: Path) -> int:
    """Rewrite a legacy (or mixed) log as compact records at a NEW path; returns row count."""
    store = ParameterStore(dst)
    count = 0
    with dst.open("xb") as out:
        for row in iter_trials(src, verify_legacy=True, include_external=False):
            row = dict(row)
            row["params_ref"] = store.put(row.pop("params"), preserve_order=True)
            out.write(dump_row(row))
            count += 1
        out.flush()
        os.fsync(out.fileno())
    sync_directory(dst.parent)
    return count


# --------------------------------------------------------------------------- external logs


def external_path(path: Path) -> Path:
    return path.with_name(path.name + ".external.json")


def assert_disjoint_logs(base: Path, sources: Sequence[Path]) -> None:
    """Reject sources that share a file, an inode, or any invocation line with each other
    or the base log: a duplicated invocation must not inflate ``calls``.

    An identical line (same second, params and metrics) is treated as a duplicate; that is
    the fail-closed reading of a copied or re-attached log.
    """
    logs = [base.resolve(), *(source.resolve() for source in sources)]
    if len(set(logs)) != len(logs):
        raise ValueError("duplicate/nested external trial log")
    inodes = {(stat.st_dev, stat.st_ino) for stat in (log.stat() for log in logs)}
    if len(inodes) != len(logs):
        raise ValueError("external trial log overlaps another log (same file)")
    with tempfile.TemporaryDirectory(prefix="qik-trial-overlap-") as folder:
        with closing(sqlite3.connect(str(Path(folder) / "lines.sqlite"))) as db:
            db.execute("CREATE TABLE lines (d BLOB PRIMARY KEY, log INTEGER) WITHOUT ROWID")
            for index, log in enumerate(logs):
                with log.open("rb") as stream:
                    for line in stream:
                        body = line.strip()
                        if not body:
                            continue
                        digest = hashlib.blake2b(body, digest_size=32).digest()
                        added = db.execute(
                            "INSERT OR IGNORE INTO lines VALUES (?, ?)", (digest, index)
                        )
                        if added.rowcount:
                            continue
                        (owner,) = db.execute(
                            "SELECT log FROM lines WHERE d = ?", (digest,)
                        ).fetchone()
                        if owner != index:
                            raise ValueError(
                                f"external trial log overlaps another log (shared invocation): {log}"
                            )


def _ensure_disjoint(
    base_ref: dict[str, Any], source_refs: list[dict[str, Any]], root: Path
) -> None:
    key = (base_ref["sha256"], *(ref["sha256"] for ref in source_refs))
    if key in _DISJOINT_OK:
        return
    assert_disjoint_logs(ref_path(base_ref, root), [ref_path(ref, root) for ref in source_refs])
    _DISJOINT_OK.add(key)


def external_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = external_path(path)
    if not manifest_path.exists():
        return None
    root = manifest_path.parent
    with manifest_path.open() as stream:
        value = json.load(stream)
    if value["version"] != 1 or verify_file(value["base"], root) != path.resolve():
        raise ValueError("invalid external trial manifest base")
    index = verify_file(value["index"], root)
    with index.open() as stream:
        indexed = json.load(stream)
    if indexed != {k: v for k, v in value.items() if k != "index"}:
        raise ValueError("external trial index/manifest mismatch")
    seen = {path.resolve()}
    for ref in value["sources"]:
        source = verify_file(ref, root)
        if source in seen or external_path(source).exists():
            raise ValueError("duplicate/nested external trial log")
        seen.add(source)
    _ensure_disjoint(value["base"], value["sources"], root)
    if "accounting" in value:
        verify_file(value["accounting"]["ledger"], root)
    for ref in value.get("peer_ledgers", []):
        source = verify_file(ref, root)
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                note = json.loads(row["note"])
                scientific = (
                    ParameterStore(source).expand(note["scientific_ref"])
                    if "scientific_ref" in note
                    else note["scientific"]
                )
                if not hash_matches(scientific, note["params_hash"]):
                    raise ValueError("external peer ledger identity changed")
    return value


def _closed_csv(path: Path) -> None:
    if path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                raise ValueError(f"partial external ledger retained: {path}")


def publish_external_manifest(
    log: Path,
    sources: Sequence[Path],
    *,
    ledger: Path,
    additional_ledger_rows: int,
    peer_ledgers: Sequence[Path] = (),
    index: Path | None = None,
) -> dict[str, Any]:
    """Publish ``<log>.external.json`` (+ index) for CLOSED disjoint source logs.

    All paths are stored relative to the manifest directory, so the tree can move.
    Publishing identical content again is idempotent; a changed index is refused.
    """
    root = external_path(log).parent
    if any(external_path(source).exists() for source in sources):
        raise ValueError("duplicate/nested external trial log")
    assert_disjoint_logs(log, sources)
    for ledger_file in (ledger, *peer_ledgers):
        _closed_csv(ledger_file)
    refs = {
        "base": file_ref(log, root),
        "sources": [file_ref(source, root) for source in sources],
        "peer_ledgers": [file_ref(peer, root) for peer in peer_ledgers],
        "ledger": file_ref(ledger, root),
    }
    value = {
        "version": 1,
        "base": refs["base"],
        "sources": refs["sources"],
        "peer_ledgers": refs["peer_ledgers"],
        "accounting": {
            "ledger": refs["ledger"],
            "additional_ledger_rows": int(additional_ledger_rows),
        },
    }
    index = index or log.with_name(log.name + ".external.index.json")
    # Sources must stay closed while accounting runs: stale hashes must never be published.
    for ref in (refs["base"], *refs["sources"], *refs["peer_ledgers"], refs["ledger"]):
        verify_file(ref, root)
    if index.exists():
        with index.open() as stream:
            if json.load(stream) != value:
                raise ValueError("previous external trial index changed")
    else:
        atomic_json(index, value)
    atomic_json(external_path(log), {**value, "index": file_ref(index, root)})
    return value


def iter_trials(
    path: Path, *, verify_legacy: bool = False, offset: int = 0, include_external: bool = True
) -> Iterator[dict[str, Any]]:
    """Stream mixed records; compact refs are verified even before scope filters."""
    manifest = external_manifest(path) if include_external else None
    if not path.exists():
        return
    store = ParameterStore(path)
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not line.endswith(b"\n"):
                    raise partial_tail_error(path) from exc
                raise
            if not line.endswith(b"\n") and ("params_ref" in row or verify_legacy):
                raise partial_tail_error(path)
            yield expand_record(row, path, verify_legacy=verify_legacy, store=store)
    if manifest:
        root = external_path(path).parent
        for ref in manifest["sources"]:
            yield from iter_trials(
                ref_path(ref, root), verify_legacy=verify_legacy, include_external=False
            )


def external_ledger_floor(path: Path, ledger: Path) -> int:
    manifest = external_manifest(path)
    if manifest is None:
        return 0
    accounting = manifest["accounting"]
    root = external_path(path).parent
    if verify_file(accounting["ledger"], root) != ledger.resolve():
        raise ValueError("external ledger floor bound to a different ledger")
    return int(accounting["additional_ledger_rows"])
