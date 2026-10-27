"""Numba engine modules behind the ``qik.vector_engine.kernel`` facade.

Module map, in dependency order (an import only ever points at an earlier entry, so
there is no cycle):

    codes     settings/rule/error/level codes, the float64 settings record, CoreContext
    fills     fill policy helpers: direction and exit masks, path order, entry expiry
    levels    level-store layout, level/risk writes, optional-column accessors
    pairs     straddle pair legs: pair reads, resting/level lookups, lifetime triggers
    decision  _core_decision: the strategy callback into the level store and account
    venue     the venue settlement matcher (command queue, protection, fills)
    single    the single-order matcher and its dispatcher
    slots     the slots matcher (trigger/stop/target updates, straddle legs)
    batch     run_columns, the parallel column runner and its Python-boundary ABI

``venue``/``single``/``slots`` all sit on ``decision``; ``batch`` runs all three.
``qik.vector_engine.kernel`` re-exports every name for compatibility.

The matchers cache to disk (``cache=True``) and Numba keys a cache entry by the source
stamp of the file that *defines* the function. Splitting the one ``kernel`` module into
modules that bind each other's constants would leave ``pairs``' entry valid while
``codes`` changed under it. ``_install_cache_locator`` therefore keys every file in this
package by ``fingerprint()`` -- the content of the whole engine -- so the split recompiles
as one unit, exactly as the single module did.

The protection has to survive Numba's fallbacks, not just its first choice: a read-only
checkout (a ``.run`` worktree, site-packages) makes the in-tree ``__pycache__`` unwritable,
and Numba then moves on to its user-wide locator, whose stamp is ``[mtime, size]`` again.
So engine files get a content-stamped twin of *every* locator Numba can pick, tried in
Numba's own order, and a function no locator can cache is saved into a directory private to
the process under a stamp no other process can reproduce, so its entry is never read back:
neither an mtime stamp nor a ``PermissionError`` on the first save (a dispatcher created
after this module imported never saw an import-time sweep). A
``NUMBA_CACHE_LOCATOR_CLASSES`` override replaces that list entirely, so it is refused at
import -- from ``os.environ`` as well as from ``numba.config``, which only snapshots the
environment when numba imports.
"""

from __future__ import annotations

import atexit
import hashlib
import inspect
import os
import secrets
import shutil
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

_ENGINE_DIR = Path(__file__).resolve().parent
# Dependency order. ``fingerprint`` keys off these names, so a new engine module must join
# both this tuple and ``ENGINE_MODULES`` below.
_ENGINE_MODULE_NAMES = (
    "codes",
    "fills",
    "levels",
    "pairs",
    "decision",
    "venue",
    "single",
    "slots",
    "batch",
)
# Modules outside this package whose source the matchers bind.
SHARED_MODULES = ("compiled", "protection", "account", "straddle", "types")
# Names the facade re-binds under another identifier: alias -> (module, attribute).
_ALIASES = {"_contracts_for": ("account", "quantity")}

_engine_digest: str | None = None
# Numba's own cache locators, in Numba's order (``numba.core.caching.CacheImpl``). Each one
# needs a content-stamped twin: a locator left unstamped is a locator that can answer for an
# engine file with the defining file's mtime. ``_install_cache_locator`` checks this against
# the list the installed numba actually ships.
_LOCATOR_NAMES = (
    "UserProvidedCacheLocator",
    "InTreeCacheLocator",
    "UserWideCacheLocator",
    "IPythonCacheLocator",
    "ZipCacheLocator",
)
_content_variants: tuple[Any, ...] = ()
# (directory, stamp) for engine files no locator can cache; created on first use, per process.
_throwaway: tuple[str, str] | None = None


def fingerprint(*extra: Any) -> str:
    """Content identity of the engine: a source change anywhere invalidates it.

    ``extra`` modules (a strategy) extend the identity to their own source.
    """
    if not extra:
        return _digest()
    added = "\n".join(inspect.getsource(module) for module in extra)
    return hashlib.sha256("\n".join((_digest(), added)).encode()).hexdigest()


def _engine_files() -> tuple[Path, ...]:
    """The source files compiled code in this package can see, named from the ``qik`` root."""
    package = _ENGINE_DIR.parent
    return (
        _ENGINE_DIR / "__init__.py",
        *(_ENGINE_DIR / f"{name}.py" for name in _ENGINE_MODULE_NAMES),
        *(package / f"{name}.py" for name in SHARED_MODULES),
    )


def _digest() -> str:
    """``fingerprint()`` with no extras, computed once per process from the files themselves.

    Files, not imported modules: a dispatcher is stamped while its module is still
    executing, so the modules that come after it in the map are not importable yet. Paths
    are relative to the ``qik`` root, so a copy of the package keeps the same identity.
    """
    global _engine_digest
    if _engine_digest is None:
        qik = _ENGINE_DIR.parent.parent
        digest = hashlib.sha256()
        for path in _engine_files():
            digest.update(path.relative_to(qik).as_posix().encode())
            digest.update(path.read_bytes())
        _engine_digest = digest.hexdigest()
    return _engine_digest


def _configured_locator_classes(config: Any) -> str:
    """``NUMBA_CACHE_LOCATOR_CLASSES`` if it is set, from the environment or numba's snapshot.

    ``numba.config`` is populated from the environment when numba imports, so a caller that
    imports numba, sets the variable and only then imports the engine leaves the snapshot
    empty -- and a later ``numba.core.config.reload_config()`` (or any reload) would hand Numba
    the override this guard never saw. Read both, so the refusal does not depend on import
    order.
    """
    return os.environ.get("NUMBA_CACHE_LOCATOR_CLASSES") or str(config.CACHE_LOCATOR_CLASSES or "")


def _install_cache_locator() -> None:
    """Key Numba's disk cache for this package by ``fingerprint()`` instead of file mtime.

    Installed before the first dispatcher below is decorated. ``EngineCacheLocator`` claims
    files inside ``vector_engine/engine`` only: every other module keeps the locator it had.
    For an engine file it offers a content-stamped twin of each of Numba's own locators, in
    Numba's order, so the cache path is still the one Numba would have used (its
    ``NUMBA_CACHE_DIR`` root when set, the in-tree ``__pycache__``, then the user-wide
    directory) and only the stamp differs. Right behind the test runtime's content locator,
    which must keep winning in tests.
    """
    from numba.core import caching, config

    # ``NUMBA_CACHE_LOCATOR_CLASSES`` replaces ``CacheImpl._locator_classes`` outright, so the
    # engine's locator is never consulted and Numba may answer an engine file from an
    # mtime-stamped entry. Refuse rather than serve stale machine code.
    configured = _configured_locator_classes(config)
    if configured:
        raise RuntimeError(
            "qik.vector_engine.engine keys its Numba disk cache by engine content, but "
            f"NUMBA_CACHE_LOCATOR_CLASSES={configured!r} replaces Numba's locator list, so an "
            "engine file could be served from an mtime-stamped entry -- stale machine code "
            "after an engine edit. Unset it before importing qik.vector_engine.engine."
        )
    # Numba's stub types its locator list as the five classes it ships; a plugin locator (what
    # ``NUMBA_CACHE_LOCATOR_CLASSES`` exists for) is outside that union by construction.
    locators: Any = caching.CacheImpl._locator_classes
    if any(getattr(locator, "_qik_engine", False) for locator in locators):
        return
    # Numba's own list, in its own order, minus locators that are already content-keyed (the
    # test runtime's). An unlisted locator would be a fallback without a content twin, so a
    # numba whose list this does not match stops here instead of risking a stale hit.
    defaults = tuple(locator for locator in locators if not getattr(locator, "_qik_content", False))
    names = [locator.__name__ for locator in defaults]
    if names != list(_LOCATOR_NAMES):
        raise RuntimeError(
            "qik.vector_engine.engine content-stamps every Numba cache locator it can fall "
            f"back to, but this numba offers {names}, not {list(_LOCATOR_NAMES)}. Teach "
            "_LOCATOR_NAMES (and so _content_locator) the new list before importing the "
            "engine: an unstamped fallback can serve stale machine code."
        )
    variants = tuple(_content_locator(locator) for locator in defaults)
    global _content_variants
    _content_variants = variants

    tested = [index for index, cls in enumerate(locators) if getattr(cls, "_qik_content", False)]
    locators.insert(tested[0] + 1 if tested else 0, EngineCacheLocator)


class EngineCacheLocator:
    """Content-stamped stand-in for whichever of Numba's locators can cache an engine file.

    Numba asks each locator in turn and takes the first that returns one; this offers every
    one of them a content-stamped twin, first usable wins, so a read-only checkout falls back
    to the user-wide directory without falling back to an mtime stamp.
    """

    _qik_engine = True

    @classmethod
    def from_function(cls, py_func: Any, py_file: str) -> Any:
        if _ENGINE_DIR not in Path(py_file).resolve().parents:
            return None
        for variant in _content_variants:
            locator = variant.from_function(py_func, py_file)
            if locator is not None:
                return locator
        # Nowhere is writable (read-only tree and no usable user-wide cache). An mtime-stamped
        # entry is not an option; a throwaway entry no other process can read is, and it needs
        # no sweep of the dispatchers that exist at import.
        return _ThrowawayLocator(py_func, py_file)


def _content_locator(base: Any) -> Any:
    """``base``, with the engine's content standing in for the defining file's mtime and size."""

    class ContentLocator(base):  # type: ignore[misc, valid-type]
        _qik_engine = True

        def get_source_stamp(self) -> str:
            return fingerprint()

    ContentLocator.__name__ = ContentLocator.__qualname__ = f"Engine{base.__name__}"
    return ContentLocator


def _throwaway_cache() -> tuple[str, str]:
    """A writable directory and a stamp no other process can reproduce.

    The fallback exists for a machine where neither the tree nor the user-wide cache is
    writable, so this must not fail -- and it must not leave anything a later process can
    read. ``mkdtemp`` gives the process a directory of its own, removed at exit, and the stamp
    carries the engine fingerprint, the pid and a random token, so an index written here can
    only ever be compared against itself. Numba compares the recorded stamp with the one the
    locator returns and treats a mismatch as an empty index: a save is a harmless write, a
    read by any other process is a miss. Within this process the source is the same source, so
    a hit is the machine code just compiled, never an older engine's. Reloading engine
    modules inside a running process is not supported (``fingerprint`` is fixed per process).
    A forked child forgets the parent's directory and makes its own; only the creating
    process removes a directory, so a child's exit cannot pull it from under the parent.
    """
    global _throwaway
    if _throwaway is None:
        directory = tempfile.mkdtemp(prefix="qik-engine-cache-")
        atexit.register(_remove_throwaway, directory, os.getpid())
        _throwaway = (directory, f"{fingerprint()}-{os.getpid()}-{secrets.token_hex(8)}")
    return _throwaway


def _remove_throwaway(directory: str, owner: int) -> None:
    if os.getpid() == owner:
        shutil.rmtree(directory, True)


def _forget_throwaway() -> None:
    global _throwaway
    _throwaway = None


os.register_at_fork(after_in_child=_forget_throwaway)


class _ThrowawayLocator:
    """Stand-in locator for an engine file no locator can cache.

    Numba builds a ``Cache`` per ``cache=True`` dispatcher and needs a locator object to do
    it, asking for the cache path and the source stamp there and then. Raising in
    ``from_function`` would fail the import itself, which a read-only filesystem does not
    warrant. Saving to ``_throwaway_cache()`` instead costs a write into the process's own
    temporary directory and can never answer a later process. This is why nothing has to
    sweep the dispatchers that exist after import: one created later behaves the same.
    """

    def __init__(self, py_func: Any, py_file: str) -> None:
        self._lineno = py_func.__code__.co_firstlineno

    def ensure_cache_path(self) -> None:
        """Re-create the throwaway directory if a forked child's exit removed it."""
        os.makedirs(_throwaway_cache()[0], exist_ok=True)

    def get_cache_path(self) -> str:
        return _throwaway_cache()[0]

    def get_source_stamp(self) -> str:
        return _throwaway_cache()[1]

    def get_disambiguator(self) -> str:
        return str(self._lineno)


_install_cache_locator()  # noqa: E402  (before the first ``cache=True`` dispatcher exists)

from . import batch, codes, decision, fills, levels, pairs, single, slots, venue  # noqa: E402

ENGINE_MODULES: tuple[ModuleType, ...] = (
    codes,
    fills,
    levels,
    pairs,
    decision,
    venue,
    single,
    slots,
    batch,
)


def imports(names: Iterable[str]) -> list[str]:
    """Import statements bringing ``names`` in from their defining modules.

    The generated matcher is a standalone module, so every global it reads must be
    imported explicitly; ``kernel`` must not be needed to compile it.
    """
    pending = set(names)
    lines: list[str] = []
    for name, statement in (("nb", "import numba as nb"), ("np", "import numpy as np")):
        if name in pending:
            pending.discard(name)
            lines.append(statement)
    for module in ENGINE_MODULES:
        lines.append(_import_line(module.__name__, pending, module.__all__))
    package = __name__.rpartition(".")[0]
    for name in SHARED_MODULES:
        shared = sys.modules.get(f"{package}.{name}")
        if shared is not None:
            lines.append(_import_line(shared.__name__, pending, tuple(vars(shared))))
    for alias, (name, attribute) in _ALIASES.items():
        if alias in pending:
            pending.discard(alias)
            lines.append(f"from {package}.{name} import {attribute} as {alias}")
    if pending:
        raise ValueError(f"no defining module for matcher globals: {sorted(pending)}")
    return [line for line in lines if line]


def _import_line(module: str, pending: set[str], available: Iterable[str]) -> str:
    chosen = sorted(pending & set(available))
    pending -= set(chosen)
    if not chosen:
        return ""
    line = f"from {module} import {', '.join(chosen)}"
    if len(line) <= 100:
        return line
    return f"from {module} import (\n" + "".join(f"    {name},\n" for name in chosen) + ")"
