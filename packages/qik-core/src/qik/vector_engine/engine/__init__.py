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
Numba's own order, and a function no locator can cache is compiled without a disk cache
rather than stamped by mtime. A ``NUMBA_CACHE_LOCATOR_CLASSES`` override replaces that list
entirely, so it is refused at import.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
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
_uncached: set[int] = set()  # id(py_func) of engine functions no locator could cache


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
    configured: str = config.CACHE_LOCATOR_CLASSES  # ty: ignore[unresolved-attribute]
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
        # Nowhere is writable (read-only tree and no usable user-wide cache). Compiling without
        # a cache is correct and slower; an mtime-stamped entry is neither. ``_disable_uncached``
        # switches these dispatchers' caches off as soon as the package has imported.
        return _UncachedLocator(py_func, py_file)


def _content_locator(base: Any) -> Any:
    """``base``, with the engine's content standing in for the defining file's mtime and size."""

    class ContentLocator(base):  # type: ignore[misc, valid-type]
        _qik_engine = True

        def get_source_stamp(self) -> str:
            return fingerprint()

    ContentLocator.__name__ = ContentLocator.__qualname__ = f"Engine{base.__name__}"
    return ContentLocator


class _UncachedLocator:
    """Placeholder locator for an engine file no locator can cache.

    Numba builds a ``Cache`` per ``cache=True`` dispatcher and needs a locator object to do
    it. ``_disable_uncached`` turns those caches off right after import, so this path is never
    read or written: the functions compile on every run instead. Raising in ``from_function``
    would fail the import itself, which a read-only filesystem does not warrant.
    """

    def __init__(self, py_func: Any, py_file: str) -> None:
        self._py_file = py_file
        self._lineno = py_func.__code__.co_firstlineno
        self._cache_path = str(_ENGINE_DIR / "__pycache__")
        _uncached.add(id(py_func))

    def ensure_cache_path(self) -> None:
        raise PermissionError(f"engine module {self._py_file} has no writable Numba cache")

    def get_cache_path(self) -> str:
        return self._cache_path

    def get_source_stamp(self) -> str:
        return fingerprint()

    def get_disambiguator(self) -> str:
        return str(self._lineno)


def _disable_uncached() -> None:
    """Compile the engine functions that found no writable cache without a disk cache."""
    if not _uncached:
        return
    from numba.core.dispatcher import Dispatcher

    for module in ENGINE_MODULES:
        for value in vars(module).values():
            if isinstance(value, Dispatcher) and id(value.py_func) in _uncached:
                value._cache.disable()
    _uncached.clear()


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

_disable_uncached()  # noqa: E402  (their caches exist only once the modules above have imported)


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
