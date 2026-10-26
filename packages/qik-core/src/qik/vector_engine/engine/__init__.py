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

    Installed before the first dispatcher below is decorated, and only for files inside
    ``vector_engine/engine``: every other module keeps Numba's own locator. The cache path
    is the one Numba would have used (its own ``NUMBA_CACHE_DIR`` directory when set, the
    in-tree ``__pycache__`` otherwise); only the stamp differs. The test runtime's
    ``_qik_content`` locator is content-keyed too, so this slots in right behind it.
    """
    from numba.core import caching, config

    # Numba's stub types its locator list as the five classes it ships; a plugin locator (what
    # ``NUMBA_CACHE_LOCATOR_CLASSES`` exists for) is outside that union by construction.
    locators: Any = caching.CacheImpl._locator_classes
    if any(getattr(locator, "_qik_engine", False) for locator in locators):
        return
    # The directory Numba would pick: its own cache root when ``NUMBA_CACHE_DIR`` was set at
    # import (``config`` reads it then), this package's ``__pycache__`` otherwise.
    cache_root = config.CACHE_DIR  # ty: ignore[unresolved-attribute] -- numba's stub omits it
    base: Any = caching.UserProvidedCacheLocator if cache_root else caching.InTreeCacheLocator
    engine_dir = _ENGINE_DIR

    class EngineCacheLocator(base):  # type: ignore[misc, valid-type]
        _qik_engine = True

        def get_source_stamp(self) -> str:
            return fingerprint()

        @classmethod
        def from_function(cls, py_func: Any, py_file: str) -> Any:
            if engine_dir not in Path(py_file).resolve().parents:
                return None
            return super().from_function(py_func, py_file)

    # Right behind the test runtime's content locator, which must keep winning in tests.
    tested = [index for index, cls in enumerate(locators) if getattr(cls, "_qik_content", False)]
    locators.insert(tested[0] + 1 if tested else 0, EngineCacheLocator)


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
