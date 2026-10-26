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
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Any

from . import batch, codes, decision, fills, levels, pairs, single, slots, venue

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
# Modules outside this package whose source the matchers bind.
SHARED_MODULES = ("compiled", "protection", "account", "straddle", "types")
# Names the facade re-binds under another identifier: alias -> (module, attribute).
_ALIASES = {"_contracts_for": ("account", "quantity")}


def owners() -> dict[str, ModuleType]:
    """Exported name -> the one engine module that defines it."""
    return {name: module for module in ENGINE_MODULES for name in module.__all__}


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


def sources(*extra: Any) -> list[str]:
    """Source of every engine module, the shared modules and ``extra`` (e.g. a strategy)."""
    return [inspect.getsource(module) for module in (*_modules(), *extra)]


def fingerprint(*extra: Any) -> str:
    """Content identity of the engine: a source change anywhere invalidates it."""
    return hashlib.sha256("\n".join(sources(*extra)).encode()).hexdigest()


def _modules() -> tuple[Any, ...]:
    import importlib

    package = __name__.rpartition(".")[0]
    shared = tuple(importlib.import_module(f"{package}.{name}") for name in SHARED_MODULES)
    return (sys.modules[__name__], *ENGINE_MODULES, *shared)
