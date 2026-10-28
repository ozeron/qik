"""The source a strategy's compiled code can reach, as one identity.

Numba compiles a strategy together with every helper it calls, wherever the helper is defined: a
``register_jitable`` function imported from another module, a closure, an ``@overload``
implementation. The cached engine and callback are keyed by source, so that source must include
all of it: an edited helper under an unedited strategy module must not be served a build made
from the old one.

``closure_digest(module)`` walks what the module's functions reference by name (globals,
attributes of the modules they import, closure cells, defaults, a dispatcher's ``py_func``, the
``@overload`` implementations registered for a function) and hashes the source file of every
module it reaches, in name order. It stops at the standard library and at installed third-party
packages (numpy, numba, pandas, ...): those are not edited in place, and their version is part of
the environment, not of the strategy. It is computed once per module per process.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Any

_THIRD_PARTY = ("site-packages", "dist-packages")
_digests: dict[str, str] = {}


def clear() -> None:
    """Forget the digests (a process that edits sources in place, as a test does)."""
    _digests.clear()


def closure_digest(module: types.ModuleType) -> str:
    """Hash of the source of ``module`` and of every first-party module its code can reach."""
    if module.__name__ not in _digests:
        _digests[module.__name__] = _closure_digest(module)
    return _digests[module.__name__]


def reachable_modules(module: types.ModuleType) -> dict[str, Path]:
    """Name -> source file of every first-party module ``module``'s functions can reach."""
    return _Walk(module).modules


def _closure_digest(module: types.ModuleType) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(reachable_modules(module).items()):
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _first_party(module: types.ModuleType) -> Path | None:
    """The module's own source file, or None for stdlib, built-in and installed packages."""
    name = module.__name__
    if name.split(".")[0] in sys.stdlib_module_names:
        return None
    file = getattr(module, "__file__", None)
    if not file or not file.endswith(".py"):
        return None
    path = Path(file).resolve()
    if any(part in _THIRD_PARTY for part in path.parts):
        return None
    return path


class _Walk:
    def __init__(self, root: types.ModuleType) -> None:
        self.modules: dict[str, Path] = {}
        self._seen: set[tuple[int, frozenset[str]]] = set()
        self._include(root)
        for value in list(vars(root).values()):
            self._visit(value, frozenset())

    def _include(self, module: types.ModuleType) -> bool:
        path = _first_party(module)
        if path is None:
            return False
        first = module.__name__ not in self.modules
        self.modules[module.__name__] = path
        return first

    def _visit(self, value: Any, names: frozenset[str]) -> None:
        key = (id(value), names)
        if key in self._seen:
            return
        self._seen.add(key)
        if isinstance(value, types.ModuleType):
            if self._include(value):
                # Only the attributes the referencing code names: ``helpers.direction``.
                for name in sorted(names):
                    if hasattr(value, name):
                        self._visit(getattr(value, name), frozenset())
            return
        py_func = getattr(value, "py_func", None)
        if isinstance(py_func, types.FunctionType):  # a numba dispatcher
            self._visit(py_func, names)
            return
        if isinstance(value, types.FunctionType):
            if self._first_party(value.__module__):
                self._function(value)
            return
        if isinstance(value, type):
            if not self._first_party(getattr(value, "__module__", None)):
                return
            self._owner(value.__module__)
            for member in vars(value).values():
                if isinstance(member, (types.FunctionType, staticmethod, classmethod)):
                    self._visit(getattr(member, "__func__", member), frozenset())
            return
        if isinstance(value, (tuple, list, frozenset, set)):
            for item in list(value):
                self._visit(item, names)
        elif isinstance(value, dict):
            for item in list(value.values()):
                self._visit(item, names)
        elif isinstance(value, types.MethodType):
            self._visit(value.__func__, names)
        elif hasattr(value, "func") and callable(value):  # functools.partial
            self._visit(value.func, names)

    @staticmethod
    def _first_party(name: str | None) -> bool:
        module = sys.modules.get(name) if name else None
        return module is not None and _first_party(module) is not None

    def _owner(self, name: str | None) -> None:
        module = sys.modules.get(name) if name else None
        if module is not None:
            self._include(module)

    def _function(self, function: types.FunctionType) -> None:
        self._owner(function.__module__)
        for overload in _overloads(function):
            self._visit(overload, frozenset())
        wrapped = getattr(function, "__wrapped__", None)
        if wrapped is not None:
            self._visit(wrapped, frozenset())
        names = _names(function.__code__)
        namespace = function.__globals__
        for name in names:
            if name in namespace:
                self._visit(namespace[name], names)
        for cell in function.__closure__ or ():
            try:
                self._visit(cell.cell_contents, names)
            except ValueError:  # an empty cell
                continue
        for default in (*(function.__defaults__ or ()), *(function.__kwdefaults__ or {}).values()):
            self._visit(default, names)


def _names(code: types.CodeType) -> frozenset[str]:
    """Every global or attribute name the code, and the code nested in it, refers to."""
    found = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            found |= _names(constant)
    return frozenset(found)


def _overloads(function: Any) -> list[Any]:
    """The ``@overload`` implementations Numba registered for ``function``, if any."""
    try:
        from numba.core.typing.templates import builtin_registry
    except ImportError:  # a Numba without the registry this reads: nothing to add
        return []
    found: list[Any] = []
    for value, typ in builtin_registry.globals:
        if value is not function:
            continue
        for template in getattr(typ, "templates", ()):
            implementation = getattr(template, "_overload_func", None)
            if implementation is not None:
                found.append(implementation)
    return found
