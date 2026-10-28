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

import ast
import hashlib
import sys
import sysconfig
import types
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any

_THIRD_PARTY = ("site-packages", "dist-packages")
_STDLIB = Path(sysconfig.get_paths()["stdlib"]).resolve()
_digests: dict[str, str] = {}


def clear() -> None:
    """Forget the digests (a process that edits sources in place, as a test does)."""
    _digests.clear()
    _closures.clear()
    _stamps.clear()
    _imports.clear()


def closure_digest(module: types.ModuleType) -> str:
    """Hash of the source of ``module`` and of every first-party module its code can reach."""
    if module.__name__ not in _digests:
        _digests[module.__name__] = _closure_digest(module)
    return _digests[module.__name__]


def reachable_modules(module: types.ModuleType) -> dict[str, Path]:
    """Name -> source file of every first-party module ``module``'s functions can reach."""
    return _Walk(module).modules


def _closure_digest(module: types.ModuleType) -> str:
    """The runtime walk's modules, plus everything those modules import (statically)."""
    modules = dict(reachable_modules(module))
    for name in list(modules):
        modules |= import_closure(name, modules[name])
    return _digest(modules)


def _digest(modules: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(modules.items()):
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


_imports: dict[Path, tuple[tuple[str, int, tuple[str, ...]], ...]] = {}
_closures: dict[str, dict[str, Path]] = {}
_stamps: dict[str, str] = {}


def import_closure(name: str, path: Path) -> dict[str, Path]:
    """The module and every first-party module it imports, transitively, read off the source.

    Static, so it needs no running code: Numba stamps a cache entry while the module is still
    being imported, before any function exists to walk. It follows every ``import`` statement in
    the file (inside functions too), the parent packages that importing runs, and the submodules a
    ``from package import name`` can name. Stops at the standard library and installed packages.
    """
    if name not in _closures:
        found: dict[str, Path] = {}
        pending = [(name, path)]
        while pending:
            module, file = pending.pop()
            if module in found:
                continue
            found[module] = file
            package = module if file.name == "__init__.py" else module.rpartition(".")[0]
            for imported, level, names in _file_imports(file):
                base = _absolute(imported, level, package)
                if base is None:
                    continue
                for candidate in _candidates(base, names):
                    located = _locate(candidate)
                    if located is not None and candidate not in found:
                        pending.append((candidate, located))
        _closures[name] = found
    return _closures[name]


def module_stamp(name: str, path: Path) -> str:
    """The identity a first-party cached function is stamped with: its module's import closure."""
    key = f"{name}:{path}"
    if key not in _stamps:
        _stamps[key] = _digest(import_closure(name, path))
    return _stamps[key]


def _file_imports(path: Path) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    """(module, relative level, imported names) of every import statement in the file."""
    if path not in _imports:
        found: list[tuple[str, int, tuple[str, ...]]] = []
        for node in ast.walk(ast.parse(path.read_bytes(), str(path))):
            if isinstance(node, ast.Import):
                found += [(alias.name, 0, ()) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                found.append((node.module or "", node.level, tuple(a.name for a in node.names)))
        _imports[path] = tuple(found)
    return _imports[path]


def _absolute(name: str, level: int, package: str) -> str | None:
    if not level:
        return name
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, *([name] if name else [])]) or None


def _candidates(base: str, names: tuple[str, ...]) -> list[str]:
    """The modules an import runs: the module, its parents, and ``base.name`` for each name."""
    parts = base.split(".")
    found = [".".join(parts[: i + 1]) for i in range(len(parts))]
    return found + [f"{base}.{name}" for name in names if name != "*"]


def _locate(name: str) -> Path | None:
    """The first-party source file of an importable module, or None. Imports nothing: a module
    is looked up on ``sys.path`` the way the import system would, because this runs while a
    module is still being imported."""
    if name.split(".")[0] in sys.stdlib_module_names:
        return None
    module = sys.modules.get(name)
    file = getattr(module, "__file__", None) if module is not None else _find(name)
    return first_party_path(file)


def _find(name: str) -> str | None:
    parts = name.split(".")
    spec = PathFinder.find_spec(parts[0])
    for end in range(2, len(parts) + 1):
        if spec is None or spec.submodule_search_locations is None:
            return None
        spec = PathFinder.find_spec(".".join(parts[:end]), list(spec.submodule_search_locations))
    return spec.origin if spec is not None and spec.has_location else None


def _first_party(module: types.ModuleType) -> Path | None:
    """The module's own source file, or None for stdlib, built-in and installed packages."""
    if module.__name__.split(".")[0] in sys.stdlib_module_names:
        return None
    return first_party_path(getattr(module, "__file__", None))


def first_party_path(file: str | None) -> Path | None:
    """The resolved path of a ``.py`` source that is neither the standard library nor installed."""
    if not file or not str(file).endswith(".py"):
        return None
    path = Path(file).resolve()
    if any(part in _THIRD_PARTY for part in path.parts) or _STDLIB in path.parents:
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
            if _first_party(value) is None:
                return  # also keeps ``hasattr`` away from a library's deprecated lazy attributes
            if self._include(value):
                # A module reached for the first time contributes everything it defines (over-
                # inclusion is safe, a missed module is not): the functions it calls into other
                # modules are followed through their globals.
                self._defined(value)
            # Every visit adds the attributes this reference names (``mid.middle``), even when the
            # module was reached before without them.
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

    def _defined(self, module: types.ModuleType) -> None:
        for value in list(vars(module).values()):
            owner = getattr(value, "py_func", value)
            if getattr(owner, "__module__", None) == module.__name__ and isinstance(
                owner, (types.FunctionType, type)
            ):
                self._visit(value, frozenset())

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
