"""Content identities of running builders, independent of source-file reloads."""

from __future__ import annotations

import dis
import hashlib
import json
import struct
import types
from typing import Any

import numpy as np
import pandas as pd


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_identity(value: np.ndarray) -> dict[str, object]:
    if value.dtype.hasobject:
        raise TypeError("feature inputs cannot have object dtype")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def normalize(value: Any) -> Any:
    """Normalize explicit numpy scalar params before BOTH invocation and identity."""
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, tuple):
        return tuple(normalize(v) for v in value)
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    scalar(value)  # validate supported values, without losing container/scalar types
    return value


def scalar(value: Any) -> object:
    """Typed JSON identity; never conflate tuple/list or bool/int/float."""
    if isinstance(value, np.generic):
        return ["numpy", str(value.dtype), scalar(value.item())]
    if value is None:
        return ["none"]
    if type(value) in (str, bool, int):
        return [type(value).__name__, value]
    if type(value) is float:
        return ["float", struct.pack(">d", value).hex()]
    if type(value) in (tuple, list):
        return [type(value).__name__, [scalar(v) for v in value]]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return ["dict", [[k, scalar(v)] for k, v in value.items()]]
    raise TypeError(f"unsupported hidden dependency {type(value).__name__}; declare inputs/params")


def _codes(code: types.CodeType) -> list[types.CodeType]:
    return [
        code,
        *(nested for c in code.co_consts if isinstance(c, types.CodeType) for nested in _codes(c)),
    ]


def _constant(value: Any) -> object:
    if isinstance(value, types.CodeType):
        return code_identity(value)
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if value is Ellipsis:
        return ["ellipsis"]
    if isinstance(value, complex):
        return ["complex", value.real.hex(), value.imag.hex()]
    if isinstance(value, tuple):
        return ["tuple", [_constant(v) for v in value]]
    if isinstance(value, frozenset):
        return ["frozenset", sorted((_constant(v) for v in value), key=repr)]
    return scalar(value)


def code_identity(code: types.CodeType) -> object:
    """Executable semantics, excluding filenames, line numbers and line tables."""
    return {
        "bytecode": code.co_code.hex(),
        "consts": [_constant(v) for v in code.co_consts],
        "names": code.co_names,
        "varnames": code.co_varnames,
        "freevars": code.co_freevars,
        "cellvars": code.co_cellvars,
        "args": [code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount],
        "flags": code.co_flags,
        "exceptiontable": code.co_exceptiontable.hex(),
    }


def function_identity(fn: Any) -> object:
    """Follow running code (including nested code), referenced helpers and cells.

    Never inspect source on disk: it may differ from the loaded executable. NumPy,
    pandas and math are versioned libraries. This is a purity contract, not a sandbox.
    """
    seen: set[int] = set()

    def visit(obj: Any) -> object:
        if isinstance(obj, types.FunctionType):
            if id(obj) in seen:
                return [obj.__module__, obj.__qualname__, "recursive"]
            seen.add(id(obj))
            codes = _codes(obj.__code__)
            names = {name for code in codes for name in code.co_names}
            instructions = [i for code in codes for i in dis.get_instructions(code)]
            refs = {i.argval for i in instructions if i.opname in {"LOAD_GLOBAL", "LOAD_NAME"}}
            forbidden = {
                "open",
                "eval",
                "exec",
                "__import__",
                "globals",
                "locals",
                "getattr",
                "setattr",
            }
            if bad := refs & forbidden:
                raise TypeError(f"{obj.__name__}: dynamic dependency {sorted(bad)} unsupported")
            if any(
                i.opname in {"IMPORT_NAME", "STORE_GLOBAL", "DELETE_GLOBAL"}
                or (i.opname == "STORE_DEREF" and i.argval in obj.__code__.co_freevars)
                for i in instructions
            ):
                raise TypeError(
                    f"{obj.__name__}: imports/global writes inside builders unsupported"
                )
            cells = {
                name: cell.cell_contents
                for name, cell in zip(obj.__code__.co_freevars, obj.__closure__ or (), strict=True)
            }
            namespace = {**obj.__globals__, **cells}
            deps: dict[str, object] = {}
            # Only LOAD_GLOBAL/LOAD_NAME resolve module bindings: an attribute such
            # as x.rolling must not capture an unrelated global named rolling.
            for name in sorted((refs | cells.keys()) & namespace.keys()):
                value = namespace[name]
                if isinstance(value, types.ModuleType):
                    root = value.__name__.split(".")[0]
                    if root in {"numpy", "pandas", "math"}:
                        deps[name] = [
                            value.__name__,
                            np.__version__
                            if root == "numpy"
                            else pd.__version__
                            if root == "pandas"
                            else "stdlib",
                        ]
                    else:
                        attrs = names & vars(value).keys()
                        if not attrs:
                            raise TypeError(f"{obj.__name__}: unresolved module {name}")
                        deps[name] = {a: visit(vars(value)[a]) for a in sorted(attrs)}
                else:
                    deps[name] = visit(value)
            return {
                "code": code_identity(obj.__code__),
                "deps": deps,
                "defaults": visit(obj.__defaults__),
                "kwdefaults": visit(obj.__kwdefaults__),
            }
        return scalar(obj)

    return visit(fn)


def backend_identity() -> object:
    """Numerical backend build/selection, independent of checkout/install paths.

    Accelerate reports no library version: the OS build binds that implementation.
    Directory hints in NumPy's build report describe build locations, not semantics.
    """
    import importlib.metadata
    import os
    import platform

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def content(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: content(v)
                for k, v in value.items()
                if k != "Python Information" and "directory" not in k.lower()
            }
        if isinstance(value, list):
            return [content(v) for v in value]
        return value

    return {
        "numpy_build": content(np.__config__.show(mode="dicts")),
        "os_build": [platform.system(), platform.release(), platform.version(), platform.machine()],
        "bottleneck": version("bottleneck"),
        "numexpr": version("numexpr"),
        "pandas_compute": {
            name: pd.get_option("compute." + name) for name in ("use_bottleneck", "use_numexpr")
        },
        "numpy_errors": np.geterr(),
        "selection": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_CORETYPE",
                "NPY_DISABLE_CPU_FEATURES",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "MKL_CBWR",
                "MKL_DEBUG_CPU_TYPE",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMEXPR_MAX_THREADS",
            )
        },
    }
