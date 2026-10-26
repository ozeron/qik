"""One cacheable matcher ABI, with a lossless, schema-specific strategy callback.

Records cross the boundary as bytes; feature lanes retain float64/int64/bool.
No strategy, schema or runtime rule value participates in the engine signature.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, get_type_hints
from uuid import uuid4

import numba as nb
import numpy as np
from numba.extending import intrinsic
from numba.typed import List

from . import engine
from .codegen import _ADAPTER_SOURCE, write_fixed_module
from .schema import BoolArray, FloatArray, IntArray, StrategySpec

ABI = nb.void(nb.int64, nb.int64)
FUNCTION = nb.types.FunctionType(ABI)
_modules: dict[tuple[Any, ...], Any] = {}


@intrinsic
def _pointer(typing_context: Any, address: Any) -> Any:
    """Rebuild a borrowed pointer; the run boundary retains every array owner."""
    signature = nb.types.voidptr(address)

    def codegen(context: Any, builder: Any, sig: Any, args: Any) -> Any:
        return builder.inttoptr(args[0], context.get_value_type(sig.return_type))

    return signature, codegen


def _load(source: str, directory: Path, prefix: str) -> Any:
    digest = hashlib.sha256(source.encode()).hexdigest()[:24]
    name = f"_qik_{prefix}_{digest}"
    path = directory / f"{name}.py"
    if not path.exists():
        temporary = directory / f"{name}.{uuid4().hex}.tmp"
        temporary.write_text(source)
        temporary.replace(path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _callback_source(step: Any) -> str:
    spec: StrategySpec = step.__strategy__
    hints = get_type_hints(spec.features)
    dtypes = {FloatArray: "np.float64", IntArray: "np.int64", BoolArray: "np.bool_"}
    fields = [
        f"nb.carray(_pointer(feats[1][{index}]), len(feats[0]), dtype={dtypes[kind]})"
        for index, kind in enumerate(hints.values())
    ]
    tree = ast.parse(_ADAPTER_SOURCE)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    function.name = "callback"
    text = ast.unparse(tree)
    text = text.replace(
        "_strategy(state, params, feats, bar, pos)",
        "_strategy(nb.carray(_pointer(state), 1, dtype=_state_dtype)[0], "
        "nb.carray(_pointer(params), 1, dtype=_params_dtype)[0], "
        + "_features_type(*lanes), bar, pos)",
    )
    # Catch inside the native callback: the FunctionType C wrapper cannot carry
    # a Python exception. Return a status before it can become unraisable.
    callback_tree = ast.parse(text)
    callback = callback_tree.body[0]
    assert isinstance(callback, ast.FunctionDef)
    callback.args.args = [ast.arg(arg="context"), ast.arg(arg="i")]
    # This context is filled once per configuration by the shared engine.
    setup = ast.parse(
        """
ctx = nb.carray(_pointer(context), 13, dtype=np.int64)
state, params = ctx[0], ctx[1]
feats = (nb.carray(_pointer(ctx[2]), ctx[11], dtype=np.int64), nb.carray(_pointer(ctx[3]), FEATURE_COUNT, dtype=np.int64))
o = nb.carray(_pointer(ctx[4]), ctx[11], dtype=np.float64)[i]
h = nb.carray(_pointer(ctx[5]), ctx[11], dtype=np.float64)[i]
lo = nb.carray(_pointer(ctx[6]), ctx[11], dtype=np.float64)[i]
c = nb.carray(_pointer(ctx[7]), ctx[11], dtype=np.float64)[i]
v = nb.carray(_pointer(ctx[8]), ctx[11], dtype=np.float64)[i]
view = nb.carray(_pointer(ctx[9]), 14, dtype=np.float64)
out = nb.carray(_pointer(ctx[10]), 31, dtype=np.float64)
account = nb.carray(_pointer(ctx[12]), 6, dtype=np.float64)
""".replace("FEATURE_COUNT", str(len(hints)))
    ).body
    # Simple local loads avoid Python 3.13 LIST_TO_TUPLE for wide constructors.
    setup += ast.parse(
        "\n".join(f"lane_{index} = {field}" for index, field in enumerate(fields))
    ).body
    groups = [
        "("
        + ", ".join(f"lane_{index}" for index in range(start, min(start + 16, len(fields))))
        + ",)"
        for start in range(0, len(fields), 16)
    ]
    setup += ast.parse("lanes = " + (" + ".join(groups) if groups else "()")).body
    callback.body = setup + [
        ast.Try(
            body=callback.body,
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name("Exception", ast.Load()),
                    name=None,
                    body=ast.parse("account[2] = 11\nreturn").body,
                )
            ],
            orelse=[],
            finalbody=[],
        ),
        ast.Return(),
    ]
    text = ast.unparse(ast.fix_missing_locations(callback_tree))
    # The callback identity carries the strategy source on top of the engine identity:
    # schema/transport definitions and every engine module are already in the latter.
    import importlib

    strategy_module = importlib.import_module(step.__module__)
    identity = engine.fingerprint(strategy_module)
    return (
        f"# Strategy source: {identity}\n"
        f"# Schemas: {spec.state.dtype.descr!r}, {spec.params.dtype.descr!r}, {hints!r}\n"
        "import numba as nb\nimport numpy as np\n"
        "from qik.vector_engine.types import Bar, Position\n"
        "from qik.vector_engine.fixed import _pointer\n"
        f"from {step.__module__} import {step.__name__} as _strategy\n"
        "_features_type = _strategy.__strategy__.features\n"
        "_state_dtype = _strategy.__strategy__.state.dtype\n"
        "_params_dtype = _strategy.__strategy__.params.dtype\n"
        # Empty Records cannot be lowered by Numba.
        "if not _state_dtype.names: _state_dtype = np.dtype([('_empty', 'i1')])\n"
        "if not _params_dtype.names: _params_dtype = np.dtype([('_empty', 'i1')])\n"
        "_strategy = nb.njit(cache=True)(_strategy)\n" + text + "\n"
    )


def _writable(array: np.ndarray) -> np.ndarray:
    """Borrow mutable storage through readonly views; copy immutable bytes safely."""
    if array.flags.writeable:
        return array
    view = array.view()
    try:
        view.flags.writeable = True
    except ValueError:
        return array.copy()
    return view


class FixedRunner:
    def __init__(self, engine: Any, callback: Any) -> None:
        self.engine = engine
        self.callback = callback
        self.functions = List.empty_list(FUNCTION)
        self.functions.append(callback)

    def run(
        self,
        states: np.recarray,
        params: np.recarray,
        feats: tuple[np.ndarray, ...],
        o: np.ndarray,
        h: np.ndarray,
        lo: np.ndarray,
        c: np.ndarray,
        volume: np.ndarray,
        days: np.ndarray,
        ends: np.ndarray,
        rules: np.ndarray,
        economic: np.ndarray,
        cash: float,
        slots: bool,
        layout: np.ndarray,
        level_rows: np.ndarray,
        store: np.ndarray,
        trades: tuple[np.ndarray, ...],
        blank: np.ndarray,
        blank_pairs: np.ndarray,
        risk_row: int,
        cores: None,
        masks: np.ndarray,
        sessions: np.ndarray,
        expiry: np.ndarray,
        pair: bool,
    ) -> tuple[Any, ...]:
        # Once per run, never once per bar. Copy the state at the same boundary as
        # the bound runner; views in the callback mutate these bytes in place.
        state_bytes = states.view(np.uint8).reshape(len(states), states.dtype.itemsize)
        params_bytes = params.view(np.uint8).reshape(len(params), params.dtype.itemsize)
        # Borrow caller storage, including overlapping views. Owners remain alive
        # through invoke; no normalization may copy feature lanes here.
        owners = tuple(feats[1:])
        if any(not a.flags.c_contiguous for a in owners):
            raise ValueError("features must be C-contiguous arrays; copying breaks aliasing")
        owners = tuple(_writable(a) for a in owners)
        addresses = np.array([a.ctypes.data for a in owners], dtype=np.int64)
        features = (np.require(feats[0], dtype=np.int64, requirements=["C", "W"]), addresses)
        # Normalize layout, mutability and scalar types: one engine signature even
        # for readonly bars, missing expiry masks and empty feature schemas.
        floats = [
            _writable(np.require(a, dtype=np.float64, requirements=["C"]))
            for a in (o, h, lo, c, volume, rules, economic)
        ]
        flags = [
            np.array(a, dtype=np.bool_, order="C", copy=True)
            for a in (days, ends, sessions, expiry)
        ]
        raw = self.engine.invoke(
            self.functions,
            state_bytes,
            params_bytes,
            features,
            *floats[:5],
            *flags[:2],
            *floats[5:],
            float(cash),
            bool(slots),
            layout,
            level_rows,
            store,
            trades,
            blank,
            blank_pairs,
            int(risk_row),
            None,
            np.array(masks, dtype=np.int8, order="C", copy=True),
            *flags[2:],
            bool(pair),
        )
        final = raw[-1].reshape(-1).view(states.dtype).view(np.recarray)
        return (*raw[:-1], final)


def runner(step: Any) -> FixedRunner:
    if step.__module__ == "__main__" or "<locals>" in step.__qualname__:
        raise ValueError("strategy must be defined at module scope in an importable module")
    directory = Path(os.environ.get("QIK_VECTOR_CACHE", Path.home() / ".cache/qik/vector_engine"))
    schema: StrategySpec = step.__strategy__
    callback_source = _callback_source(step)
    key = (step, schema.state, schema.params, schema.features, directory.resolve(), callback_source)
    if key not in _modules:
        directory.mkdir(parents=True, exist_ok=True)
        draft = directory / f"draft_fixed_{uuid4().hex}.py"
        try:
            write_fixed_module(draft)
            source = draft.read_text()
        finally:
            draft.unlink(missing_ok=True)
        tree = ast.parse(source)
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        names = [n.arg for n in run.args.args[1:]]
        source += (
            "\n@nb.njit(cache=True)\ndef invoke(callbacks, "
            + ", ".join(names)
            + "):\n    return run(callbacks[0], "
            + ", ".join(names)
            + ")\n"
        )
        engine = _load(source, directory, "fixed")
        callback = _load(callback_source, directory, "callback").callback
        callback.compile(ABI)
        _modules[key] = FixedRunner(engine, callback)
    return _modules[key]
