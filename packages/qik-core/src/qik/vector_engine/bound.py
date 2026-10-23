"""Internal source binding for Phase 2.2; generated modules own their disk cache.

Emit the shared matcher source, never a separately maintained matching algorithm.
The strategy is a module global throughout the call graph, not a callback argument.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import textwrap
from pathlib import Path
from typing import Any, cast

from . import account, compiled, kernel, protection


def write_bound_module(
    destination: Path, strategy_module: str, strategy_name: str, *, records: bool = False
) -> None:
    """Write an importable internal runner for ``step(state, params, feats, bar, pos)``.

    ``run`` has the prototype _run ABI without its first callback argument. State
    is a structured array with one independent record per parameter row. This is
    a code-generation primitive, not the public strategy decorator or Result API.
    Regenerate when engine or strategy source changes, before importing the module.
    records=True transports typed features with an int64 clock and appends final
    copied state rows to the raw output. The prototype ABI stays unchanged otherwise.
    """
    if not all(part.isidentifier() for part in strategy_module.split(".")):
        raise ValueError("strategy_module must be an importable module name")
    if not strategy_name.isidentifier():
        raise ValueError("strategy_name must be an identifier")

    def source(function: Any) -> ast.FunctionDef:
        original = getattr(function, "py_func", function)
        node = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
        assert isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list:
            assert isinstance(decorator, ast.Call)
            for keyword in decorator.keywords:
                if keyword.arg == "cache":
                    keyword.value = ast.Constant(True)
        return node

    class BindGlobals(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
            if isinstance(node.value, ast.Name) and node.value.id == "K":
                return ast.Name(node.attr.replace("_compiled_", "_"), ast.Load())
            return node

        def visit_Call(self, node: ast.Call) -> ast.expr:
            if isinstance(node.func, ast.Name) and node.func.id == "_venue_driver":
                return ast.Name("_venue_column", ast.Load())
            self.generic_visit(node)
            return node

    adapter = source(compiled._jitclass_step)
    adapter.name = "_bound_step"
    adapter.args.args[0].arg = "state"
    for node in ast.walk(adapter):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ):
            node.func = ast.Name("_strategy", ast.Load())
            node.args = [
                ast.Name(name, ast.Load()) for name in ("state", "params", "feats", "bar", "pos")
            ]

    if records:
        # Keep the clock int64 and each named feature's dtype through the ABI.
        class RecordFeatures(ast.NodeTransformer):
            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                if isinstance(node.value, ast.Name) and node.value.id == "feats":
                    return ast.parse("feats[0][i]", mode="eval").body
                return cast(ast.expr, self.generic_visit(node))

            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id == "CompiledBar":
                    return ast.Name("Bar", ast.Load())
                return node

        RecordFeatures().visit(adapter)
        for node in ast.walk(adapter):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_strategy"
            ):
                node.args[2] = ast.parse("_features_type(*feats[1:])", mode="eval").body

    decision = source(kernel._core_decision)
    for node in ast.walk(decision):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
        ):
            node.func = ast.Name("_bound_step", ast.Load())

    functions: list[ast.stmt] = [adapter, decision]
    for name in (
        "_venue_column_advance",
        "_column_advance",
        "_slots_column_advance",
        "_venue_column",
        "_column",
        "_slots_column",
    ):
        node = source(getattr(kernel, name))
        BindGlobals().visit(node)
        functions.append(node)
    runner = source(compiled._run)
    runner.name = "run"
    runner.args.args.pop(0)
    runner.body.insert(0, ast.parse("init_state = init_state.copy()").body[0])
    for node in ast.walk(runner):
        if isinstance(node, ast.Name) and node.id == "step":
            node.id = "_no_callback"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_compiled_"):
            node.attr = node.attr.replace("_compiled_", "_")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_initial_core_state"
        ):
            node.func = ast.Name("_bound_initial_state", ast.Load())
    BindGlobals().visit(runner)
    if records:
        for node in runner.body:
            if isinstance(node, ast.Return):
                assert isinstance(node.value, ast.Tuple)
                node.value.elts.append(ast.Name("init_state", ast.Load()))
    functions.append(runner)
    strategy = importlib.import_module(strategy_module)
    fingerprint = hashlib.sha256(
        "\n".join(
            inspect.getsource(module)
            for module in (kernel, compiled, protection, account, strategy)
        ).encode()
    ).hexdigest()
    imports = ", ".join(
        sorted(
            {
                node.id
                for function in functions
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and node.id in vars(kernel)
            }
        )
    )
    header = (
        "# Generated by qik.vector_engine.bound; regenerate before importing.\n"
        f"# Engine and strategy source: {fingerprint}\n"
        f"from {strategy_module} import {strategy_name} as _strategy\n"
        f"from qik.vector_engine.kernel import ({imports})\n"
        "from qik.vector_engine.compiled import CompiledBar, Position\n"
        "_no_callback = None\n"
        "@nb.njit(cache=True)\ndef _bound_initial_state(states, cores, j):\n    return states[j]\n"
    )
    if records:
        header += (
            "from qik.vector_engine.types import Bar, Position\n"
            "_features_type = _strategy.__strategy__.features\n"
            "_strategy = nb.njit(cache=True)(_strategy)\n"
        )
    text = (
        header
        + ast.unparse(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])))
        + "\n"
    )
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)
